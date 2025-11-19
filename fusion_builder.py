import torch
import torch.nn as nn
import torch.nn.functional as F

class CostVolumeBuilder(nn.Module):
    """
    Builds a differentiable cost volume by warping source features into
    the reference view across all depth hypotheses.

    This version outputs a [B, C, D, H, W] variance-based cost volume
    to match the 3D cost regularization network (32 input channels).
    """

    def __init__(self):
        super().__init__()

    def forward(self, ref_feat, src_feats, intrinsics, extrinsics, depth_values):
        """
        ref_feat:    [B, C, H, W]       (reference view features)
        src_feats:   list of [B, C, H, W] (source view features)
        intrinsics:  [B, N, 3, 3]
        extrinsics:  [B, N, 4, 4]
        depth_values:[B, D]
        """
        B, C, H, W = ref_feat.shape
        D = depth_values.shape[1]
        Nsrc = len(src_feats)

        # reference K, T
        K_ref = intrinsics[:, 0]   # [B, 3, 3]
        T_ref = extrinsics[:, 0]   # [B, 4, 4]

        # cost volume: [B, C, D, H, W]  (32 channels expected by 3D UNet)
        cost_volume = ref_feat.new_zeros(B, C, D, H, W)

        # pixel grid in ref image
        y, x = torch.meshgrid(
            torch.arange(0, H, device=ref_feat.device),
            torch.arange(0, W, device=ref_feat.device),
            indexing='ij'
        )
        ones = torch.ones_like(x)
        pix = torch.stack((x, y, ones), dim=0).float()     # [3, H, W]
        pix = pix.unsqueeze(0).repeat(B, 1, 1, 1)          # [B, 3, H, W]

        # precompute K^-1 * pix (unit ray directions in ref cam)
        Kinv_ref = torch.inverse(K_ref)                    # [B, 3, 3]
        rays = (Kinv_ref @ pix.view(B, 3, -1)).view(B, 3, H, W)  # [B, 3, H, W]

        for d in range(D):
            depth = depth_values[:, d].view(B, 1, 1, 1)    # [B,1,1,1]
            # 3D points in ref camera coordinates
            cam_pts = rays * depth                         # [B, 3, H, W]

            # to homogeneous and then to world
            ones_ = torch.ones((B, 1, H, W),
                               device=ref_feat.device,
                               dtype=ref_feat.dtype)
            cam_hom = torch.cat((cam_pts, ones_), dim=1).view(B, 4, -1)  # [B,4,HW]
            world_pts = (torch.inverse(T_ref) @ cam_hom).view(B, 4, H, W)[:, :3]
            # world_pts: [B, 3, H, W]

            # accumulate features from views: ref + warped sources
            view_feats = [ref_feat]  # list of [B, C, H, W]

            if Nsrc > 0:
                # reuse homogeneous world coords for all sources
                world_h = torch.cat(
                    (world_pts, ones_), dim=1
                ).view(B, 4, -1)   # [B,4,HW]

                for v in range(Nsrc):
                    feat_src = src_feats[v]        # [B, C, H, W]
                    K_src = intrinsics[:, v + 1]   # [B, 3, 3]
                    T_src = extrinsics[:, v + 1]   # [B, 4, 4]

                    # world -> cam_src
                    cam_src = (T_src @ world_h).view(B, 4, H, W)[:, :3]  # [B,3,H,W]

                    x_src = cam_src[:, 0]
                    y_src = cam_src[:, 1]
                    z_src = cam_src[:, 2].clamp(min=1e-6)

                    # project using intrinsics
                    fx = K_src[:, 0, 0].view(B, 1, 1)
                    fy = K_src[:, 1, 1].view(B, 1, 1)
                    cx = K_src[:, 0, 2].view(B, 1, 1)
                    cy = K_src[:, 1, 2].view(B, 1, 1)

                    u = fx * (x_src / z_src) + cx   # [B,H,W]
                    v = fy * (y_src / z_src) + cy   # [B,H,W]

                    # normalize to [-1, 1] for grid_sample
                    u_norm = 2.0 * (u / (W - 1)) - 1.0
                    v_norm = 2.0 * (v / (H - 1)) - 1.0
                    grid = torch.stack((u_norm, v_norm), dim=3)  # [B,H,W,2]

                    # bilinear sample src features
                    warped = F.grid_sample(
                        feat_src, grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=False
                    )   # [B, C, H, W]

                    view_feats.append(warped)

            # stack across views -> [B, Nviews, C, H, W]
            views_stack = torch.stack(view_feats, dim=1)
            # variance-based cost along view dimension -> [B, C, H, W]
            var_cost = views_stack.var(dim=1, unbiased=False)

            cost_volume[:, :, d] = var_cost

        return cost_volume
