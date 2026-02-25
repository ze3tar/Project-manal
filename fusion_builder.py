import torch
import torch.nn as nn
import torch.nn.functional as F

class CostVolumeBuilder(nn.Module):
    """
    Builds a differentiable cost volume by warping source features into
    the reference view across all depth hypotheses.

    Vectorized: all D depths are processed in a single batched operation
    per source view (no Python for-loop over depths).
    """

    def __init__(self):
        super().__init__()

    def forward(self, ref_feat, src_feats, intrinsics, extrinsics, depth_values,
                image_shape=None):
        """
        ref_feat:    [B, C, H, W]
        src_feats:   list of [B, C, H, W]
        intrinsics:  [B, N, 3, 3]
        extrinsics:  [B, N, 4, 4]
        depth_values:[B, D] or [B, D, H, W]
        image_shape: (img_H, img_W)
        """
        B, C, H, W = ref_feat.shape
        D = depth_values.shape[1]
        Nsrc = len(src_feats)

        # --- Scale intrinsics to feature-map resolution -----------------
        if image_shape is not None:
            img_H, img_W = image_shape
            scale_w = W / img_W
            scale_h = H / img_H
        else:
            scale_w = 1.0
            scale_h = 1.0

        intrinsics_scaled = intrinsics.clone()
        intrinsics_scaled[:, :, 0, 0] *= scale_w
        intrinsics_scaled[:, :, 1, 1] *= scale_h
        intrinsics_scaled[:, :, 0, 2] *= scale_w
        intrinsics_scaled[:, :, 1, 2] *= scale_h

        K_ref = intrinsics_scaled[:, 0]   # [B, 3, 3]
        T_ref = extrinsics[:, 0]          # [B, 4, 4]

        # --- Precompute matrices ONCE (not per depth) -------------------
        Kinv_ref = torch.linalg.inv(K_ref)        # [B, 3, 3]
        T_ref_inv = torch.linalg.inv(T_ref)       # [B, 4, 4]

        # Pixel grid [B, 3, H, W]
        y, x = torch.meshgrid(
            torch.arange(0, H, device=ref_feat.device, dtype=ref_feat.dtype),
            torch.arange(0, W, device=ref_feat.device, dtype=ref_feat.dtype),
            indexing='ij'
        )
        ones = torch.ones_like(x)
        pix = torch.stack((x, y, ones), dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 3, H, W]

        # Unit rays in ref camera: [B, 3, H, W]
        rays = (Kinv_ref @ pix.reshape(B, 3, -1)).reshape(B, 3, H, W)

        # --- Vectorize over all D depths at once -----------------------
        # depth_vals: [B, D, H, W] (expand if global)
        if depth_values.ndim == 2:
            depth_expanded = depth_values.view(B, D, 1, 1).expand(B, D, H, W)
        else:
            depth_expanded = depth_values  # already [B, D, H, W]

        # cam_pts: [B, 3, D, H, W]
        rays_5d = rays.unsqueeze(2)                          # [B, 3, 1, H, W]
        cam_pts = rays_5d * depth_expanded.unsqueeze(1)      # [B, 3, D, H, W]

        # To homogeneous [B, 4, D*H*W]
        ones_dhw = torch.ones(B, 1, D, H, W, device=ref_feat.device, dtype=ref_feat.dtype)
        cam_hom = torch.cat((cam_pts, ones_dhw), dim=1).reshape(B, 4, -1)  # [B, 4, D*H*W]

        # Ref camera → world (single matmul for ALL depths)
        world_pts = (T_ref_inv @ cam_hom)  # [B, 4, D*H*W]

        # --- Variance accumulation across views -------------------------
        # We accumulate sum and sum_sq to compute variance without storing all views
        ref_feat_expanded = ref_feat.unsqueeze(2).expand(B, C, D, H, W)  # [B, C, D, H, W]
        feat_sum = ref_feat_expanded.clone()
        feat_sq_sum = ref_feat_expanded.square()
        n_views = 1

        for v in range(Nsrc):
            feat_src = src_feats[v]                # [B, C, H, W]
            K_src = intrinsics_scaled[:, v + 1]    # [B, 3, 3]
            T_src = extrinsics[:, v + 1]           # [B, 4, 4]

            # World → source camera (single matmul for ALL depths)
            cam_src = (T_src @ world_pts)          # [B, 4, D*H*W]
            cam_src = cam_src[:, :3].reshape(B, 3, D, H, W)  # [B, 3, D, H, W]

            x_src = cam_src[:, 0]   # [B, D, H, W]
            y_src = cam_src[:, 1]
            z_src = cam_src[:, 2].clamp(min=1e-6)

            fx = K_src[:, 0, 0].reshape(B, 1, 1, 1)
            fy = K_src[:, 1, 1].reshape(B, 1, 1, 1)
            cx = K_src[:, 0, 2].reshape(B, 1, 1, 1)
            cy = K_src[:, 1, 2].reshape(B, 1, 1, 1)

            u = fx * (x_src / z_src) + cx   # [B, D, H, W]
            v_coord = fy * (y_src / z_src) + cy

            # Normalize to [-1, 1]
            u_norm = 2.0 * (u / (W - 1)) - 1.0
            v_norm = 2.0 * (v_coord / (H - 1)) - 1.0

            # Reshape grid for batched grid_sample: [B, D*H, W, 2]
            grid = torch.stack((u_norm, v_norm), dim=-1)     # [B, D, H, W, 2]
            grid = grid.reshape(B, D * H, W, 2)

            # Single grid_sample call for ALL D depths (instead of D separate calls)
            warped = F.grid_sample(
                feat_src, grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )   # [B, C, D*H, W]
            warped = warped.reshape(B, C, D, H, W)

            feat_sum = feat_sum + warped
            feat_sq_sum = feat_sq_sum + warped.square()
            n_views += 1

        # Variance = E[X^2] - E[X]^2  (compute in float32 to avoid AMP overflow)
        orig_dtype = feat_sum.dtype
        feat_sum = feat_sum.float()
        feat_sq_sum = feat_sq_sum.float()
        mean = feat_sum / n_views
        cost_volume = (feat_sq_sum / n_views - mean.square()).clamp(min=0.0)
        cost_volume = cost_volume.to(orig_dtype)  # [B, C, D, H, W]

        return cost_volume
