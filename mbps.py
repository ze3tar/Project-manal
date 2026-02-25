# mbps.py - Per-pixel cascade depth refinement
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_builder import CostVolumeBuilder
from cost_regularization import CostRegularization3DUNet

class MBPS(nn.Module):
    def __init__(self, base_channels=32, num_stages=4):
        super(MBPS, self).__init__()
        self.num_stages = num_stages
        self.cost_volume_builder = CostVolumeBuilder()
        self.stage_nets = nn.ModuleList([
            CostRegularization3DUNet(base_channels) for _ in range(num_stages)
        ])
        self.stage_depth_counts = [48, 32, 8, 4]
        self.depth_intervals = [0.125, 0.25, 0.5, 1.0]

    def forward(self, ref_feat, src_feats, intrinsics, extrinsics, depth_values):
        results = []
        current_depth_values = depth_values

        for stage_idx in range(self.num_stages):
            depth_map, prob_volume, depth_vals_used = self._process_single_stage(
                ref_feat, src_feats, intrinsics, extrinsics, current_depth_values, stage_idx
            )
            results.append((depth_map, prob_volume, depth_vals_used))

            if stage_idx < self.num_stages - 1:
                current_depth_values = self._update_depth_values(depth_map, current_depth_values, stage_idx)

        return results

    def _process_single_stage(self, ref_feat, src_feats, intrinsics, extrinsics, depth_values, stage_idx,
                              image_shape=None):
        """
        Returns: (depth_map, prob_volume, depth_values_used)
        """
        B, C, H, W = ref_feat.shape
        target_depth_count = self.stage_depth_counts[stage_idx]

        # Only resample global (2D) depth values
        if depth_values.ndim == 2 and depth_values.shape[1] != target_depth_count:
            depth_values = self._resample_depth_hypotheses(depth_values, target_depth_count)

        D = depth_values.shape[1]

        try:
            cost_volume = self.cost_volume_builder(ref_feat, src_feats, intrinsics, extrinsics, depth_values,
                                                   image_shape=image_shape)
            prob_volume = self.stage_nets[stage_idx](cost_volume)
            prob_volume = prob_volume.clamp(min=0.0)  # guard against negative variance artifacts
            prob_sum = prob_volume.sum(dim=1, keepdim=True).clamp(min=1e-8)
            prob_volume = prob_volume / prob_sum  # renormalize

            # Soft argmax: handle both global [B,D] and per-pixel [B,D,H,W]
            if depth_values.ndim == 2:
                depth_map = torch.sum(prob_volume * depth_values.view(B, D, 1, 1), dim=1)
            else:
                depth_map = torch.sum(prob_volume * depth_values, dim=1)

            return depth_map, prob_volume, depth_values

        except Exception as e:
            print(f"Warning: Stage {stage_idx} failed: {e}")
            fallback_depth = torch.ones(B, H, W, device=ref_feat.device) * depth_values.mean()
            fallback_prob = torch.ones(B, D, H, W, device=ref_feat.device) / D
            return fallback_depth, fallback_prob, depth_values

    def _resample_depth_hypotheses(self, depth_values, target_count):
        B, current_count = depth_values.shape
        if current_count == target_count:
            return depth_values
        depth_min = depth_values.min(dim=1, keepdim=True)[0]
        depth_max = depth_values.max(dim=1, keepdim=True)[0]
        new_depth_values = []
        for b in range(B):
            new_depths = torch.linspace(
                depth_min[b, 0].item(), depth_max[b, 0].item(),
                target_count, device=depth_values.device
            )
            new_depth_values.append(new_depths)
        return torch.stack(new_depth_values, dim=0)

    def _update_depth_values(self, depth_map, depth_values, stage_idx, next_feat_size=None):
        """Per-pixel depth hypotheses for next cascade stage."""
        next_count = self.stage_depth_counts[stage_idx + 1]
        cascade_scales = [4.0, 2.0, 1.0]

        # base_interval from initial depth range
        if depth_values.ndim == 2:
            drange = depth_values.max(dim=1)[0] - depth_values.min(dim=1)[0]  # [B]
            base_interval = drange / depth_values.shape[1]
        else:
            drange = depth_values.amax(dim=(1, 2, 3)) - depth_values.amin(dim=(1, 2, 3))  # [B]
            base_interval = drange / depth_values.shape[1]

        interval = base_interval * cascade_scales[stage_idx]  # [B]

        # Upsample depth to next feature resolution
        if next_feat_size is not None and depth_map.shape[-2:] != next_feat_size:
            depth_up = F.interpolate(
                depth_map.unsqueeze(1), size=next_feat_size,
                mode='bilinear', align_corners=True
            ).squeeze(1)
        else:
            depth_up = depth_map

        B, H, W = depth_up.shape
        offsets = torch.linspace(-(next_count - 1) / 2, (next_count - 1) / 2,
                                 next_count, device=depth_up.device)
        # Per-pixel hypotheses: [B, D_next, H, W]
        new_vals = depth_up.unsqueeze(1) + offsets.view(1, -1, 1, 1) * interval.view(B, 1, 1, 1)
        return new_vals.clamp(min=0.1)
