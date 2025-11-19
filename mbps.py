# mbps.py - Return depth_values used
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
        self.stage_depth_counts = [32, 16, 8, 4]
        self.depth_intervals = [0.125, 0.25, 0.5, 1.0]

    def forward(self, ref_feat, src_feats, intrinsics, extrinsics, depth_values):
        results = []
        current_depth_values = depth_values
        
        for stage_idx in range(self.num_stages):
            depth_map, prob_volume, depth_vals_used = self._process_single_stage(
                ref_feat, src_feats, intrinsics, extrinsics, current_depth_values, stage_idx
            )
            results.append((depth_map, prob_volume, depth_vals_used))  # 3-tuple
            
            if stage_idx < self.num_stages - 1:
                current_depth_values = self._update_depth_values(depth_map, current_depth_values, stage_idx)
                
        return results

    def _process_single_stage(self, ref_feat, src_feats, intrinsics, extrinsics, depth_values, stage_idx):
        """
        Returns: (depth_map, prob_volume, depth_values_used)
        """
        B, C, H, W = ref_feat.shape
        target_depth_count = self.stage_depth_counts[stage_idx]
        
        if depth_values.shape[1] != target_depth_count:
            depth_values = self._resample_depth_hypotheses(depth_values, target_depth_count)
        
        D = depth_values.shape[1]
        
        try:
            cost_volume = self.cost_volume_builder(ref_feat, src_feats, intrinsics, extrinsics, depth_values)
            prob_volume = self.stage_nets[stage_idx](cost_volume)
            depth_map = torch.sum(prob_volume * depth_values.view(B, D, 1, 1), dim=1)
            
            return depth_map, prob_volume, depth_values  # Return the actual depth_values used
            
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

    def _update_depth_values(self, depth_map, current_depth_values, stage_idx):
        B, H, W = depth_map.shape
        device = depth_map.device
        depth_mean = depth_map.view(B, -1).mean(dim=1)
        depth_std = depth_map.view(B, -1).std(dim=1)
        refinement_factor = 2.0 - (stage_idx * 0.3)
        new_depth_values = []
        next_stage_depth_count = self.stage_depth_counts[min(stage_idx + 1, len(self.stage_depth_counts) - 1)]
        for b in range(B):
            center_depth = depth_mean[b]
            range_half = torch.maximum(depth_std[b] * refinement_factor, torch.tensor(100.0, device=device))  # FIXED: min 10mm range
            print(f"DEBUG Stage {stage_idx}, Batch {b}: depth_std={depth_std[b].item():.4f}, refinement={refinement_factor:.2f}, range_half={range_half.item():.2f}")
            
            d_min = max(0.1, center_depth - range_half)
            d_max = center_depth + range_half
            new_depths = torch.linspace(d_min.item(), d_max.item(), next_stage_depth_count, device=device)
            new_depth_values.append(new_depths)
        return torch.stack(new_depth_values, dim=0)


