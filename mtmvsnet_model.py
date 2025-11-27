# mtmvsnet_model.py - Pass through 3-tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extraction import FeatureExtractor
from feature_smooth_transition import FeatureSmoothTransition
from mobile_transformer_block import MobileTransformerBlock
from edge_attention import EdgeAttentionFusion
from mbps import MBPS

class MTMVSNet(nn.Module):
    def __init__(self, base_channels=32, num_stages=4):
        super(MTMVSNet, self).__init__()
        self.num_stages = num_stages
        self.feature_extractor = FeatureExtractor(in_channels=3, base_channels=base_channels)
        self.fst = FeatureSmoothTransition(in_channels=base_channels, num_scales=4)
        self.mtb_blocks = nn.ModuleList([
            MobileTransformerBlock(in_channels=base_channels) for _ in range(4)
        ])
        self.eaf = EdgeAttentionFusion(in_channels=base_channels, num_scales=4)
        self.mbps = MBPS(base_channels=base_channels, num_stages=num_stages)

    def forward(self, images, intrinsics, extrinsics, depth_values):
        """
        Args:
            images:     (B, V, 3, H, W)
            intrinsics: (B, V, 3, 3)
            extrinsics: (B, V, 4, 4)
            depth_values: (B, D)

        Returns:
            List of tuples per stage: (depth_pred, prob_volume, depth_values_stage)
            prob_volume is softmax-normalized over the depth dimension for the
            exact depth_values_stage used by that stage.
        """
        B, N, C, H, W = images.shape

        all_view_features = []
        for view_idx in range(N):
            fpn_feats = self.feature_extractor(images[:, view_idx])
            all_view_features.append(fpn_feats)

        fst_features = []
        for view_idx in range(N):
            view_fst_feats = self.fst(all_view_features[view_idx])
            fst_features.append(view_fst_feats)

        enhanced_features = []
        ref_enhanced = []
        for scale_idx in range(4):
            source_feats_at_scale = [fst_features[src_idx][scale_idx] for src_idx in range(1, N)]
            ref_feat_enhanced = self.mtb_blocks[scale_idx](
                fst_features[0][scale_idx], source_feats_at_scale
            )
            ref_enhanced.append(ref_feat_enhanced)
        enhanced_features.append(ref_enhanced)
        
        for view_idx in range(1, N):
            view_enhanced = []
            for scale_idx in range(4):
                enhanced_feat = self.mtb_blocks[scale_idx](
                    fst_features[view_idx][scale_idx], None
                )
                view_enhanced.append(enhanced_feat)
            enhanced_features.append(view_enhanced)

        ref_final_features = self.eaf(enhanced_features[0])
        src_final_features = [enhanced_features[i] for i in range(1, N)]

        stage_to_scale = list(reversed(range(len(ref_final_features))))[: self.num_stages]
        results = []
        current_depth_values = depth_values

        for stage_idx in range(self.num_stages):
            if stage_idx >= len(stage_to_scale):
                raise ValueError(
                    f"Requested {self.num_stages} stages but only have {len(stage_to_scale)} feature scales"
                )

            scale_idx = stage_to_scale[stage_idx]
            ref_feat = ref_final_features[scale_idx]
            src_feat_list = [src_view[scale_idx] for src_view in src_final_features] if src_final_features else []

            try:
                depth_map, prob_volume, depth_vals = self.mbps._process_single_stage(
                    ref_feat, src_feat_list, intrinsics, extrinsics, current_depth_values, stage_idx
                )
                
                if depth_map.shape[-2:] != (H, W):
                    depth_map = F.interpolate(
                        depth_map.unsqueeze(1), size=(H, W), 
                        mode='bilinear', align_corners=False
                    ).squeeze(1)
                    prob_volume = F.interpolate(
                        prob_volume, size=(H, W),
                        mode='bilinear', align_corners=False
                    )
                
                results.append((depth_map, prob_volume, depth_vals))
                
                if stage_idx < self.num_stages - 1:
                    current_depth_values = self.mbps._update_depth_values(
                        depth_map, current_depth_values, stage_idx
                    )
            except Exception as e:
                print(f"Stage {stage_idx} failed: {e}")
                fallback_depth = torch.ones(B, H, W, device=images.device) * current_depth_values.mean()
                D = self.mbps.stage_depth_counts[stage_idx]
                fallback_prob = torch.ones(B, D, H, W, device=images.device) / D
                results.append((fallback_depth, fallback_prob, current_depth_values))

        return results






