import torch
import torch.nn as nn
import torch.nn.functional as F

from fruit_segmentation_head import FruitSegmentationHead
from mtmvsnet_model import MTMVSNet


class MTMVSNetWithFruit(nn.Module):
    def __init__(self, base_channels=32, num_stages=4):
        super().__init__()
        self.backbone = MTMVSNet(base_channels=base_channels, num_stages=num_stages)
        self.fruit_head = FruitSegmentationHead(in_channels=base_channels)

    def _extract_eaf_features(self, images):
        bsz, num_views, _, _, _ = images.shape
        all_view_features = []
        for view_idx in range(num_views):
            fpn_feats = self.backbone.feature_extractor(images[:, view_idx])
            all_view_features.append(fpn_feats)

        fst_features = []
        for view_idx in range(num_views):
            view_fst_feats = self.backbone.fst(all_view_features[view_idx])
            fst_features.append(view_fst_feats)

        enhanced_features = []
        ref_enhanced = []
        for scale_idx in range(4):
            source_feats_at_scale = [fst_features[src_idx][scale_idx] for src_idx in range(1, num_views)]
            ref_feat_enhanced = self.backbone.mtb_blocks[scale_idx](
                fst_features[0][scale_idx], source_feats_at_scale
            )
            ref_enhanced.append(ref_feat_enhanced)
        enhanced_features.append(ref_enhanced)

        for view_idx in range(1, num_views):
            view_enhanced = []
            for scale_idx in range(4):
                enhanced_feat = self.backbone.mtb_blocks[scale_idx](
                    fst_features[view_idx][scale_idx], None
                )
                view_enhanced.append(enhanced_feat)
            enhanced_features.append(view_enhanced)

        ref_final_features = self.backbone.eaf(enhanced_features[0])
        src_final_features = [enhanced_features[i] for i in range(1, num_views)]
        return ref_final_features, src_final_features

    def _run_depth_path(self, ref_final_features, src_final_features, intrinsics, extrinsics, depth_values, image_shape):
        _, _, height, width = image_shape
        stage_to_scale = list(reversed(range(len(ref_final_features))))[: self.backbone.num_stages]
        results = []
        current_depth_values = depth_values

        for stage_idx in range(self.backbone.num_stages):
            scale_idx = stage_to_scale[stage_idx]
            ref_feat = ref_final_features[scale_idx]
            src_feat_list = [src_view[scale_idx] for src_view in src_final_features] if src_final_features else []

            depth_map_feat, prob_volume, depth_vals = self.backbone.mbps._process_single_stage(
                ref_feat, src_feat_list, intrinsics, extrinsics, current_depth_values, stage_idx,
                image_shape=(height, width)
            )

            # Upsample to full res for output only
            depth_map_full = depth_map_feat
            prob_volume_full = prob_volume
            if depth_map_feat.shape[-2:] != (height, width):
                depth_map_full = F.interpolate(
                    depth_map_feat.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True
                ).squeeze(1)
                prob_volume_full = F.interpolate(
                    prob_volume, size=(height, width), mode="bilinear", align_corners=True
                )

            results.append((depth_map_full, prob_volume_full, depth_vals))

            # Cascade: use feature-resolution depth for per-pixel hypotheses
            if stage_idx < self.backbone.num_stages - 1:
                next_scale = stage_to_scale[stage_idx + 1]
                next_size = ref_final_features[next_scale].shape[-2:]
                current_depth_values = self.backbone.mbps._update_depth_values(
                    depth_map_feat, current_depth_values, stage_idx,
                    next_feat_size=next_size
                )

        return results

    def forward(self, images, intrinsics=None, extrinsics=None, depth_values=None,
                compute_depth=True, multi_view_fruit=False):
        bsz, num_views, _, height, width = images.shape
        ref_final_features, src_final_features = self._extract_eaf_features(images)

        fruit_logits = self.fruit_head(ref_final_features[-1])
        # Upsample fruit logits to input resolution (features are at 1/8 scale)
        if fruit_logits.shape[-2:] != (height, width):
            fruit_logits = F.interpolate(
                fruit_logits, size=(height, width), mode="bilinear", align_corners=False
            )
        fruit_mask = self.fruit_head.logits_to_mask(fruit_logits)

        # Multi-view fruit: run EAF + fruit_head on source views too
        all_view_fruit_logits = None
        if multi_view_fruit and not compute_depth:
            all_view_fruit_logits = [fruit_logits]
            for src_feats in src_final_features:
                src_eaf_feats = self.backbone.eaf(src_feats)
                src_logits = self.fruit_head(src_eaf_feats[-1])
                if src_logits.shape[-2:] != (height, width):
                    src_logits = F.interpolate(
                        src_logits, size=(height, width), mode="bilinear", align_corners=False
                    )
                all_view_fruit_logits.append(src_logits)

        depth_map = None
        prob_volume = None
        if compute_depth:
            if intrinsics is None or extrinsics is None or depth_values is None:
                raise ValueError("intrinsics, extrinsics, and depth_values are required when compute_depth=True")

            depth_results = self._run_depth_path(
                ref_final_features,
                src_final_features,
                intrinsics,
                extrinsics,
                depth_values,
                (bsz, num_views, height, width),
            )
            depth_map, prob_volume, _ = depth_results[-1]

        return {
            "depth": depth_map,
            "prob": prob_volume,
            "fruit_mask": fruit_mask,
            "fruit_logits": fruit_logits,
            "all_view_fruit_logits": all_view_fruit_logits,
        }
