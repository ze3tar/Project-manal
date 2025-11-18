# edge_attention.py - FIXED
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAttentionFusion(nn.Module):
    """
    Edge Attention Fusion (EAF) - FIXED version
    """
    def __init__(self, in_channels=32, num_scales=4):
        super(EdgeAttentionFusion, self).__init__()

        # Adjustment layers to align channels
        self.adjust_layers = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)
            for _ in range(num_scales)
        ])

        # Edge attention convs (3 fusion operations for 4 scales)
        self.edge_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_scales - 1)
        ])

    def forward(self, features):
        """
        Args:
            features: list [feat0, feat1, feat2, feat3] (high → low resolution)
                     feat0: [B, C, H, W] - full resolution
                     feat1: [B, C, H/2, W/2]
                     feat2: [B, C, H/4, W/4]
                     feat3: [B, C, H/8, W/8] - lowest resolution
        Returns:
            enhanced_features: list [feat0_enh, feat1_enh, feat2_enh, feat3_enh]
        """
        # FIXED: Process from coarse to fine, building up enhanced features
        enhanced = [None] * len(features)
        
        # Start from lowest resolution (feat3)
        current_feat = self.adjust_layers[3](features[3])
        enhanced[3] = current_feat  # Save feat3_enhanced
        
        # Process upwards: feat3 → feat2 → feat1 → feat0
        for i in range(2, -1, -1):  # i = 2, 1, 0
            high_feat = self.adjust_layers[i](features[i])
            
            # Upsample current_feat to match high_feat resolution
            upsampled = F.interpolate(
                current_feat, 
                size=high_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

            # Add + Subtract (Laplacian residual)
            feat_add = high_feat + upsampled
            feat_sub = high_feat - upsampled

            # Concatenate and enhance
            edge_input = torch.cat([feat_add, feat_sub], dim=1)
            current_feat = self.edge_convs[i](edge_input)
            
            enhanced[i] = current_feat  # Save enhanced feature

        # Return in order: [feat0, feat1, feat2, feat3]
        return enhanced


