# feature_smooth_transition.py - FIXED import
import torch
import torch.nn as nn
import torch.nn.functional as F  # ADDED THIS LINE
from torchvision.ops import DeformConv2d

class FeatureSmoothTransition(nn.Module):
    """
    Feature Smooth Transition with deformable convolutions (as per paper)
    """
    def __init__(self, in_channels=32, num_scales=4):
        super(FeatureSmoothTransition, self).__init__()
        self.num_scales = num_scales

        # Offset generators for deformable convolutions
        self.offset_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 18, 3, 1, 1),  # 2*3*3=18 offsets for 3x3 kernel
                nn.Tanh()  # Limit offset range
            )
            for _ in range(num_scales)
        ])
        
        # Deformable convolutions
        self.deform_convs = nn.ModuleList([
            DeformConv2d(in_channels, in_channels, 3, 1, 1, bias=False)
            for _ in range(num_scales)
        ])
        
        # Batch norms
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(in_channels)
            for _ in range(num_scales)
        ])

    def forward(self, features):
        """
        Args:
            features: list of [feat0, feat1, feat2, feat3]
        Returns:
            adapted_features: list of adapted features
        """
        adapted = []
        for i, feat in enumerate(features):
            # Generate offsets
            offset = self.offset_convs[i](feat)
            # Apply deformable convolution
            out = self.deform_convs[i](feat, offset)
            out = self.bns[i](out)
            out = F.relu(out)
            adapted.append(out)
        return adapted

