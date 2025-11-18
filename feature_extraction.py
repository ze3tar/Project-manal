# feature_extraction.py
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    Feature Pyramid Network (FPN) for multi-scale feature extraction
    """
    def __init__(self, in_channels=3, base_channels=32):
        super(FeatureExtractor, self).__init__()

        # Stage 0: full resolution
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 1/2 resolution
        self.stage1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Stage 2: 1/4 resolution
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # Stage 3: 1/8 resolution
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        # Reduce all stages to the same base channel size
        self.reduce0 = nn.Conv2d(base_channels, base_channels, 1, 1, 0, bias=False)
        self.reduce1 = nn.Conv2d(base_channels * 2, base_channels, 1, 1, 0, bias=False)
        self.reduce2 = nn.Conv2d(base_channels * 4, base_channels, 1, 1, 0, bias=False)
        self.reduce3 = nn.Conv2d(base_channels * 8, base_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        # Multi-scale extraction
        feat0 = self.stage0(x)   # [B, C, H, W]
        feat1 = self.stage1(feat0)  # [B, 2C, H/2, W/2]
        feat2 = self.stage2(feat1)  # [B, 4C, H/4, W/4]
        feat3 = self.stage3(feat2)  # [B, 8C, H/8, W/8]

        # Channel reduction
        feat0 = self.reduce0(feat0)
        feat1 = self.reduce1(feat1)
        feat2 = self.reduce2(feat2)
        feat3 = self.reduce3(feat3)

        return [feat0, feat1, feat2, feat3]  # From high to low resolution


