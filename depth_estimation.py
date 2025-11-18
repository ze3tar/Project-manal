import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthEstimation(nn.Module):
    def __init__(self, in_channels, num_depths=64):
        super(DepthEstimation, self).__init__()
        self.in_channels = in_channels
        self.num_depths = num_depths

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, num_depths, kernel_size=3, padding=1)

    def forward(self, fused_feature, depth_values):
        """
        fused_feature: [B, C, H, W]
        depth_values: [B, D] where D = num_depths
        """
        x = F.relu(self.conv1(fused_feature))
        x = F.relu(self.conv2(x))
        cost_volume = self.conv3(x)  # [B, D, H, W]

        prob_volume = F.softmax(cost_volume, dim=1)  # along D
        depth_values = depth_values.view(depth_values.shape[0], -1, 1, 1)  # [B, D, 1, 1]
        depth_map = torch.sum(prob_volume * depth_values, dim=1)  # [B, H, W]

        return depth_map


















