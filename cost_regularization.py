# cost_regularization.py - FIXED with size matching
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class CostRegularization3DUNet(nn.Module):
    """
    Full 3D U-Net with size-matching for skip connections
    """
    def __init__(self, in_channels=32):
        super(CostRegularization3DUNet, self).__init__()
        
        # Encoder
        self.conv0 = nn.Sequential(
            ConvBnReLU3D(in_channels, 8, 3, 1, 1),
            ConvBnReLU3D(8, 8, 3, 1, 1)
        )
        
        self.conv1 = nn.Sequential(
            ConvBnReLU3D(8, 16, 5, 2, 2),
            ConvBnReLU3D(16, 16, 3, 1, 1),
            ConvBnReLU3D(16, 16, 3, 1, 1)
        )
        
        self.conv2 = nn.Sequential(
            ConvBnReLU3D(16, 32, 5, 2, 2),
            ConvBnReLU3D(32, 32, 3, 1, 1),
            ConvBnReLU3D(32, 32, 3, 1, 1)
        )
        
        self.conv3 = nn.Sequential(
            ConvBnReLU3D(32, 64, 5, 2, 2),
            ConvBnReLU3D(64, 64, 3, 1, 1),
            ConvBnReLU3D(64, 64, 3, 1, 1)
        )
        
        # Decoder
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        
        # Refinement
        self.refine3 = ConvBnReLU3D(64, 32, 3, 1, 1)  # 32+32=64
        self.refine2 = ConvBnReLU3D(32, 16, 3, 1, 1)  # 16+16=32
        self.refine1 = ConvBnReLU3D(16, 8, 3, 1, 1)   # 8+8=16
        
        # Output
        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            prob: [B, D, H, W]
        """
        # Encoder
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        # Decoder with adaptive skip connections
        deconv3 = self.deconv3(conv3)
        # Match sizes before concat
        if deconv3.shape != conv2.shape:
            deconv3 = F.interpolate(deconv3, size=conv2.shape[2:], mode='trilinear', align_corners=False)
        concat3 = torch.cat([deconv3, conv2], dim=1)
        refine3 = self.refine3(concat3)
        
        deconv2 = self.deconv2(refine3)
        if deconv2.shape != conv1.shape:
            deconv2 = F.interpolate(deconv2, size=conv1.shape[2:], mode='trilinear', align_corners=False)
        concat2 = torch.cat([deconv2, conv1], dim=1)
        refine2 = self.refine2(concat2)
        
        deconv1 = self.deconv1(refine2)
        if deconv1.shape != conv0.shape:
            deconv1 = F.interpolate(deconv1, size=conv0.shape[2:], mode='trilinear', align_corners=False)
        concat1 = torch.cat([deconv1, conv0], dim=1)
        refine1 = self.refine1(concat1)
        
        # Output
        out = self.prob(refine1).squeeze(1)
        prob = F.softmax(out, dim=1)
        
        return prob
