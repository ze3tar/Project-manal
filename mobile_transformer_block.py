# mobile_transformer_block.py - With linear attention
import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileTransformerBlock(nn.Module):
    """
    Mobile Transformer Block with linear attention (as per paper)
    """
    def __init__(self, in_channels=64, patch_size=4, num_heads=8):
        super(MobileTransformerBlock, self).__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads

        # Local representation
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Downsample for attention
        self.downsample = nn.Conv2d(in_channels, in_channels, patch_size, patch_size)

        # Linear attention projections
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3)
        self.out_proj = nn.Linear(in_channels, in_channels)
        
        # Cross-attention projections
        self.cross_kv_proj = nn.Linear(in_channels, in_channels * 2)

        # Norms + FFN
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels)
        )

        # Upsample
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False)

        # Fusion
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0, bias=False)

    def forward(self, x, source_features=None):
        """
        Args:
            x: [B,C,H,W] reference feature
            source_features: list of [B,C,H,W] from source views
        """
        B, C, H, W = x.shape

        # Local features
        local_feat = self.local_conv(x)

        # Downsample
        x_down = self.downsample(x)
        H_d, W_d = x_down.shape[2], x_down.shape[3]

        # Flatten
        tokens = x_down.flatten(2).transpose(1, 2)  # [B, N, C]

        # Linear self-attention
        attn_out = self._linear_self_attention(tokens)
        attn_out = self.norm1(tokens + attn_out)

        # Cross-attention if source features provided
        if source_features is not None:
            for src_feat in source_features:
                src_down = F.interpolate(src_feat, size=(H_d, W_d), mode="bilinear", align_corners=False)
                src_tokens = src_down.flatten(2).transpose(1, 2)
                cross_out = self._linear_cross_attention(attn_out, src_tokens)
                attn_out = self.norm2(attn_out + cross_out)

        # Feed-forward
        ffn_out = self.ffn(attn_out)
        global_feat = attn_out + ffn_out

        # Reshape
        global_feat = global_feat.transpose(1, 2).view(B, C, H_d, W_d)

        # Upsample
        global_feat = self.upsample(global_feat)

        # Fuse
        fused = torch.cat([local_feat, global_feat], dim=1)
        output = self.fusion(fused)

        return x + output

    def _linear_self_attention(self, x):
        """
        Linear attention: φ(Q)[φ(K)^T V]
        """
        B, N, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply feature map φ (ELU + 1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention: φ(Q)[φ(K)^T V]
        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, H, D, D]
        out = torch.einsum('bhnd,bhde->bhne', q, kv)  # [B, H, N, D]
        
        # Normalize
        k_sum = k.sum(dim=2, keepdim=True)  # [B, H, 1, D]
        out = out / (torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(2)).unsqueeze(-1) + 1e-6)
        
        # Reshape
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

    def _linear_cross_attention(self, q_tokens, kv_tokens):
        """
        Linear cross-attention
        """
        B, N, C = q_tokens.shape
        
        # Q from query, K,V from source
        q = self.qkv_proj(q_tokens)[:, :, :C].reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.cross_kv_proj(kv_tokens).reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Apply feature map
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear cross-attention
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, kv)
        
        # Normalize
        k_sum = k.sum(dim=2, keepdim=True)
        out = out / (torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(2)).unsqueeze(-1) + 1e-6)
        
        # Reshape
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out





