"""Cross-scale attention for fusing features from different spatial scales.

Standard multi-head attention (Vaswani et al., "Attention Is All You Need",
NeurIPS 2017) applied in a cross-attention configuration: a query from one
scale attends to key/value features from another scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleAttention(nn.Module):
    """Multi-head cross-attention with residual connection and layer norm.

    Args:
        dim: Feature dimensionality.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        """Cross-attend from query to key_value.

        Args:
            query: (B, D) single query vector per sample.
            key_value: (B, N, D) set of N context vectors per sample.

        Returns:
            (B, D) attended + residual + normed output.
        """
        B, D = query.shape
        N = key_value.size(1)
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, 1, H, d).transpose(1, 2)
        k = self.k_proj(key_value).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(key_value).view(B, N, H, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, D)
        return self.norm(query + self.out_proj(out))


class MultiScalePyramid(nn.Module):
    """Fuse features from three scales via cross-scale attention.

    Scale 1 (global) attends to Scale 2 patches.
    Scale 2 attends to Scale 3 patches.
    Scale 3 attends to the global and Scale 2 context.

    The three attended features are concatenated and projected.

    Args:
        feature_dim: Input CLIP feature dimensionality (e.g. 512).
        hidden_dim: Internal and output dimensionality (e.g. 256).
        num_heads: Attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, feature_dim=512, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj_s1 = nn.Linear(feature_dim, hidden_dim)
        self.proj_s2 = nn.Linear(feature_dim, hidden_dim)
        self.proj_s3 = nn.Linear(feature_dim, hidden_dim)

        self.attn_1to2 = CrossScaleAttention(hidden_dim, num_heads, dropout)
        self.attn_2to3 = CrossScaleAttention(hidden_dim, num_heads, dropout)
        self.attn_3to1 = CrossScaleAttention(hidden_dim, num_heads, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, feat_s1, feat_s2, feat_s3):
        """Fuse three scale features.

        Args:
            feat_s1: (B, D) global features.
            feat_s2: (B, 4, D) scale-2 patch features.
            feat_s3: (B, 16, D) scale-3 patch features.

        Returns:
            (B, hidden_dim) L2-normalized fused feature.
        """
        f1 = self.proj_s1(feat_s1)
        f2 = self.proj_s2(feat_s2.mean(dim=1))
        f3 = self.proj_s3(feat_s3.mean(dim=1))

        kv_s2 = self.proj_s2(feat_s2)
        kv_s3 = self.proj_s3(feat_s3)

        f1_att = self.attn_1to2(f1, kv_s2)
        f2_att = self.attn_2to3(f2, kv_s3)
        # Attending to a single key always yields a weight of one and therefore
        # cannot select context. Include the medium-scale tokens so this branch
        # performs a genuine cross-scale interaction.
        global_medium_context = torch.cat([f1.unsqueeze(1), kv_s2], dim=1)
        f3_att = self.attn_3to1(f3, global_medium_context)

        fused = self.fusion(torch.cat([f1_att, f2_att, f3_att], dim=-1))
        fused = self.norm(fused)
        return F.normalize(fused, p=2, dim=1)
