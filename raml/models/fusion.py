"""Text-visual fusion via cross-attention with a learnable residual scale.

The visual feature (query) attends to two text embeddings (normal prompt,
anomaly prompt) projected into the visual feature space.  A learnable scalar
controls how much of the attended signal is mixed into the visual feature.
"""

import torch
import torch.nn as nn


class TextVisualFusion(nn.Module):
    """Fuse visual features with text prompt embeddings.

    Args:
        text_dim: Dimensionality of CLIP text embeddings (e.g. 512).
        hidden_dim: Internal/output dimensionality (e.g. 256).
        num_heads: Attention heads.
        dropout: Dropout rate.
        init_scale: Initial value for the learnable fusion scale.
    """

    def __init__(
        self, text_dim=512, hidden_dim=256, num_heads=4, dropout=0.1, init_scale=0.1
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_scale = nn.Parameter(torch.tensor(init_scale))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_features, text_features):
        """Fuse visual and text features.

        Args:
            visual_features: (B, hidden_dim) visual feature tensor.
            text_features: (B, 2, text_dim) text embeddings.
                           Dim 1 index 0 = normal prompt, 1 = anomaly prompt.

        Returns:
            (B, hidden_dim) fused feature.
        """
        B = visual_features.size(0)

        text_proj = self.text_proj(text_features)

        if text_proj.dim() == 2:
            text_proj = text_proj.unsqueeze(0).expand(B, -1, -1)

        visual_query = visual_features.unsqueeze(1)
        attn_output, _ = self.cross_attn(visual_query, text_proj, text_proj)
        attn_output = attn_output.squeeze(1)

        fused = visual_features + self.fusion_scale * attn_output
        return self.norm(fused)
