"""Multi-scale CLIP model for image-level industrial anomaly detection.

Assembles multi-scale CLIP extraction, cross-scale attention pyramid,
text-visual fusion, and a classification head into a single module.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from raml.models.multi_scale import MultiScaleExtractor
from raml.models.cross_attention import MultiScalePyramid
from raml.models.fusion import TextVisualFusion
from raml.models.prompts import build_state_prompts


class MultiScaleCLIPDetector(nn.Module):
    """Multi-scale CLIP detector with trainable image-level fusion.

    Architecture:
        1. Frozen CLIP backbone extracts multi-scale features (1x1, 2x2, 4x4).
        2. Multi-scale pyramid fuses scales via cross-attention.
        3. Text-visual fusion incorporates CLIP text embeddings.
        4. Feature refinement head + classification head produce logits.
        5. Final score blends visual classifier output with text similarity.

    Args:
        clip_model: Loaded CLIP model instance (will be frozen).
        tokenizer: CLIP tokenizer function (e.g. clip.tokenize).
        feature_dim: CLIP output dimensionality.
        hidden_dim: Internal dimensionality for trainable layers.
        num_heads: Attention heads for cross-attention modules.
        dropout: Dropout rate.
        visual_weight: Blending weight for visual score at inference.
        text_weight: Blending weight for text score at inference.
    """

    def __init__(
        self,
        clip_model,
        tokenizer,
        feature_dim=512,
        hidden_dim=256,
        num_heads=4,
        dropout=0.1,
        visual_weight=0.85,
        text_weight=0.15,
        clip_input_size=224,
        scales=(1, 2, 4),
        encode_chunk_size=64,
        combine_scale_batches=True,
        channels_last=False,
        prompt_ensemble=True,
        text_temperature=0.01,
        local_score_weight=0.5,
        local_topk=0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.visual_weight = visual_weight
        self.text_weight = text_weight
        self.prompt_ensemble = prompt_ensemble
        self.text_temperature = text_temperature
        self.local_score_weight = local_score_weight
        self.local_topk = local_topk

        if visual_weight < 0 or text_weight < 0 or visual_weight + text_weight <= 0:
            raise ValueError("inference weights must be non-negative with a positive sum")
        if text_temperature <= 0:
            raise ValueError("text_temperature must be positive")
        if not 0 <= local_score_weight <= 1:
            raise ValueError("local_score_weight must be in [0, 1]")
        if not 0 < local_topk <= 1:
            raise ValueError("local_topk must be in (0, 1]")
        if len(scales) != 3:
            raise ValueError("The detector currently requires exactly three scales")

        self.extractor = MultiScaleExtractor(
            clip_model,
            clip_input_size=clip_input_size,
            scales=scales,
            encode_chunk_size=encode_chunk_size,
            combine_scale_batches=combine_scale_batches,
            channels_last=channels_last,
        )

        for param in clip_model.parameters():
            param.requires_grad = False

        self.pyramid = MultiScalePyramid(feature_dim, hidden_dim, num_heads, dropout)
        self.fusion = TextVisualFusion(feature_dim, hidden_dim, num_heads, dropout)

        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._text_cache = {}

    def _apply(self, fn):
        self._text_cache.clear()
        return super()._apply(fn)

    def _prompts(self, category):
        return build_state_prompts(category, ensemble=self.prompt_ensemble)

    def _encode_text(self, device, category):
        """Encode normal/anomaly text prompts for a single category.

        Returns:
            (2, feature_dim) tensor: [normal_embedding, anomaly_embedding].
        """
        cache_key = (str(device), str(category))
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]

        clip = self.extractor.clip
        normal_prompts, anomaly_prompts = self._prompts(category)
        prompts = normal_prompts + anomaly_prompts
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(device)
            prompt_embeddings = F.normalize(
                clip.encode_text(tokens).float(), dim=-1
            )
            split = len(normal_prompts)
            normal_emb = F.normalize(
                prompt_embeddings[:split].mean(dim=0), dim=0
            )
            anomaly_emb = F.normalize(
                prompt_embeddings[split:].mean(dim=0), dim=0
            )
            emb = torch.stack([normal_emb, anomaly_emb], dim=0)

        self._text_cache[cache_key] = emb
        return emb

    def encode_text_batch(self, device, categories):
        """Encode text prompts for a batch of categories.

        Args:
            device: torch device.
            categories: list of category name strings (length B).

        Returns:
            (B, 2, feature_dim) tensor.
        """
        unique = set(categories)
        if len(unique) == 1:
            emb = self._encode_text(device, categories[0])
            return emb.unsqueeze(0).expand(len(categories), -1, -1)

        return torch.stack([self._encode_text(device, c) for c in categories])

    def forward_from_scale_features(self, scale_features, categories=None):
        """Run the trainable model from frozen multi-scale CLIP features.

        This entry point allows deterministic validation features to be cached
        without changing the trainable computation.

        Args:
            scale_features: Global, medium, and fine feature tensors returned by
                :class:`MultiScaleExtractor`.
            categories: Optional category names, one per sample.

        Returns:
            Model outputs with image-level scores and local patch scores.
        """
        feat_s1, feat_s2, feat_s3 = scale_features
        batch_size = feat_s1.size(0)

        pyramid_feat = self.pyramid(feat_s1, feat_s2, feat_s3)

        if categories is None:
            categories = ["object"] * batch_size

        text_emb = self.encode_text_batch(feat_s1.device, categories)
        fused = self.fusion(pyramid_feat, text_emb)

        refined = F.normalize(self.feature_head(fused), p=2, dim=1)
        logits = self.classifier(refined).squeeze(-1)
        visual_score = torch.sigmoid(logits)

        global_features = F.normalize(feat_s1, dim=-1).unsqueeze(1)
        local_features = F.normalize(torch.cat([feat_s2, feat_s3], dim=1), dim=-1)
        global_similarity = torch.einsum(
            "bnd,bkd->bnk", global_features, text_emb
        )
        local_similarity = torch.einsum(
            "bnd,bkd->bnk", local_features, text_emb
        )
        global_text_score = torch.softmax(
            global_similarity / self.text_temperature, dim=-1
        )[:, 0, 1]
        local_patch_scores = torch.softmax(
            local_similarity / self.text_temperature, dim=-1
        )[:, :, 1]
        topk = max(1, math.ceil(local_patch_scores.size(1) * self.local_topk))
        local_text_score = local_patch_scores.topk(topk, dim=1).values.mean(dim=1)
        text_score = (
            (1.0 - self.local_score_weight) * global_text_score
            + self.local_score_weight * local_text_score
        )

        weight_sum = self.visual_weight + self.text_weight
        scores = (
            self.visual_weight * visual_score + self.text_weight * text_score
        ) / weight_sum

        return {
            "logits": logits,
            "features": refined,
            "scores": scores,
            "visual_scores": visual_score,
            "text_scores": text_score,
            "local_patch_scores": local_patch_scores,
        }

    def forward(self, images, categories=None):
        """Encode images and run the image-level anomaly classifier.

        Args:
            images: (B, C, H, W) preprocessed image tensor.
            categories: Optional list of category names (length B).

        Returns:
            dict with keys:
                logits: (B,) raw classifier output.
                features: (B, hidden_dim) refined features for loss computation.
                scores: (B,) blended anomaly scores in [0, 1].
        """
        return self.forward_from_scale_features(
            self.extractor(images), categories=categories
        )


# Compatibility alias for checkpoints and downstream code created before the
# method was given a scientifically narrower working name.
RAMLModel = MultiScaleCLIPDetector
