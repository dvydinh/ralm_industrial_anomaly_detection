"""Multi-scale patch extraction for CLIP-based anomaly detection.

CLIP processes fixed 224x224 inputs.  Industrial images are typically
700-1024 pixels, so resizing causes information loss for small defects.

We split each image into a grid of non-overlapping patches at multiple
scales, encode each patch independently through CLIP, and aggregate the
resulting features.

    Scale 1: 1x1  (global)  ->  1 feature per image
    Scale 2: 2x2  (medium)  ->  4 features per image
    Scale 3: 4x4  (fine)    -> 16 features per image

This is conceptually similar to the window-based extraction in WinCLIP
(Jeong et al., CVPR 2023, Section 3.2) but applied at the image level
for classification rather than pixel-level segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_patches(images, grid_size):
    """Split images into a grid of non-overlapping patches.

    Args:
        images: (B, C, H, W) tensor.
        grid_size: Number of patches per spatial dimension.

    Returns:
        (B * grid_size^2, C, H // grid_size, W // grid_size) tensor.
    """
    if images.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")
    if grid_size < 1:
        raise ValueError("grid_size must be a positive integer")

    batch_size, channels, height, width = images.shape
    if height % grid_size or width % grid_size:
        raise ValueError(
            f"image size {(height, width)} must be divisible by grid_size={grid_size}"
        )

    patch_height = height // grid_size
    patch_width = width // grid_size

    # unfold returns (B, C, grid_h, grid_w, patch_h, patch_w). Move the
    # spatial grid ahead of channels before flattening; a direct view mixes
    # channel values from different grid cells.
    patches = images.unfold(2, patch_height, patch_height).unfold(
        3, patch_width, patch_width
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(
        batch_size * grid_size * grid_size,
        channels,
        patch_height,
        patch_width,
    )
    return patches


class MultiScaleExtractor(nn.Module):
    """Extract CLIP features at multiple spatial scales.

    The CLIP backbone is kept frozen; only the downstream projection layers
    are trainable.

    Args:
        clip_model: A CLIP model instance (frozen).
        clip_input_size: Expected input size for CLIP (default 224).
        scales: Tuple of grid sizes.  (1, 2, 4) gives 1+4+16 = 21 patches.
    """

    def __init__(
        self,
        clip_model,
        clip_input_size=224,
        scales=(1, 2, 4),
        encode_chunk_size=64,
        combine_scale_batches=True,
        channels_last=False,
    ):
        super().__init__()
        self.clip = clip_model
        self.clip_input_size = clip_input_size
        self.scales = tuple(scales)
        self.encode_chunk_size = encode_chunk_size
        self.combine_scale_batches = bool(combine_scale_batches)
        self.channels_last = bool(channels_last)

        if not self.scales or self.scales[0] != 1:
            raise ValueError("scales must start with the global 1x1 view")
        if any(scale < 1 for scale in self.scales):
            raise ValueError("all scales must be positive integers")
        if encode_chunk_size < 1:
            raise ValueError("encode_chunk_size must be positive")

    def train(self, mode=True):
        """Keep the frozen CLIP backbone in evaluation mode."""
        super().train(mode)
        self.clip.eval()
        return self

    def _encode(self, images):
        """Encode views in bounded chunks to control peak accelerator memory."""
        features = []
        for chunk in images.split(self.encode_chunk_size, dim=0):
            if self.channels_last:
                chunk = chunk.contiguous(memory_format=torch.channels_last)
            features.append(self.clip.encode_image(chunk).float())
        return torch.cat(features, dim=0)

    def _prepare_views(self, images, grid_size):
        patches = extract_patches(images, grid_size)
        size = self.clip_input_size
        if patches.shape[-2:] != (size, size):
            patches = F.interpolate(
                patches,
                size=(size, size),
                mode="bicubic",
                align_corners=False,
            )
        return patches

    @torch.no_grad()
    def forward(self, images):
        """Encode images at each scale.

        Args:
            images: (B, C, H, W) tensor, already preprocessed for CLIP.

        Returns:
            features: list of tensors, one per scale.
                Scale s with grid_size g produces (B, g*g, D).
                Scale 1 (global) produces (B, D) which we keep as-is.
        """
        batch_size = images.size(0)
        view_batches = [
            self._prepare_views(images, grid_size) for grid_size in self.scales
        ]

        if self.combine_scale_batches:
            # A single chunk stream avoids launching a separate encoder loop for
            # every scale. The chunk size still bounds peak activation memory.
            encoded = self._encode(torch.cat(view_batches, dim=0))
            feature_batches = encoded.split(
                [views.size(0) for views in view_batches], dim=0
            )
        else:
            feature_batches = [self._encode(views) for views in view_batches]

        results = []
        for grid_size, features in zip(self.scales, feature_batches):
            if grid_size == 1:
                results.append(features.view(batch_size, -1))
            else:
                results.append(features.view(batch_size, grid_size * grid_size, -1))
        return results
