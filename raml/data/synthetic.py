"""Lightweight pseudo-anomaly generation for normal-only training."""

import math
import random

from PIL import ImageEnhance, ImageOps


class CutPasteAnomaly:
    """Create a local pseudo defect by transforming and relocating a crop.

    This implementation follows the training principle of CutPaste while
    keeping the augmentation dependency-free beyond Pillow. It is intended as
    a reproducible baseline, not as a substitute for real defect annotations.
    """

    def __init__(
        self,
        min_area_ratio=0.02,
        max_area_ratio=0.15,
        min_aspect_ratio=0.3,
        max_aspect_ratio=3.3,
    ):
        if not 0 < min_area_ratio <= max_area_ratio < 1:
            raise ValueError("area ratios must satisfy 0 < min <= max < 1")
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def __call__(self, image):
        width, height = image.size
        image_area = width * height
        patch_area = random.uniform(
            self.min_area_ratio, self.max_area_ratio
        ) * image_area
        log_aspect = random.uniform(
            math.log(self.min_aspect_ratio), math.log(self.max_aspect_ratio)
        )
        aspect = math.exp(log_aspect)

        patch_width = max(2, min(width - 1, int(round(math.sqrt(patch_area * aspect)))))
        patch_height = max(2, min(height - 1, int(round(math.sqrt(patch_area / aspect)))))

        source_x = random.randint(0, max(0, width - patch_width))
        source_y = random.randint(0, max(0, height - patch_height))
        patch = image.crop(
            (source_x, source_y, source_x + patch_width, source_y + patch_height)
        )

        patch = ImageEnhance.Color(patch).enhance(random.uniform(0.6, 1.4))
        patch = ImageEnhance.Contrast(patch).enhance(random.uniform(0.7, 1.4))
        if random.random() < 0.5:
            patch = ImageOps.mirror(patch)

        target_x = random.randint(0, max(0, width - patch_width))
        target_y = random.randint(0, max(0, height - patch_height))
        augmented = image.copy()
        augmented.paste(patch, (target_x, target_y))
        return augmented
