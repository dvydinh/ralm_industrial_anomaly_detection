"""Preprocessing for high-resolution multi-scale CLIP inputs."""

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _normalization_from(preprocess):
    """Reuse the normalization supplied with a loaded CLIP checkpoint."""
    for transform in reversed(getattr(preprocess, "transforms", ())):
        if hasattr(transform, "mean") and hasattr(transform, "std"):
            return transforms.Normalize(
                mean=tuple(float(value) for value in transform.mean),
                std=tuple(float(value) for value in transform.std),
            )
    return transforms.Normalize(CLIP_MEAN, CLIP_STD)


class ResizeLongestSideAndPad:
    """Preserve the complete field of view on a fixed square canvas."""

    def __init__(self, image_size, fill):
        self.image_size = int(image_size)
        self.fill = tuple(int(value) for value in fill)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("ResizeLongestSideAndPad expects a Pillow image")
        width, height = image.size
        scale = self.image_size / max(width, height)
        resized_width = max(1, min(self.image_size, round(width * scale)))
        resized_height = max(1, min(self.image_size, round(height * scale)))
        image = transform_functional.resize(
            image,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
        )
        horizontal = self.image_size - resized_width
        vertical = self.image_size - resized_height
        padding = [
            horizontal // 2,
            vertical // 2,
            horizontal - horizontal // 2,
            vertical - vertical // 2,
        ]
        return transform_functional.pad(image, padding, fill=self.fill)


def build_multiscale_transform(preprocess, image_size=896, resize_mode="letterbox"):
    """Build a fixed-size transform that preserves detail before patching.

    The standard CLIP transform first reduces an image to the backbone input
    size. That is appropriate for a single global view but discards the detail
    needed by a multi-scale crop pipeline. This transform keeps a larger square
    view; each crop is resized to the CLIP input size inside the extractor.
    """
    if image_size < 1:
        raise ValueError("image_size must be a positive integer")

    normalization = _normalization_from(preprocess)
    if resize_mode == "letterbox":
        fill = tuple(round(float(value) * 255) for value in normalization.mean)
        spatial_transform = ResizeLongestSideAndPad(image_size, fill=fill)
    elif resize_mode == "center_crop":
        spatial_transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
            ]
        )
    else:
        raise ValueError("resize_mode must be letterbox or center_crop")

    return transforms.Compose(
        [spatial_transform, transforms.ToTensor(), normalization]
    )
