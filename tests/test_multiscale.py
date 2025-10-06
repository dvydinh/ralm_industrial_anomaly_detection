import pytest


torch = pytest.importorskip("torch")

from raml.models.multi_scale import MultiScaleExtractor, extract_patches


class FakeClip(torch.nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.seen_shapes = []

    def encode_image(self, images):
        self.seen_shapes.append(tuple(images.shape))
        means = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        offsets = torch.arange(
            self.feature_dim, device=images.device, dtype=images.dtype
        ).unsqueeze(0)
        return means + offsets


def test_extract_patches_preserves_channels_and_row_major_grid_order():
    image = torch.arange(3 * 4 * 4).reshape(1, 3, 4, 4)
    patches = extract_patches(image, grid_size=2)
    expected = torch.stack(
        [
            image[0, :, 0:2, 0:2],
            image[0, :, 0:2, 2:4],
            image[0, :, 2:4, 0:2],
            image[0, :, 2:4, 2:4],
        ]
    )
    assert torch.equal(patches, expected)


def test_extractor_resizes_every_view_and_returns_expected_shapes():
    clip = FakeClip(feature_dim=8)
    extractor = MultiScaleExtractor(
        clip,
        clip_input_size=4,
        scales=(1, 2, 4),
        encode_chunk_size=5,
    )
    outputs = extractor(torch.randn(2, 3, 8, 8))
    assert [tuple(output.shape) for output in outputs] == [
        (2, 8),
        (2, 4, 8),
        (2, 16, 8),
    ]
    assert all(shape[-2:] == (4, 4) for shape in clip.seen_shapes)


def test_extract_patches_rejects_non_divisible_shapes():
    with pytest.raises(ValueError, match="divisible"):
        extract_patches(torch.randn(1, 3, 7, 8), grid_size=4)


def test_combined_scale_batching_preserves_features():
    images = torch.randn(2, 3, 8, 8)
    combined = MultiScaleExtractor(
        FakeClip(feature_dim=8),
        clip_input_size=4,
        scales=(1, 2, 4),
        encode_chunk_size=5,
        combine_scale_batches=True,
    )(images)
    separate = MultiScaleExtractor(
        FakeClip(feature_dim=8),
        clip_input_size=4,
        scales=(1, 2, 4),
        encode_chunk_size=5,
        combine_scale_batches=False,
    )(images)
    for combined_features, separate_features in zip(combined, separate):
        assert torch.equal(combined_features, separate_features)
