import pytest
from PIL import Image


pytest.importorskip("torch")
pytest.importorskip("torchvision")

from raml.data.transforms import ResizeLongestSideAndPad


def test_letterbox_preserves_complete_non_square_field_of_view():
    image = Image.new("RGB", (40, 20), color=(0, 0, 0))
    for x in range(4):
        for y in range(20):
            image.putpixel((x, y), (255, 0, 0))

    transformed = ResizeLongestSideAndPad(32, fill=(10, 20, 30))(image)

    assert transformed.size == (32, 32)
    assert transformed.getpixel((0, 16))[0] > 200
    assert transformed.getpixel((16, 0)) == (10, 20, 30)
