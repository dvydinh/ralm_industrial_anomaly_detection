from pathlib import Path

import pytest
from PIL import Image


pytest.importorskip("torch")

from raml.data.mvtec_dataset import MVTecDataset


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(100, 120, 140)).save(path)


def _make_category(root, category):
    for index in range(10):
        _write_image(root / category / "train" / "good" / f"train_{index}.png")
    for index in range(2):
        _write_image(root / category / "test" / "good" / f"good_{index}.png")
        _write_image(root / category / "test" / "scratch" / f"bad_{index}.png")


def _paths(dataset, category):
    return {
        Path(item["path"]).name
        for item in dataset.samples
        if item["category"] == category
    }


def test_split_is_category_order_independent_and_source_disjoint(tmp_path):
    _make_category(tmp_path, "a")
    _make_category(tmp_path, "b")
    combined = MVTecDataset(
        tmp_path, categories=["a", "b"], split="val", seed=7, validation_ratio=0.2
    )
    standalone = MVTecDataset(
        tmp_path, categories=["b"], split="val", seed=7, validation_ratio=0.2
    )
    assert _paths(combined, "b") == _paths(standalone, "b")

    train = MVTecDataset(
        tmp_path, categories=["a"], split="train", seed=7, validation_ratio=0.2
    )
    validation = MVTecDataset(
        tmp_path, categories=["a"], split="val", seed=7, validation_ratio=0.2
    )
    test = MVTecDataset(tmp_path, categories=["a"], split="test", seed=7)
    train_paths = {item["path"] for item in train.samples}
    validation_paths = {item["path"] for item in validation.samples}
    test_paths = {item["path"] for item in test.samples}
    assert train_paths.isdisjoint(validation_paths)
    assert train_paths.isdisjoint(test_paths)
    assert validation_paths.isdisjoint(test_paths)


def test_synthetic_mode_duplicates_normal_samples_with_positive_labels(tmp_path):
    _make_category(tmp_path, "a")
    dataset = MVTecDataset(
        tmp_path,
        categories=["a"],
        split="train",
        seed=7,
        validation_ratio=0.2,
        synthetic_anomalies=True,
    )
    labels = [item["label"] for item in dataset.samples]
    assert labels.count(0) == labels.count(1)
    assert sum(item["synthetic"] for item in dataset.samples) == labels.count(1)
