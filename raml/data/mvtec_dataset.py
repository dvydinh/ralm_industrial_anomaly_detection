"""MVTec AD loading with leakage-free train, validation, and test splits."""

import logging
import os
import random
import zlib

from PIL import Image
from torch.utils.data import Dataset

from raml.data.synthetic import CutPasteAnomaly


logger = logging.getLogger(__name__)

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def _partition(items, validation_ratio, rng):
    items = list(items)
    rng.shuffle(items)
    if validation_ratio <= 0 or len(items) < 2:
        return items, []
    validation_count = int(round(len(items) * validation_ratio))
    validation_count = max(1, min(len(items) - 1, validation_count))
    return items[validation_count:], items[:validation_count]


class MVTecDataset(Dataset):
    """Load MVTec AD without using official test labels during standard training.

    The standard protocol partitions only the official normal training images
    into train and validation subsets, then evaluates once on the complete
    official test set. Optional pseudo anomalies are generated from a duplicate
    of each normal training or validation image.

    ``use_test_anomalies`` is retained only for explicitly labelled legacy
    supervised experiments. Results from that mode are not comparable with the
    standard unsupervised or few-normal-shot literature.
    """

    def __init__(
        self,
        data_dir,
        categories=None,
        split="train",
        transform=None,
        use_test_anomalies=False,
        train_ratio=0.8,
        seed=42,
        validation_ratio=0.1,
        synthetic_anomalies=False,
        synthetic_transform=None,
        return_resolution=False,
        effective_resolution=896,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if not 0 <= validation_ratio < 1:
            raise ValueError("validation_ratio must be in [0, 1)")
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be in (0, 1)")

        self.transform = transform
        self.split = split
        self.seed = seed
        self.return_resolution = return_resolution
        self.effective_resolution = float(effective_resolution)
        self.synthetic_transform = synthetic_transform or CutPasteAnomaly()
        self.samples = []

        categories = MVTEC_CATEGORIES if categories is None else list(categories)
        requested_categories = set(categories)
        found_categories = set()

        for category in categories:
            category_dir = os.path.join(data_dir, category)
            if not os.path.isdir(category_dir):
                logger.warning("Category directory not found: %s", category_dir)
                continue

            found_categories.add(category)
            category_seed = seed + zlib.crc32(category.encode("utf-8"))
            category_rng = random.Random(category_seed)
            train_paths = self._list_images(
                os.path.join(category_dir, "train", "good")
            )
            test_good, test_anomaly = self._test_samples(category_dir, category)

            if use_test_anomalies:
                self._add_legacy_supervised_split(
                    train_paths,
                    test_good,
                    test_anomaly,
                    category,
                    split,
                    train_ratio,
                    validation_ratio,
                    category_rng,
                )
            elif split == "test":
                self.samples.extend(test_good)
                self.samples.extend(test_anomaly)
            else:
                normal_items = [
                    {"path": path, "label": 0, "category": category}
                    for path in train_paths
                ]
                train_items, validation_items = _partition(
                    normal_items, validation_ratio, category_rng
                )
                self.samples.extend(
                    train_items if split == "train" else validation_items
                )

        missing_categories = sorted(requested_categories - found_categories)
        if missing_categories:
            raise FileNotFoundError(
                "MVTec AD category directories were not found: "
                + ", ".join(missing_categories)
            )

        if synthetic_anomalies and split in {"train", "val"}:
            pseudo_anomalies = [
                {**item, "label": 1, "synthetic": True}
                for item in self.samples
                if item["label"] == 0
            ]
            for item in self.samples:
                item.setdefault("synthetic", False)
            self.samples.extend(pseudo_anomalies)
        else:
            for item in self.samples:
                item.setdefault("synthetic", False)

        logger.info(
            "MVTecDataset split=%s: %d samples across %d categories",
            split,
            len(self.samples),
            len(found_categories),
        )
        if not self.samples:
            raise RuntimeError(
                f"No MVTec AD samples found for split={split!r} under {data_dir!r}"
            )
        if split == "test":
            labels_by_category = {
                category: {
                    item["label"]
                    for item in self.samples
                    if item["category"] == category
                }
                for category in categories
            }
            invalid = [
                category
                for category, labels in labels_by_category.items()
                if labels != {0, 1}
            ]
            if invalid:
                raise RuntimeError(
                    "Every MVTec AD test category must contain normal and anomaly "
                    "images; invalid categories: " + ", ".join(invalid)
                )

    def _add_legacy_supervised_split(
        self,
        train_paths,
        test_good,
        test_anomaly,
        category,
        split,
        train_ratio,
        validation_ratio,
        rng,
    ):
        shuffled_good = list(test_good)
        shuffled_anomaly = list(test_anomaly)
        rng.shuffle(shuffled_good)
        rng.shuffle(shuffled_anomaly)
        good_cut = int(len(shuffled_good) * train_ratio)
        anomaly_cut = int(len(shuffled_anomaly) * train_ratio)

        if split == "test":
            self.samples.extend(shuffled_good[good_cut:])
            self.samples.extend(shuffled_anomaly[anomaly_cut:])
            return

        pool = [
            {"path": path, "label": 0, "category": category}
            for path in train_paths
        ]
        pool.extend(shuffled_good[:good_cut])
        pool.extend(shuffled_anomaly[:anomaly_cut])
        normal_pool = [item for item in pool if item["label"] == 0]
        anomaly_pool = [item for item in pool if item["label"] == 1]
        normal_train, normal_validation = _partition(
            normal_pool, validation_ratio, rng
        )
        anomaly_train, anomaly_validation = _partition(
            anomaly_pool, validation_ratio, rng
        )
        if split == "train":
            self.samples.extend(normal_train + anomaly_train)
        else:
            self.samples.extend(normal_validation + anomaly_validation)

    @classmethod
    def _test_samples(cls, category_dir, category):
        test_good = []
        test_anomaly = []
        test_dir = os.path.join(category_dir, "test")
        if not os.path.isdir(test_dir):
            return test_good, test_anomaly

        for defect_type in sorted(os.listdir(test_dir)):
            defect_dir = os.path.join(test_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            label = 0 if defect_type == "good" else 1
            destination = test_good if label == 0 else test_anomaly
            destination.extend(
                {
                    "path": path,
                    "label": label,
                    "category": category,
                    "defect_type": defect_type,
                }
                for path in cls._list_images(defect_dir)
            )
        return test_good, test_anomaly

    @staticmethod
    def _list_images(directory):
        if not os.path.isdir(directory):
            return []
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        return sorted(
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if os.path.splitext(name)[1].lower() in extensions
        )

    def __len__(self):
        return len(self.samples)

    def _apply_synthetic_transform(self, image, item):
        if self.split != "val":
            return self.synthetic_transform(image)

        # Keep validation pseudo anomalies fixed across epochs without changing
        # the random state used by the training worker.
        random_state = random.getstate()
        path_seed = zlib.crc32(item["path"].encode("utf-8"))
        random.seed(self.seed + path_seed)
        try:
            return self.synthetic_transform(image)
        finally:
            random.setstate(random_state)

    def __getitem__(self, index):
        item = self.samples[index]
        try:
            image = Image.open(item["path"]).convert("RGB")
        except (OSError, IOError) as error:
            logger.error("Failed to load image %s: %s", item["path"], error)
            raise

        original_long_side = max(image.size)
        if item["synthetic"]:
            image = self._apply_synthetic_transform(image, item)
        if self.transform is not None:
            image = self.transform(image)

        output = (image, item["label"], item["category"])
        if not self.return_resolution:
            return output

        resolution_retention = min(
            1.0, self.effective_resolution / max(1, original_long_side)
        )
        return output + (resolution_retention,)
