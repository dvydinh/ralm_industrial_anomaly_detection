"""VisA loader for the official prepared one-class test split."""

import logging
import os

from PIL import Image
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


class VisADataset(Dataset):
    """Read the test split produced by the official VisA preparation script.

    ``data_dir`` may point either to ``VisA_pytorch`` or directly to its
    ``1cls`` child. Raw VisA data must first be prepared with the official
    ``utils/prepare_data.py`` command so the published split is preserved.
    """

    def __init__(
        self,
        data_dir,
        categories=None,
        transform=None,
        return_resolution=False,
        effective_resolution=896,
    ):
        self.transform = transform
        self.return_resolution = return_resolution
        self.effective_resolution = float(effective_resolution)
        self.samples = []
        categories = VISA_CATEGORIES if categories is None else list(categories)
        requested_categories = set(categories)

        roots = [data_dir, os.path.join(data_dir, "1cls")]
        prepared_root = next(
            (
                root
                for root in roots
                if any(os.path.isdir(os.path.join(root, category, "test")) for category in categories)
            ),
            None,
        )
        if prepared_root is None:
            raise FileNotFoundError(
                "Expected the official prepared VisA one-class tree under "
                f"{data_dir!r}; see the dataset preparation instructions"
            )

        found_categories = set()
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for category in categories:
            test_directory = os.path.join(prepared_root, category, "test")
            if not os.path.isdir(test_directory):
                logger.warning("VisA test directory not found: %s", test_directory)
                continue
            found_categories.add(category)

            for state in sorted(os.listdir(test_directory)):
                state_directory = os.path.join(test_directory, state)
                if not os.path.isdir(state_directory):
                    continue
                label = 0 if state.lower() in {"good", "normal"} else 1
                for name in sorted(os.listdir(state_directory)):
                    path = os.path.join(state_directory, name)
                    if os.path.splitext(name)[1].lower() not in extensions:
                        continue
                    self.samples.append(
                        {"path": path, "label": label, "category": category}
                    )

        missing_categories = sorted(requested_categories - found_categories)
        if missing_categories:
            raise FileNotFoundError(
                "Prepared VisA category directories were not found: "
                + ", ".join(missing_categories)
            )
        if not self.samples:
            raise RuntimeError(f"No VisA test samples found under {prepared_root!r}")
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
                "Every VisA test category must contain normal and anomaly images; "
                "invalid categories: " + ", ".join(invalid)
            )
        logger.info(
            "VisADataset: %d samples across %d categories",
            len(self.samples),
            len(found_categories),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        try:
            image = Image.open(item["path"]).convert("RGB")
        except (OSError, IOError) as error:
            logger.error("Failed to load image %s: %s", item["path"], error)
            raise

        original_long_side = max(image.size)
        if self.transform is not None:
            image = self.transform(image)
        output = (image, item["label"], item["category"])
        if not self.return_resolution:
            return output
        resolution_retention = min(
            1.0, self.effective_resolution / max(1, original_long_side)
        )
        return output + (resolution_retention,)
