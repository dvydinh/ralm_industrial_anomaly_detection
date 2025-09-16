"""Evaluate an MVTec-trained multi-scale CLIP checkpoint on VisA."""

import argparse
import json
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raml.data.transforms import build_multiscale_transform
from raml.data.visa_dataset import VisADataset
from raml.models.clip_loader import load_clip_model
from raml.utils.checkpointing import (
    load_checkpoint_model,
    validate_checkpoint_backbone,
)
from raml.utils.metrics import compute_per_category_metrics
from raml.utils.performance import autocast_context, configure_accelerator
from scripts.evaluate import build_model, load_checkpoint


logger = logging.getLogger("raml.eval_visa")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-scale CLIP on VisA")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--visa-root")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", default="results/result_visa_transfer.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint.get("config")
    if args.config and config is not None:
        raise ValueError(
            "A checkpoint-embedded configuration cannot be replaced; use the "
            "VisA-root and batch-size CLI overrides instead"
        )
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    if config is None:
        raise ValueError("No configuration was found in the checkpoint or CLI")

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    validate_checkpoint_backbone(checkpoint, config)
    configure_accelerator(training_config, device)
    visa_root = args.visa_root or data_config.get("visa_root")
    if not visa_root:
        raise ValueError("A prepared VisA one-class root is required")

    clip_model, clip_preprocess, tokenizer = load_clip_model(
        model_config["clip_model"],
        device,
        source=model_config.get("clip_source", "openai"),
    )
    image_size = int(data_config.get("image_size", 896))
    clip_input_size = int(model_config.get("clip_input_size", 224))
    scales = tuple(model_config.get("scales", (1, 2, 4)))
    effective_resolution = min(image_size, clip_input_size * max(scales))
    transform = build_multiscale_transform(
        clip_preprocess,
        image_size,
        resize_mode=data_config.get("resize_mode", "letterbox"),
    )
    dataset = VisADataset(
        visa_root,
        transform=transform,
        return_resolution=True,
        effective_resolution=effective_resolution,
    )
    loader_arguments = dict(
        dataset=dataset,
        batch_size=args.batch_size or config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )
    if loader_arguments["num_workers"] > 0:
        loader_arguments["prefetch_factor"] = training_config.get(
            "prefetch_factor", 2
        )
        loader_arguments["persistent_workers"] = False
    loader = DataLoader(**loader_arguments)

    model = build_model(config, clip_model, tokenizer).to(device)
    load_checkpoint_model(model, checkpoint)
    model.eval()
    labels = []
    scores = []
    categories = []
    amp_enabled = bool(training_config.get("amp", True) and device.type == "cuda")
    amp_dtype = training_config.get("amp_dtype", "float16")
    with torch.inference_mode():
        for images, batch_labels, batch_categories, _ in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled, amp_dtype):
                output = model(images, categories=list(batch_categories))
            scores.extend(output["scores"].float().cpu().tolist())
            labels.extend(int(label) for label in batch_labels.tolist())
            categories.extend(list(batch_categories))

    per_category, macro = compute_per_category_metrics(categories, labels, scores)
    result = {
        "method": "multi-scale CLIP cross-dataset transfer",
        "source_dataset": "MVTec AD",
        "target_dataset": "VisA",
        "macro": macro,
        "per_category": per_category,
        "predictions": {"category": categories, "label": labels, "score": scores},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    logger.info("VisA evaluation saved to %s", args.output)


if __name__ == "__main__":
    main()
