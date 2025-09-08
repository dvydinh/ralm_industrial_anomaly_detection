"""Evaluate a trained multi-scale CLIP checkpoint on MVTec AD."""

import argparse
import json
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raml.data.mvtec_dataset import MVTecDataset
from raml.data.transforms import build_multiscale_transform
from raml.models.clip_loader import load_clip_model
from raml.models.raml_model import MultiScaleCLIPDetector
from raml.utils.checkpointing import (
    load_checkpoint_model,
    validate_checkpoint_backbone,
)
from raml.utils.metrics import compute_per_category_metrics
from raml.utils.performance import autocast_context, configure_accelerator


logger = logging.getLogger("raml.evaluate")


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(config, clip_model, tokenizer):
    model_config = config["model"]
    return MultiScaleCLIPDetector(
        clip_model,
        tokenizer,
        feature_dim=model_config["feature_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_attention_heads"],
        dropout=model_config["dropout"],
        visual_weight=config["inference"]["visual_weight"],
        text_weight=config["inference"]["text_weight"],
        clip_input_size=model_config.get("clip_input_size", 224),
        scales=tuple(model_config.get("scales", (1, 2, 4))),
        encode_chunk_size=model_config.get("encode_chunk_size", 64),
        combine_scale_batches=model_config.get("combine_scale_batches", True),
        channels_last=model_config.get("channels_last", False),
        prompt_ensemble=model_config.get("prompt_ensemble", True),
        text_temperature=model_config.get("text_temperature", 0.01),
        local_score_weight=model_config.get("local_score_weight", 0.5),
        local_topk=model_config.get("local_topk", 0.1),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate the multi-scale CLIP detector")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional configuration override")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--mvtec-root")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", default="results/evaluation.json")
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
            "A checkpoint-embedded configuration cannot be replaced; use CLI "
            "path, category, and batch-size overrides instead"
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
    mvtec_root = args.mvtec_root or data_config["mvtec_root"]
    categories = args.categories or data_config.get("categories")
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

    dataset = MVTecDataset(
        mvtec_root,
        categories=categories,
        split="test",
        transform=transform,
        use_test_anomalies=data_config.get("use_test_anomalies_for_train", False),
        train_ratio=data_config.get("train_ratio", 0.8),
        seed=data_config.get("split_seed", 42),
        validation_ratio=data_config.get("validation_ratio", 0.1),
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
    sample_categories = []
    amp_enabled = bool(training_config.get("amp", True) and device.type == "cuda")
    amp_dtype = training_config.get("amp_dtype", "float16")
    with torch.inference_mode():
        for images, batch_labels, batch_categories, _ in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled, amp_dtype):
                output = model(images, categories=list(batch_categories))
            scores.extend(output["scores"].float().cpu().tolist())
            labels.extend(int(label) for label in batch_labels.tolist())
            sample_categories.extend(list(batch_categories))

    per_category, macro = compute_per_category_metrics(
        sample_categories,
        labels,
        scores,
        thresholds=checkpoint.get("validation_thresholds"),
    )
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "macro": macro,
        "per_category": per_category,
        "predictions": {
            "category": sample_categories,
            "label": labels,
            "score": scores,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    logger.info("Evaluation saved to %s", args.output)


if __name__ == "__main__":
    main()
