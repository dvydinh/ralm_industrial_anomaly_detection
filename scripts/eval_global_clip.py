"""Evaluate a global CLIP compositional-prompt baseline on MVTec AD."""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raml.data.mvtec_dataset import MVTecDataset
from raml.models.clip_loader import load_clip_model
from raml.models.prompts import build_state_prompts
from raml.utils.metrics import compute_per_category_metrics
from raml.utils.performance import autocast_context, configure_accelerator


logger = logging.getLogger("raml.eval_global_clip")


def main():
    parser = argparse.ArgumentParser(
        description="Global CLIP compositional-prompt baseline"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mvtec-root")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", default="results/result_global_clip.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_accelerator(training_config, device)
    clip_model, preprocess, tokenizer = load_clip_model(
        model_config["clip_model"],
        device,
        source=model_config.get("clip_source", "openai"),
    )
    clip_model.eval()

    dataset = MVTecDataset(
        args.mvtec_root or data_config["mvtec_root"],
        categories=args.categories or data_config.get("categories"),
        split="test",
        transform=preprocess,
        use_test_anomalies=False,
        seed=data_config.get("split_seed", 42),
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
    text_temperature = float(model_config.get("text_temperature", 0.01))
    text_cache = {}

    def text_prototypes(category):
        if category in text_cache:
            return text_cache[category]
        normal_prompts, anomaly_prompts = build_state_prompts(category, ensemble=True)
        prompts = normal_prompts + anomaly_prompts
        with torch.no_grad():
            embeddings = F.normalize(
                clip_model.encode_text(tokenizer(prompts).to(device)).float(),
                dim=-1,
            )
            split = len(normal_prompts)
            normal = F.normalize(embeddings[:split].mean(dim=0), dim=0)
            anomaly = F.normalize(embeddings[split:].mean(dim=0), dim=0)
            prototypes = torch.stack([normal, anomaly], dim=0)
        text_cache[category] = prototypes
        return prototypes

    labels = []
    scores = []
    categories = []
    amp_enabled = bool(training_config.get("amp", True) and device.type == "cuda")
    amp_dtype = training_config.get("amp_dtype", "float16")
    with torch.inference_mode():
        for images, batch_labels, batch_categories in loader:
            images = images.to(device, non_blocking=True)
            if model_config.get("channels_last", False):
                images = images.contiguous(memory_format=torch.channels_last)
            with autocast_context(device, amp_enabled, amp_dtype):
                image_features = F.normalize(
                    clip_model.encode_image(images).float(), dim=-1
                )
            prototypes = torch.stack(
                [text_prototypes(category) for category in batch_categories]
            )
            similarities = torch.einsum(
                "bd,bkd->bk", image_features, prototypes
            )
            batch_scores = torch.softmax(
                similarities / text_temperature, dim=-1
            )[:, 1]
            scores.extend(batch_scores.cpu().tolist())
            labels.extend(int(label) for label in batch_labels.tolist())
            categories.extend(list(batch_categories))

    per_category, macro = compute_per_category_metrics(categories, labels, scores)
    result = {
        "method": "global CLIP compositional-prompt baseline",
        "backbone": model_config["clip_model"],
        "macro": macro,
        "per_category": per_category,
        "predictions": {"category": categories, "label": labels, "score": scores},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    logger.info("Baseline results saved to %s", args.output)


if __name__ == "__main__":
    main()
