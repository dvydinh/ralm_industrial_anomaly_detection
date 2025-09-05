"""Train the multi-scale CLIP detector with an untouched official test set."""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raml.data.mvtec_dataset import MVTecDataset
from raml.data.transforms import build_multiscale_transform
from raml.losses.combined_loss import CompositeAnomalyLoss
from raml.models.clip_loader import load_clip_model
from raml.models.raml_model import MultiScaleCLIPDetector
from raml.utils.checkpointing import (
    checkpoint_backbone_metadata,
    compact_model_state_dict,
    configuration_sha256,
    load_checkpoint_model,
    validate_checkpoint_backbone,
)
from raml.utils.metrics import (
    compute_per_category_metrics,
    select_per_category_thresholds,
)
from raml.utils.performance import (
    autocast_context,
    configure_accelerator,
    cuda_device_summary,
)
from raml.utils.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    seed_worker,
)


logger = logging.getLogger("raml.train")
CHECKPOINT_SCHEMA_VERSION = 3


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoint(path, device):
    """Load trusted local checkpoints across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_config(config, args):
    """Apply command-line overrides and return the exact configuration used."""
    resolved = copy.deepcopy(config)
    if args.epochs is not None:
        resolved["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        resolved["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        resolved["training"]["learning_rate"] = args.lr
    if args.categories is not None:
        resolved["data"]["categories"] = args.categories
    if args.mvtec_root is not None:
        resolved["data"]["mvtec_root"] = args.mvtec_root
    return resolved


def validate_config(config):
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    scales = tuple(model_config.get("scales", (1, 2, 4)))
    image_size = int(data_config.get("image_size", 896))

    if len(scales) != 3 or scales[0] != 1 or any(scale < 1 for scale in scales):
        raise ValueError("model.scales must contain three positive values starting at 1")
    if any(image_size % scale for scale in scales):
        raise ValueError("data.image_size must be divisible by every model scale")
    if training_config["epochs"] < 1 or training_config["batch_size"] < 1:
        raise ValueError("training epochs and batch size must be positive")
    if training_config.get("amp_dtype", "float16") not in {"float16", "bfloat16"}:
        raise ValueError("training.amp_dtype must be float16 or bfloat16")
    if int(training_config.get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers cannot be negative")
    if int(training_config.get("prefetch_factor", 2)) < 1:
        raise ValueError("training.prefetch_factor must be positive")
    if not 0 <= data_config.get("validation_ratio", 0.1) < 1:
        raise ValueError("data.validation_ratio must be in [0, 1)")
    if (
        data_config.get("use_test_anomalies_for_train", False)
        and data_config.get("use_synthetic_anomalies", False)
    ):
        raise ValueError(
            "Choose either real test-anomaly reuse or synthetic anomalies, not both"
        )
    if not (
        data_config.get("use_test_anomalies_for_train", False)
        or data_config.get("use_synthetic_anomalies", False)
    ):
        raise ValueError(
            "The binary training objective requires synthetic or real anomaly examples"
        )


def find_mvtec_root(configured_path):
    """Resolve a configured path or a common Kaggle dataset location."""
    if os.path.isdir(configured_path):
        return os.path.abspath(configured_path)

    candidates = (
        "/data/mvtec-anomaly-detection",
        "/data/mvtec-ad",
        "/data/mvtecad",
    )
    for path in candidates:
        if os.path.isdir(os.path.join(path, "bottle")):
            return path

    if os.path.isdir("/data"):
        for root, directories, _ in os.walk("/data"):
            if "bottle" in directories and "carpet" in directories:
                return root
    raise FileNotFoundError(f"MVTec AD was not found under {configured_path!r}")


def build_manifest(datasets, dataset_root):
    manifest = {}
    for split, dataset in datasets.items():
        manifest[split] = [
            {
                "path": os.path.relpath(item["path"], dataset_root),
                "label": int(item["label"]),
                "category": item["category"],
                "synthetic": bool(item.get("synthetic", False)),
            }
            for item in dataset.samples
        ]
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return manifest, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def assert_disjoint_source_images(datasets):
    """Ensure that no original file crosses train, validation, and test splits."""
    source_paths = {
        split: {os.path.realpath(item["path"]) for item in dataset.samples}
        for split, dataset in datasets.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = source_paths[left] & source_paths[right]
        if overlap:
            example = next(iter(overlap))
            raise RuntimeError(f"Data leakage between {left} and {right}: {example}")


def make_loader(
    dataset,
    batch_size,
    shuffle,
    workers,
    pin_memory,
    seed,
    prefetch_factor=2,
    persistent_workers=False,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    arguments = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if workers > 0:
        arguments["prefetch_factor"] = prefetch_factor
        arguments["persistent_workers"] = persistent_workers
    loader = DataLoader(**arguments)
    return loader, generator


def train_epoch(model, loader, loss_fn, optimizer, scaler, device, config):
    model.train()
    amp_enabled = bool(config["training"].get("amp", True) and device.type == "cuda")
    amp_dtype = config["training"].get("amp_dtype", "float16")
    max_grad_norm = config["training"]["max_grad_norm"]
    bce_weight = config["loss"].get("bce_weight", 1.0)
    totals = {
        "loss": 0.0,
        "bce": 0.0,
        "center": 0.0,
        "margin": 0.0,
        "contrastive": 0.0,
        "active_hinge_fraction": 0.0,
        "gradient_norm": 0.0,
    }
    sample_count = 0

    for images, labels, categories, resolution_retention in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        resolution_retention = resolution_retention.to(
            device, non_blocking=True, dtype=images.dtype
        )
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled, amp_dtype):
            output = model(images, categories=list(categories))
            maccl_loss, loss_info = loss_fn(
                output["features"],
                labels,
                list(categories),
                resolution_retention=resolution_retention,
            )
            bce_loss = F.binary_cross_entropy_with_logits(output["logits"], labels)
            loss = maccl_loss + bce_weight * bce_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss: {float(loss.detach())}; details={loss_info}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        sample_count += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["bce"] += float(bce_loss.detach()) * batch_size
        totals["center"] += loss_info["center_loss"] * batch_size
        totals["margin"] += loss_info["margin_loss"] * batch_size
        totals["contrastive"] += loss_info["contrastive_loss"] * batch_size
        totals["active_hinge_fraction"] += (
            loss_info["active_hinge_fraction"] * batch_size
        )
        totals["gradient_norm"] += float(gradient_norm.detach()) * batch_size

    if sample_count == 0:
        raise RuntimeError("The training loader is empty")
    return {name: value / sample_count for name, value in totals.items()}


def evaluate(model, loader, device, config, thresholds=None, select_thresholds=False):
    model.eval()
    labels = []
    scores = []
    categories = []

    amp_enabled = bool(config["training"].get("amp", True) and device.type == "cuda")
    amp_dtype = config["training"].get("amp_dtype", "float16")
    with torch.inference_mode():
        for images, batch_labels, batch_categories, _ in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled, amp_dtype):
                output = model(images, categories=list(batch_categories))
            scores.extend(output["scores"].float().cpu().tolist())
            labels.extend(batch_labels.tolist())
            categories.extend(list(batch_categories))

    selected_thresholds = (
        select_per_category_thresholds(categories, labels, scores)
        if select_thresholds
        else thresholds
    )
    per_category, macro = compute_per_category_metrics(
        categories, labels, scores, thresholds=selected_thresholds
    )
    result = {
        "per_category": per_category,
        "macro": macro,
        "predictions": {
            "category": categories,
            "label": [int(label) for label in labels],
            "score": scores,
        },
    }
    if select_thresholds:
        result["selected_thresholds"] = selected_thresholds
    return result


def cache_feature_batches(model, loader, device, config):
    """Cache fixed validation CLIP features on CPU for repeated evaluation."""
    model.extractor.eval()
    amp_enabled = bool(config["training"].get("amp", True) and device.type == "cuda")
    amp_dtype = config["training"].get("amp_dtype", "float16")
    cached = []
    with torch.inference_mode():
        for images, labels, categories, _ in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled, amp_dtype):
                scale_features = model.extractor(images)
            cached.append(
                (
                    tuple(feature.float().cpu() for feature in scale_features),
                    labels.clone(),
                    tuple(categories),
                )
            )
    if not cached:
        raise RuntimeError("The validation feature cache is empty")
    return cached


def evaluate_cached(
    model,
    cached_batches,
    device,
    config,
    thresholds=None,
    select_thresholds=False,
):
    """Evaluate the changing trainable head from fixed cached CLIP features."""
    model.eval()
    labels = []
    scores = []
    categories = []
    amp_enabled = bool(config["training"].get("amp", True) and device.type == "cuda")
    amp_dtype = config["training"].get("amp_dtype", "float16")

    with torch.inference_mode():
        for cpu_features, batch_labels, batch_categories in cached_batches:
            scale_features = tuple(
                feature.to(device, non_blocking=True) for feature in cpu_features
            )
            with autocast_context(device, amp_enabled, amp_dtype):
                output = model.forward_from_scale_features(
                    scale_features, categories=list(batch_categories)
                )
            scores.extend(output["scores"].float().cpu().tolist())
            labels.extend(batch_labels.tolist())
            categories.extend(batch_categories)

    selected_thresholds = (
        select_per_category_thresholds(categories, labels, scores)
        if select_thresholds
        else thresholds
    )
    per_category, macro = compute_per_category_metrics(
        categories, labels, scores, thresholds=selected_thresholds
    )
    result = {
        "per_category": per_category,
        "macro": macro,
        "predictions": {
            "category": categories,
            "label": [int(label) for label in labels],
            "score": scores,
        },
    }
    if select_thresholds:
        result["selected_thresholds"] = selected_thresholds
    return result


def checkpoint_payload(
    epoch,
    model,
    loss_fn,
    optimizer,
    scheduler,
    scaler,
    config,
    manifest_hash,
    train_generator,
    best_value,
    best_epoch,
    best_model_state,
    best_validation_thresholds,
    epochs_without_improvement,
    history,
):
    tracker_state = (
        loss_fn.tracker.state_dict() if loss_fn.tracker is not None else None
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "model_trainable_state_dict": compact_model_state_dict(model),
        "best_model_trainable_state_dict": best_model_state,
        "backbone": checkpoint_backbone_metadata(config),
        "loss_state_dict": loss_fn.state_dict(),
        "difficulty_tracker_state": tracker_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "train_generator_state": train_generator.get_state(),
        "config": config,
        "config_sha256": configuration_sha256(config),
        "manifest_sha256": manifest_hash,
        "best_validation_auroc": best_value,
        "best_epoch": best_epoch,
        "validation_thresholds": best_validation_thresholds,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the multi-scale CLIP detector")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--mvtec-root")
    parser.add_argument("--resume", help="Checkpoint path for exact continuation")
    args = parser.parse_args()

    config = resolve_config(load_config(args.config), args)
    validate_config(config)
    training_config = config["training"]
    data_config = config["data"]
    model_config = config["model"]
    seed = int(training_config.get("seed", data_config.get("split_seed", 42)))
    seed_everything(seed, deterministic=training_config.get("deterministic", False))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_accelerator(training_config, device)
    pin_memory = device.type == "cuda"
    logger.info("Runtime device: %s", cuda_device_summary(device))

    mvtec_root = find_mvtec_root(data_config["mvtec_root"])
    config["data"]["mvtec_root"] = mvtec_root
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
        image_size=image_size,
        resize_mode=data_config.get("resize_mode", "letterbox"),
    )
    common_dataset_arguments = {
        "data_dir": mvtec_root,
        "categories": data_config.get("categories"),
        "transform": transform,
        "use_test_anomalies": data_config.get(
            "use_test_anomalies_for_train", False
        ),
        "train_ratio": data_config.get("train_ratio", 0.8),
        "seed": data_config.get("split_seed", seed),
        "validation_ratio": data_config.get("validation_ratio", 0.1),
        "return_resolution": True,
        "effective_resolution": effective_resolution,
    }
    use_synthetic = data_config.get("use_synthetic_anomalies", False)
    train_dataset = MVTecDataset(
        split="train",
        synthetic_anomalies=use_synthetic,
        **common_dataset_arguments,
    )
    validation_dataset = MVTecDataset(
        split="val",
        synthetic_anomalies=use_synthetic,
        **common_dataset_arguments,
    )
    test_dataset = MVTecDataset(
        split="test",
        synthetic_anomalies=False,
        **common_dataset_arguments,
    )
    datasets = {
        "train": train_dataset,
        "val": validation_dataset,
        "test": test_dataset,
    }
    assert_disjoint_source_images(datasets)

    save_directory = config["logging"]["save_dir"]
    results_directory = config["logging"]["results_dir"]
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)
    manifest, manifest_hash = build_manifest(datasets, mvtec_root)
    manifest_path = os.path.join(results_directory, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"sha256": manifest_hash, "splits": manifest},
            handle,
            indent=2,
        )

    worker_count = int(training_config.get("num_workers", 4))
    batch_size = int(training_config["batch_size"])
    prefetch_factor = int(training_config.get("prefetch_factor", 2))
    persistent_workers = bool(
        training_config.get("persistent_workers", False)
        and not training_config.get("deterministic", False)
    )
    train_loader, train_generator = make_loader(
        train_dataset,
        batch_size,
        True,
        worker_count,
        pin_memory,
        seed,
        prefetch_factor,
        persistent_workers,
    )
    validation_loader, _ = make_loader(
        validation_dataset,
        batch_size,
        False,
        worker_count,
        pin_memory,
        seed + 1,
        prefetch_factor,
        persistent_workers,
    )
    test_loader, _ = make_loader(
        test_dataset,
        batch_size,
        False,
        worker_count,
        pin_memory,
        seed + 2,
        prefetch_factor,
        persistent_workers,
    )

    actual_feature_dim = getattr(getattr(clip_model, "visual", None), "output_dim", None)
    configured_feature_dim = int(model_config["feature_dim"])
    if actual_feature_dim is not None and actual_feature_dim != configured_feature_dim:
        raise ValueError(
            f"Configured feature_dim={configured_feature_dim}, but CLIP returns "
            f"{actual_feature_dim}"
        )

    model = MultiScaleCLIPDetector(
        clip_model,
        tokenizer,
        feature_dim=configured_feature_dim,
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_attention_heads"],
        dropout=model_config["dropout"],
        visual_weight=config["inference"]["visual_weight"],
        text_weight=config["inference"]["text_weight"],
        clip_input_size=clip_input_size,
        scales=scales,
        encode_chunk_size=model_config.get("encode_chunk_size", 64),
        combine_scale_batches=model_config.get("combine_scale_batches", True),
        channels_last=model_config.get("channels_last", False),
        prompt_ensemble=model_config.get("prompt_ensemble", True),
        text_temperature=model_config.get("text_temperature", 0.01),
        local_score_weight=model_config.get("local_score_weight", 0.5),
        local_topk=model_config.get("local_topk", 0.1),
    ).to(device)
    loss_config = config["loss"]
    loss_fn = CompositeAnomalyLoss(
        feature_dim=model_config["hidden_dim"],
        margin_base=loss_config["margin_base"],
        lambda_sigma=loss_config["lambda_sigma"],
        lambda_resolution=loss_config["lambda_resolution"],
        original_resolution=loss_config["original_resolution"],
        model_resolution=loss_config["model_resolution"],
        temperature=loss_config["temperature"],
        alpha=loss_config["alpha"],
        beta=loss_config["beta"],
        gamma=loss_config["gamma"],
        difficulty_cfg=config.get("difficulty_tracker"),
    ).to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training_config["epochs"]))
    )
    amp_enabled = bool(training_config.get("amp", True) and device.type == "cuda")
    scaler_enabled = bool(
        amp_enabled and training_config.get("amp_dtype", "float16") == "float16"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    start_epoch = 1
    best_value = -math.inf
    best_epoch = None
    best_model_state = None
    best_validation_thresholds = None
    epochs_without_improvement = 0
    history = []
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        validate_checkpoint_backbone(checkpoint, config)
        recorded_config_hash = checkpoint.get("config_sha256")
        if (
            recorded_config_hash is not None
            and recorded_config_hash != configuration_sha256(config)
        ):
            raise RuntimeError("The resume checkpoint uses a different configuration")
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("The resume checkpoint uses a different data manifest")
        load_checkpoint_model(model, checkpoint)
        loss_fn.load_state_dict(checkpoint["loss_state_dict"])
        if loss_fn.tracker is not None:
            loss_fn.tracker.load_state_dict(
                checkpoint.get("difficulty_tracker_state") or {}
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        restore_rng_state(checkpoint.get("rng_state"))
        if "train_generator_state" in checkpoint:
            train_generator.set_state(checkpoint["train_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_value = float(checkpoint.get("best_validation_auroc", -math.inf))
        best_epoch = checkpoint.get("best_epoch")
        best_model_state = checkpoint.get("best_model_trainable_state_dict")
        best_validation_thresholds = checkpoint.get("validation_thresholds")
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        history = list(checkpoint.get("history", []))

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    logger.info("Parameters: %d trainable / %d total", trainable_count, total_parameters)
    logger.info(
        "Samples: train=%d, validation=%d, test=%d",
        len(train_dataset),
        len(validation_dataset),
        len(test_dataset),
    )

    validation_feature_cache = None
    if training_config.get("cache_validation_features", True):
        logger.info("Caching frozen validation features")
        validation_feature_cache = cache_feature_batches(
            model, validation_loader, device, config
        )

    best_path = os.path.join(save_directory, "best.pt")
    latest_path = os.path.join(save_directory, "latest.pt")
    patience = int(training_config.get("early_stopping_patience", 0))
    min_delta = float(training_config.get("early_stopping_min_delta", 0.0))

    for epoch in range(start_epoch, int(training_config["epochs"]) + 1):
        started_at = time.time()
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, config
        )
        if validation_feature_cache is None:
            validation_result = evaluate(
                model,
                validation_loader,
                device,
                config,
                select_thresholds=True,
            )
        else:
            validation_result = evaluate_cached(
                model,
                validation_feature_cache,
                device,
                config,
                select_thresholds=True,
            )
        validation_auroc = validation_result["macro"]["auroc"]
        if not math.isfinite(validation_auroc):
            raise RuntimeError("Validation AUROC is undefined; inspect the split manifest")
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "training": train_metrics,
                "validation": validation_result["macro"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started_at,
            }
        )
        improved = validation_auroc > best_value + min_delta
        if improved:
            best_value = validation_auroc
            best_epoch = epoch
            best_model_state = compact_model_state_dict(model)
            best_validation_thresholds = validation_result["selected_thresholds"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = checkpoint_payload(
            epoch,
            model,
            loss_fn,
            optimizer,
            scheduler,
            scaler,
            config,
            manifest_hash,
            train_generator,
            best_value,
            best_epoch,
            best_model_state,
            best_validation_thresholds,
            epochs_without_improvement,
            history,
        )
        torch.save(payload, latest_path)
        if improved:
            torch.save(payload, best_path)

        logger.info(
            "Epoch %d: loss=%.6f, validation macro AUROC=%.6f, hinge-active=%.4f",
            epoch,
            train_metrics["loss"],
            validation_auroc,
            train_metrics["active_hinge_fraction"],
        )
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info("Early stopping after %d epochs without improvement", patience)
            break

    if os.path.isfile(latest_path):
        selection_checkpoint = load_checkpoint(latest_path, device)
    elif args.resume:
        selection_checkpoint = load_checkpoint(args.resume, device)
    else:
        raise RuntimeError("No checkpoint was produced")
    best_state = selection_checkpoint.get("best_model_trainable_state_dict")
    if best_state is None:
        if os.path.isfile(best_path):
            selection_checkpoint = load_checkpoint(best_path, device)
        else:
            raise RuntimeError("The selected model state is unavailable")
    else:
        selection_checkpoint = dict(selection_checkpoint)
        selection_checkpoint["model_trainable_state_dict"] = best_state
    load_checkpoint_model(model, selection_checkpoint)
    test_result = evaluate(
        model,
        test_loader,
        device,
        config,
        thresholds=selection_checkpoint.get("validation_thresholds"),
    )

    predictions_path = os.path.join(results_directory, "test_predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as handle:
        json.dump(test_result["predictions"], handle, indent=2)
    if data_config.get("use_test_anomalies_for_train", False):
        protocol_name = "legacy target-supervised split"
    elif use_synthetic:
        protocol_name = "normal-only with CutPaste pseudo anomalies"
    else:
        protocol_name = "normal-only without positive training examples"

    result = {
        "method": "multi-scale CLIP with CutPaste-style pseudo-anomaly supervision",
        "protocol": protocol_name,
        "manifest_sha256": manifest_hash,
        "selected_epoch": int(
            selection_checkpoint.get("best_epoch", selection_checkpoint["epoch"])
        ),
        "validation": {
            "macro": next(
                entry["validation"]
                for entry in selection_checkpoint["history"]
                if entry["epoch"]
                == selection_checkpoint.get(
                    "best_epoch", selection_checkpoint["epoch"]
                )
            )
        },
        "test": {
            "macro": test_result["macro"],
            "per_category": test_result["per_category"],
        },
        "config": config,
    }
    result_path = os.path.join(results_directory, "result_multiscale_clip.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    logger.info("Untouched-test results saved to %s", result_path)


if __name__ == "__main__":
    main()
