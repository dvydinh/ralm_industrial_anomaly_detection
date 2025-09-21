"""Measure feasible batch and encoder-chunk settings on the assigned GPU."""

import argparse
import gc
import json
import os
import sys

import torch
import torch.nn.functional as F
import yaml


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raml.models.clip_loader import load_clip_model
from raml.utils.performance import (
    autocast_context,
    configure_accelerator,
    cuda_device_summary,
)
from scripts.evaluate import build_model


def benchmark_candidate(
    model,
    optimizer,
    scaler,
    device,
    image_size,
    batch_size,
    chunk_size,
    warmup_steps,
    measured_steps,
    amp_enabled,
    amp_dtype,
):
    """Benchmark a synthetic full forward/backward path for one configuration."""
    model.extractor.encode_chunk_size = chunk_size
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    labels = torch.arange(batch_size, device=device).remainder(2).float()
    categories = ["bottle"] * batch_size

    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled, amp_dtype):
            output = model(images, categories=categories)
            loss = F.binary_cross_entropy_with_logits(output["logits"], labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize(device)

    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(measured_steps):
        step()
    finished.record()
    torch.cuda.synchronize(device)

    elapsed_ms = float(started.elapsed_time(finished))
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    return {
        "batch_size": batch_size,
        "encoder_chunk_size": chunk_size,
        "step_time_ms": elapsed_ms / measured_steps,
        "images_per_second": batch_size * measured_steps * 1000.0 / elapsed_ms,
        "peak_memory_bytes": peak_memory,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-scale CLIP training compute on the assigned GPU"
    )
    parser.add_argument("--config", default="configs/notebook_env.yaml")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument(
        "--chunk-sizes", nargs="+", type=int, default=[64, 128, 256, 512]
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark")
    if args.warmup_steps < 1 or args.steps < 1:
        raise ValueError("Warmup and measured step counts must be positive")
    if any(value < 1 for value in args.batch_sizes + args.chunk_sizes):
        raise ValueError("Batch and chunk sizes must be positive")

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device("cuda")
    training_config = config["training"]
    configure_accelerator(training_config, device)
    clip_model, _, tokenizer = load_clip_model(
        config["model"]["clip_model"],
        device,
        source=config["model"].get("clip_source", "openai"),
    )
    model = build_model(config, clip_model, tokenizer).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    amp_enabled = bool(training_config.get("amp", True))
    amp_dtype = training_config.get("amp_dtype", "float16")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == "float16"
    )

    measurements = []
    for batch_size in args.batch_sizes:
        for chunk_size in args.chunk_sizes:
            try:
                measurement = benchmark_candidate(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    image_size=int(config["data"]["image_size"]),
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.steps,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            except torch.cuda.OutOfMemoryError:
                measurement = {
                    "batch_size": batch_size,
                    "encoder_chunk_size": chunk_size,
                    "status": "out_of_memory",
                }
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            measurements.append(measurement)

    feasible = [item for item in measurements if item["status"] == "ok"]
    best = max(feasible, key=lambda item: item["images_per_second"]) if feasible else None
    report = {
        "scope": "synthetic compute benchmark; data loading is excluded",
        "hardware": cuda_device_summary(device),
        "image_size": int(config["data"]["image_size"]),
        "amp_dtype": amp_dtype if amp_enabled else "disabled",
        "measurements": measurements,
        "fastest_measured_configuration": best,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
