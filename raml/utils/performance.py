"""Explicit accelerator settings shared by training and evaluation."""

from contextlib import nullcontext

import torch


AMP_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def configure_accelerator(training_config, device):
    """Configure CUDA math modes without overriding deterministic runs."""
    if device.type != "cuda":
        return

    deterministic = bool(training_config.get("deterministic", False))
    allow_tf32 = bool(training_config.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(
        training_config.get("cudnn_benchmark", True) and not deterministic
    )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def autocast_context(device, enabled, dtype_name="float16"):
    """Return an accelerator autocast context with validated precision."""
    if not enabled or device.type != "cuda":
        return nullcontext()
    if dtype_name not in AMP_DTYPES:
        raise ValueError("training.amp_dtype must be float16 or bfloat16")
    return torch.autocast(
        device_type="cuda",
        dtype=AMP_DTYPES[dtype_name],
        enabled=True,
    )


def cuda_device_summary(device):
    """Return measured runtime hardware metadata, without benchmark claims."""
    if device.type != "cuda":
        return {"device": str(device)}
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
    }
