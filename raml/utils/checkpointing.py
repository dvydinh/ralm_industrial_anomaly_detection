"""Compact checkpoint helpers for a reproducible frozen-backbone model."""

import hashlib
import json


FROZEN_BACKBONE_PREFIX = "extractor.clip."


def compact_model_state_dict(model):
    """Return model state without the reproducible frozen CLIP backbone."""
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith(FROZEN_BACKBONE_PREFIX)
    }


def load_checkpoint_model(model, checkpoint):
    """Load either the compact schema or a legacy full-model checkpoint."""
    compact_state = checkpoint.get("model_trainable_state_dict")
    if compact_state is None:
        legacy_state = checkpoint.get("model_state_dict")
        if legacy_state is None:
            raise KeyError("Checkpoint has no model state")
        model.load_state_dict(legacy_state)
        return

    incompatible = model.load_state_dict(compact_state, strict=False)
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(FROZEN_BACKBONE_PREFIX)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Compact checkpoint does not match the model: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )


def checkpoint_backbone_metadata(config):
    """Record enough information to reload the omitted frozen backbone."""
    model_config = config["model"]
    return {
        "source": model_config.get("clip_source", "openai"),
        "model": model_config["clip_model"],
        "frozen": True,
        "weights_omitted": True,
    }


def configuration_sha256(config):
    """Hash the resolved experiment configuration deterministically."""
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_checkpoint_backbone(checkpoint, config):
    """Reject a compact checkpoint loaded with a different frozen backbone."""
    recorded = checkpoint.get("backbone")
    if recorded is None:
        return
    expected = checkpoint_backbone_metadata(config)
    for field in ("source", "model"):
        if recorded.get(field) != expected[field]:
            raise RuntimeError(
                "Checkpoint backbone mismatch: "
                f"recorded {field}={recorded.get(field)!r}, "
                f"requested {field}={expected[field]!r}"
            )
