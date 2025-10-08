import pytest


torch = pytest.importorskip("torch")

from raml.utils.checkpointing import (
    compact_model_state_dict,
    configuration_sha256,
    load_checkpoint_model,
    validate_checkpoint_backbone,
)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = torch.nn.Module()
        self.extractor.clip = torch.nn.Linear(3, 3)
        self.head = torch.nn.Linear(3, 1)


def test_compact_checkpoint_omits_backbone_and_restores_trainable_state():
    source = SmallModel()
    compact = compact_model_state_dict(source)
    assert compact
    assert not any(name.startswith("extractor.clip.") for name in compact)

    target = SmallModel()
    original_backbone = {
        name: value.detach().clone()
        for name, value in target.extractor.clip.state_dict().items()
    }
    load_checkpoint_model(target, {"model_trainable_state_dict": compact})

    for source_value, target_value in zip(
        source.head.parameters(), target.head.parameters()
    ):
        assert torch.equal(source_value, target_value)
    for name, value in target.extractor.clip.state_dict().items():
        assert torch.equal(value, original_backbone[name])


def test_backbone_metadata_mismatch_is_rejected():
    checkpoint = {
        "backbone": {"source": "openai", "model": "ViT-B/16"}
    }
    config = {"model": {"clip_source": "openai", "clip_model": "ViT-B/32"}}
    with pytest.raises(RuntimeError, match="backbone mismatch"):
        validate_checkpoint_backbone(checkpoint, config)


def test_configuration_hash_is_order_independent():
    left = {"model": {"name": "clip", "size": 224}, "seed": 42}
    right = {"seed": 42, "model": {"size": 224, "name": "clip"}}
    assert configuration_sha256(left) == configuration_sha256(right)
