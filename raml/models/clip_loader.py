"""Consistent loading for OpenAI CLIP and open_clip checkpoints."""


def load_clip_model(model_name, device, source="openai"):
    """Return a CLIP model, its preprocessing transform, and its tokenizer."""
    if source == "openai":
        try:
            import clip

            model, preprocess = clip.load(model_name, device=device)
            return model, preprocess, clip.tokenize
        except ImportError:
            source = "open_clip"

    if source == "open_clip":
        try:
            import open_clip
        except ImportError as error:
            raise ImportError(
                "Install either the OpenAI CLIP package or open_clip_torch"
            ) from error

        open_clip_name = model_name.replace("/", "-")
        model, _, preprocess = open_clip.create_model_and_transforms(
            open_clip_name, pretrained="openai"
        )
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(open_clip_name)
        return model, preprocess, tokenizer

    raise ValueError("model.clip_source must be either 'openai' or 'open_clip'")
