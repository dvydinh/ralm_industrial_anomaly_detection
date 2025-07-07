"""Text prompt construction shared by models and baselines."""


NORMAL_STATES = (
    "flawless",
    "perfect",
    "normal",
    "undamaged",
    "without defect",
)
ANOMALY_STATES = (
    "damaged",
    "defective",
    "anomalous",
    "broken",
    "with a flaw",
)
PROMPT_TEMPLATES = (
    "a photo of a {}",
    "a close-up photo of a {}",
    "an industrial photo of a {}",
    "a cropped photo of a {}",
)


def category_text(category):
    """Convert dataset identifiers into natural prompt text."""
    return str(category).replace("_", " ").replace("-", " ").strip()


def build_state_prompts(category, ensemble=True):
    """Return normal and anomalous prompt lists for one category."""
    category = category_text(category)
    if not ensemble:
        return (
            [f"a photo of a flawless {category}"],
            [f"a photo of a damaged {category}"],
        )

    normal = [
        template.format(f"{state} {category}")
        for state in NORMAL_STATES
        for template in PROMPT_TEMPLATES
    ]
    anomaly = [
        template.format(f"{state} {category}")
        for state in ANOMALY_STATES
        for template in PROMPT_TEMPLATES
    ]
    return normal, anomaly
