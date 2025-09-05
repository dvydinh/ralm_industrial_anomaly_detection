"""Deprecated compatibility entry point for the global CLIP baseline."""

import warnings

from eval_global_clip import main


if __name__ == "__main__":
    warnings.warn(
        "This repository does not implement full WinCLIP. "
        "Use scripts/eval_global_clip.py here and a separately validated full "
        "implementation for a WinCLIP comparison; the public repositories found "
        "during this audit identify themselves as unofficial reimplementations.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
