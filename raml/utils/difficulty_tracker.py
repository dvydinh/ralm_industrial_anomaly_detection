import numpy as np


class DifficultyTracker:
    """Track per-category loss with exponential moving average and compute
    re-weighting factors so that harder categories receive more attention.

    The weight for category c is:
        w_c = clip(1 + scale * (L_c - L_global) / L_global, clip_min, clip_max)

    This is a simple curriculum-style heuristic; it does not originate from a
    specific paper but is a standard practice in multi-task / multi-class
    training (see e.g. GradNorm, Chen et al., ICML 2018).

    Args:
        momentum: EMA decay for running loss per category.
        weight_scale: Multiplier controlling how aggressively weights shift.
        clip_min: Minimum allowed weight.
        clip_max: Maximum allowed weight.
    """

    def __init__(self, momentum=0.7, weight_scale=0.5, clip_min=0.5, clip_max=2.0):
        self.momentum = momentum
        self.weight_scale = weight_scale
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._running_loss = {}

    def update(self, category, loss_value):
        """Update the running loss for a single category."""
        if category not in self._running_loss:
            self._running_loss[category] = loss_value
        else:
            self._running_loss[category] = (
                self.momentum * self._running_loss[category]
                + (1 - self.momentum) * loss_value
            )

    def get_weight(self, category):
        """Return the re-weighting factor for *category*."""
        if len(self._running_loss) < 2:
            return 1.0
        if category not in self._running_loss:
            return 1.0
        global_loss = np.mean(list(self._running_loss.values()))
        if global_loss < 1e-8:
            return 1.0
        raw = 1.0 + self.weight_scale * (
            self._running_loss[category] - global_loss
        ) / global_loss
        return float(np.clip(raw, self.clip_min, self.clip_max))

    def state_dict(self):
        """Return serializable tracker state for exact experiment resumption."""
        return {"running_loss": dict(self._running_loss)}

    def load_state_dict(self, state):
        """Restore state produced by :meth:`state_dict`."""
        self._running_loss = dict(state.get("running_loss", {}))
