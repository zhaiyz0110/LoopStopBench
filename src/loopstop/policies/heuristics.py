"""Heuristic stopping policies S1 through S8.

Policy configurations are supplied by ``configs/replay.yaml``. Visible-signal
keys are documented in ``schema.py``.
"""
from __future__ import annotations

from typing import Optional

from ..schema import VisibleHistory
from .base import StoppingPolicy, register


def _scores(h: VisibleHistory) -> list[Optional[float]]:
    return h.signal("score")


@register("fixed_n")
class FixedN(StoppingPolicy):
    """S1: stop at fixed horizon ``N``."""

    def __init__(self, n: int):
        self.n = n

    def decide(self, h: VisibleHistory) -> bool:
        return h.t >= self.n


@register("verifier_pass")
class VerifierPass(StoppingPolicy):
    """S2: stop when the verifier passes or its score reaches a threshold."""

    def __init__(self, score_threshold: Optional[float] = None):
        self.score_threshold = score_threshold

    def decide(self, h: VisibleHistory) -> bool:
        s = h.last.visible_signals
        if s.get("verifier_pass") is True:
            return True
        if self.score_threshold is not None:
            v = s.get("score")
            return v is not None and v >= self.score_threshold
        return False


@register("critic_approve")
class CriticApprove(StoppingPolicy):
    """S3: critic 显式输出 APPROVE 即停(writer_critic 专属)。"""

    def decide(self, h: VisibleHistory) -> bool:
        return h.last.visible_signals.get("critic_approve") is True


@register("semantic_convergence")
class SemanticConvergence(StoppingPolicy):
    """S4: stop after ``patience`` consecutive embedding distances below ``eps``."""

    def __init__(self, eps: float = 0.02, patience: int = 2):
        self.eps = eps
        self.patience = patience

    def decide(self, h: VisibleHistory) -> bool:
        dists = [d for d in h.signal("emb_dist_prev") if d is not None]
        if len(dists) < self.patience:
            return False
        return all(d < self.eps for d in dists[-self.patience :])


@register("confidence_threshold")
class ConfidenceThreshold(StoppingPolicy):
    """S5: 自评置信度 ≥ 阈值即停(react_qa 重点)。"""

    def __init__(self, tau: float = 0.9):
        self.tau = tau

    def decide(self, h: VisibleHistory) -> bool:
        c = h.last.visible_signals.get("self_confidence")
        return c is not None and c >= self.tau


@register("score_plateau")
class ScorePlateau(StoppingPolicy):
    """S6: stop after ``patience`` consecutive score increments below ``eps``."""

    def __init__(self, eps: float = 0.01, patience: int = 2):
        self.eps = eps
        self.patience = patience

    def decide(self, h: VisibleHistory) -> bool:
        vs = [v for v in _scores(h) if v is not None]
        if len(vs) < self.patience + 1:
            return False
        deltas = [vs[i + 1] - vs[i] for i in range(len(vs) - 1)]
        return all(d < self.eps for d in deltas[-self.patience :])


@register("dual_signal")
class DualSignal(StoppingPolicy):
    """S7: stop when both S4 and S6 stop."""

    def __init__(self, emb_eps: float = 0.02, score_eps: float = 0.01, patience: int = 2):
        self._s4 = SemanticConvergence(eps=emb_eps, patience=patience)
        self._s6 = ScorePlateau(eps=score_eps, patience=patience)

    def decide(self, h: VisibleHistory) -> bool:
        return self._s4.decide(h) and self._s6.decide(h)


@register("reward_prediction")
class RewardPrediction(StoppingPolicy):
    """S8: stop when a linear score forecast falls below ``threshold``."""

    def __init__(self, window: int = 4, threshold: float = 0.005, min_rounds: int = 2):
        self.window = window
        self.threshold = threshold
        self.min_rounds = min_rounds

    def decide(self, h: VisibleHistory) -> bool:
        vs = [v for v in _scores(h) if v is not None]
        if len(vs) < self.min_rounds:
            return False
        tail = vs[-self.window :]
        n = len(tail)
        if n < 2:
            return False
        # Least-squares slope estimates the per-round score increment.
        xbar = (n - 1) / 2
        ybar = sum(tail) / n
        denom = sum((i - xbar) ** 2 for i in range(n))
        slope = sum((i - xbar) * (y - ybar) for i, y in enumerate(tail)) / denom
        return slope < self.threshold
