"""VOI stopping policy based on ``P(future improvement | visible history)``.

The estimator is trained offline. This module extracts visible features,
computes a probability, and stops when it falls below ``tau``.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..schema import VisibleHistory
from .base import StoppingPolicy, register


@register("voi")
class VOIPolicy(StoppingPolicy):
    def __init__(
        self,
        model_path: Optional[str] = None,
        tau: float = 0.3,
        min_rounds: int = 1,
        predict_fn: Optional[Callable[[dict], float]] = None,
    ):
        """Load a joblib model or use an injected prediction function."""
        self.tau = tau
        self.min_rounds = min_rounds
        if predict_fn is not None:
            self._predict = predict_fn
        elif model_path is not None:
            self._predict = _load_predictor(model_path)
        else:
            raise ValueError("VOIPolicy needs model_path or predict_fn")

    def decide(self, h: VisibleHistory) -> bool:
        if h.t < self.min_rounds:
            return False
        from ..estimator.features import extract_features

        p = self._predict(extract_features(h))
        return p < self.tau


def _load_predictor(model_path: str) -> Callable[[dict], float]:
    import joblib  # Optional analysis dependency.

    bundle = joblib.load(model_path)
    model, columns = bundle["model"], bundle["columns"]

    def predict(feats: dict) -> float:
        row = [[feats.get(c, 0.0) if feats.get(c) is not None else 0.0 for c in columns]]
        return float(model.predict_proba(row)[0][1])

    return predict
