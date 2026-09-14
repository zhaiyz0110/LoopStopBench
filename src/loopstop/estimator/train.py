"""Train the gradient-boosted VOI estimator and feature ablations.

The saved joblib bundle contains ``model`` and ``columns`` for use by
``policies/voi.py``. This module requires the optional analysis dependencies.
"""
from __future__ import annotations

from pathlib import Path

from .dataset import Example, to_matrix
from .features import ALL_FEATURES, FEATURE_GROUPS


def train_gbt(train: list[Example], val: list[Example], model_out: str | Path) -> dict:
    import joblib

    X_tr, y_tr, columns = to_matrix(train)
    X_va, y_va, _ = to_matrix(val)

    model = _make_gbt()
    model.fit(X_tr, y_tr)

    metrics = {
        "auroc_val": auroc(y_va, [p[1] for p in model.predict_proba(X_va)]),
        "n_train": len(train),
        "n_val": len(val),
        "model_type": type(model).__name__,
    }
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": columns}, model_out)
    return metrics


def ablate_groups(train: list[Example], val: list[Example]) -> dict[str, float]:
    """Drop each feature group in turn and report validation AUROC."""
    results = {}
    for group, feats in [("full", [])] + list(FEATURE_GROUPS.items()):
        keep = [c for c in ALL_FEATURES if c not in feats]
        model = _make_gbt()
        X_tr = _project(train, keep)
        X_va = _project(val, keep)
        model.fit(X_tr, [e.label for e in train])
        results[f"drop_{group}" if feats else "full"] = auroc(
            [e.label for e in val], [p[1] for p in model.predict_proba(X_va)]
        )
    return results


def calibration_bins(y_true: list[int], p: list[float], n_bins: int = 10) -> list[dict]:
    """校准图(reliability diagram)数据。"""
    bins: list[list] = [[] for _ in range(n_bins)]
    for yt, pi in zip(y_true, p):
        bins[min(int(pi * n_bins), n_bins - 1)].append((yt, pi))
    out = []
    for i, b in enumerate(bins):
        if b:
            out.append(
                {
                    "bin": i,
                    "mean_pred": sum(x[1] for x in b) / len(b),
                    "frac_pos": sum(x[0] for x in b) / len(b),
                    "n": len(b),
                }
            )
    return out


def auroc(y_true: list[int], scores: list[float]) -> float:
    """纯 python AUROC(Mann-Whitney), 避免评测路径依赖 sklearn。"""
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p_ in pos:
        for n_ in neg:
            if p_ > n_:
                wins += 1
            elif p_ == n_:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _make_gbt():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=0,
            verbose=-1,  # 静默: bootstrap 内重拟合上千次, 否则日志刷屏埋掉结果
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=400, random_state=0)


def _project(examples: list[Example], columns: list[str]) -> list[list[float]]:
    return [
        [0.0 if e.features.get(c) is None else float(e.features[c]) for c in columns]
        for e in examples
    ]
