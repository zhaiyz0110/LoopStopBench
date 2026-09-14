"""Trajectory metrics used by the analysis and figure scripts."""
from __future__ import annotations

import math
import statistics
from typing import Optional

from ..schema import Trajectory


def final_neq_best_rate(trajs: list[Trajectory], delta: float = 0.05) -> float:
    """Fraction of trajectories for which ``max(r_t) - r_T > delta``."""
    hits = sum(1 for t in trajs if max(_r(t)) - _r(t)[-1] > delta)
    return hits / len(trajs) if trajs else 0.0


def peak_round_distribution(trajs: list[Trajectory]) -> dict[int, int]:
    """Histogram of the earliest round attaining maximum hidden quality."""
    hist: dict[int, int] = {}
    for t in trajs:
        r = _r(t)
        peak = r.index(max(r)) + 1
        hist[peak] = hist.get(peak, 0) + 1
    return dict(sorted(hist.items()))


def mean_quality_curve(trajs: list[Trajectory]) -> list[tuple[int, float, float]]:
    """Return round, mean hidden quality, and sample standard deviation."""
    T = trajs[0].T
    out = []
    for i in range(T):
        vals = [_r(t)[i] for t in trajs]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append((i + 1, statistics.fmean(vals), sd))
    return out


def marginal_improvement_curve(trajs: list[Trajectory]) -> list[tuple[int, float]]:
    """Return mean one-step hidden-quality change by round."""
    T = trajs[0].T
    out = []
    for i in range(T - 1):
        out.append((i + 1, statistics.fmean(_r(t)[i + 1] - _r(t)[i] for t in trajs)))
    return out


def proxy_truth_corr_by_round(trajs: list[Trajectory]) -> list[tuple[int, Optional[float]]]:
    """Return the cross-trajectory Pearson correlation of score and quality by round."""
    T = trajs[0].T
    out = []
    for i in range(T):
        pairs = [
            (t.steps[i].visible_signals.get("score"), _r(t)[i])
            for t in trajs
            if t.steps[i].visible_signals.get("score") is not None
        ]
        out.append((i + 1, _pearson(pairs) if len(pairs) >= 3 else None))
    return out


def scissor_gap_rate(trajs: list[Trajectory], delta: float = 0.05) -> float:
    """Fraction with rising visible score and falling hidden quality in the second half."""
    hits, n = 0, 0
    for t in trajs:
        r = _r(t)
        v = [s.visible_signals.get("score") for s in t.steps]
        if any(x is None for x in v):
            continue
        n += 1
        half = len(r) // 2
        if (v[-1] - v[half]) > 0 and (r[half] - r[-1]) > delta:
            hits += 1
    return hits / n if n else 0.0


def _r(traj: Trajectory) -> list[float]:
    r = traj.r_series
    if any(x is None for x in r):
        raise ValueError(f"{traj.traj_id}: hidden truth pending")
    return [float(x) for x in r]


def _pearson(pairs: list[tuple[float, float]]) -> Optional[float]:
    xs, ys = zip(*pairs)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
