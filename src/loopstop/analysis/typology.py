"""Classify trajectories from their hidden-quality sequence ``r_1..r_T``.

This classification operates on hidden quality rather than embedding space.
Overlapping categories use the following precedence:
    degrading > oscillating > plateau > monotone > other
``other`` includes curves that are still rising at the final observed round.
"""
from __future__ import annotations

from dataclasses import dataclass

TYPES = ("monotone", "plateau", "oscillating", "degrading", "other")


@dataclass
class TypologyResult:
    label: str
    peak_t: int  # 平滑序列峰值轮(1-based)
    details: dict


def moving_average(xs: list[float], w: int = 3) -> list[float]:
    """居中滑动平均, 边缘收缩窗口。"""
    n = len(xs)
    out = []
    half = w // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def _reversal_count(s: list[float], delta: float) -> int:
    """幅度 > delta 的方向反转次数: 在平滑序列的摆动段上计数。"""
    reversals = 0
    direction = 0  # +1 升, -1 降
    anchor = s[0]
    for v in s[1:]:
        move = v - anchor
        if direction == 0:
            if abs(move) > delta:
                direction = 1 if move > 0 else -1
                anchor = v
        elif direction == 1:
            if v > anchor:
                anchor = v
            elif anchor - v > delta:
                reversals += 1
                direction = -1
                anchor = v
        else:
            if v < anchor:
                anchor = v
            elif v - anchor > delta:
                reversals += 1
                direction = 1
                anchor = v
    return reversals


def classify(
    r: list[float], eps: float = 0.01, delta: float = 0.05, window: int = 3
) -> TypologyResult:
    if len(r) < 4:
        raise ValueError("trajectory too short to classify (need >= 4 rounds)")
    s = moving_average([float(x) for x in r], w=window)
    T = len(s)
    diffs = [s[i + 1] - s[i] for i in range(T - 1)]
    peak_t = max(range(T), key=lambda i: s[i]) + 1
    details: dict = {"smoothed": s, "peak_t": peak_t}

    # 劣化型: 峰值显著高于末轮, 且峰值出现在前 2/3 段
    if max(s) - s[-1] > delta and peak_t <= (2 * T) / 3:
        return TypologyResult("degrading", peak_t, details)

    # 振荡型: 平滑后仍存在 >= 2 次幅度 > delta 的方向反转
    rev = _reversal_count(s, delta)
    details["reversals"] = rev
    if rev >= 2:
        return TypologyResult("oscillating", peak_t, details)

    # 平台型: 存在 k < T/2, 其后增量持续 < eps
    k_plateau = None
    for k in range(T - 1):
        if all(d < eps for d in diffs[k:]):
            k_plateau = k + 1  # 1-based
            break
    details["k_plateau"] = k_plateau
    if k_plateau is not None and k_plateau < T / 2:
        return TypologyResult("plateau", peak_t, details)

    # 单调收敛型: 单调不减(容差 eps/2), 且末段增量 < eps
    tol = eps / 2
    monotone = all(d >= -tol for d in diffs)
    tail = diffs[-min(3, len(diffs)) :]
    converged = sum(tail) / len(tail) < eps
    if monotone and converged:
        return TypologyResult("monotone", peak_t, details)

    return TypologyResult("other", peak_t, details)


def type_proportions(series_list: list[list[float]], **kw) -> dict[str, float]:
    counts = {t: 0 for t in TYPES}
    for r in series_list:
        counts[classify(r, **kw).label] += 1
    n = max(len(series_list), 1)
    return {t: c / n for t, c in counts.items()}
