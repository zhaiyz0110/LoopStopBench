"""Build training examples for the VOI estimator.

The label is ``y_t = 1[max_{s>t} r_s > r_t + delta]``. Labels use hidden
quality during offline training, while features use only visible history.
The deterministic 60/20/20 split groups all trajectories for a task together.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..schema import Trajectory
from .features import ALL_FEATURES, extract_features


@dataclass
class Example:
    features: dict
    label: int
    task_id: str
    traj_id: str
    loop_type: str
    t: int


def build_examples(trajs: list[Trajectory], delta: float = 0.05) -> list[Example]:
    out = []
    for traj in trajs:
        r = [traj.r(t) for t in range(1, traj.T + 1)]
        for t in range(1, traj.T):  # 最后一轮没有"继续"可言, 不产样本
            label = int(max(r[t:]) > r[t - 1] + delta)
            out.append(
                Example(
                    features=extract_features(traj.visible(t)),
                    label=label,
                    task_id=traj.task_id,
                    traj_id=traj.traj_id,
                    loop_type=traj.loop_type,
                    t=t,
                )
            )
    return out


def split_by_task(
    examples: list[Example], ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> dict[str, list[Example]]:
    """task_id 哈希 → 确定性划分, 与运行顺序无关。"""
    assert abs(sum(ratios) - 1.0) < 1e-9
    out: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for ex in examples:
        h = int(hashlib.md5(ex.task_id.encode()).hexdigest(), 16) % 10_000 / 10_000
        if h < ratios[0]:
            out["train"].append(ex)
        elif h < ratios[0] + ratios[1]:
            out["val"].append(ex)
        else:
            out["test"].append(ex)
    return out


def split_leave_one_loop_out(
    examples: list[Example], held_out_loop: str
) -> dict[str, list[Example]]:
    """Train on two loop families and hold out the third."""
    return {
        "train": [e for e in examples if e.loop_type != held_out_loop],
        "test": [e for e in examples if e.loop_type == held_out_loop],
    }


def to_matrix(examples: list[Example]) -> tuple[list[list[float]], list[int], list[str]]:
    """Convert feature dictionaries to a fixed-order matrix, filling ``None`` with 0."""
    X = [
        [0.0 if e.features.get(c) is None else float(e.features[c]) for c in ALL_FEATURES]
        for e in examples
    ]
    y = [e.label for e in examples]
    return X, y, list(ALL_FEATURES)
