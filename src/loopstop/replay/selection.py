from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass, field

from ..policies import build_policy, policy_label
from ..schema import Trajectory
from .evaluator import replay_policy, utility


def task_fold(task_id: str, k: int = 2) -> int:
    """Return the deterministic task-level fold used for policy evaluation."""
    if k < 2:
        raise ValueError("k must be at least 2")
    return int(hashlib.md5(task_id.encode()).hexdigest(), 16) % k


@dataclass
class CrossFittedPolicyResult:
    mean_utility: float
    fold_winners: dict[int, str]
    per_traj_utility: dict[str, float] = field(repr=False)

    @property
    def label(self) -> str:
        labels = set(self.fold_winners.values())
        if len(labels) == 1:
            return next(iter(labels))
        detail = ",".join(f"f{fold}:{label}" for fold, label in sorted(self.fold_winners.items()))
        return f"OOF[{detail}]"


def heuristic_utility_table(
    trajs: list[Trajectory], policy_specs: list[dict], lam: float, mode: str
) -> dict[str, dict[str, float]]:
    """Precompute each simple policy's utility on every trajectory."""
    table: dict[str, dict[str, float]] = {}
    for spec in policy_specs:
        if spec.get("type") == "voi":
            continue
        try:
            policy = build_policy(spec)
            label = policy_label(policy)
            table[label] = {
                traj.traj_id: utility(traj, replay_policy(policy, traj), lam, mode)
                for traj in trajs
            }
        except Exception:
            continue
    return table


def crossfit_best_from_utilities(
    trajs: list[Trajectory],
    utility_by_policy: dict[str, dict[str, float]],
    k_outer: int = 2,
) -> CrossFittedPolicyResult:
    """Select a policy on outer-train utilities and score it on outer-test tasks."""
    folds: dict[int, list[Trajectory]] = {fold: [] for fold in range(k_outer)}
    for traj in trajs:
        folds[task_fold(traj.task_id, k_outer)].append(traj)

    observed_utilities: list[float] = []
    per_traj: dict[str, float] = {}
    winners: dict[int, str] = {}

    for test_fold in range(k_outer):
        test = folds[test_fold]
        train = [traj for fold in range(k_outer) if fold != test_fold for traj in folds[fold]]
        if not train or not test:
            continue

        best_label, best_train_utility = None, -math.inf
        for label, values in utility_by_policy.items():
            available = [values[traj.traj_id] for traj in train if traj.traj_id in values]
            if available:
                train_utility = statistics.fmean(available)
                if train_utility > best_train_utility:
                    best_label = label
                    best_train_utility = train_utility

        if best_label is None:
            raise ValueError(f"no heuristic policy could be evaluated for outer fold {test_fold}")

        winners[test_fold] = best_label
        values = utility_by_policy[best_label]
        for traj in test:
            if traj.traj_id not in values:
                continue
            value = float(values[traj.traj_id])
            observed_utilities.append(value)
            per_traj[traj.traj_id] = value

    if not observed_utilities:
        return CrossFittedPolicyResult(float("nan"), winners, per_traj)
    return CrossFittedPolicyResult(statistics.fmean(observed_utilities), winners, per_traj)


def crossfit_best_heuristic(
    trajs: list[Trajectory],
    policy_specs: list[dict],
    lam: float,
    mode: str,
    k_outer: int = 2,
) -> CrossFittedPolicyResult:
    """Select on outer-train tasks and score the winner on outer-test tasks.

    Repeating this function inside each task-cluster bootstrap resample also
    repeats winner selection, so its uncertainty is included in the interval.
    """
    table = heuristic_utility_table(trajs, policy_specs, lam, mode)
    return crossfit_best_from_utilities(trajs, table, k_outer)
