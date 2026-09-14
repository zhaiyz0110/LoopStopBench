"""Offline replay evaluation on full-horizon trajectories.

Utility is ``U(t_stop) = r_selected - lambda * cost(t_stop)``. The
``stop_at_last`` mode returns the output at the stopping round, while
``stop_and_select`` returns the highest visible-scoring output observed by
that round. Both modes charge cost through the stopping round.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

from ..policies.base import StoppingPolicy, policy_label
from ..schema import Trajectory

MODES = ("stop_at_last", "stop_and_select")


def select_round(traj: Trajectory, t_stop: int, mode: str = "stop_at_last") -> int:
    if mode == "stop_at_last":
        return t_stop
    if mode == "stop_and_select":
        scores = [s.visible_signals.get("score") for s in traj.steps[:t_stop]]
        if all(v is None for v in scores):
            return t_stop
        best = max((v for v in scores if v is not None))
        return next(i + 1 for i, v in enumerate(scores) if v == best)  # 平分取最早轮
    raise ValueError(f"unknown mode {mode!r}")


def utility(traj: Trajectory, t_stop: int, lam: float, mode: str = "stop_at_last") -> float:
    return traj.r(select_round(traj, t_stop, mode)) - lam * traj.cost(t_stop)


def replay_policy(policy: StoppingPolicy, traj: Trajectory) -> int:
    """前向扫描: 每轮结束后询问策略。策略不停则跑满 T(与采集时行为一致)。"""
    policy.reset()
    for t in range(1, traj.T + 1):
        if policy.decide(traj.visible(t)):
            return t
    return traj.T


def oracle_stop(traj: Trajectory, lam: float, mode: str = "stop_at_last") -> tuple[int, float]:
    """Return the hindsight-optimal stop using all hidden qualities."""
    best_t, best_u = 1, utility(traj, 1, lam, mode)
    for t in range(2, traj.T + 1):
        u = utility(traj, t, lam, mode)
        if u > best_u:
            best_t, best_u = t, u
    return best_t, best_u


@dataclass
class PolicyResult:
    policy: str
    lam: float
    mode: str
    mean_utility: float
    mean_regret: float
    regret_ci: tuple[float, float]
    mean_stop_round: float
    mean_quality: float
    mean_cost: float
    n: int
    per_traj: list[dict] = field(repr=False, default_factory=list)


def evaluate_policy(
    policy: StoppingPolicy,
    trajs: list[Trajectory],
    lam: float,
    mode: str = "stop_at_last",
    bootstrap_n: int = 2000,
    seed: int = 0,
) -> PolicyResult:
    rows = []
    for traj in trajs:
        t_stop = replay_policy(policy, traj)
        u = utility(traj, t_stop, lam, mode)
        _, u_star = oracle_stop(traj, lam, mode)
        sel = select_round(traj, t_stop, mode)
        rows.append(
            {
                "traj_id": traj.traj_id,
                "task_id": traj.task_id,
                "loop_type": traj.loop_type,
                "t_stop": t_stop,
                "t_selected": sel,
                "utility": u,
                "regret": u_star - u,
                "quality": traj.r(sel),
                "cost": traj.cost(t_stop),
            }
        )
    regrets = [r["regret"] for r in rows]
    return PolicyResult(
        policy=policy_label(policy),
        lam=lam,
        mode=mode,
        mean_utility=statistics.fmean(r["utility"] for r in rows),
        mean_regret=statistics.fmean(regrets),
        regret_ci=bootstrap_ci(regrets, n=bootstrap_n, seed=seed),
        mean_stop_round=statistics.fmean(r["t_stop"] for r in rows),
        mean_quality=statistics.fmean(r["quality"] for r in rows),
        mean_cost=statistics.fmean(r["cost"] for r in rows),
        n=len(rows),
        per_traj=rows,
    )


def evaluate_suite(
    policies: list[StoppingPolicy],
    trajs: list[Trajectory],
    lams: list[float],
    mode: str = "stop_at_last",
) -> list[PolicyResult]:
    """Evaluate every policy and cost-weight combination."""
    pending = [t.traj_id for t in trajs if not t.has_truth()]
    if pending:
        raise ValueError(f"{len(pending)} trajectories have pending hidden truth, e.g. {pending[:3]}")
    return [evaluate_policy(p, trajs, lam, mode) for lam in lams for p in policies]


def oracle_row(trajs: list[Trajectory], lam: float, mode: str = "stop_at_last") -> dict:
    """Compute the oracle reference point in quality, cost, and utility."""
    us, ts, qs, cs = [], [], [], []
    for traj in trajs:
        t, u = oracle_stop(traj, lam, mode)
        us.append(u)
        ts.append(t)
        qs.append(traj.r(select_round(traj, t, mode)))
        cs.append(traj.cost(t))
    return {
        "policy": "oracle",
        "lam": lam,
        "mean_utility": statistics.fmean(us),
        "mean_stop_round": statistics.fmean(ts),
        "mean_quality": statistics.fmean(qs),
        "mean_cost": statistics.fmean(cs),
    }


def bootstrap_ci(
    values: list[float], n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if len(values) < 2:
        v = values[0] if values else 0.0
        return (v, v)
    rng = random.Random(seed)
    k = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=k)) for _ in range(n))
    lo = means[int(alpha / 2 * n)]
    hi = means[min(int((1 - alpha / 2) * n), n - 1)]
    return (lo, hi)
