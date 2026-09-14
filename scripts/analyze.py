"""Report trajectory types, peak rounds, changes, and score-quality correlations."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.analysis import (
    classify,
    final_neq_best_rate,
    marginal_improvement_curve,
    mean_quality_curve,
    peak_round_distribution,
    proxy_truth_corr_by_round,
    scissor_gap_rate,
    type_proportions,
)
from loopstop.utils.io import load_trajectories


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--out", default=None, help="可选: 每轨迹分型明细 CSV")
    args = ap.parse_args()

    trajs = load_trajectories(args.traj)
    ready = [t for t in trajs if t.has_truth()]
    print(f"trajectories: {len(trajs)} total, {len(ready)} with truth "
          f"({len(trajs) - len(ready)} pending offline judging)")
    if not ready:
        sys.exit("no trajectory has complete hidden truth yet")

    fnb = final_neq_best_rate(ready, delta=args.delta)
    print(f"\n== final≠best rate (δ={args.delta}): {fnb:.1%}")

    print("\n== typology proportions:")
    for label, p in type_proportions([t.r_series for t in ready],
                                     eps=args.eps, delta=args.delta).items():
        print(f"  {label:12s} {p:.1%}")

    print("\n== peak round distribution:")
    for t, c in peak_round_distribution(ready).items():
        print(f"  t={t:2d}  {'#' * c} {c}")

    print("\n== E[r_t] curve:")
    for t, mean, sd in mean_quality_curve(ready):
        print(f"  t={t:2d}  {mean:.3f} ± {sd:.3f}")

    print("\n== marginal improvement E[r_(t+1)−r_t]:")
    for t, d in marginal_improvement_curve(ready):
        print(f"  t={t:2d}→{t+1:<2d}  {d:+.4f}")

    print("\n== corr(v_t, r_t) by round:")
    for t, c in proxy_truth_corr_by_round(ready):
        print(f"  t={t:2d}  {'n/a' if c is None else f'{c:+.3f}'}")

    print(f"\n== score-quality divergence rate: {scissor_gap_rate(ready, args.delta):.1%}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["traj_id", "task_id", "loop_type", "model_cfg", "seed",
                        "type", "peak_t", "r_series"])
            for t in ready:
                res = classify([float(x) for x in t.r_series],
                               eps=args.eps, delta=args.delta)
                w.writerow([t.traj_id, t.task_id, t.loop_type,
                            t.steps[0].model_cfg, t.steps[0].seed,
                            res.label, res.peak_t, json.dumps(t.r_series)])
        print(f"\nper-trajectory detail → {args.out}")


if __name__ == "__main__":
    main()
