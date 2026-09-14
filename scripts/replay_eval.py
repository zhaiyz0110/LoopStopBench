"""Replay stopping policies on shared trajectories and export regret and Pareto data."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from loopstop.policies import build_policy, policy_label
from loopstop.replay import evaluate_suite, oracle_row
from loopstop.utils.io import load_trajectories


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--replay-config", default="configs/replay.yaml")
    ap.add_argument("--mode", default="stop_at_last", choices=["stop_at_last", "stop_and_select"])
    ap.add_argument("--exclude-tasks", default="",
                    help="Comma-separated task IDs to exclude (for example, Mbpp_793)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with open(args.replay_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    excluded = {task_id.strip() for task_id in args.exclude_tasks.split(",") if task_id.strip()}
    all_trajs = [t for t in load_trajectories(args.traj) if t.has_truth()]
    trajs = [t for t in all_trajs if t.task_id not in excluded]
    if not trajs:
        sys.exit("no trajectories with complete hidden truth")
    removed = len(all_trajs) - len(trajs)
    tasks = len({t.task_id for t in trajs})
    print(f"replaying on {len(trajs)} trajectories / {tasks} tasks, mode={args.mode}, "
          f"excluded={removed}")

    policies = []
    for spec in cfg["policies"]:
        try:
            policies.append(build_policy(spec))
        except Exception as e:  # For example, a missing optional VOI checkpoint.
            print(f"[skip] {spec.get('label', spec['type'])}: {e}")

    results = evaluate_suite(policies, trajs, lams=cfg["lambdas"], mode=args.mode)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.traj).stem

    # Regret table.
    with open(out_dir / f"regret_{stem}_{args.mode}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "lambda", "mean_regret", "regret_ci_lo", "regret_ci_hi",
                    "mean_utility", "mean_quality", "mean_cost", "mean_stop_round", "n"])
        for r in results:
            w.writerow([r.policy, r.lam, f"{r.mean_regret:.4f}",
                        f"{r.regret_ci[0]:.4f}", f"{r.regret_ci[1]:.4f}",
                        f"{r.mean_utility:.4f}", f"{r.mean_quality:.4f}",
                        f"{r.mean_cost:.2f}", f"{r.mean_stop_round:.2f}", r.n])

    # Quality-cost points, including the oracle reference.
    with open(out_dir / f"pareto_{stem}_{args.mode}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "lambda", "mean_quality", "mean_cost"])
        for r in results:
            w.writerow([r.policy, r.lam, f"{r.mean_quality:.4f}", f"{r.mean_cost:.2f}"])
        for lam in cfg["lambdas"]:
            o = oracle_row(trajs, lam, args.mode)
            w.writerow(["oracle", lam, f"{o['mean_quality']:.4f}", f"{o['mean_cost']:.2f}"])

    # Console summary for the middle cost weight.
    lam_mid = cfg["lambdas"][min(1, len(cfg["lambdas"]) - 1)]
    print(f"\n== regret @ λ={lam_mid} (mode={args.mode}):")
    subset = sorted((r for r in results if r.lam == lam_mid), key=lambda r: r.mean_regret)
    for r in subset:
        print(f"  {r.policy:10s} regret={r.mean_regret:.4f} "
              f"[{r.regret_ci[0]:.4f},{r.regret_ci[1]:.4f}]  "
              f"stop@{r.mean_stop_round:.1f}  quality={r.mean_quality:.3f}")
    print(f"\ncsv → {out_dir}/")


if __name__ == "__main__":
    main()
