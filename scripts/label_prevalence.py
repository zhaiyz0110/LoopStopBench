"""Report continuation-label prevalence for the pooled and inner-fit samples."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.replay import task_fold, utility
from loopstop.utils.io import load_trajectories


def label_counts(trajs, lam: float, mode: str) -> tuple[int, int]:
    positive = 0
    total = 0
    for traj in trajs:
        utilities = [utility(traj, t, lam, mode) for t in range(1, traj.T + 1)]
        for index in range(traj.T - 1):
            positive += int(max(utilities[index + 1:]) > utilities[index] + 1e-9)
            total += 1
    return positive, total


def inner_fit_sample(trajs, test_fold: int):
    train = [traj for traj in trajs if task_fold(traj.task_id) != test_fold]
    fit = [traj for traj in train if task_fold(traj.task_id + ":inner") == 0]
    validation = [traj for traj in train if task_fold(traj.task_id + ":inner") == 1]
    return fit if fit and validation else train


def rate(positive: int, total: int) -> float:
    return positive / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", nargs="+", required=True)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.005)
    parser.add_argument("--mode", default="stop_at_last",
                        choices=["stop_at_last", "stop_and_select"])
    parser.add_argument("--exclude-tasks", default="")
    parser.add_argument("--out", default="results/label_prevalence.csv")
    args = parser.parse_args()

    excluded = {task_id.strip() for task_id in args.exclude_tasks.split(",") if task_id.strip()}
    rows = []
    for path in args.traj:
        trajs = [
            traj for traj in load_trajectories(path)
            if traj.has_truth() and traj.task_id not in excluded
        ]
        if not trajs:
            print(f"[skip] {path}: no trajectories with complete hidden truth")
            continue

        all_pos, all_total = label_counts(trajs, args.lam, args.mode)
        fold_counts = [label_counts(inner_fit_sample(trajs, fold), args.lam, args.mode)
                       for fold in (0, 1)]
        row = {
            "loop": Path(path).stem,
            "trajectories": len(trajs),
            "tasks": len({traj.task_id for traj in trajs}),
            "all_positive": all_pos,
            "all_total": all_total,
            "all_rate": rate(all_pos, all_total),
            "fold0_fit_positive": fold_counts[0][0],
            "fold0_fit_total": fold_counts[0][1],
            "fold0_fit_rate": rate(*fold_counts[0]),
            "fold1_fit_positive": fold_counts[1][0],
            "fold1_fit_total": fold_counts[1][1],
            "fold1_fit_rate": rate(*fold_counts[1]),
        }
        rows.append(row)
        print(
            f"{row['loop']}: n={row['trajectories']}, tasks={row['tasks']}; "
            f"all={all_pos}/{all_total} ({row['all_rate']:.1%}); "
            f"fold0-fit={fold_counts[0][0]}/{fold_counts[0][1]} "
            f"({row['fold0_fit_rate']:.1%}); "
            f"fold1-fit={fold_counts[1][0]}/{fold_counts[1][1]} "
            f"({row['fold1_fit_rate']:.1%})"
        )

    if not rows:
        sys.exit("no prevalence rows produced")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"csv -> {output}")


if __name__ == "__main__":
    main()
