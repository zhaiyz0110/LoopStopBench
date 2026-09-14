"""Measure association between judge scores and output-length changes.

The analysis regresses relative score ``r_rel`` on the character-count
difference between each candidate and its first-round anchor. It expects
trajectories produced by the pairwise or dual judging protocols.

用法:
  python scripts/analyze_judge_bias.py --traj data/writing_judged_v2.jsonl
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.utils.io import load_trajectories


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """最小二乘 y = a·x + b, 返回 (斜率, 截距)。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    return a, my - a * mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--field", default="r_rel", help="回归的分数字段: r_rel(global) 或 r_local")
    args = ap.parse_args()

    trajs = load_trajectories(args.traj)
    len_delta, rel, per_genre = [], [], {}
    for t in trajs:
        anchor_len = len(t.steps[0].output)
        genre = t.steps[0].task_id.split("_")[0]
        for s in t.steps:
            if s.t == 1:
                continue
            v = s.hidden_truth.get(args.field)
            if v is None:
                continue
            d = len(s.output) - anchor_len
            len_delta.append(float(d))
            rel.append(float(v))
            per_genre.setdefault(genre, [[], []])
            per_genre[genre][0].append(float(d))
            per_genre[genre][1].append(float(v))

    if len(len_delta) < 3:
        sys.exit(f"字段 {args.field!r} 样本不足(需 pairwise/dual 裁判产物)")

    r = pearson(len_delta, rel)
    a, b = linfit(len_delta, rel)
    print(f"n={len(len_delta)}  field={args.field}")
    print(f"corr(长度差, 相对分) = {r:+.3f}")
    print(f"线性拟合: 相对分 ≈ {a*1000:+.3f}·(长度差/1000字符) {b:+.2f}")
    print("参考范围: |corr|>0.4 表示较强关联; |corr|<0.2 表示较弱关联")
    print("\n分体裁:")
    for g, (xs, ys) in sorted(per_genre.items()):
        if len(xs) >= 3:
            print(f"  {g:10s} n={len(xs):3d}  corr={pearson(xs, ys):+.3f}")


if __name__ == "__main__":
    main()
