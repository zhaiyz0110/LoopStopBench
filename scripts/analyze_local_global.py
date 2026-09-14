"""Compare local and cumulative quality changes along a trajectory.

The dual writing protocol records quality relative to both the first output
and the previous output. Code quality uses the observed round-to-round
difference. The report covers mean curves, sign agreement, signal
correlations, and the round with the largest local gain.

Local change by mode:
  code    : r_t − r_{t-1}
  writing : r_local − 0.5          (r_local=0.5+相对分/10)

Examples:
  python scripts/analyze_local_global.py --mode writing \\
      --traj data/m2_writing_small_judged.jsonl
  python scripts/analyze_local_global.py --mode code \\
      --traj data/m2_code_8b.jsonl
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.estimator.features import extract_features
from loopstop.utils.io import load_trajectories


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return float("nan")
    xs2, ys2 = zip(*pairs)
    n = len(xs2)
    mx, my = statistics.fmean(xs2), statistics.fmean(ys2)
    sx = sum((x - mx) ** 2 for x in xs2) ** 0.5
    sy = sum((y - my) ** 2 for y in ys2) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs2, ys2)) / (sx * sy)


def local_series(traj, mode: str) -> list[float]:
    """返回 t=2..T 的单步边际(index 0 对应 t=2)。"""
    if mode == "code":
        r = [traj.r(t) for t in range(1, traj.T + 1)]
        return [r[i] - r[i - 1] for i in range(1, len(r))]
    # writing: r_local 已是 [0,1](0.5=与上一轮相当), 减 0.5 得边际
    out = []
    for s in traj.steps[1:]:
        rl = s.hidden_truth.get("r_local")
        out.append(None if rl is None else rl - 0.5)
    return out


def global_marginal(traj) -> list[float]:
    """global 边际 r_t − r_{t-1}, t=2..T。"""
    r = [traj.r(t) for t in range(1, traj.T + 1)]
    return [r[i] - r[i - 1] for i in range(1, len(r))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["code", "writing"])
    ap.add_argument("--traj", required=True)
    ap.add_argument("--eps", type=float, default=0.0, help="边际过零判定的死区(默认 0)")
    args = ap.parse_args()

    trajs = load_trajectories(args.traj)
    T = trajs[0].T

    if args.mode == "writing" and all(
        s.hidden_truth.get("r_local") is None for t in trajs for s in t.steps[1:]
    ):
        sys.exit("写作 local 分析需要 dual 协议判分(hidden_truth.r_local 缺失)")

    print(f"# local vs global 停止尺度 (mode={args.mode}, n={len(trajs)} trajectories)\n")

    # 1) 两条曲线(按轮平均), t=2..T
    print("== E[单步边际] by round: global(r_t−r_{t-1}) vs local ==")
    print(f"{'t':>3}{'global':>12}{'local':>12}")
    g_zero = l_zero = None
    for i in range(T - 1):
        gvals = [global_marginal(t)[i] for t in trajs]
        lvals = [v for v in (local_series(t, args.mode)[i] for t in trajs) if v is not None]
        gm = statistics.fmean(gvals)
        lm = statistics.fmean(lvals) if lvals else float("nan")
        print(f"{i + 2:>3}{gm:>+12.4f}{lm:>+12.4f}")
        if g_zero is None and gm <= args.eps:
            g_zero = i + 2
        if l_zero is None and not math.isnan(lm) and lm <= args.eps:
            l_zero = i + 2
    print(f"\n过零点(平均边际首次 ≤{args.eps}): global t={g_zero}  local t={l_zero}")
    print("  → local 过零点 = '平均而言第几轮起单步不再值得继续'(VOI 停止的目标时机)")

    # 2) 符号一致性: global 累计方向 vs local 单步方向
    same = opp = gpos_lneg = 0
    for t in trajs:
        r = [t.r(k) for k in range(1, t.T + 1)]
        loc = local_series(t, args.mode)
        for i in range(t.T - 1):
            if loc[i] is None:
                continue
            gsign = 1 if (r[i + 1] - r[0]) > 0 else -1 if (r[i + 1] - r[0]) < 0 else 0
            lsign = 1 if loc[i] > 0 else -1 if loc[i] < 0 else 0
            if gsign == 0 or lsign == 0:
                continue
            if gsign == lsign:
                same += 1
            else:
                opp += 1
                if gsign > 0 and lsign < 0:
                    gpos_lneg += 1
    tot = same + opp
    if tot:
        print(f"\n== 符号一致性 (n={tot}) ==")
        print(f"  同号(累计与单步方向一致) = {same / tot:.1%}")
        print(f"  异号 = {opp / tot:.1%}  其中'累计在涨但这一步在退' = {gpos_lneg / tot:.1%}")
        print("  → 异号率高 = 整体改进中夹杂大量单步倒退(振荡循环特征)")

    # Association between the local-improvement label and visible features.
    print("\n== local>0 与可见信号的单变量相关 ==")
    labels, feat_cols = [], {}
    for t in trajs:
        loc = local_series(t, args.mode)
        for i in range(t.T - 1):  # 该步对应停在 t=i+1 后是否还有单步正向改进
            if loc[i] is None:
                continue
            labels.append(1.0 if loc[i] > 0 else 0.0)
            f = extract_features(t.visible(i + 1))
            for k in ("dv", "emb_dist_prev", "edit_dist_prev", "v", "failure_jaccard_prev"):
                feat_cols.setdefault(k, []).append(f.get(k))
    for k, col in feat_cols.items():
        print(f"  corr(local>0, {k:<20}) = {pearson(col, labels):+.3f}")
    print("  → |corr| 越大表示该单变量与 local>0 的线性关联越强")

    # 4) local 峰值轮分布
    print("\n== local 峰值轮 argmax(单步边际最大的轮次) 分布 ==")
    hist: dict[int, int] = {}
    for t in trajs:
        loc = local_series(t, args.mode)
        valid = [(i, v) for i, v in enumerate(loc) if v is not None]
        if valid:
            peak = max(valid, key=lambda p: p[1])[0] + 2
            hist[peak] = hist.get(peak, 0) + 1
    for tt in sorted(hist):
        print(f"  t={tt:>3}  {'#' * hist[tt]} {hist[tt]}")


if __name__ == "__main__":
    main()
