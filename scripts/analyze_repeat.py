"""Analyze repeated critic queries stored by ``rescore_visible.py``.

The report compares single-query and averaged signal correlations, estimates
single-query and aggregate reliability with ICC(1), and checks whether the
number of valid parses is correlated with hidden quality.

Examples:
  python scripts/analyze_repeat.py --traj data/m2_writing_small_rep8_1.jsonl
  python scripts/analyze_repeat.py --traj data/m2_writing_mid_rep8.jsonl
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.utils.io import load_trajectories


def pearson(pairs):
    """Return Pearson correlation and sample count after removing missing pairs."""
    xy = [(x, y) for x, y in pairs if x is not None and y is not None]
    n = len(xy)
    if n < 3:
        return None, n
    mx = statistics.fmean(x for x, _ in xy)
    my = statistics.fmean(y for _, y in xy)
    sxx = sum((x - mx) ** 2 for x, _ in xy)
    syy = sum((y - my) ** 2 for _, y in xy)
    sxy = sum((x - mx) * (y - my) for x, y in xy)
    if sxx <= 0 or syy <= 0:
        return None, n
    return sxy / (sxx ** 0.5 * syy ** 0.5), n


def icc_oneway(groups):
    """Estimate single-measure ICC(1) from repeated scores for each step."""
    groups = [g for g in groups if len(g) >= 2]
    k = len(groups)
    N = sum(len(g) for g in groups)
    if k < 2 or N <= k:
        return None, None, None
    grand = statistics.fmean(x for g in groups for x in g)
    gmeans = [statistics.fmean(g) for g in groups]
    ssb = sum(len(g) * (gm - grand) ** 2 for g, gm in zip(groups, gmeans))
    ssw = sum((x - gm) ** 2 for g, gm in zip(groups, gmeans) for x in g)
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    m0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)  # 不等组大小的有效重复数
    denom = msb + (m0 - 1) * msw
    if denom == 0:
        return None, None, None
    r1 = (msb - msw) / denom
    return r1, m0, statistics.fmean(len(g) for g in groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--field-name", default="score_rep8",
                    help="聚合可见分字段(其 __samples/__m 为 <field>__samples / <field>__m)")
    ap.add_argument("--deployed-field", default="score", help="采集期部署的单次 temp0 可见分字段")
    args = ap.parse_args()

    field, dep = args.field_name, args.deployed_field
    trajs = [t for t in load_trajectories(args.traj) if t.has_truth()]
    if not trajs:
        sys.exit("无带真值的轨迹")

    rows = []
    for tr in trajs:
        for s in tr.steps:
            vs = s.visible_signals
            samples = vs.get(f"{field}__samples") or []
            m = vs.get(f"{field}__m")
            rows.append({
                "r": s.hidden_truth.get("r"),
                "v_dep": vs.get(dep),
                "v_single": samples[0] if samples else None,  # 一个代表性 temp0.7 单次采样
                "v_m8": vs.get(field),
                "m": m if m is not None else (len(samples) if samples else None),
                "samples": samples,
            })

    loop = trajs[0].loop_type
    n_steps, n_traj = len(rows), len(trajs)

    # Sampling completeness.
    ms = [x["m"] for x in rows if x["m"] is not None]
    mean_m = statistics.fmean(ms) if ms else 0.0
    max_m = max(ms) if ms else 0
    n_fallback = sum(1 for x in rows if not x["samples"])
    sds = [statistics.pstdev(x["samples"]) for x in rows if len(x["samples"]) >= 2]
    mean_sd = statistics.fmean(sds) if sds else 0.0
    fail_rate = (1 - mean_m / max_m) if max_m else 0.0

    # Critic reliability.
    r1, m0, mbar = icc_oneway([x["samples"] for x in rows])
    rM = (mbar * r1 / (1 + (mbar - 1) * r1)) if (r1 is not None and (1 + (mbar - 1) * r1) != 0) else None

    # Correlation between visible score and hidden quality.
    c_dep, n_dep = pearson([(x["v_dep"], x["r"]) for x in rows])
    c_single, n_single = pearson([(x["v_single"], x["r"]) for x in rows])
    c_m8, n_m8 = pearson([(x["v_m8"], x["r"]) for x in rows])

    # Association between valid-response count and score or quality.
    c_mr, _ = pearson([(x["m"], x["r"]) for x in rows])
    c_mdep, _ = pearson([(x["m"], x["v_dep"]) for x in rows])

    def f(v):
        return "n/a" if v is None else f"{v:.4f}"

    print(f"== analyze_repeat: {Path(args.traj).stem} (loop={loop}, n_traj={n_traj}, n_steps={n_steps}) ==")
    print("[采样健康度]")
    print(f"  每步有效采样 m: 均值 {mean_m:.2f} / {max_m}  (作废率 {fail_rate:.1%}, 以观测最大采样数为分母)")
    print(f"  全废兜底步数(改用部署分): {n_fallback}")
    print(f"  组内 SD(critic 单次抖动): 均值 {mean_sd:.4f}")
    print("[critic 自身信度(重复采样估计, 单向 ICC(1))]")
    print(f"  单次信度 r1 = {f(r1)}" + ("" if r1 is None else f"   (有效重复数 m0={m0:.2f})"))
    if rM is not None:
        print(f"  M={mbar:.1f} 次合成信度 rM (Spearman-Brown) = {rM:.4f}")
    else:
        print("  M 次合成信度 rM = n/a")
    print("[corr(信号, 真值 r) — pooled over steps]")
    print(f"  部署 temp0 单次    corr = {f(c_dep)}  (n={n_dep})")
    print(f"  temp0.7 单次        corr = {f(c_single)}  (n={n_single})")
    print(f"  temp0.7 M=8 均值    corr = {f(c_m8)}  (n={n_m8})")
    if c_single is not None and c_m8 is not None:
        print(f"  → 同温度'平均'净效果 Δcorr(M=8 − 单次) = {c_m8 - c_single:+.4f}")
    print("[失败选择性偏差]")
    print(f"  corr(有效采样数 m, r)      = {f(c_mr)}   (≈0 → 失败与质量无关, 聚合无偏)")
    print(f"  corr(有效采样数 m, 部署分) = {f(c_mdep)}")


if __name__ == "__main__":
    main()
