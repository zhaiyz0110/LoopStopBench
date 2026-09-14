"""Analyze stopping regret by task difficulty and generator size.

For code, trajectories are joined to model-specific difficulty tiers and
first-round pass rates. The script reports tier means, same-task model
differences, and a regression on difficulty and model size. For writing, it
reports model-group differences with a trajectory-level length correlation.
The regression implementation uses only the Python standard library.

Code example:
  python scripts/analyze_difficulty.py --mode code \\
      --traj data/m2_code_8b.jsonl:gen_8b \\
             data/m2_code_32b.jsonl:gen_32b \\
      --task-pool data/raw/code_tasks_m2.json
Writing example:
  python scripts/analyze_difficulty.py --mode writing \\
      --traj data/m2_writing_small_judged.jsonl:small \\
             data/m2_writing_mid_judged.jsonl:mid
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.replay import oracle_stop, utility
from loopstop.utils.io import load_trajectories

TIERS = ("easy", "medium", "hard")


# Standard-library numerical helpers.

def ols(rows: list[list[float]], y: list[float]) -> list[float]:
    """正规方程解 (XᵀX)⁻¹Xᵀy; rows 已含截距列。返回系数向量。"""
    n, p = len(rows), len(rows[0])
    xtx = [[sum(rows[k][i] * rows[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    xty = [sum(rows[k][i] * y[k] for k in range(n)) for i in range(p)]
    return _solve(xtx, xty)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """高斯消元(带部分主元)。a 会被就地修改。"""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        if abs(m[c][c]) < 1e-12:
            m[c][c] = 1e-12
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def standardize(xs: list[float]) -> list[float]:
    mu = statistics.fmean(xs)
    sd = statistics.pstdev(xs) or 1.0
    return [(x - mu) / sd for x in xs]


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def parse_specs(specs: list[str]) -> list[tuple[str, str]]:
    """"path:label" → (path, label)。"""
    out = []
    for s in specs:
        path, _, label = s.rpartition(":")
        if not path:
            raise ValueError(f"--traj 需要 path:label 形式, 收到 {s!r}")
        out.append((path, label))
    return out


def parse_exclusions(value: str) -> set[str]:
    return {task_id.strip() for task_id in value.split(",") if task_id.strip()}


def write_summary(path: str, rows: list[dict]) -> None:
    fields = [
        "section",
        "metric",
        "tier",
        "model",
        "comparison",
        "value",
        "n",
        "lambda",
        "reference_policy",
        "excluded_tasks",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\ncsv -> {output}")


def traj_regret(traj, lam: float) -> float:
    """该轨迹的 oracle regret: U(π*) − U(参考策略)。
    参考策略取 S1-n5(固定 5 轮), 与 replay 表可比且不依赖可见信号可得性。"""
    ref_t = min(5, traj.T)
    _, u_star = oracle_stop(traj, lam)
    return u_star - utility(traj, ref_t, lam)


# --------------------------------- 代码侧 ---------------------------------

def run_code(args) -> None:
    pool = {t["task_id"]: t for t in json.load(open(args.task_pool, encoding="utf-8"))}
    lam = args.lam
    excluded = parse_exclusions(args.exclude_tasks)
    summary = []
    # 长表: 每条轨迹一行
    recs = []  # (model, tier, pass_rate, regret, task_id, final_neq_best)
    for path, label in parse_specs(args.traj):
        for traj in load_trajectories(path):
            if traj.task_id in excluded:
                continue
            meta = pool.get(traj.task_id)
            if meta is None:
                continue
            tier = meta.get("difficulty_tier", {}).get(label)
            rate = meta.get("first_round_pass_rate", {}).get(label)
            if tier is None or rate is None:
                continue
            r = [traj.r(t) for t in range(1, traj.T + 1)]
            recs.append({
                "model": label, "tier": tier, "rate": float(rate),
                "regret": traj_regret(traj, lam), "task_id": traj.task_id,
                "fnb": int(max(r) - r[-1] > args.delta),
            })
    if not recs:
        sys.exit("无可用记录, 检查 task_pool 是否含 difficulty_tier/first_round_pass_rate")

    models = sorted({r["model"] for r in recs})
    print(f"# 代码侧难度×规模归因 (λ={lam}, 参考策略 S1-n5, n={len(recs)})\n")

    # 列联表: 行=难度层, 列=模型, 值=平均 regret
    print("== mean oracle regret: 难度层 × 模型 ==")
    print(f"{'tier':<8}" + "".join(f"{m:>12}" for m in models) + f"{'Δ(size)':>12}")
    for tier in TIERS:
        cell = {}
        counts = {}
        for m in models:
            vals = [r["regret"] for r in recs if r["tier"] == tier and r["model"] == m]
            cell[m] = statistics.fmean(vals) if vals else None
            counts[m] = len(vals)
            if vals:
                summary.append({
                    "section": "tier",
                    "metric": "mean_regret",
                    "tier": tier,
                    "model": m,
                    "comparison": "",
                    "value": f"{cell[m]:.8f}",
                    "n": len(vals),
                    "lambda": lam,
                    "reference_policy": "S1-n5",
                    "excluded_tasks": ",".join(sorted(excluded)),
                })
        row = f"{tier:<8}"
        for m in models:
            row += f"{cell[m]:>12.4f}" if cell[m] is not None else f"{'—':>12}"
        if all(cell[m] is not None for m in models) and len(models) == 2:
            size_delta = cell[models[1]] - cell[models[0]]
            row += f"{size_delta:>+12.4f}"
            summary.append({
                "section": "tier",
                "metric": "model_difference",
                "tier": tier,
                "model": "",
                "comparison": f"{models[1]}-{models[0]}",
                "value": f"{size_delta:.8f}",
                "n": min(counts.values()),
                "lambda": lam,
                "reference_policy": "S1-n5",
                "excluded_tasks": ",".join(sorted(excluded)),
            })
        print(row)

    # 沿列的 difficulty 梯度(固定模型)
    print("\n== difficulty 梯度(固定模型, hard−easy 的 regret 差) ==")
    for m in models:
        e = [r["regret"] for r in recs if r["tier"] == "easy" and r["model"] == m]
        h = [r["regret"] for r in recs if r["tier"] == "hard" and r["model"] == m]
        if e and h:
            difficulty_delta = statistics.fmean(h) - statistics.fmean(e)
            print(f"  {m}: hard−easy = {difficulty_delta:+.4f}")
            summary.append({
                "section": "difficulty",
                "metric": "hard_minus_easy",
                "tier": "",
                "model": m,
                "comparison": "hard-easy",
                "value": f"{difficulty_delta:.8f}",
                "n": len(e) + len(h),
                "lambda": lam,
                "reference_policy": "S1-n5",
                "excluded_tasks": ",".join(sorted(excluded)),
            })

    # 同题配对(控制任务身份)
    if len(models) == 2:
        by_task = defaultdict(dict)
        for r in recs:
            by_task[r["task_id"]][r["model"]] = r
        paired = [(v[models[0]], v[models[1]]) for v in by_task.values() if len(v) == 2]
        diffs = [b["regret"] - a["regret"] for a, b in paired]
        if diffs:
            paired_mean = statistics.fmean(diffs)
            paired_t = paired_mean / (statistics.pstdev(diffs) / len(diffs) ** 0.5 or 1)
            print(f"\n== 同题配对 (n={len(diffs)}, 控制任务身份) ==")
            print(f"  同题上 {models[1]}−{models[0]} 平均 regret 差 = {paired_mean:+.4f}")
            print(f"  (配对 t 类比: mean/se = {paired_t:.2f})")
            for metric, value in (("paired_mean_difference", paired_mean), ("paired_t_analogue", paired_t)):
                summary.append({
                    "section": "paired",
                    "metric": metric,
                    "tier": "",
                    "model": "",
                    "comparison": f"{models[1]}-{models[0]}",
                    "value": f"{value:.8f}",
                    "n": len(diffs),
                    "lambda": lam,
                    "reference_policy": "S1-n5",
                    "excluded_tasks": ",".join(sorted(excluded)),
                })

    # 回归: regret ~ 难度(1−pass_rate) + 规模(0/1)
    print("\n== 回归 regret ~ 难度(1−首轮通过率) + 规模 (标准化系数) ==")
    diff_x = standardize([1 - r["rate"] for r in recs])
    size_x = standardize([models.index(r["model"]) for r in recs]) if len(models) == 2 else [0.0] * len(recs)
    y = [r["regret"] for r in recs]
    X = [[1.0, diff_x[i], size_x[i]] for i in range(len(recs))]
    b = ols(X, y)
    corr_difficulty = pearson([1-r["rate"] for r in recs], y)
    corr_size = pearson([models.index(r["model"]) for r in recs], y)
    print(f"  截距={b[0]:+.4f}   难度系数={b[1]:+.4f}   规模系数={b[2]:+.4f}")
    print(f"  单变量 corr: regret~难度={corr_difficulty:+.3f}  regret~规模={corr_size:+.3f}")
    for metric, model, value in (
        ("standardized_coefficient", "difficulty", b[1]),
        ("standardized_coefficient", "model_size", b[2]),
        ("pearson_correlation", "difficulty", corr_difficulty),
        ("pearson_correlation", "model_size", corr_size),
    ):
        summary.append({
            "section": "regression",
            "metric": metric,
            "tier": "",
            "model": model,
            "comparison": "",
            "value": f"{value:.8f}",
            "n": len(recs),
            "lambda": lam,
            "reference_policy": "S1-n5",
            "excluded_tasks": ",".join(sorted(excluded)),
        })
    # Compare coefficient magnitudes without assuming which term dominates.
    diff_c, size_c = abs(b[1]), abs(b[2])
    if len(models) == 2 and max(diff_c, size_c) > 0:
        if size_c >= 2 * diff_c:
            verdict = "模型规模系数的绝对值至少是难度系数的两倍"
        elif diff_c >= 2 * size_c:
            verdict = "难度系数的绝对值至少是模型规模系数的两倍"
        else:
            verdict = "两个标准化系数处于同一数量级"
        print(f"\n系数比较: {verdict}。同题配对结果另行报告。")

    if args.out:
        write_summary(args.out, summary)


# --------------------------------- 写作侧 ---------------------------------

def run_writing(args) -> None:
    lam = args.lam
    recs = []  # (pairing, regret, length_slope, quality_final)
    for path, label in parse_specs(args.traj):
        for traj in load_trajectories(path):
            anchor_len = len(traj.steps[0].output)
            # 该轨迹内 长度差 与 r 的关系斜率(正=越长评分越高)
            xs = [len(s.output) - anchor_len for s in traj.steps[1:]]
            ys = [traj.r(t) for t in range(2, traj.T + 1)]
            slope = pearson(xs, ys) if len(xs) >= 3 else 0.0
            recs.append({
                "pairing": label, "regret": traj_regret(traj, lam),
                "length_slope": slope if slope == slope else 0.0,
            })
    pairings = sorted({r["pairing"] for r in recs})
    print(f"# 写作侧 规模×长度归因 (λ={lam}, 参考策略 S1-n5, n={len(recs)})\n")
    print("== 按配对(作者规模)平均 ==")
    for p in pairings:
        sub = [r for r in recs if r["pairing"] == p]
        print(f"  {p:<8} regret={statistics.fmean(r['regret'] for r in sub):.4f}  "
              f"轨迹内长度-质量相关均值={statistics.fmean(r['length_slope'] for r in sub):+.3f}  n={len(sub)}")
    if len(pairings) == 2:
        print("\n== 回归 regret ~ 作者规模 + 长度斜率 (标准化系数) ==")
        size_x = standardize([pairings.index(r["pairing"]) for r in recs])
        len_x = standardize([r["length_slope"] for r in recs])
        y = [r["regret"] for r in recs]
        b = ols([[1.0, size_x[i], len_x[i]] for i in range(len(recs))], y)
        print(f"  截距={b[0]:+.4f}   规模系数={b[1]:+.4f}   长度斜率系数={b[2]:+.4f}")
        # Compare the length and model-size coefficient magnitudes.
        size_c, len_c = abs(b[1]), abs(b[2])
        if len_c >= 0.5 * size_c and len_c > 0.01:
            verdict = "长度斜率系数与模型规模系数处于同一数量级"
        else:
            verdict = "模型规模系数较大，长度斜率系数低于比较阈值"
        print(f"\n系数比较: {verdict}。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["code", "writing"])
    ap.add_argument("--traj", nargs="+", required=True, help="path:label (代码 label=模型端点名, 写作 label=配对名)")
    ap.add_argument("--task-pool", default="data/raw/code_tasks_m2.json", help="代码侧: 带难度层的任务池")
    ap.add_argument("--lam", type=float, default=0.005)
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument(
        "--exclude-tasks",
        default="Mbpp_793",
        help="comma-separated task IDs excluded from the analysis",
    )
    ap.add_argument("--out", default=None, help="optional CSV summary path (code mode)")
    args = ap.parse_args()
    (run_code if args.mode == "code" else run_writing)(args)


if __name__ == "__main__":
    main()
