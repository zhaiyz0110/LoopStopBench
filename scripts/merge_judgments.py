"""Merge separately generated writing judgments.

每个输入文件由 judge_writing.py --only-judge <member> 产出, 其 hidden_truth.judge_rels
只含一个裁判成员。本脚本按 (traj_id, t) 对齐, 合并各文件的 judge_rels /
judge_rels_local, 重算:
    r_rel   = median(所有裁判的 global 相对分)
    r       = 0.5 + r_rel / 10
    r_local = median(所有裁判的 local 相对分)   # 仅 dual 协议有
其余字段(output/visible_signals/r_abs 等)取第一个文件。

The report includes the distribution of per-step |max−min| disagreement and
the number of steps with disagreement greater than two points. This script is
offline and does not call an API.

用法:
  python scripts/merge_judgments.py \\
      --inputs m2_writing_qwen.jsonl m2_writing_llama.jsonl \\
      --out m2_writing_judged.jsonl
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.utils.io import append_steps, read_steps


def _numeric_members(rels: dict) -> list[float]:
    """Return numeric panel-member scores, excluding calibration entries."""
    return [v for k, v in rels.items() if k != "api_calibration" and isinstance(v, (int, float))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="各单裁判判分 jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base: dict[tuple, object] = {}       # (traj_id, t) → 第一个文件的 StepRecord
    g_rels: dict[tuple, dict] = {}       # 合并的 global judge_rels
    l_rels: dict[tuple, dict] = {}       # 合并的 local judge_rels

    for path in args.inputs:
        for s in read_steps(path):
            key = (s.traj_id, s.t)
            if key not in base:
                base[key] = s
                g_rels[key] = {}
                l_rels[key] = {}
            g_rels[key].update(s.hidden_truth.get("judge_rels", {}))
            l_rels[key].update(s.hidden_truth.get("judge_rels_local", {}))

    out_path = Path(args.out)
    if out_path.exists():
        sys.exit(f"{out_path} 已存在, 请先删除或换名(避免重复追加)")

    disagreements = []
    for key in sorted(base, key=lambda k: (k[0], k[1])):
        s = base[key]
        gvals = _numeric_members(g_rels[key])
        gmed = statistics.median(gvals) if gvals else 0.0
        ht = dict(s.hidden_truth)
        ht["judge_rels"] = g_rels[key]
        ht["r_rel"] = gmed
        ht["r"] = 0.5 + gmed / 10.0
        ht["pending"] = False
        ht["n_judges"] = len(gvals)
        lvals = _numeric_members(l_rels[key])
        if lvals:
            ht["judge_rels_local"] = l_rels[key]
            ht["r_local"] = statistics.median(lvals)
        s.hidden_truth = ht
        append_steps(out_path, [s])
        if len(gvals) >= 2 and s.t > 1:
            disagreements.append(max(gvals) - min(gvals))

    print(f"merged {len(base)} steps from {len(args.inputs)} judges → {out_path}", flush=True)
    if disagreements:
        n = len(disagreements)
        mean_d = statistics.fmean(disagreements)
        gt2 = sum(1 for d in disagreements if d > 2)
        print(f"裁判一致性(global, t>1): n={n}  mean|max−min|={mean_d:.2f}  "
              f">2 档分歧={gt2} ({gt2 / n:.1%})")
        print("较大的逐步分歧可用于确定额外校准样本")


if __name__ == "__main__":
    main()
