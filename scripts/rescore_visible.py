"""Rescore stored outputs with another critic or query protocol.

Each score is added to ``visible_signals`` without changing hidden quality.
The resulting files can be compared with ``oracle_visible.py --score-field``.
Successive runs may add scores from several critic models:
  # 8B(读原始轨迹)
  python scripts/rescore_visible.py --traj m2_writing_small_judged.jsonl \\
      --critic gen_8b --field-name score_critic_gen_8b --out ws_r1.jsonl
  # 32B(读上一步产物, 保留 8B 字段)
  python scripts/rescore_visible.py --traj ws_r1.jsonl \\
      --critic gen_32b --field-name score_critic_gen_32b --out ws_r2.jsonl
  # 72B(切 large/solo 档后)
  python scripts/rescore_visible.py --traj ws_r2.jsonl \\
      --critic gen_72b --field-name score_critic_gen_72b --out m2_ws_rescored.jsonl

With ``--repeat M``, the script stores the mean score and the corresponding
``__sd``, ``__samples``, and ``__m`` fields. These fields support reliability
and repeated-query analyses.
  python scripts/rescore_visible.py --traj m2_writing_small_judged.jsonl \\
      --critic gen_32b --field-name score_rep8 --repeat 8 --temperature 0.7 --out ws_rep8.jsonl
  python scripts/oracle_visible.py --traj ws_rep8.jsonl --score-field score_rep8 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.llm import client_from_config, load_models_config
from loopstop.loops.writer_critic import CRITIC_SYSTEM, CRITIC_TMPL
from loopstop.utils.io import append_steps, load_trajectories, read_steps


def score_output(client, assignment: str, essay: str, temperature: float = 0.0) -> Optional[float]:
    """Return an absolute rubric score in [0, 1].

    A positive temperature supports independent repeated queries whose valid
    scores are averaged by the caller.
    Invalid JSON or a nonnumeric ``overall`` value returns ``None`` and is
    excluded from the repeated-query aggregate.
    """
    res = client.chat(
        [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": CRITIC_TMPL.format(assignment=assignment, essay=essay)},
        ],
        temperature=temperature,
        max_tokens=800,
    )
    return _safe_overall(res.text)


def _safe_overall(text: str) -> Optional[float]:
    """从 critic 响应稳健取 overall∈[0,1]; 无法解析/非数值 → None(不兜底成 0.5, 避免污染聚合)。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ov = d.get("overall")
    if isinstance(ov, bool):          # True/False 会被 float() 误当 1.0/0.0
        return None
    try:
        return float(ov) / 10.0       # 接受 int/float 及 "8" 这类数值串
    except (TypeError, ValueError):   # 评语塞进 overall / 缺字段 → 作废
        return None


# Pairwise scoring uses the hidden-quality prompt with a different model. It
# compares each round with round one and maps the result to the same scale.
PAIRWISE_SYSTEM = "You are an impartial writing judge comparing two drafts. Respond only with the JSON requested."

PAIRWISE_TMPL = """Two drafts were written for the same assignment. Compare Draft B against
Draft A on how well they fulfil the assignment (clarity, argument strength, structure, style).

Assignment:
{assignment}

Draft A:
---
{draft_a}
---

Draft B:
---
{draft_b}
---

Rate how much better Draft B is compared to Draft A as one integer from -5 to +5:
-5 = B is much worse, 0 = same quality, +5 = B is much better.
Respond with exactly this JSON: {{"relative": <integer -5..5>}}"""


def score_pairwise(client, assignment: str, anchor: str, candidate: str, key: str,
                   temperature: float = 0.0) -> float:
    """candidate(round t) vs anchor(round 1) 相对分 → 0.5 + x/10。顺序按 key 哈希翻转去偏。"""
    flip = int(hashlib.md5(f"{key}:order".encode()).hexdigest(), 16) % 2 == 1
    a, b = (candidate, anchor) if flip else (anchor, candidate)
    res = client.chat(
        [
            {"role": "system", "content": PAIRWISE_SYSTEM},
            {"role": "user", "content": PAIRWISE_TMPL.format(assignment=assignment, draft_a=a, draft_b=b)},
        ],
        temperature=temperature,
        max_tokens=60,
    )
    m = re.search(r"-?\d+", res.text)
    x = max(-5, min(5, int(m.group(0)) if m else 0))
    x = -x if flip else x
    return 0.5 + x / 10.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--critic", required=True, help="models.yaml 的端点名(如 gen_8b/gen_32b/gen_72b)")
    ap.add_argument("--field-name", required=True, help="写入 visible_signals 的新字段名")
    ap.add_argument("--assignments", default="data/raw/writing_tasks_m2.jsonl")
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="显式续采已存在的 --out(确认它是同一 --traj/--critic 的部分产物)")
    ap.add_argument("--protocol", default="absolute", choices=["absolute", "pairwise"],
                    help="absolute: rubric 绝对分; pairwise: vs 首轮的相对分(与隐藏真值同口径)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Query each output M times and store the mean; M>1 requires temperature>0")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="critic 采样温度; --repeat>1 时必须 >0 才能产生独立样本(贪心解码下 M 次全同)")
    args = ap.parse_args()

    if args.repeat < 1:
        sys.exit("--repeat 必须 ≥1")
    if args.repeat > 1 and args.temperature <= 0.0:
        sys.exit("--repeat>1 需要 --temperature>0, 否则贪心解码下 M 次调用完全相同、取均值无意义。")

    cfg = load_models_config(args.models_config)
    client = client_from_config(cfg, args.critic)
    assignments = {
        d["task_id"]: d["assignment"]
        for d in map(json.loads, filter(str.strip, open(args.assignments, encoding="utf-8")))
    }

    out_path = Path(args.out)
    if out_path.exists() and not args.resume:
        sys.exit(f"{out_path} 已存在。删除它, 或加 --resume 显式续采"
                 "(确认它确是同一 --traj/--critic 的部分产物, 否则会得到残缺/串档文件)。")
    done = {(s.traj_id, s.t) for s in read_steps(out_path)} if out_path.exists() else set()

    trajs = load_trajectories(args.traj)
    anchor = {t.traj_id: t.steps[0].output for t in trajs}  # pairwise 用: 每轨迹首轮草稿
    work = [s for t in trajs for s in t.steps if (s.traj_id, s.t) not in done]
    print(f"{len(work)} steps to rescore with critic={args.critic} protocol={args.protocol} "
          f"repeat={args.repeat} temp={args.temperature} → field {args.field_name}", flush=True)

    def score_once(step) -> float:
        if args.protocol == "pairwise":
            return 0.5 if step.t == 1 else score_pairwise(
                client, assignments[step.task_id], anchor[step.traj_id], step.output,
                f"{step.traj_id}:{step.t}", temperature=args.temperature)
        return score_output(client, assignments[step.task_id], step.output,
                            temperature=args.temperature)

    def do(step):
        vals = [v for v in (score_once(step) for _ in range(args.repeat)) if v is not None]
        # 全废(极少): 退回采集期部署分, 保持与 M=1 相同覆盖度; 否则取有效采样均值
        agg = statistics.fmean(vals) if vals else step.visible_signals.get("score")
        step.visible_signals[args.field_name] = agg
        if args.repeat > 1:  # 侧字段: 供离线算 critic 自身信度(Spearman-Brown); 不在特征列表, 被 extract_features 忽略
            step.visible_signals[f"{args.field_name}__m"] = len(vals)   # 有效重采样次数; 0=用了部署分兜底
            step.visible_signals[f"{args.field_name}__sd"] = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
            step.visible_signals[f"{args.field_name}__samples"] = vals
        return step

    n = 0
    failed_draws = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for step in pool.map(do, work):
            append_steps(out_path, [step])  # 主线程顺序落盘, 保留已有可见字段
            n += 1
            if args.repeat > 1:
                failed_draws += args.repeat - int(step.visible_signals.get(f"{args.field_name}__m", args.repeat))
            if n % 100 == 0:
                print(f"rescored {n}/{len(work)}...", flush=True)
    print(f"done: {n} steps → {out_path}", flush=True)
    if args.repeat > 1 and n:
        print(f"作废采样(解析失败已丢弃): {failed_draws}/{n * args.repeat} "
              f"({failed_draws / (n * args.repeat):.1%})", flush=True)


if __name__ == "__main__":
    main()
