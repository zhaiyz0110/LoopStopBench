"""Add hidden writing-quality judgments to stored trajectories.

pairwise (default):
    每轮输出与同轨迹第 1 轮草稿盲比, 裁判输出相对整数分 x ∈ [-5, +5]
    (−5 差得多 / 0 相当 / +5 好得多), 映射 r = 0.5 + x/10 ∈ [0,1];
    第 1 轮自比恒为 0.5, 不消耗裁判调用。
    两稿呈现顺序按 (traj_id, t) 哈希确定性随机, 位置偏置跨样本对消。
    Existing absolute scores are retained as hidden_truth.r_abs.

dual:
    每步同时做两次对比评审: global(vs 首轮草稿, 累计改进 → r/r_rel)
    与 local(vs 上一轮, 边际改进 → r_local)。
    local 偏好链非传递、无法由 global 积分得到, 两个尺度信息互补。

absolute:
    离线盲评, 每轮独立 1-10 分, 裁判团中位数 /10 写回 r。仅填充 pending 步。

裁判团配置见 models 配置的 judge_panel。API 校准结果写入 hidden_truth，
但不参与中位数。

Example with two judges run separately and then merged:
  # Qwen judge
  python scripts/judge_writing.py --protocol dual --only-judge judge_qwen72b \\
      --traj m2_writing.jsonl --out m2_writing_qwen.jsonl \\
      --models-config configs/models_m2_judge.yaml --assignments writing_tasks_m2.jsonl
  # Run the same command with the Llama judge, then merge both outputs:
  python scripts/merge_judgments.py --inputs m2_writing_qwen.jsonl m2_writing_llama.jsonl \\
      --out m2_writing_judged.jsonl
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.llm import client_from_config, load_models_config
from loopstop.utils.io import append_steps, load_trajectories, read_steps

# Absolute-score protocol.

JUDGE_SYSTEM = "You are an impartial essay judge. Respond only with the JSON requested."

JUDGE_TMPL = """Score this essay against the assignment on a 1-10 scale
(consider clarity, argument strength, structure, style; judge the essay on its own,
you have no other versions to compare with).

Assignment:
{assignment}

Essay:
---
{essay}
---

Respond with exactly: {{"score": <1-10>}}"""

# Pairwise protocol.

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
Respond with exactly: {{"relative": <integer -5..5>}}"""


def judge_score(client, assignment: str, essay: str) -> float:
    res = client.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_TMPL.format(assignment=assignment, essay=essay)},
        ],
        temperature=0.0,
        max_tokens=50,
    )
    m = re.search(r"\d+(?:\.\d+)?", res.text)
    return min(max(float(m.group(0)) if m else 5.0, 1.0), 10.0)


def judge_pairwise(client, assignment: str, anchor: str, candidate: str, key: str) -> int:
    """返回 candidate 相对 anchor 的整数分 ∈ [-5, 5]。呈现顺序按 key 哈希翻转。"""
    flip = int(hashlib.md5(f"{key}:order".encode()).hexdigest(), 16) % 2 == 1
    a, b = (candidate, anchor) if flip else (anchor, candidate)
    res = client.chat(
        [
            {"role": "system", "content": PAIRWISE_SYSTEM},
            {
                "role": "user",
                "content": PAIRWISE_TMPL.format(assignment=assignment, draft_a=a, draft_b=b),
            },
        ],
        temperature=0.0,
        max_tokens=60,
    )
    m = re.search(r"-?\d+", res.text)
    x = max(-5, min(5, int(m.group(0)) if m else 0))  # 解析失败按 0(相当)
    return -x if flip else x  # 翻转时 B=anchor, candidate 在 A 位, 取反


def _calib_hit(traj_id: str, t: int, fraction: float) -> bool:
    h = int(hashlib.md5(f"{traj_id}:{t}".encode()).hexdigest(), 16) % 100
    return h < fraction * 100


def score_step(step, traj, assignment, panel, calib_client, calib_frac, protocol):
    """对单个 step 打分并写入 step.hidden_truth。各 step 相互独立(只依赖 traj 内已知的
    首轮/上一轮文本), 可并发。单裁判模式(--only-judge)时 panel 只含一个成员。"""
    if protocol == "absolute":
        if step.hidden_truth.get("r") is None:
            scores = {m: judge_score(c, assignment, step.output) for m, c in panel.items()}
            step.hidden_truth = {
                "r": statistics.median(scores.values()) / 10.0,
                "pending": False,
                "judge_scores": scores,
            }
            if calib_client and _calib_hit(step.traj_id, step.t, calib_frac):
                step.hidden_truth["judge_scores"]["api_calibration"] = judge_score(
                    calib_client, assignment, step.output
                )
        return step

    # pairwise / dual
    anchor = traj.steps[0].output
    gkey = f"{step.traj_id}:{step.t}:g"
    if step.t == 1:
        g_rels = {m: 0 for m in panel}  # 首轮自比恒平, 不消耗调用
        l_rels = {m: 0 for m in panel} if protocol == "dual" else None
    else:
        g_rels = {  # global: 与首轮草稿比, 累计改进
            m: judge_pairwise(c, assignment, anchor, step.output, gkey) for m, c in panel.items()
        }
        if protocol == "dual":  # local: 与上一轮比, 边际改进(VOI 标签)
            prev = traj.steps[step.t - 2].output
            lkey = f"{step.traj_id}:{step.t}:l"
            l_rels = {
                m: judge_pairwise(c, assignment, prev, step.output, lkey) for m, c in panel.items()
            }
        else:
            l_rels = None
    g_med = statistics.median(g_rels.values())
    step.hidden_truth = {
        "r": 0.5 + g_med / 10.0,
        "r_rel": g_med,  # global 相对首轮
        "r_abs": step.hidden_truth.get("r"),  # v1 绝对分留档
        "pending": False,
        "judge_rels": g_rels,
    }
    if l_rels is not None:
        step.hidden_truth["r_local"] = statistics.median(l_rels.values())
        step.hidden_truth["judge_rels_local"] = l_rels
    if calib_client and step.t > 1 and _calib_hit(step.traj_id, step.t, calib_frac):
        step.hidden_truth["judge_rels"]["api_calibration"] = judge_pairwise(
            calib_client, assignment, anchor, step.output, gkey
        )
    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--assignments", default="data/raw/writing_tasks.jsonl")
    ap.add_argument("--protocol", default="pairwise",
                    choices=["pairwise", "absolute", "dual"],
                    help="dual scores each step against both the first and previous drafts")
    ap.add_argument("--only-judge", default=None,
                    help="Use one judge endpoint; merge separate outputs with merge_judgments")
    ap.add_argument("--workers", type=int, default=8,
                    help="并发判分请求数; vLLM 靠并发攒 batch, TP=4 有充足 KV 余量")
    args = ap.parse_args()

    cfg = load_models_config(args.models_config)
    members = [args.only_judge] if args.only_judge else cfg["judge_panel"]["members"]
    panel = {name: client_from_config(cfg, name) for name in members}
    if args.only_judge:  # 单裁判先后跑时不触发 API 校准
        calib_client, calib_frac = None, 0.0
    else:
        calib = cfg["judge_panel"].get("api_calibration", {})
        calib_client = client_from_config(cfg, calib["member"]) if calib else None
        calib_frac = float(calib.get("fraction", 0.0))

    assignments = {
        d["task_id"]: d["assignment"]
        for d in map(json.loads, filter(str.strip, open(args.assignments, encoding="utf-8")))
    }

    out_path = Path(args.out)
    done = {(s.traj_id, s.t) for s in read_steps(out_path)} if out_path.exists() else set()

    # 收集待判步骤(全部独立), 并发提交; 结果在主线程顺序落盘, 无需写锁
    trajs = load_trajectories(args.traj)
    work = [(t, s) for t in trajs for s in t.steps if (s.traj_id, s.t) not in done]
    print(f"{len(work)} steps to judge, members={members}, workers={args.workers}", flush=True)

    def do(item):
        traj, step = item
        return score_step(step, traj, assignments[traj.task_id], panel, calib_client, calib_frac, args.protocol)

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for step in pool.map(do, work):
            append_steps(out_path, [step])
            n += 1
            if n % 50 == 0:
                print(f"judged {n}/{len(work)} steps...", flush=True)

    print(f"done: {n} steps → {out_path}", flush=True)


if __name__ == "__main__":
    main()
