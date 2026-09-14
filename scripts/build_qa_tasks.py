"""Build ReAct QA tasks from the HotpotQA distractor split.

Each task contains ten paragraphs. The default filter retains bridge questions
with at least two supporting paragraphs and removes yes/no answers. Hidden
quality is evaluated with exact match and F1. Output records have the form:
  {task_id, question, gold_answers:[answer], answer_eval:"em_f1",
   context:[10 段落], type, level, n_support}

Example:
  python scripts/build_qa_tasks.py --n 200 --out data/raw/qa_tasks_m2.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _iter_examples(local_json, split):
    """Normalize local JSON and Hugging Face records to a common schema.

    Local JSON uses the official HotpotQA schema:
    context=[[title,[句子]],...], supporting_facts=[[title,sent_id],...], _id);
    Hugging Face records store context as ``{title, sentences}`` arrays.
    """
    if local_json:
        raw = json.load(open(local_json, encoding="utf-8"))
        for ex in raw:
            yield {
                "id": str(ex.get("_id") or ex.get("id")),
                "question": ex["question"], "answer": ex["answer"],
                "type": ex.get("type", ""), "level": ex.get("level", ""),
                "context_pairs": [(c[0], c[1]) for c in ex["context"]],
                "sup_titles": {sf[0] for sf in ex["supporting_facts"]},
            }
    else:
        from datasets import load_dataset

        for ex in load_dataset("hotpot_qa", "distractor", split=split):
            yield {
                "id": str(ex["id"]), "question": ex["question"], "answer": ex["answer"],
                "type": ex["type"], "level": ex["level"],
                "context_pairs": list(zip(ex["context"]["title"], ex["context"]["sentences"])),
                "sup_titles": set(ex["supporting_facts"]["title"]),
            }


def build(examples, n, seed, include_comparison, levels):
    tasks = []
    for ex in examples:
        if len(ex["sup_titles"]) < 2:            # Require at least two supporting passages.
            continue
        if not include_comparison and ex["type"] != "bridge":
            continue
        if levels and ex["level"] not in levels:
            continue
        ans = str(ex["answer"]).strip()
        if not ans or ans.lower() in ("yes", "no"):
            continue
        context = [f"{t}: {' '.join(s)}" for t, s in ex["context_pairs"]]
        tasks.append({
            "task_id": "hotpot_" + ex["id"],
            "question": ex["question"],
            "gold_answers": [ans],
            "answer_eval": "em_f1",
            "context": context,
            "type": ex["type"],
            "level": ex["level"],
            "n_support": len(ex["sup_titles"]),
        })
    # Deterministic sampling independent of input order.
    tasks.sort(key=lambda t: hashlib.md5(f"{seed}:{t['task_id']}".encode()).hexdigest())
    return tasks[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-json", default=None,
                    help="离线: 官方 HotpotQA JSON 路径(如 hotpot_dev_distractor_v1.json);缺省走 HF 联网下载")
    ap.add_argument("--include-comparison", action="store_true", help="也纳入 comparison 型(默认只 bridge)")
    ap.add_argument("--levels", nargs="+", default=None, help="难度过滤, 如 medium hard(默认全部)")
    ap.add_argument("--out", default="data/raw/qa_tasks_m2.jsonl")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} 已存在(--force 覆盖)。")

    tasks = build(_iter_examples(args.local_json, args.split), args.n, args.seed,
                  args.include_comparison, args.levels)
    from collections import Counter
    print(f"built {len(tasks)} multi-hop QA tasks  "
          f"type={dict(Counter(t['type'] for t in tasks))}  "
          f"level={dict(Counter(t['level'] for t in tasks))}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
