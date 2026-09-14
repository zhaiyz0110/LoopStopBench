"""Collect full-horizon trajectories and resume by ``(task, seed)`` pair.

Example:
  python scripts/collect.py --loop code_repair --pair code_small \\
      --seeds 0 1 2 --limit 30 --out data/pilot_code.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from loopstop.llm import client_from_config, load_models_config
from loopstop.loops import HARNESSES
from loopstop.utils.io import append_steps, existing_traj_ids


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_embed_fn(model_name: str):
    """Build the shared embedding function used by convergence policies."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        lock = threading.Lock()

        def embed(text: str) -> list:
            with lock:
                return model.encode([text[:4000]], show_progress_bar=False)[0].tolist()

        return embed
    except ImportError:
        print("[warn] sentence-transformers not installed; emb_dist_prev will be None", flush=True)
        return None


def build_harness(
    loop: str,
    pair_cfg: dict,
    loop_cfg: dict,
    models_cfg: dict,
    embedding_model: str,
):
    embed_fn = make_embed_fn(embedding_model)
    gen = client_from_config(models_cfg, pair_cfg["generator"])
    if loop == "writer_critic":
        critic = client_from_config(models_cfg, pair_cfg["critic"])
        return HARNESSES[loop](gen, loop_cfg, critic_client=critic, embed_fn=embed_fn)
    if loop == "react_qa":
        # With no injected search function, react_qa uses its local BM25 index.
        return HARNESSES[loop](gen, loop_cfg, search_fn=None, embed_fn=embed_fn)
    return HARNESSES[loop](gen, loop_cfg, embed_fn=embed_fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", required=True, choices=list(HARNESSES))
    ap.add_argument("--pair", required=True, help="configs/models.yaml 的 pairs 键")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=None, help="任务数上限(pilot 用)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--loop-config", default=None, help="缺省 configs/<loop>.yaml")
    ap.add_argument(
        "--embedding-model",
        default=os.environ.get("LOOPSTOP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help=(
            "sentence-transformers model ID or local path; defaults to "
            "LOOPSTOP_EMBEDDING_MODEL, then the public all-MiniLM-L6-v2 ID"
        ),
    )
    ap.add_argument("--workers", type=int, default=4,
                    help="并发轨迹数; vLLM 靠并发请求吃满吞吐。react_qa 有 task 级共享状态, 强制串行")
    args = ap.parse_args()

    models_cfg = load_models_config(args.models_config)
    loop_cfg_path = args.loop_config or f"configs/{args.loop}.yaml"
    with open(loop_cfg_path, encoding="utf-8") as f:
        loop_cfg = yaml.safe_load(f)
    if args.limit:
        loop_cfg["limit"] = args.limit

    pair_cfg = models_cfg["pairs"][args.pair]
    harness = build_harness(
        args.loop,
        pair_cfg,
        loop_cfg,
        models_cfg,
        args.embedding_model,
    )
    tasks = harness.load_tasks()
    workers = 1 if args.loop == "react_qa" else max(1, args.workers)

    # The stable prefix identifies completed task, seed, and model-pair jobs.
    done_prefixes = {tid.rsplit(":", 1)[0] for tid in existing_traj_ids(args.out)}
    jobs = [
        (task, seed)
        for task in tasks
        for seed in args.seeds
        if f"{harness.loop_type}:{task.task_id}:{args.pair}:s{seed}" not in done_prefixes
    ]
    total = len(tasks) * len(args.seeds)
    print(f"loaded {len(tasks)} tasks → {len(jobs)} pending / {total} total "
          f"(resume skipped {total - len(jobs)}), workers={workers}", flush=True)

    write_lock = threading.Lock()  # append_steps 非线程安全, 串行化落盘
    counter = {"done": 0, "fail": 0}

    def run_job(task, seed):
        prefix = f"{harness.loop_type}:{task.task_id}:{args.pair}:s{seed}"
        steps = harness.run_task(task, seed=seed, model_cfg_name=args.pair)
        with write_lock:
            append_steps(args.out, steps)  # 整条轨迹一次性追加(原子性)
        return prefix, len(steps)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, task, seed): (task, seed) for task, seed in jobs}
        for fut in as_completed(futures):
            task, seed = futures[fut]
            try:
                prefix, T = fut.result()
                counter["done"] += 1
                print(f"[{counter['done']}/{len(jobs)}] {prefix}  T={T}", flush=True)
            except Exception:
                counter["fail"] += 1
                print(f"[FAIL] {task.task_id} s{seed}\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)

    print(f"done: {counter['done']} collected, {counter['fail']} failed, out={args.out}", flush=True)


if __name__ == "__main__":
    main()
