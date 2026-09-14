"""Add semantic features to a trajectory file.

The script encodes each output, optionally with critic feedback, applies PCA,
and stores ``emb_0`` through ``emb_{K-1}`` in ``visible_signals``. PCA does not
use outcome labels. Policy fitting remains cross-fitted in
``oracle_visible.py``.

Example:
  python scripts/add_embed_features.py --traj data/m2_writing_small_judged.jsonl \\
      --dims 16 --include-feedback --out data/ws_emb.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from loopstop.utils.io import append_steps, load_trajectories


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", type=int, default=16, help="PCA 主成分数 K")
    ap.add_argument("--include-feedback", action="store_true",
                    help="拼接 critic feedback_text 一起编码(默认只编码草稿输出)")
    ap.add_argument(
        "--embedding-model",
        "--model",
        dest="embedding_model",
        default=os.environ.get("LOOPSTOP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="sentence-transformers model ID or local path",
    )
    args = ap.parse_args()

    if Path(args.out).exists():
        sys.exit(f"{args.out} 已存在,请先删除或换名(避免重复追加)。")

    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    enc = SentenceTransformer(args.embedding_model)
    trajs = load_trajectories(args.traj)
    steps = [s for t in trajs for s in t.steps]  # 已按 traj 分组、轮次有序

    texts = []
    for s in steps:
        txt = s.output or ""
        if args.include_feedback:
            txt = txt + "\n[CRITIC]\n" + str(s.visible_signals.get("feedback_text", ""))
        texts.append(txt[:4000])

    print(
        f"encoding {len(texts)} step outputs with {args.embedding_model} ...",
        flush=True,
    )
    X = np.asarray(enc.encode(texts, batch_size=64, show_progress_bar=True))
    k = min(args.dims, X.shape[1], max(1, X.shape[0] - 1))
    Z = PCA(n_components=k, random_state=0).fit_transform(X)  # 无监督, 无标签泄漏

    for s, z in zip(steps, Z):
        for i, v in enumerate(z):
            s.visible_signals[f"emb_{i}"] = float(v)

    append_steps(args.out, steps)
    print(f"done: {len(steps)} steps, {k} emb dims (include_feedback={args.include_feedback}) → {args.out}", flush=True)


if __name__ == "__main__":
    main()
