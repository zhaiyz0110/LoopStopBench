"""ReAct question-answering loop over HotpotQA distractor contexts.

Each round proposes a search query, an answer, and self-confidence. The
default search uses an inline BM25 implementation over the ten paragraphs
provided with each question. An injected search function can replace it.
Visible signals include self-confidence and an optional NLI score; hidden
quality is answer F1, with exact match recorded separately.
"""
from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from ..llm import LLMClient
from .base import LoopHarness, LoopTask, RoundContext

SYSTEM = """You are a research assistant answering questions with a search tool.
Each round you receive previous search results. Respond with exactly this JSON:
{"search_query": "<next query, or null if no more search needed>",
 "answer": "<your current best short answer>",
 "confidence": <0.0-1.0>}"""

ROUND_TMPL = """Question: {question}

Search results so far:
{context}

Your previous answer: {prev_answer}

Refine your answer. Respond with the JSON only."""


class ReactQALoop(LoopHarness):
    loop_type = "react_qa"

    def __init__(
        self,
        gen_client: LLMClient,
        cfg: dict,
        search_fn: Optional[Callable[[str], list[str]]] = None,
        nli_fn: Optional[Callable[[str, str], float]] = None,
        **kw,
    ):
        super().__init__(gen_client, cfg, **kw)
        # Use in-context BM25 unless an external search function is provided.
        self.search_fn = search_fn
        self.nli_fn = nli_fn
        self._ctx_cache: dict[str, list[str]] = {}  # Retrieved passages by task.

    def load_tasks(self) -> list[LoopTask]:
        """Load task records containing questions, gold answers, and contexts."""
        path = Path(self.cfg["task_file"])
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; build the QA task file first")
        tasks = [
            LoopTask(task_id=d["task_id"], payload=d)
            for d in map(json.loads, filter(str.strip, open(path, encoding="utf-8")))
        ]
        limit = self.cfg.get("limit")
        return tasks[:limit] if limit else tasks

    def build_messages(self, ctx: RoundContext) -> list[dict]:
        tid = ctx.task.task_id
        if ctx.t == 1:
            self._ctx_cache[tid] = []
        else:
            # Retrieve the previous query and add unseen passages to the context.
            prev = _parse_json(ctx.prev_outputs[-1])
            q = prev.get("search_query")
            if q:
                k = self.cfg.get("top_k", 3)
                if self.search_fn is not None:
                    snippets = self.search_fn(q)[:k]
                else:
                    snippets = _bm25_topk(q, ctx.task.payload.get("context", []), k)
                for s in snippets:
                    if s not in self._ctx_cache[tid]:
                        self._ctx_cache[tid].append(s)
        context = "\n---\n".join(self._ctx_cache[tid][-12:]) or "(none yet)"
        prev_answer = _parse_json(ctx.prev_outputs[-1]).get("answer", "(none)") if ctx.prev_outputs else "(none)"
        return [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": ROUND_TMPL.format(
                    question=ctx.task.payload["question"], context=context, prev_answer=prev_answer
                ),
            },
        ]

    def parse_output(self, raw: str) -> str:
        return raw.strip()  # Preserve the JSON response for later parsing.

    def verify(self, task: LoopTask, output: str) -> dict:
        d = _parse_json(output)
        answer = str(d.get("answer", ""))
        conf = float(d.get("confidence", 0.0) or 0.0)
        evidence = "\n".join(self._ctx_cache.get(task.task_id, [])[-6:])
        nli = self.nli_fn(evidence, answer) if (self.nli_fn and evidence) else None
        return {
            "score": nli if nli is not None else conf,
            "self_confidence": conf,
            "nli_entail": nli,
            "feedback_text": "",
        }

    def evaluate_truth(self, task: LoopTask, output: str) -> dict:
        answer = str(_parse_json(output).get("answer", ""))
        golds = task.payload.get("gold_answers", [])
        if task.payload.get("answer_eval") == "llm_equiv":
            return {"r": None, "pending": True, "answer": answer}
        f1 = max((_f1(answer, g) for g in golds), default=0.0)
        return {"r": f1, "em": float(any(_norm(answer) == _norm(g) for g in golds))}


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _bm25_topk(query: str, paragraphs: list[str], k: int = 3, k1: float = 1.5, b: float = 0.75) -> list[str]:
    """Retrieve the top-k paragraphs with an inline Okapi BM25 scorer."""
    if not paragraphs:
        return []
    docs = [_norm(p).split() for p in paragraphs]
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / N or 1.0
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    q = _norm(query).split()
    scored = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        dl = len(d)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / avgdl))
        scored.append((s, i))
    scored.sort(reverse=True)
    return [paragraphs[i] for sc, i in scored[:k] if sc > 0]


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)
