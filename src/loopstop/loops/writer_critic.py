"""Writer-critic loop with offline hidden-quality annotation.

The author generates and revises a draft from rubric feedback supplied by a
separate critic model. The critic score is visible during the loop. Hidden
quality is filled later by ``scripts/judge_writing.py``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..llm import LLMClient
from .base import LoopHarness, LoopTask, RoundContext

AUTHOR_SYSTEM = "You are a skilled writer. Revise carefully based on the critique. Output only the essay text."

AUTHOR_GEN = """Write an essay for the following assignment.

{assignment}

Output only the essay text."""

AUTHOR_REVISE = """Assignment:
{assignment}

Your current draft:
---
{prev}
---

Critique of the current draft:
{feedback}

Revise the draft to address the critique. Output only the full revised essay."""

CRITIC_SYSTEM = "You are a strict writing critic. Respond only with the JSON object requested."

CRITIC_TMPL = """Evaluate the essay against this rubric (each 1-10):
clarity, argument strength, structure, style.

Assignment:
{assignment}

Essay:
---
{essay}
---

Respond with exactly this JSON:
{{"scores": {{"clarity": x, "argument": x, "structure": x, "style": x}},
  "overall": <1-10>,
  "critique": "<3-6 concrete, actionable points>",
  "approve": <true if no further revision needed, else false>}}"""


class WriterCriticLoop(LoopHarness):
    loop_type = "writer_critic"

    def __init__(self, gen_client: LLMClient, cfg: dict, critic_client: LLMClient, **kw):
        super().__init__(gen_client, cfg, **kw)
        self.critic = critic_client
        self.T = int(cfg.get("rounds", 20))

    def load_tasks(self) -> list[LoopTask]:
        """Load JSONL records containing ``task_id`` and ``assignment``."""
        path = Path(self.cfg["task_file"])
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; build the writing task file first"
            )
        tasks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    tasks.append(LoopTask(task_id=d["task_id"], payload=d))
        limit = self.cfg.get("limit")
        return tasks[:limit] if limit else tasks

    def build_messages(self, ctx: RoundContext) -> list[dict]:
        a = ctx.task.payload["assignment"]
        user = (
            AUTHOR_GEN.format(assignment=a)
            if ctx.t == 1
            else AUTHOR_REVISE.format(assignment=a, prev=ctx.prev_outputs[-1], feedback=ctx.prev_feedback[-1])
        )
        return [{"role": "system", "content": AUTHOR_SYSTEM}, {"role": "user", "content": user}]

    def parse_output(self, raw: str) -> str:
        return raw.strip()

    def verify(self, task: LoopTask, output: str) -> dict:
        res = self.critic.chat(
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {
                    "role": "user",
                    "content": CRITIC_TMPL.format(assignment=task.payload["assignment"], essay=output),
                },
            ],
            temperature=0.0,
            max_tokens=800,
        )
        parsed = _parse_critic_json(res.text)
        return {
            "score": parsed["overall"] / 10.0,
            "critic_approve": parsed["approve"],
            "feedback_text": parsed["critique"],
            "rubric_scores": parsed["scores"],
            "critic_tokens": res.tokens_in + res.tokens_out,
        }

    def evaluate_truth(self, task: LoopTask, output: str) -> dict:
        return {"r": None, "pending": True}  # Filled by judge_writing.py.


def _parse_critic_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {
        "overall": float(d.get("overall", 5.0)),
        "approve": bool(d.get("approve", False)),
        "critique": str(d.get("critique", text[:500])),
        "scores": d.get("scores", {}),
    }
