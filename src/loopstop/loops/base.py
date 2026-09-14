"""Base harness for full-horizon trajectory collection.

Collection always runs for ``T`` rounds; stopping policies are evaluated
offline in ``replay``. Subclasses implement task-specific loading, prompting,
verification, and hidden-quality evaluation. The base class records derived
signals and serializes each step.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..llm import ChatResult, LLMClient
from ..schema import StepRecord
from ..utils.textdiff import normalized_edit_distance, unified_diff


@dataclass
class LoopTask:
    task_id: str
    payload: dict = field(default_factory=dict)


@dataclass
class RoundContext:
    """Context passed to a loop implementation for the current round."""

    task: LoopTask
    t: int
    prev_outputs: list[str]
    prev_feedback: list[str]


class LoopHarness(ABC):
    loop_type: str = "abstract"

    def __init__(
        self,
        gen_client: LLMClient,
        cfg: dict,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        """Initialize from a loop configuration and optional text embedder."""
        self.gen = gen_client
        self.cfg = cfg
        self.embed_fn = embed_fn
        self.T = int(cfg.get("rounds", 15))
        self.temperature = float(cfg.get("temperature", 0.7))
        self.max_tokens = int(cfg.get("max_tokens", 2048))

    # Subclass interface.
    @abstractmethod
    def load_tasks(self) -> list[LoopTask]:
        """Load tasks from the configured data source."""

    @abstractmethod
    def build_messages(self, ctx: RoundContext) -> list[dict]:
        """构造第 t 轮的对话消息(t=1 为生成, t>1 为带反馈的修订)。"""

    @abstractmethod
    def parse_output(self, raw: str) -> str:
        """从模型回复中抽取本轮产出(代码块/正文/答案)。"""

    @abstractmethod
    def verify(self, task: LoopTask, output: str) -> dict:
        """Return task-specific visible signals, including ``score``."""

    @abstractmethod
    def evaluate_truth(self, task: LoopTask, output: str) -> dict:
        """Return hidden quality, or mark it pending for offline annotation."""

    # Collection orchestration.
    def run_task(self, task: LoopTask, seed: int, model_cfg_name: str) -> list[StepRecord]:
        traj_id = f"{self.loop_type}:{task.task_id}:{model_cfg_name}:s{seed}:{uuid.uuid4().hex[:8]}"
        records: list[StepRecord] = []
        prev_outputs: list[str] = []
        prev_feedback: list[str] = []
        prev_emb: Optional[list[float]] = None

        for t in range(1, self.T + 1):
            ctx = RoundContext(task=task, t=t, prev_outputs=prev_outputs, prev_feedback=prev_feedback)
            res: ChatResult = self.gen.chat(
                self.build_messages(ctx),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=seed * 1000 + t,  # 轮间去相关, 种子间可复现
            )
            output = self.parse_output(res.text)

            visible = self.verify(task, output)
            hidden = self.evaluate_truth(task, output)

            # Derived signals shared across loop implementations.
            visible.setdefault("output_len", len(output))
            visible.setdefault("mean_logprob", res.mean_logprob)
            if prev_outputs:
                visible.setdefault(
                    "edit_dist_prev", normalized_edit_distance(prev_outputs[-1], output)
                )
            else:
                visible.setdefault("edit_dist_prev", None)
            emb = self.embed_fn(output) if self.embed_fn else None
            visible.setdefault(
                "emb_dist_prev",
                _cosine_dist(prev_emb, emb) if (prev_emb is not None and emb is not None) else None,
            )

            records.append(
                StepRecord(
                    traj_id=traj_id,
                    task_id=task.task_id,
                    loop_type=self.loop_type,
                    model_cfg=model_cfg_name,
                    seed=seed,
                    t=t,
                    output=output,
                    diff_vs_prev=unified_diff(prev_outputs[-1], output) if prev_outputs else "",
                    visible_signals=visible,
                    hidden_truth=hidden,
                    tokens_in=res.tokens_in,
                    tokens_out=res.tokens_out,
                    latency_ms=res.latency_ms,
                    timestamp=time.time(),
                )
            )
            prev_outputs.append(output)
            prev_feedback.append(str(visible.get("feedback_text", "")))
            prev_emb = emb

        return records


def _cosine_dist(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return 1.0 - dot / (na * nb)
