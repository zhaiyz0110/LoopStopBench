"""Trajectory data types and the visible-history interface.

Each ``StepRecord`` represents one iteration in the JSONL data. Stopping
policies receive ``VisibleHistory`` objects, which omit ``hidden_truth``.

visible_signals 的键约定(各 harness 按可用性填写, 缺省为 None/缺键):
    score            float  主可见分数, S2/S6/S8 的依据(代码=可见测试通过率, 写作=critic 分, QA=NLI 蕴含分)
    verifier_pass    bool   可见验证器判定"通过"(代码=可见测试全绿)
    critic_approve   bool   critic 显式 APPROVE(writer_critic 专属, S3)
    self_confidence  float  智能体自报置信度 [0,1](react_qa 重点, S5)
    edit_dist_prev   float  与上一轮输出的归一化编辑距离, t=1 为 None
    emb_dist_prev    float  与上一轮输出的嵌入余弦距离, t=1 为 None
    output_len       int    输出长度(字符)
    feedback_text    str    本轮验证器/critic 的文本反馈
    failure_ids      list   本轮失败的测试名或批评点标识
    mean_logprob     float  修订模型对自身输出的平均 token logprob

hidden_truth 的键约定:
    r        float|None  真实质量 ∈ [0,1]; 写作循环采集期为 None, 由 judge_writing.py 离线回填
    pending  bool        真值待离线标注
    其余分解项(如 plus_pass_rate、逐裁判分数)自由扩展。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence

SCHEMA_VERSION = "1.1"

LOOP_TYPES = ("code_repair", "writer_critic", "react_qa")

# Cost unit: one thousand tokens; lambda is measured in quality per thousand tokens.
COST_PER_KILOTOKEN = 1.0


@dataclass
class StepRecord:
    traj_id: str
    task_id: str
    loop_type: str
    model_cfg: str
    seed: int
    t: int  # 1-based 轮次
    output: str
    diff_vs_prev: str
    visible_signals: dict
    hidden_truth: dict
    tokens_in: int
    tokens_out: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "StepRecord":
        d = json.loads(line)
        d.pop("schema_version", None)
        known = {f for f in cls.__dataclass_fields__ if f != "schema_version"}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class VisibleStep:
    """Visible fields for one step, excluding hidden truth."""

    task_id: str
    loop_type: str
    t: int
    output: str
    diff_vs_prev: str
    visible_signals: dict
    tokens_in: int
    tokens_out: int
    latency_ms: float

    @classmethod
    def from_record(cls, rec: StepRecord) -> "VisibleStep":
        return cls(
            task_id=rec.task_id,
            loop_type=rec.loop_type,
            t=rec.t,
            output=rec.output,
            diff_vs_prev=rec.diff_vs_prev,
            visible_signals=dict(rec.visible_signals),
            tokens_in=rec.tokens_in,
            tokens_out=rec.tokens_out,
            latency_ms=rec.latency_ms,
        )


class VisibleHistory(Sequence):
    """Visible information through round ``t``; the only policy input."""

    def __init__(self, steps: Iterable[VisibleStep]):
        self._steps = sorted(steps, key=lambda s: s.t)

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, i):
        return self._steps[i]

    @property
    def t(self) -> int:
        return self._steps[-1].t if self._steps else 0

    @property
    def last(self) -> VisibleStep:
        return self._steps[-1]

    def signal(self, key: str, default: Any = None) -> list:
        """按轮取某个可见信号的序列。"""
        return [s.visible_signals.get(key, default) for s in self._steps]

    def cumulative_cost(self) -> float:
        return sum(s.tokens_in + s.tokens_out for s in self._steps) / 1000.0 * COST_PER_KILOTOKEN


class Trajectory:
    """一条跑满 T 轮的轨迹。离线评测(replay/analysis/estimator)持有全量信息。"""

    def __init__(self, steps: Iterable[StepRecord]):
        self.steps: list[StepRecord] = sorted(steps, key=lambda s: s.t)
        if not self.steps:
            raise ValueError("empty trajectory")
        expected = list(range(1, len(self.steps) + 1))
        if [s.t for s in self.steps] != expected:
            raise ValueError(
                f"trajectory {self.steps[0].traj_id}: rounds not contiguous "
                f"{[s.t for s in self.steps]}"
            )

    # ---- 元信息 ----
    @property
    def traj_id(self) -> str:
        return self.steps[0].traj_id

    @property
    def task_id(self) -> str:
        return self.steps[0].task_id

    @property
    def loop_type(self) -> str:
        return self.steps[0].loop_type

    @property
    def T(self) -> int:
        return len(self.steps)

    # ---- 真值与成本 ----
    @property
    def r_series(self) -> list[Optional[float]]:
        return [s.hidden_truth.get("r") for s in self.steps]

    def r(self, t: int) -> float:
        v = self.steps[t - 1].hidden_truth.get("r")
        if v is None:
            raise ValueError(f"{self.traj_id} round {t}: hidden truth pending (run offline judging first)")
        return float(v)

    def has_truth(self) -> bool:
        return all(v is not None for v in self.r_series)

    def cost(self, t: int) -> float:
        """截止第 t 轮的累计成本(千 token)。"""
        return sum(s.tokens_in + s.tokens_out for s in self.steps[:t]) / 1000.0 * COST_PER_KILOTOKEN

    # ---- 可见视图 ----
    def visible(self, t: int) -> VisibleHistory:
        if not 1 <= t <= self.T:
            raise IndexError(f"t={t} out of range 1..{self.T}")
        return VisibleHistory(VisibleStep.from_record(s) for s in self.steps[:t])


def group_trajectories(records: Iterable[StepRecord]) -> list[Trajectory]:
    by_id: dict[str, list[StepRecord]] = {}
    for r in records:
        by_id.setdefault(r.traj_id, []).append(r)
    return [Trajectory(v) for v in by_id.values()]
