"""JSONL trajectory I/O with resumable collection."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator

from ..schema import StepRecord, Trajectory, group_trajectories


def append_steps(path: str | Path, steps: Iterable[StepRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for s in steps:
            f.write(s.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_steps(path: str | Path) -> Iterator[StepRecord]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield StepRecord.from_json(line)


def load_trajectories(path: str | Path) -> list[Trajectory]:
    return group_trajectories(read_steps(path))


def existing_traj_ids(path: str | Path) -> set[str]:
    """断点续采: 已完整落盘的 traj_id 集合(不校验轮数完整性, collect 端保证原子追加)。"""
    p = Path(path)
    if not p.exists():
        return set()
    return {s.traj_id for s in read_steps(p)}
