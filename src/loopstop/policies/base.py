"""Common stopping-policy interface.

Policies receive only ``VisibleHistory`` and are evaluated by forward scans
during offline replay.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Type

from ..schema import VisibleHistory

_REGISTRY: dict[str, Type["StoppingPolicy"]] = {}


def register(name: str) -> Callable[[Type["StoppingPolicy"]], Type["StoppingPolicy"]]:
    def deco(cls: Type["StoppingPolicy"]) -> Type["StoppingPolicy"]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


class StoppingPolicy(ABC):
    name: str = "abstract"

    @abstractmethod
    def decide(self, h: VisibleHistory) -> bool:
        """True = 在第 h.t 轮结束后停止。"""

    def reset(self) -> None:
        """新轨迹开始前清内部状态(有状态策略覆写)。"""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} {self.__dict__}>"


def build_policy(spec: dict) -> StoppingPolicy:
    """由 configs/replay.yaml 的一条策略配置构造实例。
    spec 形如 {"type": "fixed_n", "label": "S1-n5", "params": {"n": 5}}。
    """
    cls = _REGISTRY.get(spec["type"])
    if cls is None:
        raise KeyError(f"unknown policy type {spec['type']!r}; registered: {sorted(_REGISTRY)}")
    policy = cls(**spec.get("params", {}))
    if "label" in spec:
        policy.label = spec["label"]
    return policy


def policy_label(p: StoppingPolicy) -> str:
    return getattr(p, "label", p.name)
