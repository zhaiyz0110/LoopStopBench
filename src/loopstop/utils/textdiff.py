"""Output-change metrics; embedding distance is supplied by the collector."""
from __future__ import annotations

import difflib


def normalized_edit_distance(a: str, b: str) -> float:
    """1 - SequenceMatcher.ratio, 值域 [0,1]。空串对空串为 0。"""
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def unified_diff(a: str, b: str, n: int = 2) -> str:
    return "\n".join(
        difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=n)
    )


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
