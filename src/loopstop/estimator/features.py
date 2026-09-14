"""Output-side features used by the VOI estimator.

All features are computed from ``VisibleHistory`` and shared by training and
inference. Missing values remain ``None`` until the training path fills them.
``FEATURE_GROUPS`` defines the groups used in feature ablations.
"""
from __future__ import annotations

import statistics
from typing import Optional

from ..schema import VisibleHistory
from ..utils.textdiff import jaccard

FEATURE_GROUPS: dict[str, list[str]] = {
    "signal_dynamics": ["v", "dv", "ddv", "v_rollvar"],
    "output_change": ["edit_dist_prev", "emb_dist_prev", "len_change_rate"],
    "error_recurrence": ["failure_jaccard_prev", "failure_jaccard_max_hist"],
    "feedback_text": ["feedback_len", "feedback_len_change"],
    "uncertainty": ["mean_logprob", "d_mean_logprob"],
    "meta": ["t", "cum_cost", "task_len"],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


def extract_features(h: VisibleHistory) -> dict[str, Optional[float]]:
    last = h.last
    sig = last.visible_signals

    # Visible-signal dynamics.
    scores = [v for v in h.signal("score") if v is not None]
    v = scores[-1] if scores else None
    dv = scores[-1] - scores[-2] if len(scores) >= 2 else None
    ddv = (scores[-1] - 2 * scores[-2] + scores[-3]) if len(scores) >= 3 else None
    tail = scores[-4:]
    v_rollvar = statistics.pvariance(tail) if len(tail) >= 2 else None

    # Output changes.
    lens = h.signal("output_len")
    len_change_rate = None
    if len(lens) >= 2 and lens[-2] and lens[-1] is not None:
        len_change_rate = (lens[-1] - lens[-2]) / max(lens[-2], 1)

    # Failure recurrence.
    fid_seq = [set(x) if x else set() for x in h.signal("failure_ids", default=[])]
    failure_jaccard_prev = jaccard(fid_seq[-1], fid_seq[-2]) if len(fid_seq) >= 2 else None
    failure_jaccard_max_hist = (
        max(jaccard(fid_seq[-1], f) for f in fid_seq[:-1]) if len(fid_seq) >= 2 else None
    )

    # Feedback-text length features.
    fb = h.signal("feedback_text", default="")
    feedback_len = float(len(fb[-1] or ""))
    feedback_len_change = (
        float(len(fb[-1] or "") - len(fb[-2] or "")) if len(fb) >= 2 else None
    )

    # Token-level uncertainty.
    lps = [x for x in h.signal("mean_logprob") if x is not None]
    mean_logprob = lps[-1] if lps else None
    d_mean_logprob = lps[-1] - lps[-2] if len(lps) >= 2 else None

    # Context metadata.
    task_len = float(h[0].tokens_in)  # 首轮 prompt 长度作为任务长度代理

    return {
        "v": v,
        "dv": dv,
        "ddv": ddv,
        "v_rollvar": v_rollvar,
        "edit_dist_prev": sig.get("edit_dist_prev"),
        "emb_dist_prev": sig.get("emb_dist_prev"),
        "len_change_rate": len_change_rate,
        "failure_jaccard_prev": failure_jaccard_prev,
        "failure_jaccard_max_hist": failure_jaccard_max_hist,
        "feedback_len": feedback_len,
        "feedback_len_change": feedback_len_change,
        "mean_logprob": mean_logprob,
        "d_mean_logprob": d_mean_logprob,
        "t": float(h.t),
        "cum_cost": h.cumulative_cost(),
        "task_len": task_len,
    }
