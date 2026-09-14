import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from loopstop.schema import StepRecord, Trajectory


def make_traj(
    r_series,
    v_series=None,
    emb_dists=None,
    traj_id="t1",
    task_id="task1",
    loop_type="code_repair",
    tokens=(3000, 1000),
    extra_signals=None,
):
    """合成轨迹构造器: r 真值序列必给, 可见分数缺省等于真值(理想验证器)。"""
    v_series = v_series or list(r_series)
    steps = []
    for i, r in enumerate(r_series):
        t = i + 1
        sig = {
            "score": v_series[i],
            "verifier_pass": v_series[i] is not None and v_series[i] >= 0.999,
            "edit_dist_prev": None if t == 1 else 0.3,
            "emb_dist_prev": None if t == 1 else (emb_dists[i] if emb_dists else 0.1),
            "output_len": 100 + i,
            "feedback_text": f"fb{t}",
            "failure_ids": [f"f{t}"],
            "mean_logprob": -0.5,
        }
        if extra_signals:
            sig.update(extra_signals[i] if isinstance(extra_signals, list) else extra_signals)
        steps.append(
            StepRecord(
                traj_id=traj_id,
                task_id=task_id,
                loop_type=loop_type,
                model_cfg="test",
                seed=0,
                t=t,
                output=f"output round {t}",
                diff_vs_prev="",
                visible_signals=sig,
                hidden_truth={"r": r},
                tokens_in=tokens[0],
                tokens_out=tokens[1],
                latency_ms=100.0,
            )
        )
    return Trajectory(steps)


@pytest.fixture
def rising_traj():
    return make_traj([0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.86, 0.86])


@pytest.fixture
def degrading_traj():
    # 峰值在第3轮(前2/3段), 之后净下降
    return make_traj([0.2, 0.6, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45], traj_id="t2", task_id="task2")
