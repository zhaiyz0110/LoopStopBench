import pytest

from conftest import make_traj
from loopstop.estimator import (
    ALL_FEATURES,
    build_examples,
    extract_features,
    split_by_task,
    split_leave_one_loop_out,
    to_matrix,
)


def test_features_cover_all_declared(rising_traj):
    feats = extract_features(rising_traj.visible(4))
    assert set(feats) == set(ALL_FEATURES)
    assert feats["t"] == 4.0
    assert feats["v"] == 0.7
    assert feats["dv"] == pytest.approx(0.2)


def test_labels_on_degrading(degrading_traj):
    # 峰值在第3轮: t>=3 之后不再有 >r_t+δ 的改进 → label 0
    ex = build_examples([degrading_traj], delta=0.05)
    by_t = {e.t: e.label for e in ex}
    assert by_t[1] == 1  # 之后还能到 0.9
    assert by_t[3] == 0
    assert by_t[7] == 0
    assert len(ex) == degrading_traj.T - 1  # 末轮不产样本


def test_split_by_task_no_leak():
    trajs = [
        make_traj([0.1, 0.3, 0.5, 0.6], traj_id=f"t{i}", task_id=f"task{i % 7}")
        for i in range(21)
    ]
    splits = split_by_task(build_examples(trajs, delta=0.05))
    seen = {}
    for name, exs in splits.items():
        for e in exs:
            assert seen.setdefault(e.task_id, name) == name  # 同 task 不跨集合


def test_leave_one_loop_out(rising_traj):
    qa = make_traj([0.2, 0.4, 0.5, 0.55], traj_id="q1", task_id="q1", loop_type="react_qa")
    exs = build_examples([rising_traj, qa], delta=0.05)
    splits = split_leave_one_loop_out(exs, held_out_loop="react_qa")
    assert all(e.loop_type != "react_qa" for e in splits["train"])
    assert all(e.loop_type == "react_qa" for e in splits["test"])


def test_matrix_shape_and_none_fill(rising_traj):
    exs = build_examples([rising_traj], delta=0.05)
    X, y, cols = to_matrix(exs)
    assert len(X) == len(y) == len(exs)
    assert all(len(row) == len(cols) for row in X)
    assert all(isinstance(v, float) for row in X for v in row)
