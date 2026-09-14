import pytest

from conftest import make_traj
from loopstop.schema import StepRecord, Trajectory


def test_jsonl_roundtrip(rising_traj):
    step = rising_traj.steps[0]
    restored = StepRecord.from_json(step.to_json())
    assert restored == step


def test_visible_history_hides_truth(rising_traj):
    h = rising_traj.visible(3)
    assert len(h) == 3
    assert h.t == 3
    for s in h:
        assert not hasattr(s, "hidden_truth")
    assert h.signal("score") == [0.1, 0.3, 0.5]


def test_cost_is_cumulative_kilotokens(rising_traj):
    # 每步 3000+1000 = 4k token
    assert rising_traj.cost(1) == pytest.approx(4.0)
    assert rising_traj.cost(5) == pytest.approx(20.0)


def test_noncontiguous_rounds_rejected(rising_traj):
    steps = [s for s in rising_traj.steps if s.t != 3]
    with pytest.raises(ValueError, match="contiguous"):
        Trajectory(steps)


def test_pending_truth_detected():
    traj = make_traj([0.5, None, 0.7, 0.8])
    assert not traj.has_truth()
    with pytest.raises(ValueError, match="pending"):
        traj.r(2)
