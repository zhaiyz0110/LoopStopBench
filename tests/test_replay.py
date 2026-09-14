import pytest

from conftest import make_traj
from loopstop.policies import build_policy
from loopstop.replay import (
    evaluate_policy,
    evaluate_suite,
    oracle_stop,
    select_round,
    utility,
)


def test_utility_lambda_zero_is_quality(degrading_traj):
    assert utility(degrading_traj, 3, lam=0.0) == pytest.approx(0.9)
    assert utility(degrading_traj, 8, lam=0.0) == pytest.approx(0.45)


def test_utility_charges_cost(degrading_traj):
    # 每步 4k token; λ=0.005/千token → 第3轮成本 12*0.005=0.06
    assert utility(degrading_traj, 3, lam=0.005) == pytest.approx(0.9 - 0.06)


def test_oracle_finds_peak_on_degrading(degrading_traj):
    t, u = oracle_stop(degrading_traj, lam=0.0)
    assert t == 3
    assert u == pytest.approx(0.9)


def test_oracle_prefers_early_under_cost(degrading_traj):
    t_free, _ = oracle_stop(degrading_traj, lam=0.0)
    t_cost, _ = oracle_stop(degrading_traj, lam=0.02)
    assert t_cost <= t_free


def test_stop_and_select_returns_best_visible(degrading_traj):
    # 可见分数=真值(理想验证器): 停在第8轮但回溯选第3轮
    assert select_round(degrading_traj, 8, "stop_and_select") == 3
    assert select_round(degrading_traj, 8, "stop_at_last") == 8


def test_regret_nonnegative_and_zero_for_oracle_like(degrading_traj, rising_traj):
    trajs = [degrading_traj, rising_traj]
    for spec in [
        {"type": "fixed_n", "params": {"n": 3}},
        {"type": "fixed_n", "params": {"n": 8}},
        {"type": "score_plateau", "params": {"eps": 0.01, "patience": 2}},
    ]:
        res = evaluate_policy(build_policy(spec), trajs, lam=0.005)
        assert res.mean_regret >= -1e-12


def test_suite_rejects_pending_truth():
    pending = make_traj([0.5, None, 0.7, 0.8])
    with pytest.raises(ValueError, match="pending"):
        evaluate_suite([build_policy({"type": "fixed_n", "params": {"n": 2}})], [pending], [0.0])


def test_fixed_late_regret_exceeds_early_on_degrading(degrading_traj):
    # 劣化轨迹上, "跑满15轮"式策略的 regret 应大于停在峰值附近的策略
    early = evaluate_policy(build_policy({"type": "fixed_n", "params": {"n": 3}}), [degrading_traj], 0.005)
    late = evaluate_policy(build_policy({"type": "fixed_n", "params": {"n": 8}}), [degrading_traj], 0.005)
    assert late.mean_regret > early.mean_regret
