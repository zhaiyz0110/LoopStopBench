from conftest import make_traj
from loopstop.policies import build_policy
from loopstop.replay import replay_policy


def test_fixed_n():
    traj = make_traj([0.1] * 8)
    p = build_policy({"type": "fixed_n", "params": {"n": 5}})
    assert replay_policy(p, traj) == 5


def test_fixed_n_exceeding_T_runs_full(rising_traj):
    p = build_policy({"type": "fixed_n", "params": {"n": 100}})
    assert replay_policy(p, rising_traj) == rising_traj.T


def test_verifier_pass_stops_on_green():
    traj = make_traj([0.2, 0.5, 1.0, 1.0, 1.0])
    p = build_policy({"type": "verifier_pass"})
    assert replay_policy(p, traj) == 3


def test_score_plateau():
    # 分数在第4轮起停涨(增量<eps), patience=2 → 第6轮停(第4→5、5→6两个平增量)
    traj = make_traj([0.1, 0.4, 0.6, 0.7, 0.702, 0.703, 0.703, 0.703])
    p = build_policy({"type": "score_plateau", "params": {"eps": 0.01, "patience": 2}})
    assert replay_policy(p, traj) == 6


def test_semantic_convergence_uses_emb_dist():
    dists = [None, 0.5, 0.4, 0.01, 0.005, 0.004, 0.3, 0.2]
    traj = make_traj([0.5] * 8, emb_dists=dists)
    p = build_policy({"type": "semantic_convergence", "params": {"eps": 0.02, "patience": 2}})
    assert replay_policy(p, traj) == 5  # 第4、5轮连续两个 <0.02


def test_dual_signal_needs_both():
    # 嵌入已收敛但分数还在涨 → S7 不停; 分数也平了才停
    dists = [None, 0.005, 0.004, 0.003, 0.002, 0.002, 0.002, 0.002]
    scores = [0.1, 0.3, 0.5, 0.7, 0.705, 0.706, 0.706, 0.706]
    traj = make_traj([0.5] * 8, v_series=scores, emb_dists=dists)
    p = build_policy(
        {"type": "dual_signal", "params": {"emb_eps": 0.02, "score_eps": 0.01, "patience": 2}}
    )
    assert replay_policy(p, traj) == 6


def test_reward_prediction_stops_when_slope_flat():
    scores = [0.1, 0.4, 0.6, 0.7, 0.701, 0.702, 0.702, 0.702]
    traj = make_traj([0.5] * 8, v_series=scores)
    p = build_policy({"type": "reward_prediction", "params": {"window": 3, "threshold": 0.005}})
    t = replay_policy(p, traj)
    assert 5 <= t <= 7  # 斜率变平后停


def test_voi_policy_with_injected_predictor():
    from loopstop.policies.voi import VOIPolicy

    traj = make_traj([0.1, 0.3, 0.5, 0.6, 0.6, 0.6])
    # 继续价值随轮数衰减的假预测器: p = 1/t
    p = VOIPolicy(predict_fn=lambda feats: 1.0 / feats["t"], tau=0.3)
    assert replay_policy(p, traj) == 4  # 1/4 = 0.25 < 0.3
