"""Estimate the regret decomposition for a visible-signal policy class.

The reported decomposition is:
    U(oracle) − U(启发式) = [U(oracle) − U(oracle-visible)] + [U(oracle-visible) − U(启发式)]
                              residual                         recoverable

The heuristic is selected on outer-train tasks and evaluated on outer-test
tasks. Visible-policy candidates include single-feature thresholds, binned
backward induction, fixed horizons, and a cross-fitted GBT. The estimate is
conditional on this finite policy class. ``--mi`` also reports the mutual
information between the binned visible state and the continuation label.

Example:
  python scripts/oracle_visible.py --method both --mi --lambda 0.005 \\
      --traj data/m2_code_8b.jsonl data/m2_code_32b.jsonl \\
             data/m2_writing_small_judged.jsonl data/m2_writing_mid_judged.jsonl \\
      --out reproduced/oracle_visible.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from loopstop.policies import build_policy, policy_label
from loopstop.replay import (
    crossfit_best_from_utilities,
    crossfit_best_heuristic,
    evaluate_policy,
    heuristic_utility_table,
    oracle_stop,
    utility,
)
from loopstop.utils.io import load_trajectories


# Visible-state settings.

SCORE_FIELD = "score"  # Overridden by --score-field for score-source comparisons.
EMB_DIMS = 0  # Number of stored embedding components added to learned features.


def _score(traj, t):
    v = traj.steps[t - 1].visible_signals.get(SCORE_FIELD)
    return None if v is None else float(v)


def _set_score_field(trajs, field):
    """把缓存的替代信号映射到标准 score 槽，返回缺失步数。"""
    miss = 0
    for traj in trajs:
        for step in traj.steps:
            value = step.visible_signals.get(field)
            if value is None:
                miss += 1
            step.visible_signals["score"] = value
    return miss


def _include_critic_cost(trajs):
    """把 writer_critic 每轮缓存的 critic token 加入标准累计成本。"""
    missing = 0
    added_tokens = 0.0
    steps = 0
    for traj in trajs:
        if traj.loop_type != "writer_critic":
            continue
        for step in traj.steps:
            steps += 1
            value = step.visible_signals.get("critic_tokens")
            if value is None:
                missing += 1
                continue
            value = float(value)
            step.tokens_in += value
            added_tokens += value
    return steps, missing, added_tokens


def _feat(traj, t, name):
    if name == "score":
        return _score(traj, t)
    if name == "t":
        return float(t)
    if name == "dv":
        a, b = _score(traj, t), _score(traj, t - 1) if t > 1 else None
        return None if (a is None or b is None) else a - b
    return traj.steps[t - 1].visible_signals.get(name)


def _quantile_edges(vals, n_bins):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if len(vals) < n_bins:
        return np.array([0.5])
    qs = [i / n_bins for i in range(1, n_bins)]
    return np.quantile(vals, qs)


def _vbin(v, edges):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0
    return int(np.digitize(v, edges))


# Reference utilities.

def mean_oracle(trajs, lam, mode):
    return float(np.mean([oracle_stop(t, lam, mode)[1] for t in trajs]))


def best_heuristic(trajs, lam, mode, replay_cfg):
    result = crossfit_best_heuristic(trajs, replay_cfg["policies"], lam, mode)
    return result.mean_utility, result.label


# Single-feature visible policy.

def ovis_simple(trajs, lam, mode, min_round):
    feats = ["score", "dv", "edit_dist_prev", "emb_dist_prev", "t"]
    best = (-math.inf, None)
    for f in feats:
        vals = [_feat(tr, t, f) for tr in trajs for t in range(1, tr.T + 1)]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if len(vals) < 5:
            continue
        cand = np.quantile(vals, [i / 10 for i in range(1, 10)])
        for direction in (">=", "<="):
            for th in cand:
                us = []
                for tr in trajs:
                    t_stop = tr.T
                    for t in range(min_round, tr.T + 1):
                        v = _feat(tr, t, f)
                        if v is None:
                            continue
                        if (direction == ">=" and v >= th) or (direction == "<=" and v <= th):
                            t_stop = t
                            break
                    us.append(utility(tr, t_stop, lam, mode))
                mu = float(np.mean(us))
                if mu > best[0]:
                    best = (mu, (f, direction, round(float(th), 4)))
    return best


# Cross-fitted backward-induction policy.

def _fit_decision_table(train, lam, mode, edges):
    T = train[0].T
    data = []
    for tr in train:
        us = [utility(tr, t, lam, mode) for t in range(1, T + 1)]
        bins = [_vbin(_score(tr, t), edges) for t in range(1, T + 1)]
        data.append((us, bins))
    pv = [row[0][T - 1] for row in data]  # t=T: 必停, 值 = u_T
    decision = {}
    for t in range(T - 1, 0, -1):
        stop_sum, cont_sum, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
        for i, (us, bins) in enumerate(data):
            b = bins[t - 1]
            stop_sum[b] += us[t - 1]
            cont_sum[b] += pv[i]
            cnt[b] += 1
        for b in cnt:
            decision[(t, b)] = stop_sum[b] / cnt[b] >= cont_sum[b] / cnt[b]
        pv = [
            (us[t - 1] if decision.get((t, bins[t - 1]), False) else pv[i])
            for i, (us, bins) in enumerate(data)
        ]
    return decision


def _apply_table(tr, decision, edges):
    for t in range(1, tr.T):  # t=T 强制停
        if decision.get((t, _vbin(_score(tr, t), edges)), False):
            return t
    return tr.T


def ovis_backward(trajs, lam, mode, n_bins):
    if len({t.T for t in trajs}) != 1:
        raise ValueError("backward 假定同一循环内 T 一致")
    folds = defaultdict(list)
    for tr in trajs:
        folds[int(hashlib.md5(tr.task_id.encode()).hexdigest(), 16) % 2].append(tr)
    us = []
    for test_f in (0, 1):
        train, test = folds[1 - test_f], folds[test_f]
        if not train or not test:
            continue
        edges = _quantile_edges([_score(tr, t) for tr in train for t in range(1, tr.T + 1)], n_bins)
        table = _fit_decision_table(train, lam, mode, edges)
        us.extend(utility(tr, _apply_table(tr, table, edges), lam, mode) for tr in test)
    return float(np.mean(us))


# History-augmented features.

HIST_FEATURES = [
    "v", "run_max", "run_min", "v_minus_max", "n_decrease", "rounds_since_best",
    "v_lag1", "v_lag2", "v_lag3", "edit_dist_prev", "emb_dist_prev", "t", "cum_cost",
]


def _hist_features(h):
    """Return cumulative extrema, decrease counts, distance from best, and lags."""
    scores = [s for s in h.signal("score") if s is not None]
    v = scores[-1] if scores else 0.0
    run_max = max(scores) if scores else 0.0
    run_min = min(scores) if scores else 0.0
    n_dec = sum(1 for i in range(1, len(scores)) if scores[i] < scores[i - 1])
    best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
    rounds_since_best = (len(scores) - 1 - best_idx) if scores else 0

    def lag(k):
        return scores[-k] if len(scores) >= k else (scores[0] if scores else 0.0)

    sig = h.last.visible_signals
    return {
        "v": v, "run_max": run_max, "run_min": run_min, "v_minus_max": v - run_max,
        "n_decrease": float(n_dec), "rounds_since_best": float(rounds_since_best),
        "v_lag1": lag(1), "v_lag2": lag(2), "v_lag3": lag(3),
        "edit_dist_prev": sig.get("edit_dist_prev"), "emb_dist_prev": sig.get("emb_dist_prev"),
        "t": float(h.t), "cum_cost": h.cumulative_cost(),
    }


# Cross-fitted learned visible policy.

def ovis_learned(trajs, lam, mode, verbose=True, feature_cols=None, hist=False, return_breakdown=False):
    """Cross-fit the augmented visible-policy class.

    Candidates are a GBT probability threshold, a single-feature threshold,
    and a fixed horizon. A candidate is selected on each fold's validation
    tasks and applied to its test tasks. ``feature_cols`` restricts the GBT
    features, ``hist`` enables history features, and ``return_breakdown``
    returns candidate-specific utilities in addition to the selected utility.
    """
    from collections import Counter as _Counter

    from loopstop.estimator.features import ALL_FEATURES, extract_features
    from loopstop.estimator.train import _make_gbt

    base_cols = list(HIST_FEATURES) if hist else list(feature_cols if feature_cols is not None else ALL_FEATURES)
    base_featfn = _hist_features if hist else extract_features
    emb_cols = [f"emb_{i}" for i in range(EMB_DIMS)]
    cols = base_cols + emb_cols

    def featfn(h):
        d = base_featfn(h)
        if emb_cols:
            vs = h.last.visible_signals
            for c in emb_cols:
                d[c] = vs.get(c)
        return d

    def frow(feats):
        return [0.0 if feats.get(c) is None else float(feats.get(c)) for c in cols]

    def us_of(tr):
        return [utility(tr, t, lam, mode) for t in range(1, tr.T + 1)]

    def apply_tau(p, us, tau):
        for i, pi in enumerate(p):  # p[i] = P(继续更优) at 第 i+1 轮
            if pi < tau:
                return us[i]
        return us[-1]

    def simple_stop_util(tr, feat, direction, th):  # 单特征阈值规则应用到一条轨迹
        t_stop = tr.T
        for t in range(1, tr.T + 1):
            v = _feat(tr, t, feat)
            if v is None:
                continue
            if (direction == ">=" and v >= th) or (direction == "<=" and v <= th):
                t_stop = t
                break
        return utility(tr, t_stop, lam, mode)

    def best_fixed_k(val_set):
        T = max(tr.T for tr in val_set)
        us_val = [us_of(tr) for tr in val_set]
        k = max(range(T), key=lambda j: np.mean([u[min(j, len(u) - 1)] for u in us_val]))
        return k, float(np.mean([u[min(k, len(u) - 1)] for u in us_val]))

    outer = defaultdict(list)
    for tr in trajs:
        outer[int(hashlib.md5(tr.task_id.encode()).hexdigest(), 16) % 2].append(tr)

    sel_us, gbt_us, simple_us, fixed_us, winners, diag = [], [], [], [], [], []
    for test_f in (0, 1):
        train, test = outer[1 - test_f], outer[test_f]
        if not train or not test:
            continue
        inner = defaultdict(list)
        for tr in train:
            inner[int(hashlib.md5((tr.task_id + ":inner").encode()).hexdigest(), 16) % 2].append(tr)
        fit_set, val_set = (inner[0], inner[1]) if inner[0] and inner[1] else (train, train)

        # Select simple-threshold and fixed-horizon candidates on validation tasks.
        s_val, s_rule = ovis_simple(val_set, lam, mode, 1)
        kf, fixed_val = best_fixed_k(val_set)

        # Fit the GBT on fit tasks and select its threshold on validation tasks.
        X, y = [], []
        for tr in fit_set:
            us = us_of(tr)
            for t in range(1, tr.T):
                X.append(frow(featfn(tr.visible(t))))
                y.append(1 if max(us[t:]) > us[t - 1] + 1e-9 else 0)
        pos_rate = sum(y) / len(y) if y else 0.0
        gbt_ok = len(set(y)) >= 2 and s_rule is not None
        if gbt_ok:
            model = _make_gbt()
            model.fit(X, y)

            def pseq(tr):
                rows = [frow(featfn(tr.visible(t))) for t in range(1, tr.T)]
                return list(model.predict_proba(rows)[:, 1]) if rows else []

            val_cache = [(pseq(tr), us_of(tr)) for tr in val_set]
            val_p = [pi for p, _ in val_cache for pi in p]
            qs = np.quantile(val_p, [i / 20 for i in range(1, 20)]) if val_p else [0.5]
            taus = sorted({-1.0, *(float(q) for q in qs), (max(val_p) + 1e-6 if val_p else 1.0)})
            tau_star = max(taus, key=lambda tau: np.mean([apply_tau(p, us, tau) for p, us in val_cache]))
            gbt_val = float(np.mean([apply_tau(p, us, tau_star) for p, us in val_cache]))
        else:
            gbt_val = float("-inf")

        # Select among candidate classes on validation utility.
        cand_val = {"simple": s_val if s_rule is not None else float("-inf"),
                    "fixed": fixed_val, "gbt": gbt_val}
        winner = max(cand_val, key=cand_val.get)
        winners.append(winner)

        # Apply each candidate to the test trajectories.
        for tr in test:
            u_simple = simple_stop_util(tr, *s_rule) if s_rule is not None else float("nan")
            u_fixed = us_of(tr)[min(kf, tr.T - 1)]
            u_gbt = apply_tau(pseq(tr), us_of(tr), tau_star) if gbt_ok else float("nan")
            simple_us.append(u_simple)
            fixed_us.append(u_fixed)
            if gbt_ok:
                gbt_us.append(u_gbt)
            sel_us.append({"simple": u_simple, "fixed": u_fixed, "gbt": u_gbt}[winner])
        diag.append(f"fold{test_f}:pos={pos_rate:.1%},winner={winner},"
                    f"val(gbt={gbt_val:.3f},simple={s_val:.3f},fixed={fixed_val:.3f})")

    if verbose:
        print("  [augmented-class diag] " + " | ".join(diag))
    result = {
        "utility": float(np.mean(sel_us)) if sel_us else float("nan"),
        "winner": dict(_Counter(winners)),
        "gbt": float(np.mean(gbt_us)) if gbt_us else float("nan"),
        "simple": float(np.mean(simple_us)) if simple_us else float("nan"),
        "fixed": float(np.mean(fixed_us)) if fixed_us else float("nan"),
    }
    return result if return_breakdown else result["utility"]


# Mutual information.

def mutual_info_bits(trajs, lam, mode, n_bins):
    edges = _quantile_edges([_score(tr, t) for tr in trajs for t in range(1, tr.T + 1)], n_bins)
    joint = defaultdict(int)
    for tr in trajs:
        us = [utility(tr, t, lam, mode) for t in range(1, tr.T + 1)]
        for t in range(1, tr.T):
            y = 1 if max(us[t:]) > us[t - 1] + 1e-9 else 0
            joint[((t, _vbin(_score(tr, t), edges)), y)] += 1
    n = sum(joint.values())
    if n == 0:
        return float("nan")
    px, py = defaultdict(int), defaultdict(int)
    for (x, y), c in joint.items():
        px[x] += c
        py[y] += c
    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / (px[x] / n * py[y] / n))
    return mi


# Permutation test for finite-sample mutual-information bias.

def mi_permutation(trajs, lam, mode, n_bins, n_perm, seed=0):
    """Compare binned MI with a permuted-label null distribution.

    Returns observed MI, null mean, null 95th percentile, and permutation p-value.
    """
    edges = _quantile_edges([_score(tr, t) for tr in trajs for t in range(1, tr.T + 1)], n_bins)
    states, labels = [], []
    for tr in trajs:
        us = [utility(tr, t, lam, mode) for t in range(1, tr.T + 1)]
        for t in range(1, tr.T):
            states.append((t, _vbin(_score(tr, t), edges)))
            labels.append(1 if max(us[t:]) > us[t - 1] + 1e-9 else 0)

    def mi(lab):
        joint, px, py = defaultdict(int), defaultdict(int), defaultdict(int)
        n = len(lab)
        for x, y in zip(states, lab):
            joint[(x, y)] += 1
            px[x] += 1
            py[y] += 1
        return sum((c / n) * math.log2((c / n) / (px[x] / n * py[y] / n)) for (x, y), c in joint.items())

    obs = mi(labels)
    rng = np.random.default_rng(seed)
    lab = np.array(labels)
    null = np.array([mi(list(rng.permutation(lab))) for _ in range(n_perm)])
    return obs, float(null.mean()), float(np.quantile(null, 0.95)), float(np.mean(null >= obs))


# Bootstrap intervals and equivalence decisions.

def estimate_targets(trajs, lam, mode, bins, min_round, replay_cfg, need_ovis):
    """Fit policies and estimate targets within one bootstrap sample."""
    u_oracle = mean_oracle(trajs, lam, mode)
    u_heur, heur = best_heuristic(trajs, lam, mode, replay_cfg)
    out = {"u_oracle": u_oracle, "u_heur": u_heur, "total_regret": u_oracle - u_heur, "heur": heur}
    if need_ovis:
        # The visible class includes the OOF heuristic, so its empirical floor
        # cannot be lower than that heuristic under the same protocol.
        u_ovis_raw = max(
            ovis_backward(trajs, lam, mode, bins),
            ovis_learned(trajs, lam, mode, verbose=False),
        )
        recoverable_raw = u_ovis_raw - u_heur
        u_ovis = max(u_heur, u_ovis_raw)
        out["u_ovis"] = u_ovis
        out["recoverable"] = max(0.0, recoverable_raw)
        out["recoverable_raw"] = recoverable_raw
    return out


def _cluster_resample(trajs, rng):
    """Resample tasks so all seeds for a task enter or leave together."""
    by = defaultdict(list)
    for t in trajs:
        by[t.task_id].append(t)
    tasks = list(by)
    return [tr for i in rng.integers(0, len(tasks), size=len(tasks)) for tr in by[tasks[i]]]


def _bca_endpoint(theta_hat, boot, jack, nominal):
    """Compute one BCa endpoint from bootstrap and leave-one-task estimates."""
    from scipy.stats import norm

    boot = np.asarray([b for b in boot if not (isinstance(b, float) and math.isnan(b))], float)
    jack = np.asarray(jack, float)
    if len(boot) < 20:
        return float("nan")
    prop = (np.sum(boot < theta_hat) + 0.5) / (len(boot) + 1)
    z0 = norm.ppf(min(max(prop, 1e-6), 1 - 1e-6))
    d = jack.mean() - jack
    denom = 6.0 * (float(np.sum(d ** 2)) ** 1.5)
    a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0
    z = norm.ppf(nominal)
    zadj = z0 + (z0 + z) / (1 - a * (z0 + z))
    return float(np.quantile(boot, min(max(float(norm.cdf(zadj)), 0.0), 1.0)))


def run_bootstrap(trajs, loop, lam, mode, bins, min_round, replay_cfg, B, seed,
                  delta_code, delta_writing, delta_levels, force_recoverable=False):
    """Apply the three-way decision rule to two one-sided 95% BCa endpoints."""
    loop_type = trajs[0].loop_type
    is_code = loop_type == "code_repair"
    need_ovis = (not is_code) or force_recoverable
    rng = np.random.default_rng(seed)

    point = estimate_targets(trajs, lam, mode, bins, min_round, replay_cfg, need_ovis)
    boot_total, boot_recov = [], []
    for _ in range(B):
        e = estimate_targets(_cluster_resample(trajs, rng), lam, mode, bins, min_round, replay_cfg, need_ovis)
        boot_total.append(e["total_regret"])
        if need_ovis:
            boot_recov.append(e["recoverable_raw"])
    tasks = sorted({t.task_id for t in trajs})
    jack_total, jack_recov = [], []
    for tk in tasks:  # 按 task 留一(BCa 加速度)
        e = estimate_targets([t for t in trajs if t.task_id != tk], lam, mode, bins, min_round, replay_cfg, need_ovis)
        jack_total.append(e["total_regret"])
        if need_ovis:
            jack_recov.append(e["recoverable_raw"])

    print(f"\n== [bootstrap B={B}] {loop} (n={len(trajs)}, {len(tasks)} tasks) ==")
    if is_code:  # 检验对象 = 总 regret vs Δ_code
        up = _bca_endpoint(point["total_regret"], boot_total, jack_total, 0.95)
        lo = _bca_endpoint(point["total_regret"], boot_total, jack_total, 0.05)
        verdict = ("等价(上界<Δ)" if up < delta_code
                   else "证伪(下界>Δ)" if lo > delta_code
                   else "inconclusive(CI 跨 Δ)")
        print(f"  总 regret 点估={point['total_regret']:.4f}  bootstrap 均值={np.mean(boot_total):.4f}(应≈点估)")
        print(f"  总 regret 95%单侧 BCa 端点=[下界 {lo:.4f}, 上界 {up:.4f}]  "
              f"vs Δ_code={delta_code} → {verdict}")
        out = {"loop": loop, "target": "total_regret", "point": round(point["total_regret"], 4),
               "lower95": round(lo, 4), "upper95": round(up, 4), "delta": delta_code,
               "verdict": verdict, "equiv": up < delta_code,
               "n": len(trajs), "tasks": len(tasks)}
        if force_recoverable:  # 额外报 recoverable(need_ovis 已开): 让 code 也出策略次优段的三分判定
            up_r = max(0.0, _bca_endpoint(point["recoverable_raw"], boot_recov, jack_recov, 0.95))
            lo_r = max(0.0, _bca_endpoint(point["recoverable_raw"], boot_recov, jack_recov, 0.05))
            verdict = ("等价(上界<Δ)" if up_r < delta_code
                       else "证伪(下界>Δ)" if lo_r > delta_code
                       else "inconclusive(CI 跨 Δ)")
            print(f"  [--force-recoverable] recoverable 点估={point['recoverable']:.4f}  "
                  f"95% BCa [下界 {lo_r:.4f}, 上界 {up_r:.4f}]  vs Δ_code={delta_code} → {verdict}")
            out.update({"recoverable": round(point["recoverable"], 4),
                        "recoverable_raw_probe": round(point["recoverable_raw"], 4),
                        "recoverable_lower95": round(lo_r, 4), "recoverable_upper95": round(up_r, 4),
                        "recoverable_verdict": verdict})
        return out

    # Writing and QA compare recoverable gain with the configured margins.
    is_writing = loop_type == "writer_critic"
    delta_own = delta_writing if is_writing else delta_code
    up = max(0.0, _bca_endpoint(point["recoverable_raw"], boot_recov, jack_recov, 0.95))
    lo = max(0.0, _bca_endpoint(point["recoverable_raw"], boot_recov, jack_recov, 0.05))
    tr_lo = _bca_endpoint(point["total_regret"], boot_total, jack_total, 0.05)
    tr_up = _bca_endpoint(point["total_regret"], boot_total, jack_total, 0.95)
    print(f"  recoverable 点估={point['recoverable']:.4f}  "
          f"未截断 probe={point['recoverable_raw']:.4f}  bootstrap 均值={np.mean(boot_recov):.4f}")
    print(f"  recoverable 95%单侧 BCa 端点=[下界 {lo:.4f}, 上界 {up:.4f}]")
    print(f"  总 regret 点估={point['total_regret']:.4f}  单侧95%下界(BCa) = {tr_lo:.4f}  单侧95%上界(BCa) = {tr_up:.4f}")
    for d in delta_levels:
        verdict = ("等价(上界<Δ)" if up < d
                   else "证伪(下界>Δ)" if lo > d
                   else "inconclusive(CI 跨 Δ)")
        print(f"    Δ={d}: {verdict}")
    if tr_lo > delta_own:
        frac_up = up / point["total_regret"] if point["total_regret"] > 1e-9 else float("nan")
        print(f"  recoverable 比例 ≲ {frac_up:.1%}(总 regret CI 下界 {tr_lo:.4f} > Δ={delta_own}, 比例可报)")
    else:
        print(f"  recoverable 比例: n/a — 总 regret CI 下界 {tr_lo:.4f} ≤ Δ={delta_own}")
    return {"loop": loop, "target": "recoverable", "point": round(point["recoverable"], 4),
            "raw_probe_point": round(point["recoverable_raw"], 4),
            "lower95": round(lo, 4), "upper95": round(up, 4),
            "verdict": ("equivalent" if up < delta_own
                        else "falsified" if lo > delta_own else "inconclusive"),
            "total_regret_point": round(point["total_regret"], 4),
            "total_regret_lower95": round(tr_lo, 4), "total_regret_upper95": round(tr_up, 4),
            "equiv_005": up < delta_writing, "equiv_002": up < 0.02, "equiv_010": up < 0.10,
            "n": len(trajs), "tasks": len(tasks)}


def run_score_field_comparison(trajs, loop, field_a, field_b, lam, mode, bins,
                               min_round, replay_cfg, B, seed, margin):
    """Compare score fields with paired task resampling and policy refitting."""
    if B < 20:
        raise ValueError("--compare-score-fields 需要 --bootstrap >= 20")

    sources = {}
    for field in (field_a, field_b):
        values = {}
        missing = 0
        for traj in trajs:
            for step in traj.steps:
                value = step.visible_signals.get(field)
                if value is None:
                    missing += 1
                values[(traj.traj_id, step.t)] = value
        if missing:
            raise ValueError(f"score field {field!r} 缺失 {missing} 步")
        sources[field] = values

    def estimate(subset, field):
        values = sources[field]
        for traj in subset:
            for step in traj.steps:
                step.visible_signals["score"] = values[(traj.traj_id, step.t)]
        return estimate_targets(subset, lam, mode, bins, min_round, replay_cfg, need_ovis=True)

    point_a = estimate(trajs, field_a)
    point_b = estimate(trajs, field_b)
    point = point_b["recoverable_raw"] - point_a["recoverable_raw"]

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(B):
        sample = _cluster_resample(trajs, rng)
        est_a = estimate(sample, field_a)
        est_b = estimate(sample, field_b)
        boot.append(est_b["recoverable_raw"] - est_a["recoverable_raw"])

    tasks = sorted({traj.task_id for traj in trajs})
    jack = []
    for task_id in tasks:
        subset = [traj for traj in trajs if traj.task_id != task_id]
        est_a = estimate(subset, field_a)
        est_b = estimate(subset, field_b)
        jack.append(est_b["recoverable_raw"] - est_a["recoverable_raw"])

    lo = _bca_endpoint(point, boot, jack, 0.05)
    up = _bca_endpoint(point, boot, jack, 0.95)
    if lo > margin:
        verdict = "B 相对 A 有 material increase(下界>+Δ)"
    elif up < -margin:
        verdict = "B 相对 A 有 material decrease(上界<-Δ)"
    elif lo > -margin and up < margin:
        verdict = "exploratory equivalence(CI 完全落在 ±Δ)"
    else:
        verdict = "inconclusive(CI 未完全落在 ±Δ)"

    print(f"\n== [paired score-field bootstrap B={B}] {loop} "
          f"({field_b} - {field_a}, {len(tasks)} tasks) ==")
    print(f"  raw recoverable(A) = {point_a['recoverable_raw']:+.4f}")
    print(f"  raw recoverable(B) = {point_b['recoverable_raw']:+.4f}")
    print(f"  paired Δrecoverable 点估={point:+.4f}  bootstrap 均值={np.mean(boot):+.4f}")
    print(f"  paired 95%单侧 BCa 端点=[下界 {lo:+.4f}, 上界 {up:+.4f}]")
    print(f"  探索性判读: {verdict}; writing Δ={margin}")

    return {
        "loop": loop, "field_a": field_a, "field_b": field_b,
        "raw_recoverable_a": round(point_a["recoverable_raw"], 4),
        "raw_recoverable_b": round(point_b["recoverable_raw"], 4),
        "paired_delta": round(point, 4), "lower95": round(lo, 4),
        "upper95": round(up, 4), "margin": margin, "verdict": verdict,
        "bootstrap_mean": round(float(np.mean(boot)), 4),
        "n": len(trajs), "tasks": len(tasks),
    }


# Feature and history comparisons.

def run_ablation(trajs, loop, lam, mode, bins, min_round, replay_cfg):
    """Report single-group and cumulative feature-group recoverable utility."""
    from loopstop.estimator.features import FEATURE_GROUPS

    u_heur, _ = best_heuristic(trajs, lam, mode, replay_cfg)
    # Use the GBT component so the curve reflects only changes in GBT features.
    standalone = {g: ovis_learned(trajs, lam, mode, False, feature_cols=cols, return_breakdown=True)["gbt"] - u_heur
                  for g, cols in FEATURE_GROUPS.items()}
    order = sorted(FEATURE_GROUPS, key=lambda g: standalone[g], reverse=True)

    print(f"\n== [ablation] {loop} (n={len(trajs)}) 单组 recoverable ==")
    for g in order:
        print(f"  {g:16s} {standalone[g]:+.4f}")
    print(f"  按单组结果排序累加 recoverable(k):")
    rows, cols_acc = [], []
    for k, g in enumerate(order, 1):
        cols_acc += FEATURE_GROUPS[g]
        rec = ovis_learned(trajs, lam, mode, False, feature_cols=cols_acc, return_breakdown=True)["gbt"] - u_heur
        print(f"    k={k} (+{g}) → {rec:+.4f}")
        rows.append({"loop": loop, "kind": "cumulative", "k": k, "added_group": g,
                     "recoverable": round(rec, 4)})
    for g in order:
        rows.append({"loop": loop, "kind": "standalone", "k": 1, "added_group": g,
                     "recoverable": round(standalone[g], 4)})
    return rows


def run_history(trajs, loop, lam, mode, bins, min_round, replay_cfg):
    """Compare Markov and history-augmented feature sets."""
    u_heur, _ = best_heuristic(trajs, lam, mode, replay_cfg)
    rec_markov = ovis_learned(trajs, lam, mode, False, return_breakdown=True)["gbt"] - u_heur
    rec_hist = ovis_learned(trajs, lam, mode, False, hist=True, return_breakdown=True)["gbt"] - u_heur
    print(f"\n== [history] {loop} (n={len(trajs)}) ==")
    print(f"  recoverable(Markov 全特征) = {rec_markov:+.4f}")
    print(f"  recoverable(历史增强)      = {rec_hist:+.4f}   Δ={rec_hist - rec_markov:+.4f}")
    return [{"loop": loop, "recoverable_markov": round(rec_markov, 4),
             "recoverable_hist": round(rec_hist, 4), "delta": round(rec_hist - rec_markov, 4)}]


# Model-class comparison.

def _best_heur_per_traj(trajs, lam, mode, replay_cfg):
    """OOF 选择的启发式及其 held-out 逐轨迹效用。"""
    result = crossfit_best_heuristic(trajs, replay_cfg["policies"], lam, mode)
    return result.mean_utility, result.label, result.per_traj_utility


def run_model_class(trajs, loop, lam, mode, replay_cfg, seed, delta_code, delta_writing,
                    label_kind="utility", quality_delta=0.05, B=1000, ladder=None):
    """Compare GBT, GRU, LSTM, and a truth-fed positive control out of fold."""
    from loopstop.estimator import seqmodel
    from loopstop.estimator.features import ALL_FEATURES, extract_features

    is_writing = trajs[0].loop_type == "writer_critic"
    delta = delta_writing if is_writing else delta_code
    heur_table = heuristic_utility_table(trajs, replay_cfg["policies"], lam, mode)
    bh = crossfit_best_from_utilities(trajs, heur_table)
    if not heur_table or math.isnan(bh.mean_utility):
        print(f"[skip model-class] {loop}: 无可用启发式")
        return []
    u_heur, heur_lab = bh.mean_utility, bh.label
    cols = list(ALL_FEATURES)
    if ladder is None:
        ladder = [("gbt", None, False, False), ("gru", 8, False, False), ("gru", 16, False, False),
                  ("gru", 32, False, False), ("gru", 64, False, False), ("gru", 128, False, False),
                  ("lstm", 64, False, False), ("gru", 64, True, True)]  # Truth-fed positive control.
    tasks = sorted({t.task_id for t in trajs})
    by_task = defaultdict(list)
    for t in trajs:
        by_task[t.task_id].append(t)
    rng = np.random.default_rng(seed)

    print(f"\n== [model-class ladder] {loop} (n={len(trajs)}, {len(tasks)} tasks, "
          f"λ={lam}, Δ={delta}, label={label_kind}) ==")
    print(f"  best heur: {heur_lab}  U={u_heur:.4f}")
    rows = []
    for kind, hid, truth, bidir in ladder:
        util_by = seqmodel.oof_stop_utils(
            trajs, extract_features, cols, lam, mode, label_kind=label_kind,
            delta=quality_delta, with_truth=truth, bidir=bidir,
            learner=("gbt" if kind == "gbt" else "rnn"),
            hidden=hid or 32, rnn_type=("lstm" if kind == "lstm" else "gru"), seed=seed)
        common = [t for t in trajs if util_by.get(t.traj_id) is not None]
        if not common:
            print(f"  {kind}: 无有效 OOF 输出,跳过")
            continue

        def rec_of(subset, _u=util_by):
            m = float(np.mean([_u[t.traj_id] for t in subset]))
            h = crossfit_best_from_utilities(subset, heur_table).mean_utility
            return m - h

        point = rec_of(common)
        boot = []
        for _ in range(B):
            samp = [t for i in rng.integers(0, len(tasks), len(tasks)) for t in by_task[tasks[i]]
                    if util_by.get(t.traj_id) is not None]
            if samp:
                boot.append(rec_of(samp))
        jack = [rec_of([t for t in common if t.task_id != tk]) for tk in tasks
                if any(t.task_id != tk for t in common)]
        up = _bca_endpoint(point, boot, jack, 0.95)
        lo = _bca_endpoint(point, boot, jack, 0.05)
        vd = "accept(上界<Δ)" if up < delta else "falsify(下界>Δ)" if lo > delta else "inconclusive"
        name = f"{kind}{('-h' + str(hid)) if hid else ''}{'+truth' if truth else ''}{'(bidi)' if bidir else ''}"
        tag = "  ← positive control (双向, 喂真值)" if truth else ""
        print(f"  {name:14s} rec={point:+.4f}  95%[{lo:+.4f}, {up:+.4f}]  → {vd}{tag}")
        rows.append({"loop": loop, "member": name, "label": label_kind,
                     "recoverable": round(point, 4), "lo95": round(lo, 4), "up95": round(up, 4),
                     "delta": delta, "verdict": vd, "best_heur": heur_lab, "n": len(common)})
    return rows


# Main entry point.

def verdict(recoverable, total_gap, ovis_below_heur):
    if total_gap < 0.005:
        return "总 regret 近 0，估计的可分解空间很小"
    if ovis_below_heur:
        return ("o-vis 下界估计低于 best-heuristic → 估计受离散粒度所限、偏松; "
                "至少说明启发式已接近该下界, 需更细状态或 simple 上界佐证")
    r = recoverable
    tag = ("可见策略类与启发式之间存在较大估计差距" if r >= 0.5 else
           "可见策略类与启发式之间存在中等估计差距" if r >= 0.2 else
           "测试策略类中的 recoverable 较小")
    return f"recoverable={r:.0%} → {tag}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--replay-config", default="configs/replay.yaml")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.005)
    ap.add_argument("--mode", default="stop_at_last", choices=["stop_at_last", "stop_and_select"])
    ap.add_argument("--include-critic-cost", action="store_true",
                    help="writing 成本敏感性: generation token 外再计入每轮 critic_tokens")
    ap.add_argument("--method", default="backward",
                    choices=["simple", "backward", "learned", "all"],
                    help="learned: cross-fitted full-feature estimate; all: compute all methods")
    ap.add_argument("--bins", type=int, default=4, help="score 分位箱数(backward/mi)")
    ap.add_argument("--min-round", type=int, default=1, help="simple 规则最早停止轮")
    ap.add_argument("--mi", action="store_true", help="额外算互信息(bits)")
    ap.add_argument("--out", default="results/oracle_visible.csv")
    # Bootstrap intervals and equivalence decisions.
    ap.add_argument("--bootstrap", type=int, default=0, help="B>0 时出单侧 95% BCa 上界 + 等价性判读")
    ap.add_argument("--force-recoverable", action="store_true",
                    help="对 code 循环也算 o-vis + recoverable 的 BCa(三分判定);默认 code 只报总 regret")
    ap.add_argument("--exclude-tasks", default="", help="逗号分隔 task_id 剔除(如 Mbpp_793)")
    ap.add_argument("--delta-code", type=float, default=0.02, help="code 总 regret 的 Δ(1 个 plus 用例)")
    ap.add_argument("--delta-writing", type=float, default=0.05, help="writing recoverable 的 Δ(半个 rubric 点)")
    ap.add_argument("--delta-levels", type=float, nargs="+", default=[0.02, 0.05, 0.10],
                    help="writing 三档 Δ 报告")
    ap.add_argument("--seed", type=int, default=0)
    # Feature and history comparisons.
    ap.add_argument("--ablate", action="store_true", help="特征组消融(recoverable 是否随特征饱和)")
    ap.add_argument("--history", action="store_true", help="历史增强特征 vs Markov 全特征")
    # Model-class comparison.
    ap.add_argument("--model-class", action="store_true",
                    help="Compare GBT, GRU, LSTM, and a truth-fed control out of fold")
    ap.add_argument("--mc-label", default="utility", choices=["utility", "quality"],
                    help="Label for model comparison: utility or quality minus delta")
    ap.add_argument("--mc-bootstrap", type=int, default=1000)
    # Visible-score source comparison.
    ap.add_argument("--score-field", default="score",
                    help="Use the specified visible-signal field as score")
    ap.add_argument("--compare-score-fields", nargs=2, metavar=("FIELD_A", "FIELD_B"),
                    help="Paired bootstrap of raw recoverable(B) minus raw recoverable(A)")
    # Embedding features and the MI permutation test.
    ap.add_argument("--emb-dims", type=int, default=0,
                    help="Add the first K stored embedding components to learned features")
    ap.add_argument("--mi-permute", type=int, default=0, help="MI permutation count; 0 disables the test")
    args = ap.parse_args()
    global EMB_DIMS
    EMB_DIMS = args.emb_dims
    excl = {x for x in args.exclude_tasks.split(",") if x}
    boot_rows = []

    with open(args.replay_config, encoding="utf-8") as f:
        replay_cfg = yaml.safe_load(f)

    rows = []
    for path in args.traj:
        trajs = [t for t in load_trajectories(path) if t.has_truth() and t.task_id not in excl]
        loop = Path(path).stem
        if not trajs:
            print(f"[skip] {loop}: 无完整真值")
            continue

        cost_scope = "generation-only"
        if args.include_critic_cost:
            critic_steps, missing, added = _include_critic_cost(trajs)
            if critic_steps and missing:
                raise ValueError(f"{loop}: critic_tokens 缺失 {missing}/{critic_steps} 步")
            cost_scope = "author+critic"
            print(f"[{loop}] cost scope = {cost_scope}; added critic tokens={added:.0f} "
                  f"over {critic_steps} steps")

        if args.compare_score_fields:
            field_a, field_b = args.compare_score_fields
            paired_row = run_score_field_comparison(
                trajs, loop, field_a, field_b, args.lam, args.mode, args.bins,
                args.min_round, replay_cfg, args.bootstrap, args.seed, args.delta_writing)
            paired_row["cost_scope"] = cost_scope
            rows.append(paired_row)
            continue

        if args.score_field != "score":
            miss = _set_score_field(trajs, args.score_field)
            print(f"[{loop}] score ← {args.score_field}(缺失 {miss} 步)")

        if args.ablate:
            rows.extend(run_ablation(trajs, loop, args.lam, args.mode, args.bins, args.min_round, replay_cfg))
            continue
        if args.history:
            rows.extend(run_history(trajs, loop, args.lam, args.mode, args.bins, args.min_round, replay_cfg))
            continue
        if args.model_class:
            rows.extend(run_model_class(trajs, loop, args.lam, args.mode, replay_cfg, args.seed,
                                        args.delta_code, args.delta_writing,
                                        label_kind=args.mc_label, quality_delta=0.05,
                                        B=args.mc_bootstrap))
            continue

        if args.bootstrap > 0:
            boot_row = run_bootstrap(
                trajs, loop, args.lam, args.mode, args.bins, args.min_round, replay_cfg,
                args.bootstrap, args.seed, args.delta_code, args.delta_writing, args.delta_levels,
                force_recoverable=args.force_recoverable)
            boot_row["cost_scope"] = cost_scope
            boot_rows.append(boot_row)

        u_oracle = mean_oracle(trajs, args.lam, args.mode)
        u_heur, heur_label = best_heuristic(trajs, args.lam, args.mode, replay_cfg)

        ovis = {}
        if args.method in ("simple", "all"):
            mu, rule = ovis_simple(trajs, args.lam, args.mode, args.min_round)
            ovis["simple"] = mu
            ovis["simple_rule"] = rule
        if args.method in ("backward", "all"):
            ovis["backward"] = ovis_backward(trajs, args.lam, args.mode, args.bins)
        if args.method in ("learned", "all"):
            lb = ovis_learned(trajs, args.lam, args.mode, return_breakdown=True)
            ovis["learned"] = lb["utility"]
            ovis["learned_breakdown"] = lb

        # The decomposition uses cross-fitted candidates. In-sample simple is
        # reported only when explicitly requested as a diagnostic.
        formal_methods = ("simple",) if args.method == "simple" else ("backward", "learned")
        probe_cands = {k: v for k, v in ovis.items() if k in formal_methods}
        raw_policy_gap = max(probe_cands.values()) - u_heur
        cands = dict(probe_cands)
        cands["heuristic_floor"] = u_heur
        u_ovis = max(cands.values())
        ovis_used = max(cands, key=cands.get)
        info_gap = u_oracle - u_ovis
        policy_gap = u_ovis - u_heur
        total_gap = u_oracle - u_heur
        recoverable = policy_gap / total_gap if total_gap > 1e-9 else float("nan")
        mi = mutual_info_bits(trajs, args.lam, args.mode, args.bins) if args.mi else None

        print(f"\n== {loop} (n={len(trajs)}, λ={args.lam}, {args.mode}) ==")
        print(f"  U(oracle)        = {u_oracle:.4f}")
        if "simple" in ovis:
            print(f"  U(o-vis simple)  = {ovis['simple']:.4f}   rule={ovis['simple_rule']}")
        if "backward" in ovis:
            print(f"  U(o-vis backward)= {ovis['backward']:.4f}")
        if "learned_breakdown" in ovis:
            lb = ovis["learned_breakdown"]
            print(f"  增广可见策略类(交叉拟合)  GBT={lb['gbt']:.4f}  simple={lb['simple']:.4f}  "
                  f"fixed={lb['fixed']:.4f}  → selected={lb['utility']:.4f}  winner={lb['winner']}")
        print(f"  U(best heur)     = {u_heur:.4f}   ({heur_label})")
        print(f"  → o-vis 采用最紧下界: {ovis_used} = {u_ovis:.4f}")
        print(f"  信息不足 regret  = {info_gap:.4f}")
        print(f"  策略次优 regret  = {policy_gap:.4f}")
        if raw_policy_gap < 0:
            print(f"  未截断 probe 差值 = {raw_policy_gap:.4f}  (区间计算保留该值)")
        print(f"  总 regret        = {total_gap:.4f}")
        if mi is not None:
            print(f"  MI(可见;继续更优)= {mi:.4f} bits")
        if args.mi_permute > 0:
            obs, nmean, n95, pval = mi_permutation(trajs, args.lam, args.mode, args.bins, args.mi_permute)
            sig = "显著>偏差底" if pval < 0.05 else "落在 shuffle 噪声内(不显著)"
            print(f"  MI 置换检验: 观测={obs:.4f} 零分布均值={nmean:.4f} 95%={n95:.4f} p={pval:.3f} → {sig}")
        if EMB_DIMS > 0:
            print(f"  (learned 已含 {EMB_DIMS} 维文本 embedding 特征)")
        print(f"  判读: {verdict(recoverable, total_gap, u_ovis < u_heur)}")

        rows.append({
            "loop": loop, "n": len(trajs), "lambda": args.lam, "mode": args.mode,
            "cost_scope": cost_scope,
            "U_oracle": round(u_oracle, 4),
            "U_ovis_simple": round(ovis["simple"], 4) if "simple" in ovis else "",
            "U_ovis_backward": round(ovis["backward"], 4) if "backward" in ovis else "",
            "U_ovis_learned": round(ovis["learned"], 4) if "learned" in ovis else "",
            "learned_gbt": round(ovis["learned_breakdown"]["gbt"], 4) if "learned_breakdown" in ovis else "",
            "learned_simple": round(ovis["learned_breakdown"]["simple"], 4) if "learned_breakdown" in ovis else "",
            "learned_fixed": round(ovis["learned_breakdown"]["fixed"], 4) if "learned_breakdown" in ovis else "",
            "U_ovis_used": round(u_ovis, 4), "ovis_method": ovis_used,
            "U_best_heur": round(u_heur, 4), "best_heur": heur_label,
            "info_gap": round(info_gap, 4), "policy_gap": round(policy_gap, 4),
            "policy_gap_raw_probe": round(raw_policy_gap, 4),
            "total_gap": round(total_gap, 4),
            "recoverable_frac": round(recoverable, 4) if not math.isnan(recoverable) else "",
            "mi_bits": round(mi, 4) if mi is not None else "",
        })

    if rows:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\ncsv → {out}")

    if boot_rows:
        keys = []
        for r in boot_rows:
            keys += [k for k in r if k not in keys]
        bout = Path(args.out).with_name(Path(args.out).stem + "_bootstrap.csv")
        with open(bout, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in boot_rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"bootstrap csv → {bout}")


if __name__ == "__main__":
    main()
