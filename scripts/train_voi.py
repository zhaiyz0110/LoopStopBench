"""Train and evaluate the value-of-information estimator.

The default path reports AUROC. ``--utility-eval`` performs leave-one-family-out
evaluation against the cross-fitted heuristic baseline and reports clustered
BCa endpoints. The oracle threshold is selected on the held-out family and is
therefore diagnostic. The transfer threshold is selected on held-in validation
tasks and then applied unchanged to the held-out family.

Example:
  python scripts/train_voi.py --traj data/m2_code_8b.jsonl data/m2_code_32b.jsonl \\
      data/m2_writing_small_judged.jsonl data/m2_writing_mid_judged.jsonl \\
      data/qa_m2.jsonl \\
      --utility-eval --held-out-loop all --lambda 0.005 --bootstrap 1000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.estimator import build_examples, split_by_task, split_leave_one_loop_out
from loopstop.estimator.train import ablate_groups, auroc, calibration_bins, train_gbt
from loopstop.policies import build_policy, policy_label
from loopstop.policies.voi import _load_predictor
from loopstop.replay import crossfit_best_from_utilities
from loopstop.replay.evaluator import replay_policy, utility
from loopstop.utils.io import load_trajectories

# Reuse the task-clustered BCa implementation used by oracle_visible.
from oracle_visible import _bca_endpoint, _cluster_resample

LOOP_TYPES = ["code_repair", "writer_critic", "react_qa"]


# Leave-one-family-out utility evaluation.

def _split_trajs_by_task(trajs, frac_train=0.8):
    """轨迹级 task 哈希划分(与 dataset.split_by_task 同哈希, 同 task 不跨集合)。"""
    tr_, va_ = [], []
    for t in trajs:
        h = int(hashlib.md5(t.task_id.encode()).hexdigest(), 16) % 10_000 / 10_000
        (tr_ if h < frac_train else va_).append(t)
    return tr_, va_


def _precompute_voi(trajs, pred, lam, mode):
    """每轨迹缓存: 各轮效用 util_at_t[tid][t-1], 及 GBT 预测概率 voi_p[tid][t-1]。
    使 τ 扫描与 bootstrap 变成纯查表, 不重放模型。"""
    from loopstop.estimator.features import extract_features

    util_at_t, voi_p = {}, {}
    for tr in trajs:
        util_at_t[tr.traj_id] = [utility(tr, t, lam, mode) for t in range(1, tr.T + 1)]
        voi_p[tr.traj_id] = [float(pred(extract_features(tr.visible(t)))) for t in range(1, tr.T + 1)]
    return util_at_t, voi_p


def _precompute_heur(trajs, policies, lam, mode):
    """每启发式策略每轨迹的效用(点估 + bootstrap 共用)。"""
    hu = {}
    for pol in policies:
        lab = policy_label(pol)
        hu[lab] = {tr.traj_id: utility(tr, replay_policy(pol, tr), lam, mode) for tr in trajs}
    return hu


def _voi_util(trajs, util_at_t, voi_p, tau, min_rounds):
    us = []
    for tr in trajs:
        pv = voi_p[tr.traj_id]
        t_stop = len(pv)
        for t in range(1, len(pv) + 1):
            if t >= min_rounds and pv[t - 1] < tau:
                t_stop = t
                break
        us.append(util_at_t[tr.traj_id][t_stop - 1])
    return float(np.mean(us)) if us else float("nan")


def _heur_util(trajs, heur_util):
    """在 outer-train task 选启发式，并在 outer-test task 报告效用。"""
    result = crossfit_best_from_utilities(trajs, heur_util)
    return result.mean_utility, result.label


def run_utility_eval(all_trajs, replay_cfg, args):
    tau_grid = [round(x, 4) for x in np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)]
    delta_of = {"code_repair": args.delta_code, "writer_critic": args.delta_writing,
                "react_qa": args.delta_qa}
    loops = LOOP_TYPES if args.held_out_loop in (None, "all") else [args.held_out_loop]

    # 启发式基线策略集(跳过 voi 型: 不用 replay.yaml 里未训练的 voi)
    policies = []
    for spec in replay_cfg["policies"]:
        if spec.get("type") == "voi":
            continue
        try:
            policies.append(build_policy(spec))
        except Exception as e:
            print(f"[skip heur] {spec.get('label', spec.get('type'))}: {e}")

    rows = []
    for L in loops:
        held_out = [t for t in all_trajs if t.loop_type == L]
        held_in = [t for t in all_trajs if t.loop_type != L]
        if not held_out or not held_in:
            print(f"\n[skip] held-out={L}: held_out={len(held_out)} held_in={len(held_in)}(缺一侧)")
            continue
        delta = delta_of[L]

        # Split held-in tasks between GBT fitting and threshold validation.
        gbt_trajs, tauval_trajs = _split_trajs_by_task(held_in, 0.8)
        model_out = f"models_ckpt/xloop_voi_{L}.joblib"
        m = train_gbt(build_examples(gbt_trajs, delta=args.delta),
                      build_examples(tauval_trajs, delta=args.delta), model_out)
        pred = _load_predictor(model_out)

        # Cache held-out and threshold-validation utilities and predictions.
        util_ho, voip_ho = _precompute_voi(held_out, pred, args.lam, args.mode)
        heur_ho = _precompute_heur(held_out, policies, args.lam, args.mode)
        util_tv, voip_tv = _precompute_voi(tauval_trajs, pred, args.lam, args.mode)

        # Select the transfer threshold on held-in validation tasks.
        tau_transfer = max(tau_grid, key=lambda tau: _voi_util(tauval_trajs, util_tv, voip_tv, tau, args.min_rounds))

        # Point estimates on the held-out family.
        u_heur, heur_lab = _heur_util(held_out, heur_ho)
        u_voi_tr = _voi_util(held_out, util_ho, voip_ho, tau_transfer, args.min_rounds)
        u_voi_or = max(_voi_util(held_out, util_ho, voip_ho, tau, args.min_rounds) for tau in tau_grid)
        rec_tr, rec_or = u_voi_tr - u_heur, u_voi_or - u_heur

        # Predictive accuracy on the held-out family.
        ho_ex = build_examples(held_out, delta=args.delta)
        from loopstop.estimator.dataset import to_matrix
        import joblib
        model = joblib.load(model_out)["model"]
        Xte, yte, _ = to_matrix(ho_ex)
        auroc_te = auroc(yte, [p[1] for p in model.predict_proba(Xte)])

        # Resample held-out tasks. Re-select the oracle threshold; keep the
        # transfer threshold and fitted GBT fixed.
        def stat(sub):
            uh, _ = _heur_util(sub, heur_ho)
            uo = max(_voi_util(sub, util_ho, voip_ho, tau, args.min_rounds) for tau in tau_grid)
            ut = _voi_util(sub, util_ho, voip_ho, tau_transfer, args.min_rounds)
            return uo - uh, ut - uh

        res = {"loop": L, "n": len(held_out), "tasks": len({t.task_id for t in held_out}),
               "lam": args.lam, "delta": delta, "u_heur": round(u_heur, 4), "best_heur": heur_lab,
               "tau_transfer": tau_transfer, "auroc_test": round(auroc_te, 4),
               "u_voi_oracle": round(u_voi_or, 4), "rec_oracle": round(rec_or, 4),
               "u_voi_transfer": round(u_voi_tr, 4), "rec_transfer": round(rec_tr, 4)}

        print(f"\n== [跨循环效用] held-out={L} (n={res['n']}, {res['tasks']} tasks, λ={args.lam}, Δ={delta}) ==")
        print(f"  U(best heur)         = {u_heur:.4f}   ({heur_lab})")
        print(f"  U(VOI oracle-τ 扫优)  = {u_voi_or:.4f}   rec_oracle   = {rec_or:+.4f}")
        print(f"  U(VOI transfer-τ={tau_transfer:.2f}) = {u_voi_tr:.4f}   rec_transfer = {rec_tr:+.4f}")
        print(f"  AUROC(held-out)     = {auroc_te:.4f}")

        if args.bootstrap >= 20:
            rng = np.random.default_rng(args.seed)
            boot_o, boot_t = [], []
            for _ in range(args.bootstrap):
                bo, bt = stat(_cluster_resample(held_out, rng))
                boot_o.append(bo)
                boot_t.append(bt)
            tasks = sorted({t.task_id for t in held_out})
            jack_o, jack_t = [], []
            for tk in tasks:
                sub = [t for t in held_out if t.task_id != tk]
                jo, jt = stat(sub)
                jack_o.append(jo)
                jack_t.append(jt)
            up_o = _bca_endpoint(rec_or, boot_o, jack_o, 0.95)
            lo_o = _bca_endpoint(rec_or, boot_o, jack_o, 0.05)
            up_t = _bca_endpoint(rec_tr, boot_t, jack_t, 0.95)
            lo_t = _bca_endpoint(rec_tr, boot_t, jack_t, 0.05)

            def verdict(lo, up):
                if lo > delta:
                    return f"material increase (lower {lo:.4f} > Delta={delta})"
                if up < delta:
                    return f"equivalent (upper {up:.4f} < Delta={delta})"
                return f"inconclusive (interval [{lo:.4f},{up:.4f}] crosses Delta={delta})"

            print(f"  [bootstrap B={args.bootstrap}] oracle-τ   rec 95% CI = [{lo_o:.4f}, {up_o:.4f}] → {verdict(lo_o, up_o)}")
            print(f"  [bootstrap B={args.bootstrap}] transfer-τ rec 95% CI = [{lo_t:.4f}, {up_t:.4f}] → {verdict(lo_t, up_t)}")
            res.update({"rec_oracle_lo95": round(lo_o, 4), "rec_oracle_up95": round(up_o, 4),
                        "rec_transfer_lo95": round(lo_t, 4), "rec_transfer_up95": round(up_t, 4),
                        "verdict_oracle": verdict(lo_o, up_o), "verdict_transfer": verdict(lo_t, up_t)})
        rows.append(res)

    if rows:
        out = Path(args.util_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = sorted({k for r in rows for k in r})
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\ncsv → {out}")


# Leave-one-family-out classification evaluation.

def run_auroc(trajs, args) -> None:
    examples = build_examples(trajs, delta=args.delta)
    print(f"{len(trajs)} trajectories → {len(examples)} examples "
          f"(pos rate {sum(e.label for e in examples) / len(examples):.2%})")

    if args.held_out_loop:
        splits = split_leave_one_loop_out(examples, args.held_out_loop)
        s = split_by_task(splits["train"], (0.8, 0.2, 0.0))
        train, val, test = s["train"], s["val"], splits["test"]
        print(f"leave-one-loop-out: held out {args.held_out_loop!r}, test n={len(test)}")
    else:
        s = split_by_task(examples)
        train, val, test = s["train"], s["val"], s["test"]

    metrics = train_gbt(train, val, args.model_out)
    print(f"trained {metrics['model_type']}  AUROC(val)={metrics['auroc_val']:.4f}")

    import joblib
    from loopstop.estimator.dataset import to_matrix

    bundle = joblib.load(args.model_out)
    X_te, y_te, _ = to_matrix(test)
    p_te = [p[1] for p in bundle["model"].predict_proba(X_te)]
    print(f"AUROC(test)={auroc(y_te, p_te):.4f}  n={len(test)}")
    print("calibration bins:", json.dumps(calibration_bins(y_te, p_te), indent=1))

    if args.ablation:
        print("\n== Feature ablation (drop one group, validation AUROC):")
        for k, v in ablate_groups(train, val).items():
            print(f"  {k:24s} {v:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True, help="一个或多个轨迹 JSONL(三循环合并)")
    ap.add_argument("--delta", type=float, default=0.05, help="标签阈值 y_t=1[max_{s>t} r_s > r_t+δ]")
    ap.add_argument("--model-out", default="models_ckpt/voi_gbt.joblib")
    ap.add_argument("--held-out-loop", default=None, help="留出的循环类型; --utility-eval 下可为 'all'(跑 3 折)")
    ap.add_argument("--ablation", action="store_true", help="Run feature-group ablation for AUROC")
    ap.add_argument("--exclude-tasks", default="Mbpp_793", help="逗号分隔 task_id 剔除(预注册)")
    # Utility evaluation uses the same leave-one-family-out split.
    ap.add_argument("--utility-eval", action="store_true", help="改跑跨循环部署效用 recoverable(而非 AUROC)")
    ap.add_argument("--replay-config", default="configs/replay.yaml")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.005)
    ap.add_argument("--mode", default="stop_at_last", choices=["stop_at_last", "stop_and_select"])
    ap.add_argument("--bootstrap", type=int, default=0, help="B≥20 出轨迹级 BCa CI + 预注册判读")
    ap.add_argument("--tau-min", type=float, default=0.05)
    ap.add_argument("--tau-max", type=float, default=0.95)
    ap.add_argument("--tau-step", type=float, default=0.05)
    ap.add_argument("--min-rounds", type=int, default=1, help="VOI 最早可停轮")
    ap.add_argument("--delta-code", type=float, default=0.02, help="留出=code 的效用 Δ")
    ap.add_argument("--delta-writing", type=float, default=0.05, help="留出=writing 的效用 Δ")
    ap.add_argument("--delta-qa", type=float, default=0.02, help="留出=qa 的效用 Δ")
    ap.add_argument("--util-out", default="results/xloop_voi.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exclude = {x for x in args.exclude_tasks.split(",") if x}
    trajs = [t for path in args.traj for t in load_trajectories(path)
             if t.has_truth() and t.task_id not in exclude]

    if args.utility_eval:
        with open(args.replay_config, encoding="utf-8") as f:
            replay_cfg = yaml.safe_load(f)
        loops = sorted({t.loop_type for t in trajs})
        print(f"{len(trajs)} trajectories, loop types = {loops}")
        run_utility_eval(trajs, replay_cfg, args)
    else:
        if args.held_out_loop == "all":
            sys.exit("--held-out-loop all 仅在 --utility-eval 下有效")
        run_auroc(trajs, args)


if __name__ == "__main__":
    main()
