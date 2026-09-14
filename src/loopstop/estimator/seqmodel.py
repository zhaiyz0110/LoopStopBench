"""Tabular and recurrent models for the model-class comparison.

All learners use the same visible features, labels, two-fold outer split,
inner validation split, and threshold grid. Each trajectory is scored by a
model that was not trained on its task.

The utility label is ``1[max_{s>t} U(s) > U(t)]``; the optional quality label
is ``1[max_{s>t} r_s > r_t + delta]``. The truth-fed bidirectional model is a
positive control and is not a deployable visible-signal policy. PyTorch is
imported only when a recurrent model is fitted.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

TOL = 1e-9


# Per-trajectory feature, utility, and quality arrays.

def _arrays(traj, featfn, cols, lam, mode, with_truth):
    from loopstop.replay.evaluator import utility
    us = np.array([utility(traj, t, lam, mode) for t in range(1, traj.T + 1)], dtype="float64")
    r = np.array([traj.r(t) for t in range(1, traj.T + 1)], dtype="float64")
    rows = []
    for t in range(1, traj.T + 1):
        d = featfn(traj.visible(t))
        row = [0.0 if d.get(c) is None else float(d.get(c)) for c in cols]
        if with_truth:                       # Positive control: append current hidden quality.
            row.append(float(r[t - 1]))
        rows.append(row)
    return np.asarray(rows, dtype="float32"), us, r


def _labels(us, r, label_kind, delta):
    """y[t-1] 对应第 t 轮(1..T-1);末轮无"继续",不产标签。"""
    T = len(us)
    y = np.zeros(T, dtype="float32")
    for t in range(1, T):
        if label_kind == "quality":
            y[t - 1] = 1.0 if r[t:].max() > r[t - 1] + delta else 0.0
        else:  # Utility label.
            y[t - 1] = 1.0 if us[t:].max() > us[t - 1] + TOL else 0.0
    return y


def _stop_util(p, us, tau, min_round=1):
    """p[i]=第 i+1 轮 P(继续更优);首个 < tau 处停,否则跑满 T。"""
    for t in range(1, len(us) + 1):
        if t >= min_round and t <= len(p) and p[t - 1] < tau:
            return float(us[t - 1])
    return float(us[-1])


def _masked_auroc(P, Y, M):
    from loopstop.estimator.train import auroc
    sel = M > 0.5
    a = auroc(Y[sel].astype(int).tolist(), P[sel].tolist())
    return 0.5 if (isinstance(a, float) and np.isnan(a)) else a


# Tabular GBT and recurrent learners.

def _fit_predict_gbt(fit_trajs, featfn, cols, lam, mode, label_kind, delta, with_truth):
    from loopstop.estimator.train import _make_gbt
    X, y = [], []
    for tr in fit_trajs:
        M, us, r = _arrays(tr, featfn, cols, lam, mode, with_truth)
        lab = _labels(us, r, label_kind, delta)
        for t in range(1, tr.T):
            X.append(M[t - 1]); y.append(lab[t - 1])
    if len({int(v) for v in y}) < 2:
        return None
    model = _make_gbt()
    model.fit(np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32"))

    def predict(tr):
        M, _, _ = _arrays(tr, featfn, cols, lam, mode, with_truth)
        rows = M[: tr.T - 1]
        if len(rows) == 0:
            return np.zeros(0, dtype="float32")
        return model.predict_proba(rows)[:, 1]
    return predict


def _fit_predict_rnn(fit_trajs, val_trajs, featfn, cols, lam, mode, label_kind, delta,
                     with_truth, hidden, rnn_type, seed, bidir=False):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    mats = [_arrays(tr, featfn, cols, lam, mode, with_truth)[0] for tr in fit_trajs]
    allrows = np.concatenate(mats, axis=0)
    mu = allrows.mean(axis=0).astype("float32")
    sd = allrows.std(axis=0).astype("float32")
    sd[sd < 1e-6] = 1.0
    D = allrows.shape[1]

    def pack(trajs):
        Tmax = max(tr.T for tr in trajs)
        Xs, Ys, Ms = [], [], []
        for tr in trajs:
            M, us, r = _arrays(tr, featfn, cols, lam, mode, with_truth)
            M = (M - mu) / sd
            y = _labels(us, r, label_kind, delta)
            pad = Tmax - tr.T
            Xs.append(np.pad(M, ((0, pad), (0, 0))))
            Ys.append(np.pad(y, (0, pad)))
            m = np.zeros(Tmax, dtype="float32"); m[: max(tr.T - 1, 0)] = 1.0
            Ms.append(m)
        return (torch.tensor(np.stack(Xs), dtype=torch.float32),
                torch.tensor(np.stack(Ys), dtype=torch.float32),
                torch.tensor(np.stack(Ms), dtype=torch.float32))

    Xtr, Ytr, Mtr = pack(fit_trajs)
    Xva, Yva, Mva = pack(val_trajs)
    RNN = torch.nn.LSTM if rnn_type == "lstm" else torch.nn.GRU
    rnn = RNN(D, hidden, num_layers=1, batch_first=True, bidirectional=bidir)
    # Bidirectional mode exposes the future only for the positive control.
    head = torch.nn.Linear(hidden * (2 if bidir else 1), 1)
    opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")

    def logits(X):
        o, _ = rnn(X)
        return head(o).squeeze(-1)

    best_auc, best_state, wait, patience = -1.0, None, 0, 20
    for _ in range(300):
        rnn.train(); head.train(); opt.zero_grad()
        lg = logits(Xtr)
        loss = (lossf(lg, Ytr) * Mtr).sum() / Mtr.sum().clamp(min=1.0)
        loss.backward(); opt.step()
        rnn.eval(); head.eval()
        with torch.no_grad():
            pv = torch.sigmoid(logits(Xva)).numpy()
        auc = _masked_auroc(pv, Yva.numpy(), Mva.numpy())
        if auc > best_auc + 1e-4:
            best_auc, wait = auc, 0
            best_state = ([p.detach().clone() for p in rnn.parameters()],
                          [p.detach().clone() for p in head.parameters()])
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        for p, s in zip(rnn.parameters(), best_state[0]):
            p.data.copy_(s)
        for p, s in zip(head.parameters(), best_state[1]):
            p.data.copy_(s)
    rnn.eval(); head.eval()

    def predict(tr):
        M, _, _ = _arrays(tr, featfn, cols, lam, mode, with_truth)
        M = (M - mu) / sd
        with torch.no_grad():
            o, _ = rnn(torch.tensor(M[None, ...], dtype=torch.float32))
            p = torch.sigmoid(head(o).squeeze(-1)).numpy()[0]
        return p[: tr.T - 1]
    return predict


def _select_tau(val_trajs, predict, lam, mode):
    from loopstop.replay.evaluator import utility
    prof, allp = [], []
    for tr in val_trajs:
        p = predict(tr)
        us = [utility(tr, t, lam, mode) for t in range(1, tr.T + 1)]
        prof.append((p, us)); allp.extend(list(p))
    if not allp:
        return 0.5
    qs = np.quantile(allp, [i / 20 for i in range(1, 20)])
    taus = sorted({-1.0, *(float(q) for q in qs), float(max(allp)) + 1e-6})
    return max(taus, key=lambda tau: float(np.mean([_stop_util(p, us, tau) for p, us in prof])))


# Nested out-of-fold evaluation.

def oof_stop_utils(trajs, featfn, cols, lam, mode, label_kind="utility", delta=0.05,
                   with_truth=False, learner="gru", hidden=32, rnn_type="gru",
                   bidir=False, seed=0, k_outer=2):
    """Return out-of-fold stopping utility by trajectory.

    Each outer fold is scored by a model trained on other tasks. The stopping
    threshold is selected on the corresponding inner validation set and then
    applied to the outer test fold.
    """
    outer = defaultdict(list)
    for tr in trajs:
        outer[int(hashlib.md5(tr.task_id.encode()).hexdigest(), 16) % k_outer].append(tr)
    out = {}
    for tf in range(k_outer):
        train = [tr for f in range(k_outer) if f != tf for tr in outer[f]]
        test = outer[tf]
        if not train or not test:
            for tr in test:
                out[tr.traj_id] = None
            continue
        inner = defaultdict(list)
        for tr in train:
            inner[int(hashlib.md5((tr.task_id + ":inner").encode()).hexdigest(), 16) % 2].append(tr)
        fit_set, val_set = (inner[0], inner[1]) if inner[0] and inner[1] else (train, train)
        if learner == "gbt":
            predict = _fit_predict_gbt(fit_set, featfn, cols, lam, mode, label_kind, delta, with_truth)
        else:
            predict = _fit_predict_rnn(fit_set, val_set, featfn, cols, lam, mode, label_kind,
                                       delta, with_truth, hidden, rnn_type, seed, bidir)
        if predict is None:
            for tr in test:
                out[tr.traj_id] = None
            continue
        tau = _select_tau(val_set, predict, lam, mode)
        from loopstop.replay.evaluator import utility
        for tr in test:
            p = predict(tr)
            us = [utility(tr, t, lam, mode) for t in range(1, tr.T + 1)]
            out[tr.traj_id] = _stop_util(p, us, tau)
    return out
