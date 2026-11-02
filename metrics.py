"""Evaluation metrics, temperature scaling, Pareto utilities and exact 3-D hypervolume."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

EPS = 1e-12


def mc_probs(logits, T=1.0):
    """logits: (K, N, C) MC-dropout logits -> mean softmax probabilities (N, C)."""
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    return p.mean(0)


def nll(p, y):
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + EPS)))


def entropy(p):
    return -(p * np.log(p + EPS)).sum(1)


def ece(p, y, n_bins=15):
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(p, y, n_bins=15):
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return np.array(out)


def risk_curve(p, y):
    """Selective risk after sorting by predictive entropy (most confident first)."""
    order = np.argsort(entropy(p), kind="stable")
    err = (p.argmax(1) != y)[order].astype(float)
    return np.cumsum(err) / np.arange(1, len(y) + 1)


def aurc(p, y):
    return float(risk_curve(p, y).mean())


def eaurc(p, y):
    n = len(y)
    e = int((p.argmax(1) != y).sum())
    j = np.arange(1, n + 1)
    opt = np.maximum(0, j - (n - e)) / j
    return float(aurc(p, y) - opt.mean())


def risk_at_coverage(p, y, cov):
    r = risk_curve(p, y)
    k = max(1, int(np.ceil(cov * len(y))))
    return float(r[k - 1])


def error_auroc(p, y):
    err = (p.argmax(1) != y).astype(int)
    if err.min() == err.max():
        return float("nan")
    return float(roc_auc_score(err, entropy(p)))


#def macro_f1(p, y):
#    return float(f1_score(y, p.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0))
def macro_f1(p, y):
    """Macro-F1 over classes 0-2, identical to sklearn f1_score(average='macro', zero_division=0)."""
    yh = np.asarray(p).argmax(1)
    y = np.asarray(y)
    f1s = []
    for c in (0, 1, 2):
        tp = np.sum((yh == c) & (y == c))
        fp = np.sum((yh == c) & (y != c))
        fn = np.sum((yh != c) & (y == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))

def objectives(p, y):
    """Search objectives (all minimised): 1 - Macro-F1, NLL, AURC."""
    return [1.0 - macro_f1(p, y), nll(p, y), aurc(p, y)]


def patient_level(p, y, pids):
    up = np.unique(pids)
    P = np.stack([p[pids == u].mean(0) for u in up])
    Y = np.array([np.bincount(y[pids == u]).argmax() for u in up])
    return P, Y


def all_metrics(p, y, pids=None):
    yh = p.argmax(1)
    m = dict(
        acc=accuracy_score(y, yh), bal_acc=balanced_accuracy_score(y, yh),
        macro_f1=macro_f1(p, y),
        macro_prec=precision_score(y, yh, average="macro", labels=[0, 1, 2], zero_division=0),
        macro_rec=recall_score(y, yh, average="macro", labels=[0, 1, 2], zero_division=0),
        mcc=matthews_corrcoef(y, yh), nll=nll(p, y), ece=ece(p, y), aurc=aurc(p, y),
        eaurc=eaurc(p, y), err_auroc=error_auroc(p, y),
    )
    for c in (1.0, 0.95, 0.90, 0.80):
        m[f"risk{int(round(c * 100))}"] = risk_at_coverage(p, y, c)
    if pids is not None:
        P, Y = patient_level(p, y, pids)
        m["patient_acc"] = accuracy_score(Y, P.argmax(1))
        m["patient_macro_f1"] = macro_f1(P, Y)
    return {k: float(v) for k, v in m.items()}


def fit_temperature(logits_val, y_val):
    def f(logT):
        return nll(mc_probs(logits_val, float(np.exp(logT))), y_val)
    r = minimize_scalar(f, bounds=(np.log(0.05), np.log(10.0)), method="bounded")
    return float(np.exp(r.x))


# ------------------------------------------------------------------ Pareto utilities
def nd_ranks(F):
    F = np.asarray(F, float)
    n = len(F)
    ranks = np.full(n, -1)
    remaining = set(range(n))
    r = 0
    while remaining:
        front = [i for i in remaining
                 if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in remaining if j != i)]
        for i in front:
            ranks[i] = r
        remaining -= set(front)
        r += 1
    return ranks


def crowding(F):
    F = np.asarray(F, float)
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        o = np.argsort(F[:, k], kind="stable")
        rng = F[o[-1], k] - F[o[0], k]
        d[o[0]] = d[o[-1]] = np.inf
        if rng <= 0:
            continue
        d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / rng
    return d


def nondominated_mask(F):
    return nd_ranks(F) == 0


def ideal_point_select(F):
    """Index (into F) of the nondominated point nearest the normalised ideal point."""
    F = np.asarray(F, float)
    nd = np.where(nondominated_mask(F))[0]
    G = F[nd]
    lo, hi = G.min(0), G.max(0)
    Gn = (G - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    return int(nd[np.argmin(np.sqrt((Gn ** 2).sum(1)))])


def _hv2d(P, ref):
    P = P[np.all(P < ref, 1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0], kind="stable")]
    area, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            area += (ref[0] - x) * (best_y - y)
            best_y = y
    return area


def hypervolume3d(F, lo, hi, ref=1.1):
    """Exact hypervolume of minimisation objectives after fixed min-max normalisation."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return 0.0
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    P = np.clip((F - lo) / (hi - lo), 0, None)
    r = np.full(3, ref)
    P = P[np.all(P < r, 1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 2], kind="stable")]
    vol = 0.0
    for k in range(len(P)):
        z_next = P[k + 1, 2] if k + 1 < len(P) else r[2]
        dz = z_next - P[k, 2]
        if dz > 0:
            vol += _hv2d(P[: k + 1, :2], r[:2]) * dz
    return float(vol / ref ** 3)  # scaled to [0, 1]
