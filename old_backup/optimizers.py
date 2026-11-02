"""Search methods with a common batch ask/tell interface over the unit cube.

All optimisers are deterministic given their seed and the sequence of told values, which is what
lets an interrupted job be resumed by replaying its log.
"""
import warnings

import numpy as np

from config import DIM, SPACE, repair
from metrics import crowding, nd_ranks

EPS = 1e-12


class BaseOpt:
    def __init__(self, S, seed):
        self.S = S
        self.pop = S.pop
        self.n_iter = S.budget // S.pop
        self.T = max(self.n_iter - 1, 1)
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def _random(self):
        return repair(self.rng.random((self.pop, DIM)))


class RandomOpt(BaseOpt):
    def ask(self):
        return self._random()

    def tell(self, X, F):
        self.t += 1


class OptunaOpt(BaseOpt):
    def __init__(self, S, seed, kind):
        super().__init__(S, seed)
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        self.multi = kind in ("motpe", "nsga2")
        if kind == "nsga2":
            sampler = optuna.samplers.NSGAIISampler(population_size=self.pop, seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=self.pop, multivariate=True)
        if self.multi:
            self.study = optuna.create_study(directions=["minimize"] * 3, sampler=sampler)
        else:
            self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.pending = []

    def ask(self):
        self.pending, X = [], []
        for _ in range(self.pop):
            tr = self.study.ask()
            x = []
            for j, spec in enumerate(SPACE):
                if spec[1] == "cat":
                    n = len(spec[2])
                    x.append(tr.suggest_int(f"x{j}", 0, n - 1) / (n - 1))
                else:
                    x.append(tr.suggest_float(f"x{j}", 0.0, 1.0))
            self.pending.append(tr)
            X.append(x)
        return repair(np.array(X))

    def tell(self, X, F):
        for tr, f in zip(self.pending, F):
            val = [float(v) for v in f] if self.multi else float(f[0] + 1e-6 * f[1])
            self.study.tell(tr, val)
        self.pending = []
        self.t += 1


class PSO(BaseOpt):
    w, c1, c2, vmax = 0.7298, 1.49618, 1.49618, 0.2

    def ask(self):
        if self.t == 0:
            self.X = self._random()
            self.V = self.rng.uniform(-self.vmax, self.vmax, (self.pop, DIM))
        return self.X.copy()

    def tell(self, X, F):
        f = F[:, 0] + 1e-6 * F[:, 1]
        if self.t == 0:
            self.pb, self.pbf = X.copy(), f.copy()
        else:
            better = f < self.pbf
            self.pb[better], self.pbf[better] = X[better], f[better]
        g = self.pb[np.argmin(self.pbf)]
        r1, r2 = self.rng.random((2, self.pop, DIM))
        self.V = self.w * self.V + self.c1 * r1 * (self.pb - X) + self.c2 * r2 * (g - X)
        self.V = np.clip(self.V, -self.vmax, self.vmax)
        self.X = repair(X + self.V)
        self.t += 1


def _gwo_move(x, leaders, a, rng, weights=None):
    """Standard GWO pull towards each leader (Eqs. 9-10), combined with optional weights."""
    moves = []
    for xl in leaders:
        r1, r2 = rng.random(DIM), rng.random(DIM)
        A, C = 2 * a * r1 - a, 2 * r2
        moves.append(xl - A * np.abs(C * xl - x))
    moves = np.array(moves)
    w = np.full(len(leaders), 1.0 / len(leaders)) if weights is None else weights
    return (w[:, None] * moves).sum(0)


class GWO(BaseOpt):
    """Single-objective GWO on 1 - Macro-F1 with the same cosine schedule as the MO variants."""

    def __init__(self, S, seed):
        super().__init__(S, seed)
        self.hx, self.hf = [], []

    def ask(self):
        if self.t == 0:
            return self._random()
        a = float(np.clip(1 + np.cos(np.pi * self.t / self.T), 0, 2))
        order = np.argsort(self.hf, kind="stable")[:3]
        leaders = [self.hx[i] for i in order]
        return repair(np.array([_gwo_move(x, leaders, a, self.rng) for x in self.X]))

    def tell(self, X, F):
        self.X = X.copy()
        for x, f in zip(X, F):
            self.hx.append(x.copy())
            self.hf.append(f[0] + 1e-6 * f[1])
        self.t += 1


class MOGWO(BaseOpt):
    """Pareto-GWO / PU-GWO family.

    signal: none | aurc | nll | random | shuffled | rank  (source of the per-candidate Û)
    leader_w: objective-quality leader weighting (Eqs. 15-17), else equal weights
    momentum: momentum update (Eqs. 18-19), else direct move to the weighted target
    bound: "standard" (a <= 2) or "extended" (a <= 2(1+rho))
    """

    def __init__(self, S, seed, signal, leader_w, momentum, rho, bound):
        super().__init__(S, seed)
        self.signal, self.leader_w, self.momentum = signal, leader_w, momentum
        self.rho, self.bound = rho, bound
        self.AX = np.zeros((0, DIM))
        self.AF = np.zeros((0, 3))

    def _u_hat(self):
        F = self.F

        def mm(v):
            r = v.max() - v.min()
            return np.zeros_like(v) if r < 1e-12 else (v - v.min()) / (r + EPS)

        s = self.signal
        if s == "none":
            return np.zeros(self.pop)
        if s == "aurc":
            return mm(F[:, 2])
        if s == "nll":
            return mm(F[:, 1])
        if s == "random":
            return self.rng.random(self.pop)
        if s == "shuffled":
            return self.rng.permutation(mm(F[:, 2]))
        if s == "rank":
            r = nd_ranks(F).astype(float)
            return r / max(r.max(), 1.0)
        raise ValueError(s)

    def a_values(self):
        cos_f = 1 + np.cos(np.pi * self.t / self.T)
        amax = 2.0 if self.bound == "standard" else 2.0 * (1 + self.rho)
        return np.clip(cos_f * (1 + self.rho * self._u_hat()), 0, amax)

    def _pick_leaders(self):
        m = len(self.AX)
        cd = crowding(self.AF) if m > 0 else np.array([])
        fin = cd[np.isfinite(cd)]
        cap = 2 * fin.max() if len(fin) and fin.max() > 0 else 1.0
        w = np.where(np.isfinite(cd), cd, cap) + 1e-9
        idx = self.rng.choice(m, size=3, replace=m < 3, p=w / w.sum())
        return idx

    def ask(self):
        if self.t == 0:
            self.V = np.zeros((self.pop, DIM))
            return self._random()
        a = self.a_values()
        lo, hi = self.AF.min(0), self.AF.max(0)
        AFn = (self.AF - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
        mu = self.S.mu_min + (self.S.mu_max - self.S.mu_min) * (self.t - 1) / max(self.T - 1, 1)
        newX = []
        for i, x in enumerate(self.X):
            li = self._pick_leaders()
            weights = None
            if self.leader_w:
                Sl = AFn[li].mean(1)
                e = np.exp(-self.S.tau * (Sl - Sl.min()))
                weights = e / e.sum()
            z = _gwo_move(x, self.AX[li], a[i], self.rng, weights)
            if self.momentum:
                self.V[i] = mu * self.V[i] + (1 - mu) * (z - x)
                newX.append(x + self.V[i])
            else:
                newX.append(z)
        return repair(np.array(newX))

    def tell(self, X, F):
        self.X, self.F = X.copy(), np.asarray(F, float).copy()
        AX = np.vstack([self.AX, X])
        AF = np.vstack([self.AF, self.F])
        keep = nd_ranks(AF) == 0
        AX, AF = AX[keep], AF[keep]
        _, uniq = np.unique(np.round(AX, 12), axis=0, return_index=True)
        uniq = np.sort(uniq)
        AX, AF = AX[uniq], AF[uniq]
        while len(AX) > self.S.archive_cap:
            j = int(np.argmin(crowding(AF)))
            AX, AF = np.delete(AX, j, 0), np.delete(AF, j, 0)
        self.AX, self.AF = AX, AF
        self.t += 1


def make_optimizer(spec, S, seed):
    k = spec["kind"]
    if k == "random":
        return RandomOpt(S, seed)
    if k in ("tpe", "motpe", "nsga2"):
        return OptunaOpt(S, seed, k)
    if k == "pso":
        return PSO(S, seed)
    if k == "gwo":
        return GWO(S, seed)
    if k == "mogwo":
        return MOGWO(S, seed, spec["signal"], spec["leader_w"], spec["momentum"],
                     spec["rho"], spec["bound"])
    raise ValueError(k)
