"""Search space, experiment settings, method variants and job priority tiers."""
import json
import math
import os
from dataclasses import asdict, dataclass, fields

import numpy as np

CLASSES = ["glioma", "meningioma", "pituitary"]

# (name, kind, lo, hi) or (name, "cat", choices). Candidates live in the unit cube [0,1]^6.
SPACE = [
    ("lr", "log", 1e-5, 1e-3),
    ("dropout", "lin", 0.10, 0.60),
    ("weight_decay", "log", 1e-6, 1e-3),
    ("batch_size", "cat", [16, 32, 64]),
    ("unfrozen_stages", "cat", [1, 2, 3, 4]),
    ("label_smoothing", "lin", 0.0, 0.15),
]
DIM = len(SPACE)

# Configuration of the untuned "fixed ResNet-50" baseline. Report these values in the paper.
DEFAULT_CONFIG = dict(lr=1e-4, dropout=0.5, weight_decay=1e-4, batch_size=32,
                      unfrozen_stages=4, label_smoothing=0.0)


def repair(u):
    """Clip to the unit cube and snap categorical coordinates to their grid."""
    u = np.clip(np.array(u, dtype=float), 0.0, 1.0)
    for j, spec in enumerate(SPACE):
        if spec[1] == "cat":
            n = len(spec[2])
            u[..., j] = np.round(u[..., j] * (n - 1)) / (n - 1)
    return u


def decode(u):
    u = repair(u)
    cfg = {}
    for j, spec in enumerate(SPACE):
        name, kind = spec[0], spec[1]
        if kind == "log":
            lo, hi = math.log10(spec[2]), math.log10(spec[3])
            cfg[name] = float(10 ** (lo + u[j] * (hi - lo)))
        elif kind == "lin":
            cfg[name] = float(spec[2] + u[j] * (spec[3] - spec[2]))
        else:
            cfg[name] = spec[2][int(round(u[j] * (len(spec[2]) - 1)))]
    return cfg


@dataclass
class Settings:
    run_dir: str = "runs"
    cache: str = "data/figshare_224.npz"
    arch: str = "resnet50"
    pretrained: bool = True
    img_size: int = 224
    n_folds: int = 5            # outer patient-grouped folds (test ~20% of patients each)
    split_seed: int = 2024
    base_seed: int = 11
    budget: int = 120           # candidate evaluations per search run
    pop: int = 10               # population / batch size of every optimiser
    promote_k: int = 3          # fixed number promoted to stage 2 per batch (same for all methods)
    s1_epochs: int = 10
    s1_patience: int = 3
    s2_epochs: int = 25         # cosine LR schedule spans s2_epochs; stage 1 is a truncation of it
    s2_patience: int = 5
    final_epochs: int = 40
    final_patience: int = 8
    mc_search: int = 10
    mc_final: int = 30
    archive_cap: int = 20
    tau: float = 1.0
    mu_min: float = 0.2
    mu_max: float = 0.8
    eval_batch: int = 256
    # Fixed normalisation bounds for hypervolume: 1-MacroF1, NLL, AURC. Reference point = hv_ref in normalised space.
    hv_lo: tuple = (0.0, 0.0, 0.0)
    hv_hi: tuple = (0.25, 0.80, 0.08)
    hv_ref: float = 1.1
    device: str = "cuda"


PROFILES = {
    "default": {},
    # Confirmation run of the redesigned method: a fresh patient split and fresh training seeds,
    # everything else identical to "default". Use with a NEW run directory.
    "confirm": dict(split_seed=7, base_seed=101, run_dir="runs_confirm"),
    # Closer to the fidelity described in the original manuscript (roughly 1.4x slower).
    "paper": dict(s1_epochs=12, s1_patience=4, s2_epochs=35, s2_patience=7,
                  final_epochs=50, final_patience=10),
    # Tiny CPU run on synthetic data, only for checking the pipeline end to end.
    "smoke": dict(arch="resnet18", pretrained=False, img_size=32, n_folds=3, budget=20,
                  pop=10, promote_k=3, s1_epochs=2, s1_patience=2, s2_epochs=3, s2_patience=2,
                  final_epochs=3, final_patience=2, mc_search=3, mc_final=5, eval_batch=64,
                  device="cpu", cache="data/smoke.npz", run_dir="runs_smoke"),
}


def make_settings(profile="default", **overrides):
    s = Settings(**PROFILES[profile])
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, v)
    return s


def save_settings(S):
    os.makedirs(S.run_dir, exist_ok=True)
    with open(os.path.join(S.run_dir, "settings.json"), "w") as f:
        json.dump(asdict(S), f, indent=2)


def load_settings(run_dir):
    p = os.path.join(run_dir, "settings.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found. Run `python run.py prepare ...` first.")
    with open(p) as f:
        d = json.load(f)
    names = {f.name for f in fields(Settings)}
    S = Settings(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})
    S.run_dir = run_dir
    return S


# ----------------------------------------------------------------------------------------------
# Methods. kind: fixed | random | tpe | pso | gwo | nsga2 | motpe | mogwo
# selector: "scalar" = best validation Macro-F1; "ideal" = nearest normalised ideal point on the
# nondominated set of all evaluated candidates (identical rule for every multiobjective method).
# ----------------------------------------------------------------------------------------------
def _mogwo(signal="aurc", leader_w=True, momentum=True, rho=0.5, bound="standard", leader_unc=False):
    return dict(kind="mogwo", selector="ideal", signal=signal, leader_w=leader_w,
                momentum=momentum, rho=rho, bound=bound, leader_unc=leader_unc)


METHODS = {
    "fixed": dict(kind="fixed", selector=None),
    "random": dict(kind="random", selector="scalar"),
    "tpe": dict(kind="tpe", selector="scalar"),
    "pso": dict(kind="pso", selector="scalar"),
    "gwo": dict(kind="gwo", selector="scalar"),
    "nsga2": dict(kind="nsga2", selector="ideal"),
    "motpe": dict(kind="motpe", selector="ideal"),
    "pareto_gwo": _mogwo(signal="none", leader_w=False, momentum=False, rho=0.0),
    "pu_gwo": _mogwo(),
    # Controls for the uncertainty-guided movement (Comment 4)
    "pu_rho0": _mogwo(signal="none", rho=0.0),          # leave-one-out: no feedback, LW + momentum kept
    "pu_random_u": _mogwo(signal="random"),
    "pu_shuffled_u": _mogwo(signal="shuffled"),
    "pu_nll_u": _mogwo(signal="nll"),
    "pu_rank_u": _mogwo(signal="rank"),
    "pu_no_lw": _mogwo(leader_w=False),
    "pu_no_mom": _mogwo(momentum=False),                 # also the "+ leader weights" ablation step
    "abl_aurc": _mogwo(leader_w=False, momentum=False),  # "Pareto-GWO + AURC movement" ablation step
    "pu_rho025": _mogwo(rho=0.25),
    "pu_rho075": _mogwo(rho=0.75),
    "pu_extbound": _mogwo(bound="extended"),
    # ---- Redesign candidates (round 2). Development: folds 0-1 of the original run, validation only.
    # R1: rank-based AURC signal instead of min-max (robust to the few-error noise in validation AURC).
    "pu_r1_rank": _mogwo(signal="aurc_rank"),
    # R2: uncertainty-weighted leader selection only (no step-size feedback): low-AURC archive
    #     members are more likely to be chosen as alpha/beta/delta, pulling the pack towards them.
    "pu_r2_lead": _mogwo(signal="none", rho=0.0, leader_unc=True),
    # R3: both.
    "pu_r3_both": _mogwo(signal="aurc_rank", leader_unc=True),
}

METHOD_LABELS = {
    "fixed": "ResNet-50 (fixed)", "random": "Random Search", "tpe": "TPE", "pso": "PSO",
    "gwo": "GWO", "nsga2": "NSGA-II", "motpe": "MOTPE", "pareto_gwo": "Pareto-GWO",
    "pu_gwo": "PU-GWO", "pu_rho0": "PU-GWO, ρ=0 (no feedback)",
    "pu_random_u": "PU-GWO, random Û", "pu_shuffled_u": "PU-GWO, shuffled Û",
    "pu_nll_u": "PU-GWO, NLL-based Û", "pu_rank_u": "PU-GWO, Pareto-rank Û",
    "pu_no_lw": "PU-GWO − leader weights", "pu_no_mom": "PU-GWO − momentum",
    "abl_aurc": "Pareto-GWO + AURC movement", "pu_rho025": "PU-GWO, ρ=0.25",
    "pu_rho075": "PU-GWO, ρ=0.75", "pu_extbound": "PU-GWO, extended bound",
    "pu_r1_rank": "R1: rank-based AURC Û", "pu_r2_lead": "R2: AURC-weighted leaders",
    "pu_r3_both": "R3: rank Û + AURC-weighted leaders",
}

# Jobs run tier by tier, fold-major inside a tier, so the most important comparisons finish first.
TIERS = {
    1: ["fixed", "pu_gwo", "pareto_gwo", "pu_random_u", "pu_rho0"],
    2: ["gwo", "nsga2", "motpe", "tpe"],
    3: ["abl_aurc", "pu_no_mom", "pu_no_lw", "pu_shuffled_u", "pu_nll_u", "random", "pso"],
    4: ["pu_rho025", "pu_rho075", "pu_rank_u", "pu_extbound"],
    5: ["pu_r1_rank", "pu_r2_lead", "pu_r3_both"],   # redesign development
}


def job_list(S, tiers=None, methods=None, folds=None):
    tiers = tiers or sorted(TIERS)
    folds = list(range(S.n_folds)) if folds is None else folds
    jobs = []
    for t in tiers:
        for fold in folds:
            for m in TIERS[t]:
                if methods and m not in methods:
                    continue
                jobs.append((m, fold))
    return jobs


def job_id(method, fold):
    return f"{method}__fold{fold}"
