"""Build every table and figure from saved logs and prediction files. Nothing is interpolated.

Outputs (CSV + LaTeX tables, PDF + PNG figures) go to the --out directory, together with
ANALYSIS_NOTES.md, which lists which jobs were available and any step that failed.

Robustness changes (v2):
  * every table and figure is built in its own try/except, so one failure cannot stop the rest;
  * figures are drawn before the (slow) statistics;
  * the patient-level bootstrap uses only NumPy (no scikit-learn/SciPy calls inside the loop);
  * the statistics table is saved after every comparison, with progress printed.
"""
import glob
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import CLASSES, METHOD_LABELS, SPACE, load_settings  # noqa: E402
from data import load_cache, load_splits, split_table  # noqa: E402
from metrics import (all_metrics, aurc, ece, fit_temperature, hypervolume3d,  # noqa: E402
                     mc_probs, nll, reliability_bins, risk_curve)
from search import JsonlLog  # noqa: E402

MAIN = ["fixed", "random", "tpe", "pso", "gwo", "nsga2", "motpe", "pareto_gwo", "pu_gwo"]
ABLATION = ["gwo", "pareto_gwo", "abl_aurc", "pu_no_mom", "pu_gwo"]
ABLATION_LABELS = {"gwo": "GWO", "pareto_gwo": "Pareto-GWO", "abl_aurc": "+ AURC-guided movement",
                   "pu_no_mom": "+ leader weights", "pu_gwo": "+ momentum (full PU-GWO)"}
CONTROLS = ["pu_gwo", "pu_rho0", "pu_random_u", "pu_shuffled_u", "pu_nll_u", "pu_rank_u",
            "pu_no_lw", "pu_no_mom", "pu_extbound"]
RHO = [("pu_rho0", 0.0), ("pu_rho025", 0.25), ("pu_gwo", 0.5), ("pu_rho075", 0.75)]
COLORS = {"fixed": "#7f7f7f", "gwo": "#ff7f0e", "pareto_gwo": "#2ca02c", "pu_gwo": "#d62728",
          "nsga2": "#9467bd", "motpe": "#8c564b", "tpe": "#1f77b4", "random": "#bcbd22",
          "pso": "#17becf"}


def say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------ pure-NumPy metrics
def macro_f1_np(p, y):
    """Macro-F1 over classes 0-2; identical to sklearn f1_score(average='macro', zero_division=0)."""
    yh = np.asarray(p).argmax(1)
    y = np.asarray(y)
    f1s = []
    for c in (0, 1, 2):
        tp = np.sum((yh == c) & (y == c))
        fp = np.sum((yh == c) & (y != c))
        fn = np.sum((yh != c) & (y == c))
        d = 2 * tp + fp + fn
        f1s.append(0.0 if d == 0 else 2.0 * tp / d)
    return float(np.mean(f1s))


# ------------------------------------------------------------------------------ loading
def hv_curve(recs, S):
    latest, curve = {}, []
    for r in recs:
        if r.get("type") == "eval":
            latest[r["cand"]] = (r["f1_loss"], r["nll"], r["aurc"])
        elif r.get("type") == "batch_end":
            F = np.array(list(latest.values()))
            curve.append((len(latest), hypervolume3d(F, S.hv_lo, S.hv_hi, S.hv_ref)))
    return curve


def load_jobs(S):
    _, _, pids, _ = load_cache(S.cache)
    jobs = []
    dirs = sorted(glob.glob(os.path.join(S.run_dir, "*__fold*")))
    for n, jd in enumerate(dirs):
        if not os.path.exists(os.path.join(jd, "done.json")):
            continue
        fin = json.load(open(os.path.join(jd, "final.json")))
        P = np.load(os.path.join(jd, "preds.npz"))
        recs = JsonlLog(os.path.join(jd, "evals.jsonl")).read()
        evals = [r for r in recs if r.get("type") == "eval"]
        pt = mc_probs(P["test_logits"])
        T = fit_temperature(P["val_logits"], P["y_val"])
        pt_ts = mc_probs(P["test_logits"], T)
        yt, pid_t = P["y_test"], pids[P["test_idx"]]
        m = all_metrics(pt, yt, pid_t)
        m.update(nll_ts=nll(pt_ts, yt), ece_ts=ece(pt_ts, yt), aurc_ts=aurc(pt_ts, yt), T=T)
        curve = hv_curve(recs, S)
        last = {}
        for r in evals:
            last[r["cand"]] = r
        search_sec = sum(r.get("gpu_sec", 0.0) for r in evals)
        epochs = sum(r.get("epochs_this_stage", 0) for r in evals) + fin["final_epochs_run"]
        peak = max([r.get("peak_mem_gb", 0.0) for r in evals] + [fin.get("final_peak_mem_gb", 0.0)])
        jobs.append(dict(method=fin["method"], fold=fin["fold"], metrics=m, probs=pt, probs_ts=pt_ts,
                         y=yt, pid=pid_t, curve=curve, config=fin["config"],
                         hv=curve[-1][1] if curve else np.nan, search_h=search_sec / 3600,
                         final_h=fin["final_gpu_sec"] / 3600, epochs=epochs, peak_gb=peak,
                         n_evals=len(last)))
        if (n + 1) % 20 == 0:
            say(f"  loaded {n + 1}/{len(dirs)} job folders")
    return jobs


def by_method(jobs):
    d = {}
    for j in jobs:
        d.setdefault(j["method"], {})[j["fold"]] = j
    return d


# ------------------------------------------------------------------------------ table helpers
def ms(vals, scale=1.0, digits=3):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float) * scale
    if len(v) == 0:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def tex_escape(s):
    s = str(s)
    for a, b in [("±", r"$\pm$"), ("ρ", r"$\rho$"), ("Û", r"$\hat{U}$"), ("−", "--"),
                 ("%", r"\%"), ("_", r"\_"), ("&", r"\&"), ("↑", r"$\uparrow$"), ("↓", r"$\downarrow$")]:
        s = s.replace(a, b)
    return s


def write_table(df, out, name, caption):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False, encoding="utf-8-sig")
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + tex_escape(caption) + "}",
             r"\label{tab:" + name + "}", r"\small",
             r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say(f"  wrote {name}")


def metric_table(M, methods, spec, labels=None):
    rows = []
    for m in methods:
        if m not in M:
            continue
        js = list(M[m].values())
        row = {"Method": (labels or METHOD_LABELS).get(m, m), "Folds": len(js)}
        for col, key, scale, dg in spec:
            vals = [j[key] if key in ("hv", "search_h", "final_h", "epochs", "peak_gb")
                    else j["metrics"][key] for j in js]
            row[col] = ms(vals, scale, dg)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ statistics
def pooled(M, method, folds):
    js = [M[method][f] for f in folds]
    keys = np.concatenate([np.array([f"{j['fold']}:{p}" for p in j["pid"]]) for j in js])
    return (np.concatenate([j["probs"] for j in js]), np.concatenate([j["y"] for j in js]), keys)


STAT_METRICS = [("Macro-F1 (pp)", lambda p, y: 100 * macro_f1_np(p, y), "macro_f1", 100),
                ("NLL", nll, "nll", 1), ("ECE", ece, "ece", 1), ("AURC", aurc, "aurc", 1)]


def wilcoxon_p(da, db):
    if len(da) < 5 or not np.any(da != db):
        return np.nan
    try:
        from scipy import stats
        return float(stats.wilcoxon(da, db).pvalue)
    except Exception:
        return np.nan


def paired_stats(M, ref, others, n_boot, out, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    name = "table11_paired_statistics"
    caption = ("PU-GWO minus comparator: per-fold differences, pooled difference with 95% "
               f"patient-level bootstrap CI ({n_boot} resamples), and Wilcoxon signed-rank p "
               "(with 5 folds the smallest possible two-sided p is 0.0625).")
    for k, o in enumerate(others):
        if o not in M or ref not in M:
            continue
        folds = sorted(set(M[ref]) & set(M[o]))
        if not folds:
            continue
        say(f"  statistics {k + 1}/{len(others)}: {METHOD_LABELS[ref]} vs {METHOD_LABELS[o]}")
        pa, y, keys = pooled(M, ref, folds)
        pb, yb, _ = pooled(M, o, folds)
        if not np.array_equal(y, yb):
            raise ValueError(f"test labels differ between {ref} and {o}; folds are not paired")
        _, inv = np.unique(keys, return_inverse=True)
        groups = [np.where(inv == g)[0] for g in range(inv.max() + 1)]
        boots = {nm: np.empty(n_boot) for nm, *_ in STAT_METRICS}
        for b in range(n_boot):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            pai, pbi, yi = pa[idx], pb[idx], y[idx]
            for nm, fn, *_ in STAT_METRICS:
                boots[nm][b] = fn(pai, yi) - fn(pbi, yi)
        for nm, fn, key, scale in STAT_METRICS:
            da = np.array([M[ref][f]["metrics"][key] for f in folds]) * scale
            db = np.array([M[o][f]["metrics"][key] for f in folds]) * scale
            d = da - db
            p_w = wilcoxon_p(da, db)
            ci = "–"
            if n_boot > 0:
                lo, hi = np.percentile(boots[nm], [2.5, 97.5])
                ci = f"[{lo:+.4f}, {hi:+.4f}]"
            better = (d > 0).sum() if key == "macro_f1" else (d < 0).sum()
            rows.append({"Comparison": f"{METHOD_LABELS[ref]} − {METHOD_LABELS[o]}", "Metric": nm,
                         "Mean fold diff": f"{d.mean():+.4f}",
                         "Pooled diff": f"{fn(pa, y) - fn(pb, y):+.4f}",
                         "95% patient-bootstrap CI": ci,
                         "Folds better": f"{int(better)}/{len(d)}",
                         "Wilcoxon p": "–" if np.isnan(p_w) else f"{p_w:.3f}"})
        write_table(pd.DataFrame(rows), out, name, caption)  # save after every comparison
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ figures
def savefig(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    say(f"  wrote {name}")


def fig_clipping(out, rho=0.5):
    t = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for u in (0, 0.25, 0.5, 0.75, 1.0):
        ax.plot(t, np.clip((1 + np.cos(np.pi * t)) * (1 + rho * u), 0, 2), label=f"Û = {u}")
    ts = np.arccos((1 - rho) / (1 + rho)) / np.pi
    ax.axvline(ts, ls="--", c="k", lw=0.8)
    ax.text(ts + 0.01, 0.2, f"t* = {ts:.2f}", fontsize=8)
    ax.set(xlabel="Normalised search progress t/T", ylabel="Exploration coefficient a",
           title=f"Analytical clipping behaviour (ρ = {rho})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, out, "fig_clipping_analytical")


def fig_reliability(M, out, methods=("fixed", "gwo", "pareto_gwo", "pu_gwo")):
    methods = [m for m in methods if m in M]
    if not methods:
        return
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.2, 6), gridspec_kw=dict(height_ratios=[3, 1]),
                                  sharex=True)
    ax.plot([0, 1], [0, 1], ":", c="k", lw=1, label="Ideal")
    for m in methods:
        p, y, _ = pooled(M, m, sorted(M[m]))
        b = reliability_bins(p, y)
        ax.plot(b[:, 0], b[:, 1], "o-", ms=4, c=COLORS.get(m),
                label=f"{METHOD_LABELS[m]} (ECE {ece(p, y):.3f})")
        if m == methods[-1]:
            axh.hist(p.max(1), bins=15, range=(0, 1), color=COLORS.get(m), alpha=0.7)
    ax.set(ylabel="Empirical accuracy", title="Reliability diagram (pooled test predictions, all folds)",
           xlim=(0.3, 1.0), ylim=(0.0, 1.02))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    axh.set(xlabel="Mean predicted confidence", ylabel=f"Count\n({METHOD_LABELS[methods[-1]]})")
    savefig(fig, out, "fig_reliability")


def fig_risk_coverage(M, out, methods=("fixed", "gwo", "pareto_gwo", "nsga2", "pu_gwo")):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in methods:
        if m not in M:
            continue
        p, y, _ = pooled(M, m, sorted(M[m]))
        r = risk_curve(p, y)
        cov = np.arange(1, len(y) + 1) / len(y)
        k = cov >= 0.5
        ax.plot(100 * cov[k], 100 * r[k], c=COLORS.get(m), label=METHOD_LABELS[m])
    ax.set(xlabel="Coverage (%)", ylabel="Selective risk (%)",
           title="Risk–coverage (pooled test predictions)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, out, "fig_risk_coverage")


def fig_hv(M, out, methods=("gwo", "pareto_gwo", "nsga2", "motpe", "pu_gwo")):
    methods = [m for m in methods if m in M]
    if not methods:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    means = [np.mean([j["hv"] for j in M[m].values()]) for m in methods]
    sds = [np.std([j["hv"] for j in M[m].values()], ddof=1) if len(M[m]) > 1 else 0 for m in methods]
    a1.bar(range(len(methods)), means, yerr=sds, capsize=4, color=[COLORS.get(m) for m in methods])
    a1.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=20)
    a1.set(ylabel="Final validation hypervolume", title="(a) Final hypervolume (mean ± SD)")
    lo = min(means) - 2 * max(sds + [0.01])
    a1.set_ylim(max(0, lo), None)
    for m in methods:
        curves = [np.array(j["curve"]) for j in M[m].values() if j["curve"]]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        C = np.stack([c[:n] for c in curves])
        x, mu = C[0, :, 0], C[:, :, 1].mean(0)
        sd = C[:, :, 1].std(0, ddof=1) if len(C) > 1 else np.zeros_like(mu)
        a2.plot(x, mu, "o-", ms=3, c=COLORS.get(m), label=METHOD_LABELS[m])
        a2.fill_between(x, mu - sd, mu + sd, color=COLORS.get(m), alpha=0.15)
    a2.set(xlabel="Candidate evaluations", ylabel="Validation hypervolume",
           title="(b) Hypervolume after each batch (logged)")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    savefig(fig, out, "fig_hypervolume")


def fig_ablation(M, out):
    chain = [m for m in ABLATION if m in M]
    rho = [(m, r) for m, r in RHO if m in M]
    if len(chain) < 2 and len(rho) < 2:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    if len(chain) >= 2:
        def stat(m, key, f):
            v = [j["metrics"][key] for j in M[m].values()]
            return f(v)
        xs = [stat(m, "aurc", np.mean) for m in chain]
        ys = [100 * stat(m, "macro_f1", np.mean) for m in chain]
        xe = [stat(m, "aurc", lambda v: np.std(v, ddof=1)) if len(M[m]) > 1 else 0 for m in chain]
        ye = [100 * stat(m, "macro_f1", lambda v: np.std(v, ddof=1)) if len(M[m]) > 1 else 0 for m in chain]
        a1.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o-", capsize=3)
        for x, y, m in zip(xs, ys, chain):
            a1.annotate(ABLATION_LABELS[m], (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        a1.invert_xaxis()
        a1.set(xlabel="Test AURC (lower is better)", ylabel="Test Macro-F1 (%)",
               title="Cumulative ablation (mean ± SD over folds)")
        a1.grid(alpha=0.3)
    if len(rho) >= 2:
        r = [x for _, x in rho]
        f1 = [100 * np.mean([j["metrics"]["macro_f1"] for j in M[m].values()]) for m, _ in rho]
        au = [np.mean([j["metrics"]["aurc"] for j in M[m].values()]) for m, _ in rho]
        a2.plot(r, f1, "o-", label="Macro-F1 (%)")
        a2.set(xlabel="Uncertainty influence ρ", ylabel="Test Macro-F1 (%)", title="Sensitivity to ρ")
        a3 = a2.twinx()
        a3.plot(r, au, "s--", c="C1", label="AURC")
        a3.set_ylabel("Test AURC")
        a2.legend(loc="upper left", fontsize=8)
        a3.legend(loc="upper right", fontsize=8)
        a2.grid(alpha=0.3)
    savefig(fig, out, "fig_ablation_rho")


def fig_confusion(M, out, method="pu_gwo"):
    if method not in M:
        return None
    p, y, _ = pooled(M, method, sorted(M[method]))
    C = np.zeros((3, 3), int)
    for t, q in zip(y, p.argmax(1)):
        C[t, q] += 1
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(C, cmap="Blues")
    for i in range(3):
        for k in range(3):
            ax.text(k, i, C[i, k], ha="center", va="center",
                    color="white" if C[i, k] > C.max() / 2 else "black")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"{METHOD_LABELS[method]} (pooled over {len(M[method])} folds)")
    savefig(fig, out, "fig_confusion_pugwo")
    return C


def fig_temperature(M, out, methods=("fixed", "gwo", "pareto_gwo", "nsga2", "pu_gwo")):
    methods = [m for m in methods if m in M]
    if not methods:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(methods))
    for ax, key, title in zip(axs, ("nll", "ece"), ("Test NLL", "Test ECE")):
        pre = [np.mean([j["metrics"][key] for j in M[m].values()]) for m in methods]
        post = [np.mean([j["metrics"][key + "_ts"] for j in M[m].values()]) for m in methods]
        ax.bar(x - 0.2, pre, 0.4, label="As trained")
        ax.bar(x + 0.2, post, 0.4, label="+ temperature scaling")
        ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=20)
        ax.set_title(title)
        ax.legend(fontsize=8)
    savefig(fig, out, "fig_temperature_scaling")


# ------------------------------------------------------------------------------ main
def main(run_dir, out, n_boot=1000):
    S = load_settings(run_dir)
    os.makedirs(out, exist_ok=True)
    failures = []

    def step(label, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            tb = traceback.format_exc()
            failures.append(f"## {label}\n```\n{tb}```")
            say(f"  !! {label} FAILED (details in ANALYSIS_NOTES.md):\n{tb.strip().splitlines()[-1]}")
            plt.close("all")
            return None

    say("loading jobs ...")
    jobs = load_jobs(S)
    M = by_method(jobs)
    say(f"loaded {len(jobs)} completed jobs")
    notes = [f"# Analysis notes\n\nCompleted jobs: {len(jobs)}\n"]
    for m in sorted(M):
        notes.append(f"- {METHOD_LABELS.get(m, m)}: folds {sorted(M[m])}")

    def table1():
        _, labels, pids, _ = load_cache(S.cache)
        folds = load_splits(os.path.join(run_dir, "splits.json"))
        write_table(pd.DataFrame(split_table(folds, labels, pids)), out, "table1_split_composition",
                    "Patient-grouped fold composition (patients and images per class).")

    def table2():
        rows = []
        for spec in SPACE:
            rng = (f"{{{', '.join(map(str, spec[2]))}}}" if spec[1] == "cat"
                   else f"{spec[2]:g}–{spec[3]:g} ({spec[1]})")
            rows.append({"Quantity": spec[0], "Range or value": rng})
        for k in ("budget", "pop", "promote_k", "s1_epochs", "s1_patience", "s2_epochs", "s2_patience",
                  "final_epochs", "final_patience", "mc_search", "mc_final", "archive_cap", "tau",
                  "mu_min", "mu_max", "n_folds"):
            rows.append({"Quantity": k, "Range or value": str(getattr(S, k))})
        rows.append({"Quantity": "HV bounds (1−F1, NLL, AURC)",
                     "Range or value": f"{S.hv_lo} – {S.hv_hi}, ref {S.hv_ref}"})
        write_table(pd.DataFrame(rows), out, "table2_search_space", "Search space and fixed settings.")

    say("tables ...")
    step("table1", table1)
    step("table2", table2)
    step("fig_clipping", fig_clipping, out)

    if not jobs:
        with open(os.path.join(out, "ANALYSIS_NOTES.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(notes) + "\nNo completed jobs yet.\n")
        return

    main_spec = [("Macro-F1 (%) ↑", "macro_f1", 100, 2), ("MCC ↑", "mcc", 1, 3), ("NLL ↓", "nll", 1, 3),
                 ("ECE ↓", "ece", 1, 3), ("AURC ↓", "aurc", 1, 4), ("E-AURC ↓", "eaurc", 1, 4),
                 ("Error AUROC ↑", "err_auroc", 1, 3)]
    step("table3", lambda: write_table(metric_table(M, MAIN, main_spec), out, "table3_main_results",
                                       "Test performance, mean ± SD over patient-grouped folds."))
    step("table3b", lambda: write_table(
        metric_table(M, MAIN, [("NLL", "nll", 1, 3), ("NLL + TS", "nll_ts", 1, 3),
                               ("ECE", "ece", 1, 3), ("ECE + TS", "ece_ts", 1, 3),
                               ("AURC", "aurc", 1, 4), ("AURC + TS", "aurc_ts", 1, 4),
                               ("Temperature", "T", 1, 2)]),
        out, "table3b_temperature_scaling",
        "Effect of post-hoc temperature scaling fitted on the validation fold."))
    step("table4", lambda: write_table(
        metric_table(M, MAIN, [("Accuracy (%)", "acc", 100, 2), ("Balanced acc. (%)", "bal_acc", 100, 2),
                               ("Macro-precision (%)", "macro_prec", 100, 2),
                               ("Macro-recall (%)", "macro_rec", 100, 2),
                               ("Patient acc. (%)", "patient_acc", 100, 2),
                               ("Patient Macro-F1 (%)", "patient_macro_f1", 100, 2)]),
        out, "table4_secondary_metrics", "Secondary image-level and patient-level metrics."))
    step("table5", lambda: write_table(
        metric_table(M, MAIN, [(f"{c}% coverage", f"risk{c}", 100, 2) for c in (100, 95, 90, 80)]),
        out, "table5_risk_at_coverage", "Classification risk (%) at fixed coverage."))
    abl_spec = [("Macro-F1 (%)", "macro_f1", 100, 2), ("NLL", "nll", 1, 3), ("AURC", "aurc", 1, 4),
                ("Val. HV", "hv", 1, 3)]
    step("table6", lambda: write_table(metric_table(M, ABLATION, abl_spec, ABLATION_LABELS), out,
                                       "table6_cumulative_ablation", "Cumulative component ablation."))
    step("table7", lambda: write_table(
        metric_table(M, CONTROLS, abl_spec[:3] + [("ECE", "ece", 1, 3), ("E-AURC", "eaurc", 1, 4),
                                                  ("Val. HV", "hv", 1, 3)]),
        out, "table7_controls", "Controls for the uncertainty-guided movement and leave-one-out ablations."))

    def table7b():
        rho_rows = metric_table(M, [m for m, _ in RHO], abl_spec)
        if len(rho_rows):
            rho_map = {METHOD_LABELS[m]: r for m, r in RHO}
            rho_rows.insert(1, "ρ", [rho_map[x] for x in rho_rows["Method"]])
            write_table(rho_rows, out, "table7b_rho_sensitivity", "Sensitivity to the uncertainty influence ρ.")
    step("table7b", table7b)

    def table8():
        C = fig_confusion(M, out)
        if C is not None:
            rows = [{"Class": CLASSES[c], "Correct / total": f"{C[c, c]}/{C[c].sum()}",
                     "Sensitivity (%)": f"{100 * C[c, c] / C[c].sum():.1f}"} for c in range(3)]
            write_table(pd.DataFrame(rows), out, "table8_class_sensitivity",
                        "PU-GWO class sensitivity from the confusion matrix pooled over all folds.")
    step("table8 + confusion figure", table8)

    step("table9", lambda: write_table(
        metric_table(M, MAIN + [m for m in CONTROLS if m not in MAIN],
                     [("Search GPU-h", "search_h", 1, 2), ("Final GPU-h", "final_h", 1, 2),
                      ("Epochs trained", "epochs", 1, 0), ("Peak mem. (GB)", "peak_gb", 1, 1)]),
        out, "table9_compute_cost", "Measured compute per run (mean ± SD over folds)."))

    def table10():
        rows = []
        for m in MAIN:
            for f, j in sorted(M.get(m, {}).items()):
                c = j["config"]
                rows.append({"Method": METHOD_LABELS[m], "Fold": f, "LR": f"{c['lr']:.2e}",
                             "Dropout": f"{c['dropout']:.2f}", "WD": f"{c['weight_decay']:.2e}",
                             "Batch": c["batch_size"], "Unfrozen": c["unfrozen_stages"],
                             "Label sm.": f"{c['label_smoothing']:.3f}"})
        write_table(pd.DataFrame(rows), out, "table10_selected_configs",
                    "Selected configuration per method and fold.")
    step("table10", table10)

    def table_hv():
        rows = []
        for m in [x for x in MAIN + CONTROLS + ["abl_aurc", "pu_rho025", "pu_rho075"]
                  if x in M and x != "fixed"]:
            for j in M[m].values():
                if not j["curve"]:
                    continue
                c = np.array(j["curve"])
                k = int(np.argmax(c[:, 1] >= 0.95 * c[-1, 1]))
                rows.append(dict(method=m, fold=j["fold"], hv=c[-1, 1], evals95=c[k, 0]))
        if rows:
            d = pd.DataFrame(rows).drop_duplicates(["method", "fold"])
            write_table(pd.DataFrame([{"Method": METHOD_LABELS[m], "Final val. HV": ms(g.hv, 1, 3),
                                       "Evals to 95% of own final HV": ms(g.evals95, 1, 0)}
                                      for m, g in d.groupby("method", sort=False)]),
                        out, "table_hv_summary", "Validation hypervolume summary (from logged batches).")
    step("table_hv_summary", table_hv)

    say("figures ...")
    step("fig_reliability", fig_reliability, M, out)
    step("fig_risk_coverage", fig_risk_coverage, M, out)
    step("fig_hypervolume", fig_hv, M, out)
    step("fig_ablation_rho", fig_ablation, M, out)
    step("fig_temperature_scaling", fig_temperature, M, out)

    say(f"paired statistics with {n_boot} patient-level bootstrap resamples (slowest step) ...")
    others = [m for m in ["pareto_gwo", "pu_random_u", "pu_shuffled_u", "pu_rho0", "pu_nll_u", "nsga2",
                          "motpe", "gwo", "tpe", "random", "pso", "fixed"] if m in M]
    if "pu_gwo" in M:
        step("table11 paired statistics", paired_stats, M, "pu_gwo", others, n_boot, out)

    notes.append("\nAll values are computed from preds.npz and evals.jsonl. ECE in the reliability "
                 "figure is computed on pooled predictions, so it differs slightly from the fold mean in "
                 "Table 3. With 5 folds the smallest possible two-sided Wilcoxon p is 0.0625; rely on the "
                 "bootstrap CIs for uncertainty.")
    if failures:
        notes.append("\n# Steps that failed\n")
        notes.extend(failures)
    with open(os.path.join(out, "ANALYSIS_NOTES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")
    if failures:
        say(f"\nfinished with {len(failures)} failed step(s); see {out}/ANALYSIS_NOTES.md")
    else:
        say(f"\nwrote all tables and figures to {out}/")


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else "runs"
    od = sys.argv[2] if len(sys.argv) > 2 else "results"
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    main(rd, od, n_boot=nb)