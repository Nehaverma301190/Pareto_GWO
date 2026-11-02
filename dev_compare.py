"""Development step: compare redesign candidates on VALIDATION data only.

Usage:
  python dev_compare.py --run-dir runs [--folds 0 1 2 3 4]

Reads final.json, the validation logits in preds.npz and the search logs. It never reads
the test arrays, so the design choice cannot be influenced by test results.

Decision rule (fixed before running; report it in the paper):
  Among R1, R2 and R3, pick the variant with the lowest mean validation AURC of the final model,
  provided its mean validation Macro-F1 is no more than 0.5 points below PU-GWO with rho = 0.
  If no variant has lower mean validation AURC than BOTH the original PU-GWO and rho = 0,
  none is taken forward and the paper is written from the first experiment (option A).
"""
import argparse
import json
import os

import numpy as np

from metrics import aurc, eaurc, hypervolume3d, mc_probs, nll
from config import METHOD_LABELS, load_settings

REFERENCE = ["pareto_gwo", "pu_gwo", "pu_rho0"]
CANDIDATES = ["pu_r1_rank", "pu_r2_lead", "pu_r3_both"]


def macro_f1_np(p, y):
    yh = p.argmax(1)
    f1s = []
    for c in (0, 1, 2):
        tp = np.sum((yh == c) & (y == c))
        fp = np.sum((yh == c) & (y != c))
        fn = np.sum((yh != c) & (y == c))
        d = 2 * tp + fp + fn
        f1s.append(0.0 if d == 0 else 2.0 * tp / d)
    return float(np.mean(f1s))


def val_hv(job_dir, S):
    latest = {}
    with open(os.path.join(job_dir, "evals.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "eval":
                latest[r["cand"]] = (r["f1_loss"], r["nll"], r["aurc"])
    return hypervolume3d(np.array(list(latest.values())), S.hv_lo, S.hv_hi, S.hv_ref)


def load(run_dir, S, method, fold):
    jd = os.path.join(run_dir, f"{method}__fold{fold}")
    if not os.path.exists(os.path.join(jd, "done.json")):
        return None
    with np.load(os.path.join(jd, "preds.npz")) as P:   # only validation keys are touched
        p = mc_probs(P["val_logits"])
        y = P["y_val"]
    return dict(f1=100 * macro_f1_np(p, y), nll=nll(p, y), aurc=aurc(p, y), eaurc=eaurc(p, y),
                hv=val_hv(jd, S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    S = load_settings(a.run_dir)

    R = {}
    for m in REFERENCE + CANDIDATES:
        rows = [load(a.run_dir, S, m, f) for f in a.folds]
        R[m] = [r for r in rows if r is not None]

    print(f"VALIDATION metrics of the final models, folds {a.folds} (test data not read)\n")
    print(f"{'method':38s} {'n':>2s} {'Macro-F1':>9s} {'NLL':>7s} {'AURC':>8s} {'E-AURC':>8s} {'val HV':>7s}")
    means = {}
    for m in REFERENCE + CANDIDATES:
        rs = R[m]
        if not rs:
            print(f"{METHOD_LABELS[m]:38s}  0   (not run yet)")
            continue
        mu = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        means[m] = mu
        print(f"{METHOD_LABELS[m]:38s} {len(rs):2d} {mu['f1']:9.2f} {mu['nll']:7.3f} "
              f"{mu['aurc']:8.4f} {mu['eaurc']:8.4f} {mu['hv']:7.3f}")

    missing = [m for m in REFERENCE + CANDIDATES if len(R[m]) < len(a.folds)]
    if missing:
        print(f"\nNot all folds finished for: {', '.join(missing)}. Decide only when all are done.")
        return

    print("\nDecision rule (fixed in advance):")
    base_f1 = means["pu_rho0"]["f1"]
    bar = min(means["pu_gwo"]["aurc"], means["pu_rho0"]["aurc"])
    eligible = [m for m in CANDIDATES
                if means[m]["aurc"] < bar and means[m]["f1"] >= base_f1 - 0.5]
    for m in CANDIDATES:
        ok_a = means[m]["aurc"] < bar
        ok_f = means[m]["f1"] >= base_f1 - 0.5
        print(f"  {METHOD_LABELS[m]:38s} AURC below {bar:.4f}: {'yes' if ok_a else 'no'}; "
              f"Macro-F1 within 0.5 of rho=0 ({base_f1:.2f}): {'yes' if ok_f else 'no'}")
    if eligible:
        best = min(eligible, key=lambda m: means[m]["aurc"])
        print(f"\nSELECTED for confirmation: {best}  ({METHOD_LABELS[best]})")
        print("Freeze this choice now. Do not change it after seeing confirmation results.")
    else:
        print("\nNo redesign met the rule. Do not run the confirmation; write the paper from the "
              "first experiment (option A).")


if __name__ == "__main__":
    main()
