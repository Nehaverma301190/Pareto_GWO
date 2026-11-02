"""Run one (method, fold) job: resumable search, promotion, final selection, retraining, test prediction."""
import glob
import json
import logging
import os
import time

import numpy as np

from config import DEFAULT_CONFIG, METHODS, decode
from engine import predict_mc, run_training
from metrics import crowding, ece, ideal_point_select, mc_probs, nd_ranks, objectives
from optimizers import make_optimizer

log = logging.getLogger("pugwo")


class JsonlLog:
    def __init__(self, path):
        self.path = path

    def read(self):
        recs = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # partially written last line after a crash
        return recs

    def append(self, rec):
        rec = dict(rec, time=time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _promotion(F, selector, k):
    """Indices (within batch) promoted to stage 2: the same count k for every method."""
    F = np.asarray(F)
    if selector == "scalar":
        order = np.lexsort((F[:, 1], F[:, 0]))
    else:
        r, cd = nd_ranks(F), crowding(F)
        order = sorted(range(len(F)), key=lambda i: (r[i], -cd[i]))
    return [int(i) for i in list(order)[:k]]


def _select(F, selector):
    F = np.asarray(F)
    if selector == "scalar":
        return int(np.lexsort((F[:, 1], F[:, 0]))[0])
    return ideal_point_select(F)


def run_job(method, fold, S, data, splits, device, job_dir, heartbeat=None):
    spec = METHODS[method]
    split = splits[fold]
    os.makedirs(os.path.join(job_dir, "ckpt"), exist_ok=True)
    for tmp in glob.glob(os.path.join(job_dir, "**", "*.tmp"), recursive=True):
        os.remove(tmp)  # partial checkpoint writes from an interrupted run
    L = JsonlLog(os.path.join(job_dir, "evals.jsonl"))
    recs = L.read()
    yv = data.y.cpu().numpy()[np.asarray(split["val"])]
    fold_seed = S.base_seed + 1000 * fold  # shared across methods -> paired comparisons

    def cand_seed(c):
        return S.base_seed * 100000 + fold * 1000 + c

    def find(**kw):
        for r in recs:
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    def evaluate(c, b, stage, u, cfg, target, patience):
        ck = os.path.join(job_dir, "ckpt", f"c{c:04d}.pt")
        res = run_training(cfg, S, data, split, cand_seed(c), target, S.s2_epochs, patience,
                           S.mc_search, device, ckpt_path=ck, heartbeat=heartbeat)
        p = mc_probs(res["val_logits"])
        obj = objectives(p, yv)
        rec = dict(type="eval", cand=c, batch=b, stage=stage, u=[float(v) for v in u], config=cfg,
                   f1_loss=obj[0], nll=obj[1], aurc=obj[2], val_ece=ece(p, yv),
                   epochs_run=res["epochs_run"], epochs_this_stage=res["epochs_this_call"],
                   stopped=res["stopped"], best_epoch=res["best_epoch"],
                   gpu_sec=res["gpu_sec"], gpu_sec_cum=res["gpu_sec_total"],
                   peak_mem_gb=res["peak_mem_gb"])
        L.append(rec)
        recs.append(rec)
        log.info(f"[{method} f{fold}] cand {c:3d} s{stage} ep={res['epochs_run']:2d} "
                 f"F1={1 - obj[0]:.4f} NLL={obj[1]:.4f} AURC={obj[2]:.4f} "
                 f"({res['gpu_sec']:.0f}s) {cfg}")
        return rec

    # --------------------------------------------------------------------------- search
    if spec["kind"] == "fixed":
        cfg, sel = dict(DEFAULT_CONFIG), None
    else:
        opt = make_optimizer(spec, S, fold_seed)
        latest = {}
        for b in range(S.budget // S.pop):
            X = opt.ask()
            logged = find(type="batch_start", batch=b)
            if logged is None:
                L.append(dict(type="batch_start", batch=b, X=X.tolist()))
                recs.append(dict(type="batch_start", batch=b, X=X.tolist()))
            elif not np.allclose(np.array(logged["X"]), X, atol=1e-9):
                raise RuntimeError(f"Replay mismatch in {job_dir} batch {b}. The optimiser or library "
                                   f"version changed; delete this job directory to restart the job.")
            cands = [b * S.pop + i for i in range(S.pop)]
            for i, c in enumerate(cands):
                r = find(type="eval", cand=c, stage=1)
                if r is None:
                    r = evaluate(c, b, 1, X[i], decode(X[i]), S.s1_epochs, S.s1_patience)
                latest[c] = r
            pr = find(type="promotion", batch=b)
            if pr is None:
                F1 = [[latest[c]["f1_loss"], latest[c]["nll"], latest[c]["aurc"]] for c in cands]
                ids = [cands[i] for i in _promotion(F1, spec["selector"], S.promote_k)]
                pr = dict(type="promotion", batch=b, cands=ids)
                L.append(pr)
                recs.append(pr)
            for c in pr["cands"]:
                r2 = find(type="eval", cand=c, stage=2)
                if r2 is None:
                    r1 = latest[c]
                    if r1["stopped"]:  # already converged in stage 1: nothing to continue
                        r2 = dict(r1, stage=2, epochs_this_stage=0, gpu_sec=0.0, continued=False)
                        L.append(r2)
                        recs.append(r2)
                    else:
                        i = cands.index(c)
                        r2 = evaluate(c, b, 2, X[i], decode(X[i]), S.s2_epochs, S.s2_patience)
                latest[c] = r2
            for p in glob.glob(os.path.join(job_dir, "ckpt", "c*.pt*")):
                os.remove(p)
            F = np.array([[latest[c]["f1_loss"], latest[c]["nll"], latest[c]["aurc"]] for c in cands])
            opt.tell(X, F)
            if find(type="batch_end", batch=b) is None:
                L.append(dict(type="batch_end", batch=b))
                recs.append(dict(type="batch_end", batch=b))
        allc = sorted(latest)
        Fall = [[latest[c]["f1_loss"], latest[c]["nll"], latest[c]["aurc"]] for c in allc]
        sel = allc[_select(Fall, spec["selector"])]
        cfg = latest[sel]["config"]

    # --------------------------------------------------------------------------- final model
    ck = os.path.join(job_dir, "final_ckpt.pt")
    log.info(f"[{method} f{fold}] final retraining with {cfg}")
    res = run_training(cfg, S, data, split, S.base_seed + 1000 * fold + 777, S.final_epochs,
                       S.final_epochs, S.final_patience, S.mc_final, device, ckpt_path=ck,
                       save_every_epoch=True, keep_best_weights=True, heartbeat=heartbeat)
    amp = device.type == "cuda"
    model = res["model"]
    val_logits = predict_mc(model, data, split["val"], S.mc_final, S.eval_batch, device, amp)
    test_logits = predict_mc(model, data, split["test"], S.mc_final, S.eval_batch, device, amp)
    y = data.y.cpu().numpy()
    np.savez_compressed(os.path.join(job_dir, "preds.npz"), val_logits=val_logits,
                        test_logits=test_logits, val_idx=np.array(split["val"]),
                        test_idx=np.array(split["test"]), y_val=y[split["val"]],
                        y_test=y[split["test"]])
    final = dict(method=method, fold=fold, selected_cand=sel, config=cfg,
                 final_epochs_run=res["epochs_run"], final_best_epoch=res["best_epoch"],
                 final_gpu_sec=res["gpu_sec_total"], final_peak_mem_gb=res["peak_mem_gb"])
    if sel is not None:
        r = [x for x in recs if x.get("type") == "eval" and x["cand"] == sel][-1]
        final.update(val_f1_loss=r["f1_loss"], val_nll=r["nll"], val_aurc=r["aurc"])
    with open(os.path.join(job_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    if os.path.exists(ck):
        os.remove(ck)
    with open(os.path.join(job_dir, "done.json"), "w") as f:
        json.dump(dict(finished=time.time()), f)
    log.info(f"[{method} f{fold}] done")
