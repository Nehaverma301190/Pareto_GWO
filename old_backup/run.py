"""Command-line entry point.

  python run.py prepare  --raw /path/to/figshare --run-dir runs [--profile default|paper|smoke]
  python run.py benchmark --run-dir runs
  python run.py worker   --run-dir runs [--tiers 1 2] [--methods pu_gwo ...] [--folds 0 1]
  python run.py status   --run-dir runs
  python run.py analyze  --run-dir runs --out results
"""
import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
import traceback

import numpy as np

from config import (DEFAULT_CONFIG, TIERS, job_id, job_list, load_settings, make_settings,
                    save_settings)

log = logging.getLogger("pugwo")


def setup_logging(run_dir, name):
    os.makedirs(run_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(run_dir, f"{name}.log"))):
        h.setFormatter(fmt)
        log.addHandler(h)


def get_device(S):
    import torch
    if S.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Check the PyTorch install (needs a CUDA build).")
    dev = torch.device(S.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return dev


def load_everything(S):
    from data import GPUData, load_cache, load_splits
    dev = get_device(S)
    images, labels, pids, ids = load_cache(S.cache)
    splits = load_splits(os.path.join(S.run_dir, "splits.json"))
    return dev, GPUData(images, labels, dev), splits, (labels, pids, ids)


# ------------------------------------------------------------------------------------ prepare
def cmd_prepare(a):
    from data import load_cache, make_fake, make_splits, prepare_figshare, save_splits, split_table
    S = make_settings(a.profile, run_dir=a.run_dir, cache=a.cache)
    if os.path.exists(os.path.join(S.run_dir, "settings.json")) and not a.force:
        raise SystemExit(f"{S.run_dir}/settings.json exists. Use --force only if no jobs have run yet.")
    if not os.path.exists(S.cache):
        if a.profile == "smoke":
            make_fake(S.cache, size=S.img_size)
        else:
            if not a.raw:
                raise SystemExit("--raw is required to build the cache")
            prepare_figshare(a.raw, S.cache, S.img_size)
    _, labels, pids, _ = load_cache(S.cache)
    folds = make_splits(labels, pids, S.n_folds, S.split_seed)
    save_settings(S)
    save_splits(os.path.join(S.run_dir, "splits.json"), folds)
    print("\nfold partition patients images  (glioma/meningioma/pituitary patients)")
    for r in split_table(folds, labels, pids):
        print(f"{r['fold']:4d} {r['partition']:9s} {r['patients']:8d} {r['images']:6d}  "
              f"{r['glioma_patients']}/{r['meningioma_patients']}/{r['pituitary_patients']}")
    print(f"\nsettings saved to {S.run_dir}/settings.json")


# ------------------------------------------------------------------------------------ benchmark
def cmd_benchmark(a):
    from engine import run_training
    S = load_settings(a.run_dir)
    setup_logging(S.run_dir, "benchmark")
    dev, data, splits, _ = load_everything(S)
    times = {}
    for bs in (16, 32, 64):
        for u in (1, 4):
            cfg = dict(DEFAULT_CONFIG, batch_size=bs, unfrozen_stages=u)
            run_training(cfg, S, data, splits[0], 0, 1, S.s2_epochs, 99, S.mc_search, dev)  # warm-up
            r = run_training(cfg, S, data, splits[0], 0, 2, S.s2_epochs, 99, S.mc_search, dev)
            times[(bs, u)] = r["gpu_sec"] / 2
            log.info(f"batch {bs:2d} unfrozen {u}: {times[(bs, u)]:.1f} s/epoch, "
                     f"peak {r['peak_mem_gb']:.1f} GB")
    ep = float(np.mean(list(times.values())))
    # rough assumption: early stopping ends ~80% of the way through each stage
    cand_s1 = 0.8 * S.s1_epochs * ep + 3
    cand_s2 = 0.8 * (S.s2_epochs - S.s1_epochs) * ep
    search = S.budget * cand_s1 + (S.budget // S.pop) * S.promote_k * cand_s2
    final = 0.8 * S.final_epochs * ep
    per_run = (search + final) / 3600
    log.info(f"\nmean {ep:.1f} s/epoch -> about {per_run:.2f} GPU-hours per search run "
             f"({final / 3600:.2f} h for the fixed baseline)")
    total = 0.0
    for t in sorted(TIERS):
        n = len(TIERS[t]) * S.n_folds
        h = sum((final / 3600 if m == "fixed" else per_run) for m in TIERS[t]) * S.n_folds
        total += h
        log.info(f"tier {t}: {n:3d} jobs, ~{h:5.1f} h (cumulative ~{total:5.1f} h with one worker)")
    log.info("These are estimates. `python run.py status` reports real throughput once jobs finish.")


# ------------------------------------------------------------------------------------ worker
def _pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return True
    if os.name == "nt":
        # os.kill(pid, 0) would TERMINATE the process on Windows, so query it instead.
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def try_claim(jd, stale_after=7200):
    os.makedirs(jd, exist_ok=True)
    lock = os.path.join(jd, "lock")
    if os.path.exists(lock):
        try:
            with open(lock) as f:
                info = json.load(f)
        except Exception:
            info = {}
        stale = (time.time() - os.path.getmtime(lock) > stale_after) or (
            info.get("host") == socket.gethostname() and not _pid_alive(info.get("pid")))
        if not stale:
            return False
        try:
            os.remove(lock)
        except FileNotFoundError:
            pass
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump(dict(host=socket.gethostname(), pid=os.getpid(), start=time.time()), f)
    return True


def release(jd):
    try:
        os.remove(os.path.join(jd, "lock"))
    except FileNotFoundError:
        pass


def free_gb(path):
    import shutil
    return shutil.disk_usage(path).free / 2 ** 30


def _is_disk_error(exc):
    msg = str(exc)
    return (isinstance(exc, OSError) and getattr(exc, "errno", None) == 28) or \
        "file write failed" in msg or "No space left" in msg


def cmd_worker(a):
    from search import run_job
    S = load_settings(a.run_dir)
    setup_logging(S.run_dir, f"worker_{socket.gethostname()}_{os.getpid()}")
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    dev, data, splits, _ = load_everything(S)
    jobs = job_list(S, a.tiers, a.methods, a.folds)
    log.info(f"worker {os.getpid()} on {dev}: {len(jobs)} jobs in scope")
    while True:
        ran = False
        for m, f in jobs:
            jd = os.path.join(S.run_dir, job_id(m, f))
            if os.path.exists(os.path.join(jd, "done.json")):
                continue
            fails = os.path.join(jd, "failures.txt")
            if (not a.retry_failed and os.path.exists(fails)
                    and len(open(fails).read().splitlines()) >= 2):
                continue
            if free_gb(S.run_dir) < a.min_free_gb:
                log.error(f"only {free_gb(S.run_dir):.1f} GB free in {S.run_dir} "
                          f"(need {a.min_free_gb} GB). Free some space and restart the worker.")
                return
            if not try_claim(jd):
                continue
            ran = True
            lock = os.path.join(jd, "lock")
            t0 = time.time()
            try:
                log.info(f"=== start {job_id(m, f)}")
                run_job(m, f, S, data, splits, dev, jd,
                        heartbeat=lambda: os.path.exists(lock) and os.utime(lock))
                log.info(f"=== finished {job_id(m, f)} in {(time.time() - t0) / 3600:.2f} h")
            except Exception as exc:
                if _is_disk_error(exc):
                    log.error("disk full while writing a checkpoint. The job is NOT marked failed; "
                              "free some space and restart the worker to resume it.")
                    release(jd)
                    return
                tb = traceback.format_exc()
                log.error(f"job {job_id(m, f)} failed:\n{tb}")
                with open(fails, "a") as fh:
                    fh.write(time.ctime() + " " + tb.strip().splitlines()[-1] + "\n")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                release(jd)
            break  # rescan from the top so higher-priority jobs are always taken first
        if not ran:
            log.info("no claimable jobs left in scope")
            return


# ------------------------------------------------------------------------------------ status
def cmd_status(a):
    from search import JsonlLog
    S = load_settings(a.run_dir)
    jobs = job_list(S)
    done_h, n_done = [], 0
    print(f"{'job':28s} {'state':10s} {'evals':>5s}")
    for m, f in jobs:
        jd = os.path.join(S.run_dir, job_id(m, f))
        recs = JsonlLog(os.path.join(jd, "evals.jsonl")).read()
        n = sum(1 for r in recs if r.get("type") == "eval" and r.get("stage") == 1)
        if os.path.exists(os.path.join(jd, "done.json")):
            state = "done"
            n_done += 1
            fin = json.load(open(os.path.join(jd, "final.json")))
            if m != "fixed":
                done_h.append((sum(r.get("gpu_sec", 0) for r in recs if r.get("type") == "eval")
                               + fin["final_gpu_sec"]) / 3600)
        elif os.path.exists(os.path.join(jd, "lock")):
            state = "running"
        elif os.path.exists(os.path.join(jd, "failures.txt")):
            state = "failed"
        elif recs:
            state = "partial"
        else:
            state = "pending"
        if state != "pending" or a.all:
            print(f"{job_id(m, f):28s} {state:10s} {n:5d}")
    print(f"\n{n_done}/{len(jobs)} jobs done")
    if done_h:
        mean_h = float(np.mean(done_h))
        left = sum(1 for m, f in jobs if m != "fixed"
                   and not os.path.exists(os.path.join(S.run_dir, job_id(m, f), "done.json")))
        print(f"measured {mean_h:.2f} GPU-h per search run; ~{left * mean_h:.1f} GPU-h for the "
              f"remaining {left} search runs (divide by the number of parallel workers)")


def cmd_analyze(a):
    import analyze
    analyze.main(a.run_dir, a.out, n_boot=a.n_boot)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--raw")
    q.add_argument("--run-dir", default=None)
    q.add_argument("--cache", default=None)
    q.add_argument("--profile", default="default", choices=["default", "paper", "smoke"])
    q.add_argument("--force", action="store_true")
    for name in ("benchmark", "worker", "status", "analyze"):
        q = sub.add_parser(name)
        q.add_argument("--run-dir", default="runs")
        if name == "worker":
            q.add_argument("--tiers", type=int, nargs="*")
            q.add_argument("--methods", nargs="*")
            q.add_argument("--folds", type=int, nargs="*")
            q.add_argument("--min-free-gb", type=float, default=8.0,
                           help="stop cleanly if less free disk space than this")
            q.add_argument("--retry-failed", action="store_true",
                           help="also retry jobs that already failed twice")
        if name == "status":
            q.add_argument("--all", action="store_true")
        if name == "analyze":
            q.add_argument("--out", default="results")
            q.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    {"prepare": cmd_prepare, "benchmark": cmd_benchmark, "worker": cmd_worker,
     "status": cmd_status, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
