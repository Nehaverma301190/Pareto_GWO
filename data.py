"""Figshare brain-tumour data: preprocessing cache, patient-grouped folds, GPU-resident batches."""
import glob
import json
import math
import os
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from config import CLASSES

# Figshare labels: 1 = meningioma, 2 = glioma, 3 = pituitary  ->  0 glioma, 1 meningioma, 2 pituitary
FIGSHARE_TO_IDX = {2: 0, 1: 1, 3: 2}


def _extract_zips(raw_dir):
    changed = True
    while changed:
        changed = False
        for z in glob.glob(os.path.join(raw_dir, "**", "*.zip"), recursive=True):
            marker = z + ".extracted"
            if os.path.exists(marker):
                continue
            print(f"extracting {z}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(os.path.splitext(z)[0])
            open(marker, "w").close()
            changed = True


def _read_mat(path):
    import h5py
    with h5py.File(path, "r") as f:
        cj = f["cjdata"]
        label = int(np.array(cj["label"]).squeeze())
        pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).flatten() if int(c) != 0).strip()
        img = np.array(cj["image"], dtype=np.float32).T  # MATLAB column-major
    return label, pid, img


def _to_uint8(img, size):
    img = img - img.min()
    img = img / max(float(img.max()), 1e-8)
    t = torch.from_numpy(img)[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return (t[0, 0].clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def prepare_figshare(raw_dir, out_path, size):
    _extract_zips(raw_dir)
    files = [p for p in glob.glob(os.path.join(raw_dir, "**", "*.mat"), recursive=True)
             if os.path.basename(p).lower() != "cvind.mat"]
    if not files:
        raise SystemExit(f"No .mat files found under {raw_dir}")

    def key(p):
        b = os.path.splitext(os.path.basename(p))[0]
        return (0, int(b)) if b.isdigit() else (1, b)

    files = sorted(set(files), key=key)
    imgs, labels, pids, ids = [], [], [], []
    for i, p in enumerate(files):
        lab, pid, img = _read_mat(p)
        imgs.append(_to_uint8(img, size))
        labels.append(FIGSHARE_TO_IDX[lab])
        pids.append(pid)
        ids.append(os.path.splitext(os.path.basename(p))[0])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}")
    _save_cache(out_path, np.stack(imgs), np.array(labels), np.array(pids), np.array(ids))


def make_fake(out_path, size=32, n_patients=36, per_patient=8, seed=0):
    """Synthetic data with a learnable class signal, for pipeline smoke tests only."""
    rng = np.random.default_rng(seed)
    imgs, labels, pids, ids = [], [], [], []
    yy, xx = np.mgrid[0:size, 0:size] / size
    centers = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
    k = 0
    for p in range(n_patients):
        c = p % 3
        for _ in range(per_patient):
            cy, cx = centers[c]
            cy += rng.normal(0, 0.08)
            cx += rng.normal(0, 0.08)
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 0.01)
            img = 0.6 * blob + 0.4 * rng.random((size, size))
            imgs.append((img / img.max() * 255).astype(np.uint8))
            labels.append(c)
            pids.append(f"P{p:03d}")
            ids.append(str(k))
            k += 1
    _save_cache(out_path, np.stack(imgs), np.array(labels), np.array(pids), np.array(ids))


def _save_cache(out_path, images, labels, pids, ids):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, images=images, labels=labels, pids=pids, ids=ids)
    print(f"saved {out_path}: {len(labels)} images, {len(np.unique(pids))} patients")
    for c, name in enumerate(CLASSES):
        m = labels == c
        print(f"  {name:11s} images={int(m.sum()):5d} patients={len(np.unique(pids[m]))}")


def load_cache(path):
    d = np.load(path, allow_pickle=False)
    return d["images"], d["labels"].astype(np.int64), d["pids"].astype(str), d["ids"].astype(str)


def make_splits(labels, pids, n_folds, seed):
    """Outer StratifiedGroupKFold for test; inner grouped split of the rest into train/val."""
    from sklearn.model_selection import StratifiedGroupKFold
    idx = np.arange(len(labels))
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for f, (trval, test) in enumerate(outer.split(idx, labels, groups=pids)):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1 + f)
        tr_rel, va_rel = next(inner.split(trval, labels[trval], groups=pids[trval]))
        tr, va = trval[tr_rel], trval[va_rel]
        assert not (set(pids[tr]) & set(pids[va]) or set(pids[tr]) & set(pids[test])
                    or set(pids[va]) & set(pids[test])), "patient leakage"
        folds.append(dict(train=tr.tolist(), val=va.tolist(), test=test.tolist()))
    return folds


def save_splits(path, folds):
    with open(path, "w") as f:
        json.dump(folds, f)


def load_splits(path):
    with open(path) as f:
        return json.load(f)


def split_table(folds, labels, pids):
    rows = []
    for f, sp in enumerate(folds):
        for part in ("train", "val", "test"):
            ix = np.array(sp[part])
            row = dict(fold=f, partition=part, patients=len(np.unique(pids[ix])), images=len(ix))
            for c, name in enumerate(CLASSES):
                m = labels[ix] == c
                row[f"{name}_patients"] = len(np.unique(pids[ix][m]))
                row[f"{name}_images"] = int(m.sum())
            rows.append(row)
    return rows


class GPUData:
    """Keeps all uint8 images on the device; augmentation runs batched on the device."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, images, labels, device):
        self.device = device
        self.x = torch.from_numpy(images).to(device)
        self.y = torch.from_numpy(labels).to(device)
        self.mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1)

    def batch(self, idx, augment):
        x = self.x[idx].float().div_(255.0).unsqueeze(1)
        if augment:
            x = gpu_augment(x)
        x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last)


def gpu_augment(x, max_rot_deg=10.0, max_shift=0.05, scale=(0.9, 1.1), p_flip=0.5):
    B, dev = x.shape[0], x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    sc = scale[0] + torch.rand(B, device=dev) * (scale[1] - scale[0])
    # affine_grid uses coordinates in [-1, 1], so a 5% shift of the width is 0.1
    tx = (torch.rand(B, device=dev) * 2 - 1) * max_shift * 2
    ty = (torch.rand(B, device=dev) * 2 - 1) * max_shift * 2
    flip = torch.where(torch.rand(B, device=dev) < p_flip, -1.0, 1.0)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1),
                         torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
