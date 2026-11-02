"""Model construction, resumable training with early stopping on validation NLL, MC-dropout prediction."""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from metrics import mc_probs, nll

_PRETRAINED = {}


def pretrained_state(arch, use_pretrained):
    if not use_pretrained:
        return None
    if arch not in _PRETRAINED:
        w = torchvision.models.get_model_weights(arch).DEFAULT
        print(f"loading ImageNet weights {w}")
        _PRETRAINED[arch] = w.get_state_dict(progress=True)
    return _PRETRAINED[arch]


def pretrained_name(arch, use_pretrained):
    return str(torchvision.models.get_model_weights(arch).DEFAULT) if use_pretrained else "none"


class Net(nn.Module):
    def __init__(self, arch, n_classes, dropout, state):
        super().__init__()
        bb = getattr(torchvision.models, arch)(weights=None)
        if state is not None:
            bb.load_state_dict(state)
        feat = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.head = nn.Linear(feat, n_classes)
        self.p = float(dropout)


def configure_trainable(model, unfrozen):
    bb = model.backbone
    layers = [bb.layer1, bb.layer2, bb.layer3, bb.layer4]
    for p in bb.parameters():
        p.requires_grad = False
    for layer in layers[4 - unfrozen:]:
        for p in layer.parameters():
            p.requires_grad = True
    return [bb.conv1, bb.bn1] + layers[: 4 - unfrozen]  # modules kept frozen (BN in eval mode)


def _autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp)


@torch.no_grad()
def predict_mc(model, data, idx, K, bs, device, amp):
    """Returns MC-dropout logits of shape (K, N, C); dropout only in the classifier head."""
    model.eval()
    out = []
    idx_t = torch.as_tensor(idx, device=device)
    for s in range(0, len(idx), bs):
        x = data.batch(idx_t[s:s + bs], augment=False)
        with _autocast(device, amp):
            f = model.backbone(x)
        f = f.float()
        out.append(torch.stack([model.head(F.dropout(f, model.p, training=True)) for _ in range(K)]))
    return torch.cat(out, 1).cpu().numpy().astype(np.float32)


def _save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_training(cfg, S, data, split, seed, target_epochs, sched_epochs, patience, mc_k,
                 device, ckpt_path=None, save_every_epoch=False, keep_best_weights=False,
                 heartbeat=None):
    """Train (or continue training) one configuration up to `target_epochs`.

    The cosine schedule always spans `sched_epochs`, so a stage-1 run is a truncation of the
    stage-2 run and continuing from the stage-1 checkpoint equals training straight through.
    Early stopping and model selection use validation NLL of the MC-averaged prediction.
    """
    amp = device.type == "cuda"
    state = None
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    model = Net(S.arch, 3, cfg["dropout"], pretrained_state(S.arch, S.pretrained))
    model = model.to(device, memory_format=torch.channels_last)
    frozen = configure_trainable(model, int(cfg["unfrozen_stages"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sched_epochs)

    ep, best, best_epoch, bad, stopped = 0, float("inf"), -1, 0, False
    best_val_logits, best_weights, prev_sec = None, None, 0.0
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        ep, best, best_epoch = state["epoch"], state["best"], state["best_epoch"]
        bad, stopped = state["bad"], state["stopped"]
        best_val_logits, best_weights = state["best_val_logits"], state.get("best_weights")
        prev_sec = state.get("elapsed", 0.0)
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])

    tr = torch.as_tensor(split["train"], device=device)
    va = np.asarray(split["val"])
    yv = data.y[torch.as_tensor(va, device=device)].cpu().numpy()
    bs = int(cfg["batch_size"])
    n_batches = max(1, len(tr) // bs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    epochs_this_call = 0

    def snapshot():
        return dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    epoch=ep, best=best, best_epoch=best_epoch, bad=bad, stopped=stopped,
                    best_val_logits=best_val_logits, best_weights=best_weights,
                    elapsed=prev_sec + time.perf_counter() - t0, rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)

    while ep < target_epochs and not stopped:
        model.train()
        for m in frozen:
            m.eval()
        perm = tr[torch.randperm(len(tr), device=device)]
        for k in range(n_batches):
            b = perm[k * bs:(k + 1) * bs]
            x = data.batch(b, augment=True)
            with _autocast(device, amp):
                f = model.backbone(x)
            logits = model.head(F.dropout(f.float(), model.p, training=True))
            loss = F.cross_entropy(logits, data.y[b], label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        val_logits = predict_mc(model, data, va, mc_k, S.eval_batch, device, amp)
        v = nll(mc_probs(val_logits), yv)
        if not np.isfinite(v):
            v = float("inf")
        if v < best - 1e-6:
            best, best_epoch, bad, best_val_logits = v, ep, 0, val_logits
            if keep_best_weights:
                best_weights = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        ep += 1
        epochs_this_call += 1
        if bad >= patience:
            stopped = True
        if save_every_epoch and ckpt_path:
            _save_atomic(snapshot(), ckpt_path)
        if heartbeat:
            heartbeat()

    if best_val_logits is None:  # diverged from the first epoch
        best_val_logits = predict_mc(model, data, va, mc_k, S.eval_batch, device, amp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sec = time.perf_counter() - t0
    if ckpt_path and not save_every_epoch and not stopped and ep < sched_epochs and epochs_this_call > 0:
        _save_atomic(snapshot(), ckpt_path)  # needed only if this candidate may be promoted
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
    if keep_best_weights and best_weights is not None:
        model.load_state_dict(best_weights)
    return dict(model=model, epochs_run=ep, epochs_this_call=epochs_this_call, stopped=stopped,
                best_epoch=best_epoch, best_nll=best, val_logits=best_val_logits,
                gpu_sec=sec, gpu_sec_total=prev_sec + sec, peak_mem_gb=peak)
