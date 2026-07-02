"""Shared training / evaluation loop for all variants.

`train_one(cfg)` builds the model + optimizer(s) for `cfg["variant"]`, trains on freshly
generated MAD noisy-recall batches, evaluates on a fixed validation set, and returns a
result dict (same schema across variants for easy aggregation).
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import MADRecallConfig, make_batch, make_instance, make_val_batches
from .models import build_model, orthogonalize_projections_

DEFAULTS = dict(
    dim=94,
    noise_vocab_size=16,
    frac_noise=0.8,
    batch_size=16,
    val_batches=16,
    eval_every=200,
    lr=3e-3,
    steps=2000,
    weight_decay=0.01,
    grad_clip=1.0,
    pogo_targets="q,k",  # which head-block projections to keep orthogonal (pogo variant)
    probe=False,     # log C-spectrum + gradient-coherence probes at each eval (trajectory-neutral)
    ckpt_dir=None,   # if set, save the final model state_dict here
    lr_schedule="cosine",  # "cosine" (T_max=steps) or "constant"
    qk_lr_mult=1.0,  # LR multiplier for the q/k projections (read-path optimizer boost)
    anneal_start=0,  # if >0: linearly anneal the NS read toward the raw read
    anneal_end=0,    # over steps [anneal_start, anneal_end] (read_alpha 1 -> 0)
    anneal_gate_acc=0.0,   # if >0: escape-gated anneal — start annealing after the first eval
    anneal_gate_dwell=200,  # with val_acc >= gate, waiting `dwell` steps to consolidate,
    anneal_gate_span=500,   # then read_alpha 1 -> 0 over `span` steps. Constant-LR runs extend
    #                         the budget so a late escaper still finishes at alpha=0.
    data_seed=None,  # if set, decouples the data stream from `seed` (init lottery vs data lottery)
)

_TARGET_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    mask = y >= 0
    if not mask.any():
        return 0.0
    return (logits.argmax(dim=-1)[mask] == y[mask]).float().mean().item()


@torch.no_grad()
def evaluate(model, batches, vocab_size) -> tuple[float, float]:
    model.eval()
    losses, accs = [], []
    for x, y in batches:
        logits = model(x)
        losses.append(F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100).item())
        accs.append(accuracy(logits, y))
    return float(np.mean(losses)), float(np.mean(accs))


def _make_sched(opt, lr_schedule, steps):
    if lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)


def _build_optimizers(variant, model, lr, weight_decay, steps, pogo_targets,
                      lr_schedule="cosine", qk_lr_mult=1.0):
    """Return (optimizers, scheduler, pre_step_hook). pre_step_hook runs each step before .step()."""
    if not variant.startswith("pogo"):
        # filter keeps frozen-projection variants (randk/randqk) out of the optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        if qk_lr_mult != 1.0:
            qk_ids = {id(model.layer.q_proj.weight), id(model.layer.k_proj.weight)}
            groups = [
                {"params": [p for p in params if id(p) not in qk_ids]},
                {"params": [p for p in params if id(p) in qk_ids], "lr": lr * qk_lr_mult},
            ]
            opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
        else:
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return [opt], _make_sched(opt, lr_schedule, steps), (lambda: None)

    # pogo variant: split params into {orthogonal q/k blocks -> POGO} and {rest -> AdamW}
    from pogo import POGO, base

    targets = tuple(_TARGET_MAP[t.strip()] for t in pogo_targets.split(",") if t.strip())
    pogo_params = orthogonalize_projections_(model, targets)
    pogo_ids = {id(p) for p in pogo_params}
    other_params = [p for p in model.parameters() if id(p) not in pogo_ids]

    opt_main = torch.optim.AdamW(other_params, lr=lr, weight_decay=weight_decay)
    # VectorAdam base keeps POGO adaptive like AdamW; weight_decay=0 so decay doesn't fight orthogonality.
    opt_pogo = POGO(pogo_params, base.VectorAdam(betas=(0.9, 0.999)), lr=lr, weight_decay=0.0)
    sched = _make_sched(opt_main, lr_schedule, steps)

    def pre_step_hook():
        # mirror the cosine-annealed lr onto POGO so both param sets share the same schedule
        cur = opt_main.param_groups[0]["lr"]
        for g in opt_pogo.param_groups:
            g["lr"] = cur

    return [opt_main, opt_pogo], sched, pre_step_hook


def train_one(cfg: dict) -> dict:
    c = {**DEFAULTS, **cfg}
    variant = c["variant"]
    device = torch.device(c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(c["seed"])

    mad = MADRecallConfig(c["vocab_size"], c["seq_len"], c["seq_len"], c["noise_vocab_size"], c["frac_noise"])
    sample_x, _ = make_instance(mad, np.random.default_rng(c["seed"]), is_training=True)
    mad.context_length = len(sample_x)

    model = build_model(variant, mad, c["dim"]).to(device)
    params = sum(p.numel() for p in model.parameters())

    optimizers, sched, pre_step_hook = _build_optimizers(
        variant, model, c["lr"], c["weight_decay"], c["steps"], c["pogo_targets"],
        lr_schedule=c["lr_schedule"], qk_lr_mult=c["qk_lr_mult"],
    )

    ds = c["data_seed"] if c["data_seed"] is not None else c["seed"]
    train_rng = np.random.default_rng(ds)
    val_batches = make_val_batches(mad, c["batch_size"], c["val_batches"], device, seed=10_000 + ds)

    best = {"best_val_acc": -1.0, "best_val_loss": None, "best_step": None}
    final = {}
    trace = []
    probe_trace = []
    anneal_trigger = None  # escape-gated anneal: step of the first eval clearing the gate
    total_steps = c["steps"]
    start = time.time()
    step = 0
    while step < total_steps:
        step += 1
        model.train()
        if hasattr(model, "read_alpha"):
            if c["anneal_start"]:
                span = max(1, c["anneal_end"] - c["anneal_start"])
                model.read_alpha = min(1.0, max(0.0, 1.0 - (step - c["anneal_start"]) / span))
            elif anneal_trigger is not None:
                a0 = anneal_trigger + c["anneal_gate_dwell"]
                span = max(1, c["anneal_gate_span"])
                model.read_alpha = min(1.0, max(0.0, 1.0 - (step - a0) / span))
        x, y = make_batch(c["batch_size"], device, mad, train_rng, is_training=True)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, mad.vocab_size), y.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        pre_step_hook()
        for opt in optimizers:
            opt.step()
        sched.step()

        if step == 1 or step % c["eval_every"] == 0 or step == total_steps:
            val_loss, val_acc = evaluate(model, val_batches, mad.vocab_size)
            if val_acc > best["best_val_acc"]:
                best = {"best_val_acc": val_acc, "best_val_loss": val_loss, "best_step": step}
            final = {"final_val_acc": val_acc, "final_val_loss": val_loss}
            alpha_tag = (f" alpha {model.read_alpha:.3f}"
                         if hasattr(model, "read_alpha") and (c["anneal_start"] or c["anneal_gate_acc"]) else "")
            trace.append(f"step {step} acc {val_acc:.5g} loss {val_loss:.4g} t {time.time() - start:.0f}s{alpha_tag}")
            if (c["anneal_gate_acc"] and anneal_trigger is None and hasattr(model, "read_alpha")
                    and val_acc >= c["anneal_gate_acc"]):
                anneal_trigger = step
                if c["lr_schedule"] == "constant":
                    # extend so the anneal completes + a settling tail lands a final alpha=0 eval;
                    # cosine budgets are left alone (stepping past T_max would reheat the LR)
                    total_steps = max(total_steps, step + c["anneal_gate_dwell"]
                                      + c["anneal_gate_span"] + c["eval_every"])
            if c["probe"]:
                from .probe import probe_step

                probe_trace.append(probe_step(model, step, c, mad, val_batches, device))

    ckpt_name = None
    if c["ckpt_dir"]:
        import os

        os.makedirs(c["ckpt_dir"], exist_ok=True)
        regime = c.get("regime", f"v{c['vocab_size']}_s{c['seq_len']}")
        ckpt_name = f"{regime}_{variant}_seed{c['seed']}_steps{c['steps']}.pt"
        torch.save(
            {"state_dict": model.state_dict(),
             "cfg": {k: v for k, v in c.items() if k not in ("ckpt_dir", "device")}},
            os.path.join(c["ckpt_dir"], ckpt_name),
        )

    return {
        "variant": variant,
        "regime": c.get("regime", f"v{c['vocab_size']}_s{c['seq_len']}"),
        "seed": c["seed"],
        "data_seed": ds,
        "params": params,
        "dim": c["dim"],
        "vocab_size": c["vocab_size"],
        "seq_len": c["seq_len"],
        "context_length": mad.context_length,
        "noise_vocab_size": c["noise_vocab_size"],
        "frac_noise": c["frac_noise"],
        "batch_size": c["batch_size"],
        "steps": c["steps"],
        "steps_run": step,
        "anneal_trigger_step": anneal_trigger,
        "final_read_alpha": float(model.read_alpha) if hasattr(model, "read_alpha") else None,
        "lr": c["lr"],
        "pogo_targets": c["pogo_targets"] if variant.startswith("pogo") else None,
        **best,
        **final,
        "trace_tail": trace[-6:],
        "trace": trace,
        "probe": probe_trace or None,
        "ckpt": ckpt_name,
    }
