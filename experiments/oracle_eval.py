"""Oracle-query evaluation: localize plateau failure to STORAGE vs LOOKUP.

For each checkpoint, evaluate recall twice on the training val set:
  - native q : the model's own query projection (normal eval)
  - oracle q : at each query position t, replace q_t with the key vector k_u the model
    itself computed at the stored pair's position u (the last prior non-query occurrence
    of the query token; --offset 1 uses u+1, the value position, where conv mixing has
    folded the pair together and k(x)v was written into C).

If a FAILED model jumps to high accuracy under oracle queries, its memory C stored the
associations fine and only the learned lookup (q path) is broken — the read-path story.
If it stays at chance, nothing usable was stored.

  uv run python -m experiments.oracle_eval --ckpt-dir runs/ckpts/probe2k --device cpu
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from memrec.data import MADRecallConfig, make_instance, make_val_batches
from memrec.models import build_model
from memrec.ns import orthogonalize_memory
from memrec.train import accuracy, set_seed


def forward_q_override(model, tokens, q_override=None):
    """Manual parallel forward (same math as NSMLSTM.memory_parallel / the baseline cell),
    with optional per-position query substitution. Uses the model's trained read
    (NS if the model has ortho_mode, raw otherwise)."""
    layer = model.layer
    cell = layer.mlstm_cell
    x = model.embedding(tokens)
    x_norm = model.norm(x)
    x_inner = layer.proj_up(x_norm)
    x_mlstm, z = torch.split(x_inner, layer.config._inner_embedding_dim, dim=-1)
    x_conv = layer.conv_act_fn(layer.conv1d(x_mlstm))
    q = layer.q_proj(x_conv)
    k = layer.k_proj(x_conv)
    v = layer.v_proj(x_mlstm)

    b, s, h = q.shape
    nh = cell.config.num_heads
    dh = h // nh
    gate_input = torch.cat([q, k, v], dim=-1)  # gates always use the native q
    qh = q.view(b, s, nh, dh).transpose(1, 2)
    kh = k.view(b, s, nh, dh).transpose(1, 2)
    vh = v.view(b, s, nh, dh).transpose(1, 2)
    if q_override is not None:  # dict (b_idx, t) -> source position u: q_t := k_u
        qh = qh.clone()
        for (bi, t), u in q_override.items():
            qh[bi, :, t] = kh[bi, :, u]

    igate = cell.igate(gate_input).transpose(-1, -2).unsqueeze(-1)
    fgate = cell.fgate(gate_input).transpose(-1, -2).unsqueeze(-1)
    log_fg = F.logsigmoid(fgate).squeeze(-1)
    igate = igate.squeeze(-1)
    cum_fg = torch.cumsum(log_fg, dim=-1)
    log_w = cum_fg.unsqueeze(-1) - cum_fg.unsqueeze(-2) + igate.unsqueeze(-2)
    causal = torch.tril(torch.ones(s, s, device=q.device, dtype=torch.bool)).view(1, 1, s, s)
    log_w = torch.where(causal, log_w, -torch.inf)
    m_state = torch.maximum(log_w.max(dim=-1).values, cum_fg)
    w = torch.where(causal, torch.exp(log_w - m_state.unsqueeze(-1)), torch.zeros_like(log_w))

    k_scaled = kh / (dh**0.5)
    kv = k_scaled.unsqueeze(-1) @ vh.unsqueeze(-2)
    c_state = torch.einsum("bhts,bhsde->bhtde", w, kv)
    n_state = torch.einsum("bhts,bhsd->bhtd", w, k_scaled)

    if getattr(model, "ortho_mode", None):
        c_read = orthogonalize_memory(c_state, model.ortho_mode, model.ortho_steps)
    else:
        c_read = c_state
    h_num = torch.einsum("bhtd,bhtde->bhte", qh, c_read)
    qn = torch.einsum("bhtd,bhtd->bht", qh, n_state).unsqueeze(-1)
    denom = torch.maximum(qn.abs(), torch.exp(-m_state).unsqueeze(-1)) + 1e-6
    h_state = cell.outnorm(h_num / denom).transpose(1, 2).reshape(b, s, -1)

    h_skip = h_state + layer.learnable_skip * x_conv
    out = h_skip * layer.ogate_act_fn(z)
    out = x + layer.dropout(layer.proj_down(out))
    return model.head(model.post_norm(out))


def oracle_map(x, y, offset):
    """(b_idx, query_pos) -> stored-pair source position (last prior non-query occurrence
    of the query token, + offset), clipped to < query_pos."""
    out = {}
    for bi in range(x.shape[0]):
        last = {}
        xs, ys = x[bi].tolist(), y[bi].tolist()
        for t, tok in enumerate(xs):
            if ys[t] != -100 and tok in last:
                u = min(last[tok] + offset, t - 1)
                out[(bi, t)] = u
            if ys[t] == -100:
                last[tok] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--offsets", default="0,1")
    args = ap.parse_args()
    device = torch.device(args.device)
    offsets = [int(o) for o in args.offsets.split(",")]

    for path in sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt"))):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["cfg"]
        set_seed(cfg["seed"])
        mad = MADRecallConfig(cfg["vocab_size"], cfg["seq_len"], cfg["seq_len"],
                              cfg["noise_vocab_size"], cfg["frac_noise"])
        sx, _ = make_instance(mad, np.random.default_rng(cfg["seed"]), is_training=True)
        mad.context_length = len(sx)
        model = build_model(cfg["variant"], mad, cfg["dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        val = make_val_batches(mad, cfg["batch_size"], args.val_batches, device,
                               seed=10_000 + cfg["seed"])
        accs = {"native": []}
        for o in offsets:
            accs[f"oracle+{o}"] = []
        with torch.no_grad():
            for x, y in val:
                accs["native"].append(accuracy(forward_q_override(model, x), y))
                for o in offsets:
                    logits = forward_q_override(model, x, oracle_map(x, y, o))
                    accs[f"oracle+{o}"].append(accuracy(logits, y))
        row = {"ckpt": os.path.basename(path), "variant": cfg["variant"], "seed": cfg["seed"],
               **{k: round(float(np.mean(v)), 4) for k, v in accs.items()}}
        print(json.dumps(row))


if __name__ == "__main__":
    main()
