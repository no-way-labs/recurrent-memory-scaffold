"""Ridge-probe decode of the memory matrix C: how much key-addressable association is
linearly recoverable from a model's own memory state, independent of its learned readout?

For each checkpoint and each query position t (target y_t != -100) the feature is
phi_t = concat_heads(k_u^T C_t): the model's OWN stored key at the pair's write position u
slicing the raw (pre-orthogonalization) per-head memory C_t at query time. A ridge decoder
is fit on a train split of sequences and scored on held-out sequences; the wrong-key
control repeats the pipeline with k taken from a different stored pair in the same
sequence. High decode accuracy in a behaviorally-failed model = storage is intact and
the readout is what never got learned.

  --offset 1 (default) uses the value position u+1, where conv mixing has folded the
  pair together and k(x)v was written into C; --offset 0 uses the key position.

  uv run python -m experiments.c_decode --ckpt-dir runs/ckpts/probe2k \
      --out runs/c_decode_probe2k.jsonl --device cpu
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
from memrec.train import set_seed

from .oracle_eval import oracle_map


@torch.no_grad()
def qkv_and_gates(model, tokens):
    """Shared front half of the parallel form (same math as probe.memory_state_last)."""
    layer = model.layer
    cell = layer.mlstm_cell
    x_norm = model.norm(model.embedding(tokens))
    x_inner = layer.proj_up(x_norm)
    x_mlstm, _z = torch.split(x_inner, layer.config._inner_embedding_dim, dim=-1)
    x_conv = layer.conv_act_fn(layer.conv1d(x_mlstm))
    q = layer.q_proj(x_conv)
    k = layer.k_proj(x_conv)
    v = layer.v_proj(x_mlstm)
    b, s, h = q.shape
    nh = cell.config.num_heads
    dh = h // nh
    gate_input = torch.cat([q, k, v], dim=-1)
    kh = k.view(b, s, nh, dh).transpose(1, 2)
    vh = v.view(b, s, nh, dh).transpose(1, 2)
    igate = cell.igate(gate_input).transpose(-1, -2)               # [b, nh, s]
    log_fg = F.logsigmoid(cell.fgate(gate_input).transpose(-1, -2))
    return kh, vh, igate, torch.cumsum(log_fg, dim=-1)


@torch.no_grad()
def memory_at(kh, vh, igate, cum_fg, bi, ts):
    """Raw per-head memory C_t for one sequence at positions ts. Returns [nq, nh, dh, dh]."""
    nh, s, dh = kh.shape[1], kh.shape[2], kh.shape[3]
    t_idx = torch.tensor(ts, dtype=torch.long, device=kh.device)
    cf, ig = cum_fg[bi], igate[bi]                                  # [nh, s]
    log_w = cf[:, t_idx].unsqueeze(-1) - cf.unsqueeze(1) + ig.unsqueeze(1)  # [nh, nq, s]
    causal = (torch.arange(s, device=kh.device).view(1, 1, s) <= t_idx.view(1, -1, 1))
    log_w = torch.where(causal, log_w, torch.full_like(log_w, -torch.inf))
    m = torch.maximum(log_w.max(dim=-1).values, cf[:, t_idx])       # [nh, nq]
    w = torch.exp(log_w - m.unsqueeze(-1))
    k_scaled = kh[bi] / (dh**0.5)
    c = torch.einsum("hqs,hsd,hse->qhde", w, k_scaled, vh[bi])
    return c


def collect_samples(model, batches, offset, rng):
    """(features, wrong-key features, labels, seq_ids) over all query positions."""
    feats, wrong, labels, seq_ids = [], [], [], []
    sid = 0
    for x, y in batches:
        kh, vh, igate, cum_fg = qkv_and_gates(model, x)
        omap = oracle_map(x, y, offset)
        by_seq = {}
        for (bi, t), u in omap.items():
            by_seq.setdefault(bi, []).append((t, u))
        for bi, pairs in by_seq.items():
            ts = [t for t, _ in pairs]
            c = memory_at(kh, vh, igate, cum_fg, bi, ts)            # [nq, nh, dh, dh]
            for j, (t, u) in enumerate(pairs):
                k_u = kh[bi, :, u]                                  # [nh, dh]
                phi = torch.einsum("hd,hde->he", k_u, c[j]).reshape(-1)
                # wrong-key control: a stored key from a DIFFERENT pair in this sequence
                others = [uu for tt, uu in pairs if x[bi, tt].item() != x[bi, t].item()]
                u_w = others[rng.integers(len(others))] if others else u
                phi_w = torch.einsum("hd,hde->he", kh[bi, :, u_w], c[j]).reshape(-1)
                feats.append(phi.cpu().numpy())
                wrong.append(phi_w.cpu().numpy())
                labels.append(int(y[bi, t].item()))
                seq_ids.append(sid)
            sid += 1
    return (np.array(feats, dtype=np.float64), np.array(wrong, dtype=np.float64),
            np.array(labels), np.array(seq_ids))


def ridge_acc(X_tr, y_tr, X_te, y_te, vocab, lambdas=(0.1, 1.0, 10.0, 100.0)):
    """Standardized ridge to one-hot targets; lambda picked on a tail split of train."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    Xt, Xe = (X_tr - mu) / sd, (X_te - mu) / sd
    Y = np.eye(vocab)[y_tr]
    n_fit = int(0.8 * len(Xt))
    best, best_acc = None, -1.0
    for lam in lambdas:
        G = Xt[:n_fit].T @ Xt[:n_fit] + lam * np.eye(Xt.shape[1])
        W = np.linalg.solve(G, Xt[:n_fit].T @ Y[:n_fit])
        acc = float((np.argmax(Xt[n_fit:] @ W, axis=1) == y_tr[n_fit:]).mean())
        if acc > best_acc:
            best, best_acc = lam, acc
    G = Xt.T @ Xt + best * np.eye(Xt.shape[1])
    W = np.linalg.solve(G, Xt.T @ Y)
    return float((np.argmax(Xe @ W, axis=1) == y_te).mean()), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val-batches", type=int, default=6)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--offsets", default="1,0")
    args = ap.parse_args()
    device = torch.device(args.device)
    offsets = [int(o) for o in args.offsets.split(",")]

    out_fh = open(args.out, "w") if args.out else None
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
        row = {"ckpt": os.path.basename(path), "variant": cfg["variant"], "seed": cfg["seed"]}
        for off in offsets:
            rng = np.random.default_rng(777 + cfg["seed"])
            X, Xw, yv, sids = collect_samples(model, val, off, rng)
            cut = sids <= np.quantile(np.unique(sids), args.train_frac)
            acc, lam = ridge_acc(X[cut], yv[cut], X[~cut], yv[~cut], cfg["vocab_size"])
            acc_w, _ = ridge_acc(Xw[cut], yv[cut], Xw[~cut], yv[~cut], cfg["vocab_size"])
            row.update({f"decode_o{off}": round(acc, 4), f"wrongkey_o{off}": round(acc_w, 4),
                        f"lambda_o{off}": lam})
        row["n_samples"] = int(len(yv))
        print(json.dumps(row), flush=True)
        if out_fh:
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
    if out_fh:
        out_fh.close()


if __name__ == "__main__":
    main()
