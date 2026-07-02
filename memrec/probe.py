"""Gradient-free probes of the mLSTM memory state + gradient-coherence measurement.

Used to test the escape-preconditioning hypothesis: during the chance plateau the raw
memory matrix C = sum_t w_t k_t (x) v_t is noise-dominated / ill-conditioned, so the read
q^T C carries little gradient signal; the ns read whitens C's spectrum and raises the
gradient signal-to-noise, which is why ns variants escape earlier and at lower lr.

All probes are trajectory-neutral: they draw probe batches from their own rng stream and
zero grads afterwards, so a probed run is bit-identical to an unprobed one.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_batch


@torch.no_grad()
def memory_state_last(model, tokens: torch.Tensor) -> torch.Tensor:
    """Materialize the raw (pre-orthogonalization) memory matrix C at the final position.

    Returns [b, nh, dh, dh]. Works for MLSTMBaseline and NSMLSTM alike (same module
    layout); mirrors NSMLSTM.memory_parallel restricted to the last row of the decay
    weight matrix, so it never builds the O(s^2) or O(s*dh^2) intermediates.
    """
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

    igate = cell.igate(gate_input).transpose(-1, -2)  # [b, nh, s]
    log_fg = F.logsigmoid(cell.fgate(gate_input).transpose(-1, -2))
    cum_fg = torch.cumsum(log_fg, dim=-1)
    # last row of the parallel-form weight matrix: w_u = exp(cum_fg[-1] - cum_fg[u] + igate[u] - m)
    log_w = cum_fg[..., -1:] - cum_fg + igate
    m = torch.maximum(log_w.max(dim=-1).values, cum_fg[..., -1])
    w = torch.exp(log_w - m.unsqueeze(-1))
    k_scaled = kh / (dh**0.5)
    return torch.einsum("bhs,bhsd,bhse->bhde", w, k_scaled, vh)


@torch.no_grad()
def key_gram(model, tokens: torch.Tensor) -> torch.Tensor:
    """Gram of unit-normalized keys over the sequence: G = (1/s) Σ k̂ k̂ᵀ, [b, nh, dh, dh].

    The compressed-sensing 'measurement ensemble' probe: recall-from-superposition is sparse
    recovery, and its feasibility is governed by the conditioning/incoherence of the key
    ensemble — separately from C's own spectrum (which mixes keys with values and gates)."""
    layer = model.layer
    cell = layer.mlstm_cell
    x_norm = model.norm(model.embedding(tokens))
    x_inner = layer.proj_up(x_norm)
    x_mlstm, _z = torch.split(x_inner, layer.config._inner_embedding_dim, dim=-1)
    k = layer.k_proj(layer.conv_act_fn(layer.conv1d(x_mlstm)))
    b, s, h = k.shape
    nh = cell.config.num_heads
    kh = k.view(b, s, nh, h // nh).transpose(1, 2)
    k_hat = kh / kh.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.einsum("bhsd,bhse->bhde", k_hat, k_hat) / s


@torch.no_grad()
def spectrum_stats(c: torch.Tensor, eps: float = 1e-12) -> dict:
    """Scale-invariant singular-spectrum summaries of batched matrices [..., d, d],
    averaged over all leading dims (batch, heads)."""
    s = torch.linalg.svdvals(c.float())
    p = s / s.sum(dim=-1, keepdim=True).clamp_min(eps)
    erank = torch.exp(-(p * p.clamp_min(eps).log()).sum(dim=-1))
    log10_cond = s[..., 0].clamp_min(eps).log10() - s[..., -1].clamp_min(eps).log10()
    return {
        "erank": round(erank.mean().item(), 3),
        "top1_share": round(p[..., 0].mean().item(), 4),
        "log10_cond": round(log10_cond.mean().item(), 3),
    }


def grad_pair_stats(model, batches, vocab_size: int) -> dict:
    """Gradient coherence between two independent batches (a gradient-SNR proxy).

    Returns cosine similarity of the full flattened gradients, the same restricted to
    the q/k projections (the read path), and the mean gradient norm. Leaves model grads
    zeroed (set_to_none) on exit.
    """
    model.train()
    qk_params = {id(model.layer.q_proj.weight), id(model.layer.k_proj.weight)}
    full, qk = [], []
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
        loss.backward()
        full.append(torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]))
        qk_grads = [p.grad.reshape(-1) for p in model.parameters()
                    if p.grad is not None and id(p) in qk_params]
        qk.append(torch.cat(qk_grads) if qk_grads else None)
    model.zero_grad(set_to_none=True)
    (g1, g2), (q1, q2) = full, qk
    out = {
        "grad_cos": round(F.cosine_similarity(g1, g2, dim=0).item(), 4),
        "grad_norm": round(((g1.norm() + g2.norm()) / 2).item(), 5),
    }
    if q1 is not None:  # absent when q/k are frozen (randqk)
        out["qk_grad_cos"] = round(F.cosine_similarity(q1, q2, dim=0).item(), 4)
    return out


def probe_step(model, step: int, cfg: dict, mad, val_batches, device) -> dict:
    """One probe measurement: C spectrum on the fixed first val batch (+ post-read spectrum
    for orthogonalized-read models) and gradient coherence on two fresh probe batches drawn
    from a dedicated rng stream (never touches the training rng)."""
    from .ns import orthogonalize_memory

    # models with their own state materializer (e.g. DeltaMLSTM's erase-write scan) know best
    own = getattr(model, "memory_state_last", None)
    with torch.no_grad():
        c_last = own(val_batches[0][0]) if own else memory_state_last(model, val_batches[0][0])
    entry = {"step": step}
    entry.update({f"c_{k}": v for k, v in spectrum_stats(c_last).items()})
    if getattr(model, "ortho_mode", None):
        co = orthogonalize_memory(c_last, model.ortho_mode, model.ortho_steps)
        entry.update({f"co_{k}": v for k, v in spectrum_stats(co).items()})
    entry.update({f"k_{k}": v for k, v in spectrum_stats(key_gram(model, val_batches[0][0])).items()})
    rng = np.random.default_rng(1_000_000 * (cfg["seed"] + 1) + step)
    pb = [make_batch(cfg["batch_size"], device, mad, rng, is_training=True) for _ in range(2)]
    entry.update(grad_pair_stats(model, pb, mad.vocab_size))
    return entry
