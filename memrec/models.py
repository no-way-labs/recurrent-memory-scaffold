"""Model variants: baseline mLSTM, ns5 (orthogonalized read), and POGO-constrained projections.

The baseline / ns5 forwards mirror the reference repo exactly so results are comparable.
The pogo variant reuses the *baseline* forward unchanged; the only difference is that
selected head-block projection weights (q_proj / k_proj, each shaped [48, 4, 4]) are
initialized orthogonal and later optimized by POGO to *stay* orthogonal. This tests
whether static parameter orthogonality recovers the gains that ns5 gets from
re-orthogonalizing the evolving memory state every token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Work around xlstm importing cuda cpp_extension include paths on CPU-only hosts.
import torch.utils.cpp_extension as _cpp_extension

_orig_include_paths = _cpp_extension.include_paths


def _include_paths(*args, **kwargs):
    kwargs.pop("cuda", None)
    return _orig_include_paths(*args, **kwargs)


_cpp_extension.include_paths = _include_paths

from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig  # noqa: E402
from xlstm.components.ln import LayerNorm  # noqa: E402

from .data import MADRecallConfig  # noqa: E402
from .ns import NS_STEPS, orthogonalize_memory  # noqa: E402

# head-block projections eligible for the POGO orthogonality constraint
POGO_TARGET_MODULES = ("q_proj", "k_proj", "v_proj")


def _mlstm_layer_config(cfg: MADRecallConfig, dim: int) -> mLSTMLayerConfig:
    return mLSTMLayerConfig(
        embedding_dim=dim,
        context_length=cfg.context_length,
        num_heads=4,
        qkv_proj_blocksize=4,
        proj_factor=2.0,
        bias=False,
        dropout=0.0,
    )


class MLSTMBaseline(nn.Module):
    """One-block mLSTM baseline (== mad_mlstm_baseline.py)."""

    def __init__(self, cfg: MADRecallConfig, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, dim)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=False)
        self.layer = mLSTMLayer(_mlstm_layer_config(cfg, dim))
        self.post_norm = LayerNorm(ndim=dim, weight=True, bias=False)
        self.head = nn.Linear(dim, cfg.vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embedding(tokens)
        h = h + self.layer(self.norm(h))
        return self.head(self.post_norm(h))


class NSMLSTM(nn.Module):
    """mLSTM with an orthogonalized memory read. Default (mode='ns', steps=5) == xlstm_ns_mad.py.

    `ortho_mode`/`ortho_steps` expose the mechanism-ablation knob (direction #1).
    """

    def __init__(self, cfg: MADRecallConfig, dim: int, ortho_mode: str = "ns", ortho_steps: int = NS_STEPS,
                 ortho_apply: str = "both") -> None:
        super().__init__()
        self.ortho_mode = ortho_mode
        self.ortho_steps = ortho_steps
        # straight-through decomposition of the orthogonalized read:
        #   "both"     - forward and backward through NS (the paper's ns5)
        #   "forward"  - forward uses NS read, backward flows as if the read were raw C
        #   "backward" - forward is numerically the raw read, backward flows through NS
        self.ortho_apply = ortho_apply
        # scaffold anneal: read = alpha * NS(C) + (1-alpha) * raw C. Set per-step by the
        # training loop (1 -> 0 over the anneal window); at 0 the read is exactly baseline.
        self.read_alpha = 1.0
        self.embedding = nn.Embedding(cfg.vocab_size, dim)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=False)
        self.layer = mLSTMLayer(_mlstm_layer_config(cfg, dim))
        self.post_norm = LayerNorm(ndim=dim, weight=True, bias=False)
        self.head = nn.Linear(dim, cfg.vocab_size)

    def memory_parallel(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        cell = self.layer.mlstm_cell
        b, s, h = q.shape
        nh = cell.config.num_heads
        dh = h // nh

        gate_input = torch.cat([q, k, v], dim=-1)
        qh = q.view(b, s, nh, dh).transpose(1, 2)
        kh = k.view(b, s, nh, dh).transpose(1, 2)
        vh = v.view(b, s, nh, dh).transpose(1, 2)

        igate = cell.igate(gate_input).transpose(-1, -2).unsqueeze(-1)
        fgate = cell.fgate(gate_input).transpose(-1, -2).unsqueeze(-1)
        log_fg = F.logsigmoid(fgate).squeeze(-1)
        igate = igate.squeeze(-1)

        cum_fg = torch.cumsum(log_fg, dim=-1)
        log_weights = cum_fg.unsqueeze(-1) - cum_fg.unsqueeze(-2) + igate.unsqueeze(-2)
        causal = torch.tril(torch.ones(s, s, device=q.device, dtype=torch.bool)).view(1, 1, s, s)
        log_weights = torch.where(causal, log_weights, -torch.inf)

        content_m = log_weights.max(dim=-1).values
        m_state = torch.maximum(content_m, cum_fg)
        weights = torch.exp(log_weights - m_state.unsqueeze(-1))
        weights = torch.where(causal, weights, torch.zeros_like(weights))

        k_scaled = kh / (dh**0.5)
        kv = k_scaled.unsqueeze(-1) @ vh.unsqueeze(-2)
        c_state = torch.einsum("bhts,bhsde->bhtde", weights, kv)
        n_state = torch.einsum("bhts,bhsd->bhtd", weights, k_scaled)

        if self.read_alpha <= 0.0:
            c_read = c_state  # scaffold fully down: exactly the baseline read, no NS compute
        else:
            c_ortho = orthogonalize_memory(c_state, self.ortho_mode, self.ortho_steps)
            if self.ortho_apply == "forward":
                c_read = c_state + (c_ortho - c_state).detach()   # value: NS read; grad: raw read
            elif self.ortho_apply == "backward":
                c_read = c_ortho + (c_state - c_ortho).detach()   # value: raw read; grad: through NS
            else:
                c_read = c_ortho
            if self.read_alpha < 1.0:
                c_read = self.read_alpha * c_read + (1.0 - self.read_alpha) * c_state
        h_num = torch.einsum("bhtd,bhtde->bhte", qh, c_read)
        qn = torch.einsum("bhtd,bhtd->bht", qh, n_state).unsqueeze(-1)
        denom = torch.maximum(qn.abs(), torch.exp(-m_state).unsqueeze(-1)) + 1e-6
        h_state = h_num / denom
        return cell.outnorm(h_state).transpose(1, 2).reshape(b, s, -1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x_norm = self.norm(x)
        x_inner = self.layer.proj_up(x_norm)
        x_mlstm, z = torch.split(x_inner, self.layer.config._inner_embedding_dim, dim=-1)
        x_conv = self.layer.conv_act_fn(self.layer.conv1d(x_mlstm))
        q_all = self.layer.q_proj(x_conv)
        k_all = self.layer.k_proj(x_conv)
        v_all = self.layer.v_proj(x_mlstm)

        h_tilde = self.memory_parallel(q_all, k_all, v_all)
        h_skip = h_tilde + self.layer.learnable_skip * x_conv
        h_state = h_skip * self.layer.ogate_act_fn(z)
        h = x + self.layer.dropout(self.layer.proj_down(h_state))
        return self.head(self.post_norm(h))


class DeltaMLSTM(nn.Module):
    """mLSTM with a delta-rule write: erase the value stored under the current key before
    writing (C_t = decay * (I - k̂ k̂ᵀ) C_{t-1} + input * (k/√dh) vᵀ, β=1, no new params).

    Same modules/gates/normalizer as MLSTMBaseline — the ONLY change is the write, so this is
    the write-side counterpart of NSMLSTM's read-side fix (both target C's rank collapse).
    Implemented as a stabilized sequential scan (the erase term breaks the parallel form);
    with erase_beta=0 the scan reproduces MLSTMBaseline's forward exactly (correctness check).
    `ortho_mode/ortho_steps` optionally apply the NS read on top (variant `delta_ns<k>`).
    """

    def __init__(self, cfg: MADRecallConfig, dim: int, ortho_mode: str | None = None,
                 ortho_steps: int = 0, erase_beta: float = 1.0,
                 rls_read: bool = False, rls_lambda: float = 1e-2) -> None:
        super().__init__()
        self.ortho_mode = ortho_mode
        self.ortho_steps = ortho_steps
        self.erase_beta = erase_beta
        # RLS read: whiten the read with P ≈ (Σ k̂k̂ᵀ + λI)⁻¹ (Sherman-Morrison, ungated Gram).
        # h = qᵀ(PC)/norm(qᵀPn): the exact least-squares decode delta approximates by SGD and
        # the NS read approximates spectrally. Same O(dh²)/step cost as the delta erase.
        self.rls_read = rls_read
        self.rls_lambda = rls_lambda
        self.embedding = nn.Embedding(cfg.vocab_size, dim)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=False)
        self.layer = mLSTMLayer(_mlstm_layer_config(cfg, dim))
        self.post_norm = LayerNorm(ndim=dim, weight=True, bias=False)
        self.head = nn.Linear(dim, cfg.vocab_size)
        self._compiled_body = None  # torch.compile'd scan body (CUDA only, lazy)

    def _qkv_gates(self, tokens: torch.Tensor):
        layer = self.layer
        cell = layer.mlstm_cell
        x = self.embedding(tokens)
        x_norm = self.norm(x)
        x_inner = layer.proj_up(x_norm)
        x_mlstm, z = torch.split(x_inner, layer.config._inner_embedding_dim, dim=-1)
        x_conv = layer.conv_act_fn(layer.conv1d(x_mlstm))
        q = layer.q_proj(x_conv)
        k = layer.k_proj(x_conv)
        v = layer.v_proj(x_mlstm)
        gate_input = torch.cat([q, k, v], dim=-1)
        igate = cell.igate(gate_input).transpose(-1, -2)          # [b, nh, s]
        log_fg = F.logsigmoid(cell.fgate(gate_input).transpose(-1, -2))
        b, s, h = q.shape
        nh = cell.config.num_heads
        dh = h // nh
        qh = q.view(b, s, nh, dh).transpose(1, 2)
        kh = k.view(b, s, nh, dh).transpose(1, 2)
        vh = v.view(b, s, nh, dh).transpose(1, 2)
        return x, x_conv, z, qh, kh, vh, igate, log_fg, nh, dh

    def _scan_body(self, c, n, m, p, q_t, k_t, v_t, lf_t, ig_t):
        """One step of the stabilized delta-write recurrence (pure function of its inputs)."""
        dh = k_t.shape[-1]
        m_new = torch.maximum(lf_t + m, ig_t)
        decay = torch.exp(lf_t + m - m_new).unsqueeze(-1)                  # [b, nh, 1]
        inp = torch.exp(ig_t - m_new).unsqueeze(-1)
        if self.erase_beta or self.rls_read:
            k_hat = k_t / k_t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if self.erase_beta:
            c = c - self.erase_beta * k_hat.unsqueeze(-1) * (k_hat.unsqueeze(-2) @ c)
            n = n - self.erase_beta * k_hat * (k_hat * n).sum(-1, keepdim=True)
        k_w = k_t / (dh**0.5)
        c = decay.unsqueeze(-1) * c + inp.unsqueeze(-1) * (k_w.unsqueeze(-1) @ v_t.unsqueeze(-2))
        n = decay * n + inp * k_w
        if self.rls_read:
            pk = (p @ k_hat.unsqueeze(-1)).squeeze(-1)                     # [b, nh, dh]
            p = p - pk.unsqueeze(-1) * pk.unsqueeze(-2) / (1.0 + (k_hat * pk).sum(-1, keepdim=True).unsqueeze(-1))
            p = 0.5 * (p + p.transpose(-1, -2))                            # fp32 symmetry insurance
            c_read = p @ c
            n_read = (p @ n.unsqueeze(-1)).squeeze(-1)
        else:
            c_read = orthogonalize_memory(c, self.ortho_mode, self.ortho_steps) if self.ortho_mode else c
            n_read = n
        h_num = (q_t.unsqueeze(-2) @ c_read).squeeze(-2)                   # [b, nh, dh]
        qn = (q_t * n_read).sum(-1, keepdim=True)
        denom = torch.maximum(qn.abs(), torch.exp(-m_new).unsqueeze(-1)) + 1e-6
        return c, n, m_new, p, h_num / denom

    def memory_scan(self, tokens: torch.Tensor, return_state: bool = False):
        """Sequential delta-write scan. Returns per-position hidden states (and final C)."""
        x, x_conv, z, qh, kh, vh, igate, log_fg, nh, dh = self._qkv_gates(tokens)
        b, _, s, _ = qh.shape
        body = self._scan_body
        if tokens.is_cuda:
            if self._compiled_body is None:
                self._compiled_body = torch.compile(self._scan_body)
            body = self._compiled_body
        c = qh.new_zeros(b, nh, dh, dh)
        n = qh.new_zeros(b, nh, dh)
        # m starts at 0 so it tracks max(content, cum_fg) like the parallel form's m_state
        m = qh.new_zeros(b, nh)
        eye = torch.eye(dh, device=qh.device, dtype=qh.dtype).expand(b, nh, dh, dh)
        p = eye / self.rls_lambda if self.rls_read else eye  # unused unless rls_read
        hs = []
        for t in range(s):
            c, n, m, p, h_t = body(c, n, m, p, qh[:, :, t], kh[:, :, t], vh[:, :, t],
                                   log_fg[:, :, t], igate[:, :, t])
            hs.append(h_t)
        h_state = torch.stack(hs, dim=2)                                   # [b, nh, s, dh]
        cell = self.layer.mlstm_cell
        out = cell.outnorm(h_state).transpose(1, 2).reshape(b, s, -1)
        if return_state:
            return out, c
        return out, (x, x_conv, z)

    def memory_state_last(self, tokens: torch.Tensor) -> torch.Tensor:
        """Raw delta-written C at the final position (probe hook; pre-ortho)."""
        ortho = self.ortho_mode
        self.ortho_mode = None  # skip per-step NS: only the final raw state is needed
        try:
            _, c = self.memory_scan(tokens, return_state=True)
        finally:
            self.ortho_mode = ortho
        return c

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h_tilde, (x, x_conv, z) = self.memory_scan(tokens)
        h_skip = h_tilde + self.layer.learnable_skip * x_conv
        h_state = h_skip * self.layer.ogate_act_fn(z)
        h = x + self.layer.dropout(self.layer.proj_down(h_state))
        return self.head(self.post_norm(h))


@torch.no_grad()
def orthogonalize_projections_(model: nn.Module, targets: tuple[str, ...]) -> list[nn.Parameter]:
    """Polar-project the targeted head-block projection weights onto the (per-block) orthogonal
    group and return the list of parameters that POGO should keep orthogonal.

    Each target weight has shape [num_blocks, blk, blk]; batched SVD gives the nearest orthogonal
    matrix U V^T per block (the init suggested in the POGO README).
    """
    params: list[nn.Parameter] = []
    for name in targets:
        mod = getattr(model.layer, name)
        w = mod.weight  # [num_blocks, blk, blk]
        assert w.ndim == 3 and w.shape[-1] == w.shape[-2], f"{name} weight not square-batched: {tuple(w.shape)}"
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        w.copy_(u @ vh)
        params.append(mod.weight)
    return params


def parse_ortho_variant(variant: str):
    """Map a variant name to (mode, steps) for an orthogonalized-read model, or None.

    'ns5'/'ns' -> ('ns', 5); 'ns0','ns1','ns3','ns8' -> ('ns', k); 'polar' -> ('polar', 0).
    """
    if variant == "polar":
        return ("polar", 0)
    if variant.startswith("ns"):
        rest = variant[2:]
        if rest == "":
            return ("ns", NS_STEPS)
        if rest.isdigit():
            return ("ns", int(rest))
    return None


def build_model(variant: str, cfg: MADRecallConfig, dim: int):
    if variant in ("rls", "rls_delta", "delta_rls"):
        # RLS/whitened read (exact least-squares decode); *_delta adds the erase write too.
        return DeltaMLSTM(cfg, dim, erase_beta=(0.0 if variant == "rls" else 1.0), rls_read=True)
    if variant in ("randk", "randqk"):
        # CS/RIP control: freeze the key (and optionally query) projections at random
        # orthogonal blocks — a fixed incoherent measurement matrix the model cannot collapse.
        model = MLSTMBaseline(cfg, dim)
        targets = ("k_proj",) if variant == "randk" else ("q_proj", "k_proj")
        for param in orthogonalize_projections_(model, targets):
            param.requires_grad_(False)
        return model
    if variant.startswith("delta"):
        # delta-rule write; a read spec after the prefix (delta_ns5) adds the NS read on top.
        rest = variant[len("delta"):].lstrip("_")
        ortho = parse_ortho_variant(rest) if rest else None
        if rest and ortho is None:
            raise ValueError(f"unknown variant {variant!r}")
        mode, steps = ortho if ortho else (None, 0)
        return DeltaMLSTM(cfg, dim, ortho_mode=mode, ortho_steps=steps)
    if variant.startswith("pogo"):
        # POGO enforces parameter orthogonality via the optimizer (see train.py), not the forward.
        # A read spec after the prefix (pogo_ns5) selects the orthogonalized-read forward on TOP of
        # POGO weights -> tests whether weight- and state-orthogonality stack. Plain pogo_qk (no read
        # spec) uses the baseline forward.
        rest = variant[len("pogo"):].lstrip("_")
        ortho = parse_ortho_variant(rest)
        if ortho is not None:
            return NSMLSTM(cfg, dim, ortho_mode=ortho[0], ortho_steps=ortho[1])
        return MLSTMBaseline(cfg, dim)
    if variant == "baseline":
        return MLSTMBaseline(cfg, dim)
    if variant.endswith(("_fwd", "_bwd")):
        # straight-through cells of the {forward, backward} x {raw, NS} 2x2 (e.g. ns5_bwd)
        base_v, apply = variant.rsplit("_", 1)
        ortho = parse_ortho_variant(base_v)
        if ortho is not None:
            return NSMLSTM(cfg, dim, ortho_mode=ortho[0], ortho_steps=ortho[1],
                           ortho_apply={"fwd": "forward", "bwd": "backward"}[apply])
    ortho = parse_ortho_variant(variant)
    if ortho is not None:
        return NSMLSTM(cfg, dim, ortho_mode=ortho[0], ortho_steps=ortho[1])
    raise ValueError(f"unknown variant {variant!r}")
