"""Orthogonalization of the mLSTM memory matrix (the ns5 read) and its ablations.

The base `newton_schulz_orthogonalize` is copied from the reference repo
(mlstm-orthogonalize/xlstm_ns_mad.py) so the ns5 reproduction is bit-for-bit faithful.
`orthogonalize_memory` generalizes it into the mechanism-ablation knob (direction #1 from
the HN thread): all modes first Frobenius-normalize the memory (fixing scale), then differ
only in how hard they push the singular values toward 1 --

  - mode="ns", steps=k : k Newton-Schulz iterations. steps=0 leaves singular values as-is
    (just normalized); more steps -> closer to a full orthogonal (all singular values = 1).
    So k is a *dose* knob for how much the directions are equalized.
  - mode="polar"        : exact nearest-orthogonal U V^T via SVD (the k -> inf limit).
"""

from __future__ import annotations

import torch

NS_STEPS = 5


def newton_schulz_orthogonalize(c: torch.Tensor, steps: int = NS_STEPS, eps: float = 1e-6) -> torch.Tensor:
    """Frobenius-normalize then apply `steps` Newton-Schulz iterations (X <- 1/2 X (3I - X^T X))."""
    x = c / c.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    eye = eye.view(*([1] * (x.ndim - 2)), x.shape[-1], x.shape[-1])
    for _ in range(steps):
        x = 0.5 * x @ (3.0 * eye - x.transpose(-1, -2) @ x)
    return x


def orthogonalize_memory(c: torch.Tensor, mode: str = "ns", steps: int = NS_STEPS, eps: float = 1e-6) -> torch.Tensor:
    """Orthogonalization dose knob: mode 'ns' (Newton-Schulz, `steps` iters) or 'polar' (exact SVD U V^T)."""
    x = c / c.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    if mode == "polar":
        # nearest-orthogonal factor U V^T = X (X^T X)^{-1/2}, via robust symmetric eigh
        # (memory matrices are often rank-deficient, which makes batched SVD fail to converge).
        # Zeroing the inverse-sqrt on null directions yields the same partial isometry NS converges to.
        xtx = x.transpose(-1, -2) @ x
        lam, v = torch.linalg.eigh(xtx)
        inv_sqrt = torch.where(lam > eps, lam.clamp_min(eps).rsqrt(), torch.zeros_like(lam))
        m = v @ (inv_sqrt.unsqueeze(-1) * v.transpose(-1, -2))
        return x @ m
    if mode == "ns":
        eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        eye = eye.view(*([1] * (x.ndim - 2)), x.shape[-1], x.shape[-1])
        for _ in range(steps):
            x = 0.5 * x @ (3.0 * eye - x.transpose(-1, -2) @ x)
        return x
    raise ValueError(f"unknown ortho mode {mode!r}")
