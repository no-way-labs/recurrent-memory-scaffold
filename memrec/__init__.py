"""memrec — orthogonalization experiments for recurrent memory (mLSTM) on MAD noisy recall.

Variants under test:
  - baseline : vanilla one-block mLSTM (xLSTM), AdamW.
  - ns5      : mLSTM whose memory matrix is Newton-Schulz orthogonalized at read time
               (reproduction of "Matrix Orthogonalization Improves Memory in Recurrent Models").
  - pogo_qk  : baseline mLSTM whose q/k head-block projections are constrained to stay
               orthogonal during training via POGO (parameter orthogonality instead of
               per-token state orthogonalization).
"""

__version__ = "0.1.0"
