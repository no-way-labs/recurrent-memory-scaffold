"""Local single-run CLI (CPU/MPS) for quick correctness checks of any variant.

  uv run python -m experiments.run_local --variant pogo_qk --seq-len 128 --steps 30 --device cpu
"""

from __future__ import annotations

import argparse
import json

from memrec.train import DEFAULTS, train_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="baseline",
                   help="baseline | pogo_qk | ns<k> (e.g. ns0,ns1,ns3,ns5,ns8) | polar")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--dim", type=int, default=94)
    p.add_argument("--vocab-size", type=int, default=80)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--noise-vocab-size", type=int, default=16)
    p.add_argument("--frac-noise", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--pogo-targets", default=DEFAULTS["pogo_targets"])
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = dict(
        variant=args.variant, seed=args.seed, dim=args.dim, vocab_size=args.vocab_size,
        seq_len=args.seq_len, noise_vocab_size=args.noise_vocab_size, frac_noise=args.frac_noise,
        steps=args.steps, batch_size=args.batch_size, val_batches=args.val_batches,
        eval_every=args.eval_every, lr=args.lr, pogo_targets=args.pogo_targets, device=args.device,
    )
    result = train_one(cfg)
    for t in result["trace_tail"]:
        print(t)
    print("RESULT_JSON " + json.dumps(result, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
