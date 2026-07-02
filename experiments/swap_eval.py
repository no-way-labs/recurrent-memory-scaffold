"""Swap evaluations: re-evaluate trained checkpoints with the *other* read.

For every checkpoint in --ckpt-dir, evaluate the trained weights under:
  - raw read  : MLSTMBaseline forward (xlstm's own parallel path, no orthogonalization)
  - ns<k> read: NSMLSTM forward with the given mode/steps bolted on at read time

This separates read-time benefit from training-time benefit:
  - ns5-trained model that still solves under the raw read  -> ns was a training scaffold;
    the learned solution does not depend on the orthogonalized read (drop it at inference).
  - failed baseline rescued by the ns read at eval          -> benefit is read-time denoising.

Uses the same fixed validation set as training (seed 10_000 + train seed), so numbers are
directly comparable to the run's final_val_acc.

  uv run python -m experiments.swap_eval --ckpt-dir runs/ckpts/<tag> --out runs/swap_<tag>.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from memrec.data import MADRecallConfig, make_instance, make_val_batches
from memrec.models import MLSTMBaseline, NSMLSTM
from memrec.train import evaluate, set_seed


def build_mad(cfg: dict) -> MADRecallConfig:
    mad = MADRecallConfig(cfg["vocab_size"], cfg["seq_len"], cfg["seq_len"],
                          cfg["noise_vocab_size"], cfg["frac_noise"])
    sample_x, _ = make_instance(mad, np.random.default_rng(cfg["seed"]), is_training=True)
    mad.context_length = len(sample_x)
    return mad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--reads", default="raw,ns5", help="comma list: raw, ns<k>, polar")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    reads = [r.strip() for r in args.reads.split(",") if r.strip()]
    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
    print(f"{len(paths)} checkpoints, reads={reads}, device={device}")

    rows = []
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["cfg"]
        set_seed(cfg["seed"])
        mad = build_mad(cfg)
        val = make_val_batches(mad, cfg["batch_size"], cfg["val_batches"], device,
                               seed=10_000 + cfg["seed"])
        row = {"ckpt": os.path.basename(path), "variant": cfg["variant"], "seed": cfg["seed"],
               "regime": cfg.get("regime"), "steps": cfg["steps"],
               "trained_final_acc": None}
        for read in reads:
            if read == "raw":
                model = MLSTMBaseline(mad, cfg["dim"])
            elif read == "polar":
                model = NSMLSTM(mad, cfg["dim"], ortho_mode="polar", ortho_steps=0)
            elif read.startswith("ns") and read[2:].isdigit():
                model = NSMLSTM(mad, cfg["dim"], ortho_mode="ns", ortho_steps=int(read[2:]))
            else:
                raise ValueError(f"unknown read {read!r}")
            model.load_state_dict(ckpt["state_dict"])
            model.to(device)
            loss, acc = evaluate(model, val, mad.vocab_size)
            row[f"acc_{read}"] = round(acc, 4)
            row[f"loss_{read}"] = round(loss, 4)
        rows.append(row)
        print(json.dumps(row))

    if args.out:
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
