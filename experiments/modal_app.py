"""Modal deployment for the memrec orthogonalization experiments.

Runs baseline / ns5 / pogo_qk across MAD noisy-recall regimes and seeds on L4 GPUs,
capped at 10 concurrent containers (Modal account limit; extras queue).

  modal run experiments/modal_app.py --mode smoke
  modal run experiments/modal_app.py --mode sweep --variants baseline,ns5,pogo_qk --regimes v80_s512,v96_s768 --seeds 5
"""

from __future__ import annotations

import json
import os

import modal

app = modal.App("memrec-ortho")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2.0.2", "torch>=2.8.0", "xlstm==2.0.0", "pogo-torch>=0.1.0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("memrec")
)

GPU = os.environ.get("MEMREC_GPU", "L4")  # A100-40GB for ns5 at batch 64 (OOMs an L4)
MAX_GPUS = 10  # Modal account concurrency limit

# checkpoints land here when --ckpt-tag is set; download with `modal volume get memrec-ckpts ...`
ckpt_volume = modal.Volume.from_name("memrec-ckpts", create_if_missing=True)
CKPT_MOUNT = "/ckpts"

REGIMES = {
    "v80_s512": dict(vocab_size=80, seq_len=512),
    "v80_s768": dict(vocab_size=80, seq_len=768),
    "v88_s768": dict(vocab_size=88, seq_len=768),
    "v80_s1024": dict(vocab_size=80, seq_len=1024),
    "v96_s768": dict(vocab_size=96, seq_len=768),
    "v96_s1024": dict(vocab_size=96, seq_len=1024),
}
BASE = dict(dim=94, noise_vocab_size=16, frac_noise=0.8, batch_size=16, val_batches=16, eval_every=200, lr=3e-3)


@app.function(image=image, gpu=GPU, timeout=60 * 60, max_containers=MAX_GPUS,
              volumes={CKPT_MOUNT: ckpt_volume})
def run(cfg: dict) -> dict:
    from memrec.train import train_one

    try:
        result = train_one(cfg)
        if cfg.get("ckpt_dir"):
            ckpt_volume.commit()
        return result
    except Exception as e:  # keep one failed run from sinking the sweep
        import traceback

        return {"variant": cfg.get("variant"), "regime": cfg.get("regime"), "seed": cfg.get("seed"),
                "error": True, "msg": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}


def _summarize(results):
    import statistics as st

    ok = [r for r in results if not r.get("error")]
    by = {}
    for r in ok:
        by.setdefault((r["regime"], r["variant"]), []).append(r["final_val_acc"])
    variants = sorted({r["variant"] for r in ok})
    regimes = [r for r in REGIMES if any(k[0] == r for k in by)]
    header = f"{'regime':<12}" + "".join(f"{v:>16}" for v in variants)
    print("\n================ final val acc (mean over seeds) ================")
    print(header)
    for reg in regimes:
        row = f"{reg:<12}"
        for v in variants:
            vals = by.get((reg, v), [])
            row += f"{(st.mean(vals) * 100 if vals else float('nan')):>15.1f}%"
        print(row)
    errs = [r for r in results if r.get("error")]
    if errs:
        print(f"\n{len(errs)} errored, e.g.: {errs[0].get('regime')} {errs[0].get('variant')} -> {errs[0].get('msg')}")


@app.local_entrypoint()
def main(mode: str = "smoke", variants: str = "baseline,ns5,pogo_qk",
         regimes: str = "v80_s512,v96_s768,v96_s1024", seeds: int = 5, steps: int = 2000,
         out: str = "memrec_results.jsonl", probe: bool = False, ckpt_tag: str = "",
         lr_schedule: str = "cosine", qk_lr_mult: float = 1.0,
         anneal_start: int = 0, anneal_end: int = 0, lr: float = 0.0, data_seed: int = 0,
         batch_size: int = 0, anneal_gate_acc: float = 0.0,
         anneal_gate_dwell: int = 200, anneal_gate_span: int = 500, seed_start: int = 11,
         lrs: str = ""):
    variant_list = [v.strip() for v in variants.split(",") if v.strip()]
    extra = {"probe": probe, "lr_schedule": lr_schedule, "qk_lr_mult": qk_lr_mult,
             "anneal_start": anneal_start, "anneal_end": anneal_end,
             "anneal_gate_acc": anneal_gate_acc, "anneal_gate_dwell": anneal_gate_dwell,
             "anneal_gate_span": anneal_gate_span}
    # --lrs "1e-3,3e-3,1e-2" sweeps LRs within one invocation; --lr sets a single value
    lr_list = [float(x) for x in lrs.split(",") if x.strip()] if lrs else [lr or None]
    if len(lr_list) == 1 and lr_list[0]:
        extra["lr"] = lr_list[0]
    if data_seed:
        extra["data_seed"] = data_seed
    if batch_size:
        # hold the validation set at a fixed 256 sequences so eval noise is comparable
        # across batch arms (val set size = batch_size * val_batches)
        extra["batch_size"] = batch_size
        extra["val_batches"] = max(1, 256 // batch_size)
    if ckpt_tag:
        extra["ckpt_dir"] = f"{CKPT_MOUNT}/{ckpt_tag}"

    if mode == "smoke":
        cfgs = [
            {**BASE, **REGIMES["v80_s512"], "regime": "v80_s512", "variant": v, "seed": 11,
             "steps": 15, "eval_every": 7, "val_batches": 3}
            for v in variant_list
        ]
        for r in run.map(cfgs):
            tag = "ERR " if r.get("error") else ""
            print(f"[{tag}{r['variant']:<9}] params={r.get('params')} final_acc={r.get('final_val_acc')} "
                  f"{r.get('msg','')} trace={r.get('trace_tail')}")
        return

    regime_list = [x.strip() for x in regimes.split(",") if x.strip()]
    cfgs = []
    for reg in regime_list:
        for v in variant_list:
            for lr_v in lr_list:
                for seed in range(seed_start, seed_start + seeds):
                    cfg = {**BASE, **REGIMES[reg], **extra, "regime": reg, "variant": v,
                           "seed": seed, "steps": steps}
                    if lr_v:
                        cfg["lr"] = lr_v
                    cfgs.append(cfg)
    print(f"launching {len(cfgs)} runs on {GPU} (<= {MAX_GPUS} concurrent): "
          f"{len(regime_list)} regimes x {len(variant_list)} variants x {len(lr_list)} lrs "
          f"x {seeds} seeds, {steps} steps", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "..", "runs", out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results, done = [], 0
    with open(out_path, "w") as f:
        for r in run.map(cfgs):
            results.append(r)
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            done += 1
            acc = r.get("final_val_acc")
            accs = f"{acc * 100:.1f}%" if isinstance(acc, (int, float)) else "n/a"
            tag = "ERR " if r.get("error") else ""
            print(f"[{done}/{len(cfgs)}] {tag}{r['regime']:<10} {r['variant']:<9} seed={r.get('seed')} acc={accs}", flush=True)
    _summarize(results)
    print("\nRESULTS_JSON_DUMP " + json.dumps(results, default=str))
