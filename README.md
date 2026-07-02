# The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory

Code, per-seed results, and the manuscript for a validation and mechanism study of
**"Matrix Orthogonalization Improves Memory in Recurrent Models"** (Tambde, 2026) —
an mLSTM whose matrix memory is orthogonalized at read time by Newton–Schulz
iterations, evaluated on MAD noisy associative recall.

**Paper:** [`paper/main.pdf`](paper/main.pdf)

## What we find

The published effect replicates, but it is not a memory improvement. Training on this
task is a long chance plateau followed by a sharp escape, and the orthogonalized read
works by re-conditioning the learning problem *during* the plateau. It has three
properties:

- **Self-consistent.** An exact recursive least-squares read (the Mesa layer) reproduces
  it; straight-through halves, delta-rule writes, frozen random keys, and plain
  normalization all fail.
- **Uniform.** Across a learning-rate × hardness grid it multiplies the escape hazard
  ~6× with no detectable hardness dependence, widening the workable learning-rate
  corridor that narrows for the baseline.
- **Removable.** It rescues no failed model at inference, and annealed away on an
  escape-triggered schedule it leaves a numerically stock mLSTM at full accuracy.

Much of the published gain needs no architecture at all: solved-rate at a fixed budget
measures *escape hazard*, which follows a heat/noise law (learning-rate elasticity +3.0,
gradient-noise elasticity −1.65). Decoding the memory state directly shows failed models
carry roughly half their associations in linearly recoverable form — the plateau is a
readout failure over half-written storage. Two broader conclusions are developed in the
paper: recall benchmarks used for architecture selection partly measure trainability,
and the system is a fully instrumented model organism of "emergence."

## Layout

- `memrec/` — models (`models.py`), MAD data generator (`data.py`, vendored),
  Newton–Schulz read (`ns.py`), trajectory-neutral probes (`probe.py`), training loop
  (`train.py`).
- `experiments/` — Modal sweep app (`modal_app.py`), local CPU/MPS smoke
  (`run_local.py`), swap evaluations (`swap_eval.py`), ridge C-decode (`c_decode.py`),
  oracle-query probe (`oracle_eval.py`), and the analysis/figure scripts
  (`analyze_grid.py`, `analyze_paper.py`, `figures.py`).
- `runs/` — per-seed results as JSONL, one record per run with full evaluation traces
  and probe telemetry. Every number in the paper regenerates from these.
- `runs/ckpts/` — final checkpoints for the replication sweep (used by the swap, decode,
  and oracle probes).
- `paper/` — LaTeX source, `refs.bib`, figures, and the compiled PDF.

## Install

```bash
uv sync                          # exact pins from uv.lock (used for the paper)
# or
pip install -r requirements.txt  # direct dependencies
```

Regenerating the paper's numbers and figures from the committed `runs/*.jsonl` needs
only `numpy`, `scipy`, and `plotnine`; retraining the model additionally needs `torch`,
`xlstm`, and `pogo-torch`; the cloud sweeps need `modal`. See `requirements.txt` for the
split.

## Reproducing the paper

Every number and figure regenerates from the released `runs/*.jsonl`. Prefix commands
with `uv run` if you installed via uv.

| Paper artifact | Command |
|---|---|
| **All numbers**, §3–§9 | `python -m experiments.analyze_paper` |
| §3 replication (Fig. 1) | `python -m experiments.analyze_paper --only replication` |
| §4 schedule confound (Figs. 2–3) | `python -m experiments.analyze_paper --only schedule` |
| §5 elimination (Table 1) + dose | `python -m experiments.analyze_paper --only elimination` |
| §6 swap + scaffold anneal | `python -m experiments.analyze_paper --only swap,anneal` |
| §7 basin-map hazard fit | `python -m experiments.analyze_paper --only basin` |
| §7 full 144-run basin table | `python -m experiments.analyze_grid` |
| §8 batch escape law | `python -m experiments.analyze_paper --only batch` |
| §9 C-decode table | `python -m experiments.analyze_paper --only decode` |
| **All figures** → `paper/figs/` | `python -m experiments.figures` |
| One figure | `python -m experiments.figures --only survival` (names: `survival schedule compute basin rank anneal batch dose`) |

The C-decode table reads the committed `runs/c_decode_probe2k.jsonl`; to regenerate it
from the checkpoints, run `python -m experiments.c_decode --ckpt-dir runs/ckpts/probe2k`.

### Retraining and sweeps

```bash
# local single run (CPU/MPS), no cloud needed
python -m experiments.run_local --variant ns5 --seq-len 128 --steps 30 --device cpu
```

The full sweeps run on [Modal](https://modal.com) L4 GPUs (`experiments/modal_app.py`);
see that file's flags for the batch, schedule, dose, and anneal sweeps. The one
vocab-96 batch-64 arm uses an A100 (set `MEMREC_GPU=A100-40GB`).

## Citation

```bibtex
@misc{aquinomichaels2026scaffold,
  title  = {The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory},
  author = {Aquino-Michaels, Keston},
  year   = {2026},
  note   = {arXiv preprint; see paper/main.pdf}
}
```

## License

Code (`memrec/`, `experiments/`) is released under the MIT License ([`LICENSE`](LICENSE)).
The manuscript, figures (`paper/`), and result data (`runs/*.jsonl`) are released under
CC BY 4.0 ([`LICENSE-paper`](LICENSE-paper)).
