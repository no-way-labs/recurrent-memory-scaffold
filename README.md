# The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory

Code, per-seed results, and the manuscript for a validation and mechanism study of
**"Matrix Orthogonalization Improves Memory in Recurrent Models"** (Tambde, 2026):
an mLSTM whose matrix memory is orthogonalized at read time by Newton–Schulz
iterations, evaluated on MAD noisy associative recall.

**Paper:** [arXiv:2607.19390](https://arxiv.org/abs/2607.19390) ·
[`paper/main.pdf`](paper/main.pdf) (17 pages, 8 figures)

**arXiv v3 files:** [submission package](paper/arxiv-submission-v3.tar.gz) ·
[plain-text abstract](paper/arxiv-abstract-v3.txt)

## What we find

We replicate the recall improvement and identify the orthogonalized read as a
removable training scaffold. Training on MAD noisy recall exhibits a long
chance-level plateau followed by a sharp increase in accuracy. The orthogonalized
read improves conditioning during this plateau and can be removed after escape.
The experiments support three findings:

- **Self-consistency.** An exact recursive least-squares read (the Mesa layer)
  yields a similar benefit. Straight-through variants, delta-rule writes, frozen
  random keys, and Frobenius normalization show no improvement over baseline.
- **Escape dynamics.** Across a learning-rate × task-difficulty grid,
  orthogonalization multiplies escape hazard roughly six-fold, with no detectable
  dependence on difficulty, and widens the range of learning rates that produce
  successful runs.
- **Removal after training.** Adding orthogonalization at inference leaves
  chance-level failures unresolved. Removing it gradually after escape yields
  standard mLSTMs at near-perfect accuracy.

Schedule changes alone recover much of the reported gain. A batch-size ×
learning-rate analysis separates the effects of per-step learning rate and gradient
noise on escape hazard (elasticities +3.0 and −1.65, respectively). Direct decoding
of the memory state recovers roughly half of the associations in behaviorally
failed models, indicating a readout-learning limitation despite substantial stored
information. These results show that fixed-budget recall benchmarks are sensitive
to trainability and provide a tractable setting for investigating abrupt
behavioral transitions through measurements of internal representations.

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
- `paper/` — LaTeX source, `refs.bib`, figures, the compiled PDF, and arXiv v3
  submission files.

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
  title         = {The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory},
  author        = {Aquino-Michaels, Keston},
  year          = {2026},
  eprint        = {2607.19390},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.19390}
}
```

## License

Code (`memrec/`, `experiments/`) is released under the MIT License ([`LICENSE`](LICENSE)).
The manuscript, figures (`paper/`), and result data (`runs/*.jsonl`) are released under
CC BY 4.0 ([`LICENSE-paper`](LICENSE-paper)).
