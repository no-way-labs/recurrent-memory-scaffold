"""Analysis for the probe sweeps: escape-preconditioning hypothesis tests.

  uv run python -m experiments.analyze_probe

Inputs: runs/probe2k_s512.jsonl (baseline+ns5 @ T_max=2000, probe on)
        runs/probe4k_baseline_s512.jsonl (baseline @ T_max=4000, probe on)

Outputs (stdout):
  A. probe time series by variant x outcome group (solved / mid / flat)
  B. plateau-phase contrast ns5 vs baseline (grad SNR + C spectrum, steps <= 600)
  C. escape-step prediction from plateau probes (probe4k): Spearman rank corr.
"""

from __future__ import annotations

import json
import re


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def escape_step(row):
    for line in row["trace"]:
        m = re.match(r"step (\d+) acc ([\d.]+)", line)
        if float(m[2]) >= 0.8:
            return int(m[1])
    return None


def outcome(row):
    a = row["final_val_acc"]
    return "solved" if a >= 0.8 else ("mid" if a >= 0.2 else "flat")


def probe_at(row, step):
    for p in row["probe"]:
        if p["step"] == step:
            return p
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def perm_p(xs, ys, obs, n=20000):
    import random
    rng = random.Random(0)
    ys = ys[:]
    cnt = 0
    for _ in range(n):
        rng.shuffle(ys)
        if abs(spearman(xs, ys)) >= abs(obs):
            cnt += 1
    return cnt / n


def main():
    p2k = [r for r in load("runs/probe2k_s512.jsonl") if not r.get("error")]
    p4k = [r for r in load("runs/probe4k_baseline_s512.jsonl") if not r.get("error")]

    # ---------- A. time series by variant x outcome ----------
    steps_shown = [1, 200, 600, 1000, 1400, 2000]
    metrics = ["grad_cos", "qk_grad_cos", "c_erank", "c_log10_cond", "grad_norm"]
    print("=" * 100)
    print("A. probe2k time series (mean within variant x outcome group)")
    for variant in ["baseline", "ns5"]:
        rows = [r for r in p2k if r["variant"] == variant]
        groups = {}
        for r in rows:
            groups.setdefault(outcome(r), []).append(r)
        for g, rs in sorted(groups.items()):
            print(f"\n{variant} / {g} (n={len(rs)})")
            print(f"  {'step':>6}" + "".join(f"{m:>14}" for m in metrics))
            for s in steps_shown:
                vals = [mean([probe_at(r, s) and probe_at(r, s).get(m) for r in rs]) for m in metrics]
                print(f"  {s:>6}" + "".join(f"{v:>14.4f}" for v in vals))

    # ---------- B. plateau contrast: ns5 vs baseline, pre-escape only ----------
    print("\n" + "=" * 100)
    print("B. plateau-phase contrast (steps 200+600 averaged, only seeds still at chance at step 600)")
    for variant in ["baseline", "ns5"]:
        rows = [r for r in p2k if r["variant"] == variant]
        plateau = [r for r in rows if (escape_step(r) or 9999) > 600]
        for m in metrics:
            v = mean([mean([probe_at(r, s).get(m) for s in (200, 600)]) for r in plateau])
            print(f"  {variant:>9} {m:>13}: {v:.4f}   (n={len(plateau)})")
        if variant == "ns5":
            co = mean([mean([probe_at(r, s).get("co_erank") for s in (200, 600)]) for r in plateau])
            print(f"  {variant:>9} {'co_erank':>13}: {co:.4f}   (post-read effective rank)")

    # ---------- C. escape-step prediction from plateau probes (probe4k) ----------
    print("\n" + "=" * 100)
    print("C. probe4k: do plateau probes predict the escape step? (Spearman, permutation p)")
    pred_metrics = ["grad_cos", "qk_grad_cos", "c_erank", "c_log10_cond", "grad_norm"]
    crossers = [(r, escape_step(r)) for r in p4k]
    crossers = [(r, e) for r, e in crossers if e is not None and e > 600]
    print(f"  seeds with escape > step 600: {len(crossers)}")
    es = [e for _, e in crossers]
    for m in pred_metrics:
        xs = [mean([probe_at(r, s).get(m) for s in (200, 600)]) for r, _ in crossers]
        rho = spearman(xs, es)
        print(f"  plateau {m:>13} vs escape step: rho={rho:+.3f}  p~{perm_p(xs, es, rho):.3f}")


if __name__ == "__main__":
    main()
