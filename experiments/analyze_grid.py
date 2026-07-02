"""Basin-map grid harvest: Hazard(LR, hardness) with whitening as basin-widener.

  uv run python -m experiments.analyze_grid

Inputs: runs/grid_lr001.jsonl, grid_lr003.jsonl, grid_lr010.jsonl
  (constant LR {1e-3,3e-3,1e-2} x {v80,v88,v96}_s768 x {baseline,ns5} x 8 seeds x 3000 steps)
Tolerates partial files (slices still draining from the Modal queue).

Outputs (stdout):
  A. cell table: solved / mid / flat / destabilized counts + median escape step
  B. log-rank ns5 vs baseline within each (lr, regime) cell column
  C. discrete-time hazard fit (logistic per 200-step interval, Newton-Raphson):
     categorical LR (ref 1e-3), hardness dummies (ref v80), ns5, ns5 x hardness.
     Destabilized runs censored at collapse onset (competing risk, reported separately).
  D. W(hardness): ns5 hazard multiplier per regime from stratified fits.
"""

from __future__ import annotations

import json
import math
import os
import re

import numpy as np

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
FILES = ["grid_lr001.jsonl", "grid_lr003.jsonl", "grid_lr010.jsonl"]
EVAL_EVERY, T_MAX = 200, 3000
REGIME_ORDER = ["v80_s768", "v88_s768", "v96_s768"]


def load_grid():
    rows = []
    for f in FILES:
        p = os.path.join(RUNS, f)
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p) if l.strip()]
    return [r for r in rows if not r.get("error")]


def trace_accs(row):
    out = {}
    for line in row["trace"]:
        m = re.match(r"step (\d+) acc ([\d.eE+-]+)", line)
        out[int(m[1])] = float(m[2])
    return out


def escape_step(row, thresh=0.8):
    for s in sorted(trace_accs(row)):
        if s > 1 and trace_accs(row)[s] >= thresh:
            return s
    return None


def collapse_step(row):
    """First eval at exactly-0-ish accuracy after having been at chance (>=1%).
    Chance is ~1-3%; sustained acc < 0.5% means argmax collapsed (destabilized)."""
    accs = trace_accs(row)
    steps = sorted(s for s in accs if s > 1)
    for i, s in enumerate(steps):
        if accs[s] < 0.005 and all(accs[t] < 0.005 for t in steps[i:]):
            return s
    return None


def outcome(row):
    if row["final_val_acc"] >= 0.8:
        return "solved"
    if collapse_step(row) is not None:
        return "dead"
    return "mid" if row["final_val_acc"] >= 0.2 else "flat"


def logrank(times_a, events_a, times_b, events_b):
    """Two-group log-rank; returns (chi2, p) via chi2_1 survival function."""
    all_t = sorted({t for t, e in zip(times_a + times_b, events_a + events_b) if e})
    O_E, V = 0.0, 0.0
    for t in all_t:
        n_a = sum(1 for x in times_a if x >= t)
        n_b = sum(1 for x in times_b if x >= t)
        d_a = sum(1 for x, e in zip(times_a, events_a) if x == t and e)
        d_b = sum(1 for x, e in zip(times_b, events_b) if x == t and e)
        n, d = n_a + n_b, d_a + d_b
        if n < 2 or d == 0:
            continue
        E_a = d * n_a / n
        O_E += d_a - E_a
        V += d * (n_a / n) * (n_b / n) * (n - d) / (n - 1)
    if V == 0:
        return 0.0, 1.0
    chi2 = O_E * O_E / V
    return chi2, math.erfc(math.sqrt(chi2 / 2))  # chi2_1 sf == erfc(sqrt(x/2))


def person_period(rows):
    """Expand runs into (run, interval) records up to escape / collapse / budget."""
    recs = []
    for r in rows:
        esc, col = escape_step(r), collapse_step(r)
        end = min(x for x in (esc, col, T_MAX) if x is not None)
        for j in range(1, end // EVAL_EVERY + 1):
            t = j * EVAL_EVERY
            recs.append(dict(row=r, t=t, event=1 if (esc is not None and t == esc) else 0))
    return recs


def fit_logistic(X, y, names, iters=60):
    """Newton-Raphson logistic; returns (beta, se) or None if separation blows up."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se


def hazard_design(recs, lr_cats=True):
    names = ["const", "log_t"]
    if lr_cats:
        names += ["lr3e-3", "lr1e-2"]
    names += ["v88", "v96", "ns5", "ns5:v88", "ns5:v96"]
    X, y = [], []
    for rec in recs:
        r = rec["row"]
        ns5 = 1.0 if r["variant"] == "ns5" else 0.0
        v88 = 1.0 if r["vocab_size"] == 88 else 0.0
        v96 = 1.0 if r["vocab_size"] == 96 else 0.0
        x = [1.0, math.log(rec["t"] / 1000)]
        if lr_cats:
            x += [1.0 if abs(r["lr"] - 3e-3) < 1e-9 else 0.0,
                  1.0 if abs(r["lr"] - 1e-2) < 1e-9 else 0.0]
        x += [v88, v96, ns5, ns5 * v88, ns5 * v96]
        X.append(x)
        y.append(rec["event"])
    return np.array(X), np.array(y, float), names


def main():
    rows = load_grid()
    print(f"loaded {len(rows)} grid runs (144 expected)")

    # ---------- A. cell table ----------
    lrs = sorted({r["lr"] for r in rows})
    print("\nA. cell outcomes: solved/mid/flat/dead (n)  |  median escape step of solvers")
    hdr = f"  {'regime':<10} {'variant':<9}" + "".join(f"{f'lr={lr:g}':>22}" for lr in lrs)
    print(hdr)
    for reg in REGIME_ORDER:
        for var in ["baseline", "ns5"]:
            line = f"  {reg:<10} {var:<9}"
            for lr in lrs:
                cell = [r for r in rows if r["regime"] == reg and r["variant"] == var
                        and abs(r["lr"] - lr) < 1e-12]
                if not cell:
                    line += f"{'-':>22}"
                    continue
                cnt = {o: 0 for o in ("solved", "mid", "flat", "dead")}
                for r in cell:
                    cnt[outcome(r)] += 1
                esc = sorted(escape_step(r) for r in cell if escape_step(r))
                med = esc[len(esc) // 2] if esc else "-"
                line += f"{cnt['solved']}/{cnt['mid']}/{cnt['flat']}/{cnt['dead']} ({len(cell)}) @{med}".rjust(22)
            print(line)

    # ---------- B. log-rank ns5 vs baseline per (lr, regime) ----------
    print("\nB. log-rank ns5 vs baseline (escape times, censored at collapse/3000)")
    for lr in lrs:
        for reg in REGIME_ORDER:
            g = {}
            for var in ["baseline", "ns5"]:
                cell = [r for r in rows if r["regime"] == reg and r["variant"] == var
                        and abs(r["lr"] - lr) < 1e-12]
                times = [min(x for x in (escape_step(r), collapse_step(r), T_MAX) if x is not None)
                         for r in cell]
                events = [1 if escape_step(r) else 0 for r in cell]
                g[var] = (times, events)
            if not g["baseline"][0] or not g["ns5"][0]:
                continue
            chi2, p = logrank(*g["baseline"], *g["ns5"])
            nb, nn = sum(g["baseline"][1]), sum(g["ns5"][1])
            print(f"  lr={lr:g} {reg}: baseline {nb}/{len(g['baseline'][0])} vs "
                  f"ns5 {nn}/{len(g['ns5'][0])} escaped   chi2={chi2:.2f} p={p:.4f}")

    # ---------- C. pooled discrete-time hazard fit ----------
    recs = person_period(rows)
    X, y, names = hazard_design(recs)
    beta, se = fit_logistic(X, y, names)
    print(f"\nC. discrete-time hazard fit ({len(recs)} run-intervals, {int(y.sum())} escapes)")
    print(f"  {'term':<10} {'beta':>8} {'se':>7} {'z':>7}   HR")
    for n, b, s in zip(names, beta, se):
        print(f"  {n:<10} {b:>8.3f} {s:>7.3f} {b/s:>7.2f}   {math.exp(b):6.2f}x")

    # ---------- D. W(hardness): ns5 multiplier per regime (stratified) ----------
    print("\nD. W(hardness): ns5 hazard multiplier per regime (stratified fit, all LRs pooled)")
    for reg in REGIME_ORDER:
        sub = [rec for rec in recs if rec["row"]["regime"] == reg]
        if not sub:
            continue
        Xs, ys = [], []
        for rec in sub:
            r = rec["row"]
            Xs.append([1.0, math.log(rec["t"] / 1000),
                       1.0 if abs(r["lr"] - 3e-3) < 1e-9 else 0.0,
                       1.0 if abs(r["lr"] - 1e-2) < 1e-9 else 0.0,
                       1.0 if r["variant"] == "ns5" else 0.0])
            ys.append(rec["event"])
        Xs, ys = np.array(Xs), np.array(ys, float)
        if ys.sum() < 3:
            print(f"  {reg}: too few escapes ({int(ys.sum())})")
            continue
        b, s = fit_logistic(Xs, ys, None)
        print(f"  {reg}: W = {math.exp(b[-1]):.2f}x  (z={b[-1]/s[-1]:.2f}, {int(ys.sum())} escapes)")


if __name__ == "__main__":
    main()
