"""Regenerate every quantitative claim in paper/main.tex from runs/*.jsonl.

  uv run python -m experiments.analyze_paper            # all sections
  uv run python -m experiments.analyze_paper --only batch,dose

Sections print in paper order; missing result files are reported as PENDING so the
script runs cleanly while sweeps drain. The basin-map grid has its own analyzer
(analyze_grid.py); section `basin` here adds the no-interaction (uniform-multiplier)
hazard fit the paper quotes.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
from scipy import stats

from .analyze_grid import (collapse_step, escape_step, fit_logistic, load_grid, logrank,
                           person_period, trace_accs)

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")


def load(name):
    p = os.path.join(RUNS, name)
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return [r for r in rows if not r.get("error")]


def solved(r, thr=0.8):
    return r["final_val_acc"] >= thr


def nsolved(rows, thr=0.8):
    return sum(1 for r in rows if solved(r, thr))


def esc_by(rows, step):
    return {r["seed"] for r in rows if (escape_step(r) or 10**9) <= step}


def surv(rows, budget=None):
    """(times, events) censored at each run's own budget (or `budget`)."""
    times, events = [], []
    for r in rows:
        cap = budget or r.get("steps_run") or r["steps"]
        esc, col = escape_step(r), collapse_step(r)
        t = min(x for x in (esc, col, cap) if x is not None)
        times.append(t)
        events.append(1 if esc is not None and esc <= t else 0)
    return times, events


def two_group_hr(rows_a, rows_b, cap_a=None, cap_b=None):
    """Discrete-time hazard ratio (group B vs A) with 95% CI from a 2-group fit."""
    X, y = [], []
    for rows, cap, g in ((rows_a, cap_a, 0.0), (rows_b, cap_b, 1.0)):
        for r in rows:
            budget = cap or r.get("steps_run") or r["steps"]
            esc, col = escape_step(r), collapse_step(r)
            end = min(x for x in (esc, col, budget) if x is not None)
            for j in range(1, end // 200 + 1):
                X.append([1.0, math.log(j * 200 / 1000), g])
                y.append(1 if esc == j * 200 else 0)
    b, s = fit_logistic(np.array(X), np.array(y, float), None)
    lo, hi = math.exp(b[-1] - 1.96 * s[-1]), math.exp(b[-1] + 1.96 * s[-1])
    return math.exp(b[-1]), lo, hi


def plateau_metric(rows, key, steps=(200, 600)):
    vals = []
    for r in rows:
        pr = {p["step"]: p for p in (r.get("probe") or [])}
        v = [pr[s][key] for s in steps if s in pr and key in pr[s]]
        if v:
            vals.append(float(np.mean(v)))
    return float(np.mean(vals)) if vals else None


def sec_replication():
    print("== replication (cosine T=2000)")
    plan = load("planA_results.jsonl")
    for reg in ("v80_s512", "v80_s1024"):
        for var in ("baseline", "ns5"):
            g = [r for r in plan if r["regime"] == reg and r["variant"] == var]
            print(f"  planA {reg} {var}: {nsolved(g)}/{len(g)}")
    b = [r for r in plan if r["variant"] == "baseline"]
    n = [r for r in plan if r["variant"] == "ns5"]
    _, p = stats.fisher_exact([[nsolved(n), len(n) - nsolved(n)],
                               [nsolved(b), len(b) - nsolved(b)]])
    print(f"  pooled ns5 {nsolved(n)}/{len(n)} vs baseline {nsolved(b)}/{len(b)}  Fisher p={p:.3f}")
    probe = load("probe2k_s512.jsonl")
    for var in ("baseline", "ns5"):
        g = [r for r in probe if r["variant"] == var]
        print(f"  probe2k replicate {var}: {nsolved(g)}/{len(g)}")
    tb, eb = surv([r for r in probe if r["variant"] == "baseline"])
    tn, en = surv([r for r in probe if r["variant"] == "ns5"])
    print(f"  log-rank ns5 vs baseline (2k sched): p={logrank(tb, eb, tn, en)[1]:.3f}")


def sec_schedule():
    print("== schedule confound")
    base2k = [r for r in load("probe2k_s512.jsonl") if r["variant"] == "baseline"]
    base4k = load("budget4k_baseline_s512.jsonl")
    con = load("constlr2k_s512.jsonl")
    e2k, e4k = esc_by(base2k, 2000), esc_by(base4k, 2000)
    print(f"  baseline solved by 2000: T2k {len(e2k)}/16, T4k {len(e4k)}/16, "
          f"T4k by 4000: {nsolved(base4k)}/16")
    flips = len(e4k - e2k) + len(e2k - e4k)
    k = min(len(e4k - e2k), len(e2k - e4k))
    p = 2 * stats.binom.cdf(k, flips, 0.5) if flips else 1.0
    print(f"  McNemar (same seeds, @2000): {len(e4k - e2k)} flips fwd, {len(e2k - e4k)} back, p={min(p,1):.4f}")
    print(f"  constant LR @2000: {nsolved(con)}/16 "
          f"(+{sum(1 for r in con if 0.55 <= r['final_val_acc'] < 0.8)} mid 57-70%)")
    ns2k = [r for r in load("probe2k_s512.jsonl") if r["variant"] == "ns5"]
    print(f"  log-rank constLR vs ns5(cos2k): p={logrank(*surv(con), *surv(ns2k))[1]:.2f}")
    hr, lo, hi = two_group_hr(con, ns2k)
    print(f"  ns5-vs-constLR hazard ratio: {hr:.2f} (95% CI {lo:.2f}-{hi:.2f})")
    print(f"  log-rank constLR vs baseline(cos2k): p={logrank(*surv(base2k), *surv(con))[1]:.3f}")
    s1024_4k = load("budget4k_baseline_s1024.jsonl")
    plan1024 = [r for r in load("planA_results.jsonl")
                if r["regime"] == "v80_s1024" and r["variant"] == "baseline"]
    print(f"  s1024: {nsolved(plan1024)}/16 -> {len(esc_by(s1024_4k, 2000))}/16 -> {nsolved(s1024_4k)}/16")
    ns4k = load("ns5_4k_s512.jsonl")
    print(f"  ns5 T4k: {nsolved(ns4k)}/16; crossings MW vs base4k:", end=" ")
    a = [escape_step(r) for r in ns4k if escape_step(r)]
    b = [escape_step(r) for r in base4k if escape_step(r)]
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    print(f"median {int(np.median(a))} vs {int(np.median(b))}, p={u.pvalue:.3f}")
    print(f"  log-rank ns5 vs baseline @T4k: p={logrank(*surv(base4k), *surv(ns4k))[1]:.3f}")


def sec_elimination():
    print("== elimination table (v80_s512 cosine T2k unless noted)")
    cells = [
        ("baseline", "probe2k_s512.jsonl", "baseline"),
        ("delta", "delta2k_s512.jsonl", None),
        ("randk", "randk2k_s512.jsonl", None),
        ("qk_lr_x4", "qkboost2k_s512.jsonl", None),
        ("ns5_fwd", "st2k_s512.jsonl", "ns5_fwd"),
        ("ns5_bwd", "st2k_s512.jsonl", "ns5_bwd"),
        ("ns0", "ns02k_s512.jsonl", None),
        ("ns1", "nsdose2k_s512.jsonl", "ns1"),
        ("ns3", "nsdose2k_s512.jsonl", "ns3"),
        ("ns5", "probe2k_s512.jsonl", "ns5"),
        ("rls", "rls2k_s512.jsonl", None),
        ("constLR", "constlr2k_s512.jsonl", None),
        ("ns5+constLR", "ns5constlr2k_s512.jsonl", None),
    ]
    ref = {}
    for label, f, var in cells:
        rows = load(f)
        if rows is None:
            print(f"  {label:<12} PENDING ({f})")
            continue
        if var:
            rows = [r for r in rows if r["variant"] == var]
        qk = plateau_metric(rows, "qk_grad_cos")
        ce = plateau_metric(rows, "c_erank")
        ref[label] = rows
        print(f"  {label:<12} {nsolved(rows)}/{len(rows)}   qk_cos {qk if qk is None else round(qk,2)}   "
              f"c_erank {ce if ce is None else round(ce,2)}")
    if "ns5" in ref and "ns5_fwd" in ref and "ns5_bwd" in ref:
        st = ref["ns5_fwd"] + ref["ns5_bwd"]
        _, p = stats.fisher_exact([[nsolved(ref["ns5"]), len(ref["ns5"]) - nsolved(ref["ns5"])],
                                   [nsolved(st), len(st) - nsolved(st)]])
        print(f"  ST pooled vs ns5: Fisher p={p:.4f}")
    for lab in ("delta", "randk", "qk_lr_x4", "rls", "ns1", "ns3"):
        if lab in ref and "baseline" in ref:
            p = logrank(*surv(ref["baseline"]), *surv(ref[lab]))[1]
            print(f"  log-rank {lab} vs baseline: p={p:.4f}")
    if "rls" in ref and "ns5" in ref:
        print(f"  log-rank rls vs ns5: p={logrank(*surv(ref['ns5']), *surv(ref['rls']))[1]:.2f}")
    if "ns5+constLR" in ref and "constLR" in ref:
        p = logrank(*surv(ref["constLR"]), *surv(ref["ns5+constLR"]))[1]
        print(f"  log-rank combo vs constLR: p={p:.3f}")
    con = ref.get("constLR")
    if con:
        esc = [r for r in con if escape_step(r)]
        non = [r for r in con if not escape_step(r)]
        for tag, g in (("escapers", esc), ("non-escapers", non)):
            pr = [p_["qk_grad_cos"] for r in g for p_ in (r.get("probe") or [])
                  if p_["step"] == 600 and "qk_grad_cos" in p_]
            if pr:
                print(f"  constLR qk_cos@600 {tag}: {np.mean(pr):.2f} (n={len(g)})")


def sec_swap():
    print("== swap evaluations")
    rows = load("swap_probe2k.jsonl")
    beh = {(r["variant"], r["seed"]): r["final_val_acc"] for r in load("probe2k_s512.jsonl")}
    fb = [r for r in rows if r["variant"] == "baseline" and beh[("baseline", r["seed"])] < 0.8]
    print(f"  failed baselines rescued by NS read at inference: "
          f"{sum(1 for r in fb if r['acc_ns5'] >= 0.8)}/{len(fb)}")
    sn = [r for r in rows if r["variant"] == "ns5" and beh[("ns5", r["seed"])] >= 0.8]
    keep = [r for r in sn if r["acc_raw"] >= 0.8]
    print(f"  solved ns5 surviving raw read: {len(keep)}/{len(sn)} "
          f"(mean {100*np.mean([r['acc_ns5'] for r in sn]):.0f}% -> {100*np.mean([r['acc_raw'] for r in sn]):.0f}%)")
    sb = [r for r in rows if r["variant"] == "baseline" and beh[("baseline", r["seed"])] >= 0.8]
    d = [r["acc_ns5"] - r["acc_raw"] for r in sb]
    print(f"  NS polish on solved baselines: {100*np.mean(d):+.1f} pts (n={len(sb)})")


def sec_anneal():
    print("== scaffold anneal")
    ann = load("anneal2k_s512.jsonl")
    ok = [r for r in ann if solved(r)]
    print(f"  fixed window: {len(ok)}/16 solved at alpha=0, "
          f"acc {100*min(r['final_val_acc'] for r in ok):.1f}-{100*max(r['final_val_acc'] for r in ok):.1f} "
          f"(mean {100*np.mean([r['final_val_acc'] for r in ok]):.1f})")
    g = load("anneal_gated_s512.jsonl")
    if not g:
        print("  gated: PENDING")
        return
    ok = [r for r in g if solved(r)]
    a0 = [r for r in ok if (r.get("final_read_alpha") or 0) == 0]
    trig = [r["anneal_trigger_step"] for r in g if r.get("anneal_trigger_step")]
    print(f"  gated: {len(ok)}/{len(g)} solved, {len(a0)}/{len(ok)} at alpha=0, "
          f"acc mean {100*np.mean([r['final_val_acc'] for r in ok]):.1f}%, "
          f"triggers {sorted(trig)}")
    ext = [r for r in g if (r.get('steps_run') or 0) > r['steps']]
    print(f"  budget extensions used: {len(ext)} (max +{max((r['steps_run']-r['steps'] for r in ext), default=0)})")


def sec_basin():
    print("== basin map: uniform-multiplier hazard fit (paper's quoted numbers)")
    rows = load_grid()
    recs = person_period(rows)
    X, y, names = [], [], ["const", "log_t", "lr3e-3", "lr1e-2", "v88", "v96", "ns5"]
    for rec in recs:
        r = rec["row"]
        X.append([1.0, math.log(rec["t"] / 1000),
                  1.0 if abs(r["lr"] - 3e-3) < 1e-9 else 0.0,
                  1.0 if abs(r["lr"] - 1e-2) < 1e-9 else 0.0,
                  1.0 if r["vocab_size"] == 88 else 0.0,
                  1.0 if r["vocab_size"] == 96 else 0.0,
                  1.0 if r["variant"] == "ns5" else 0.0])
        y.append(rec["event"])
    b, s = fit_logistic(np.array(X), np.array(y, float), names)
    print(f"  ({len(recs)} run-intervals, {int(sum(y))} escapes)")
    for n_, b_, s_ in zip(names, b, s):
        print(f"  {n_:<8} HR {math.exp(b_):6.2f}x  z={b_/s_: .2f}  "
              f"CI [{math.exp(b_-1.96*s_):.2f}, {math.exp(b_+1.96*s_):.2f}]")
    # interaction model: CIs on ns5 x hardness terms (the uniformity claim's uncertainty)
    from .analyze_grid import hazard_design
    Xi, yi, ni = hazard_design(recs)
    bi, si = fit_logistic(Xi, yi, ni)
    for n_, b_, s_ in zip(ni, bi, si):
        if n_.startswith("ns5:"):
            print(f"  {n_:<8} HR {math.exp(b_):6.2f}x  "
                  f"CI [{math.exp(b_-1.96*s_):.2f}, {math.exp(b_+1.96*s_):.2f}]")
    cold_b = [r for r in rows if r["regime"] == "v88_s768" and abs(r["lr"] - 1e-3) < 1e-9
              and r["variant"] == "baseline"]
    cold_n = [r for r in rows if r["regime"] == "v88_s768" and abs(r["lr"] - 1e-3) < 1e-9
              and r["variant"] == "ns5"]
    print(f"  cold cell v88@1e-3: {nsolved(cold_b)}/8 -> {nsolved(cold_n)}/8 "
          f"log-rank p={logrank(*surv(cold_b, 3000), *surv(cold_n, 3000))[1]:.4f}")
    c3 = [r for r in rows if abs(r["lr"] - 3e-3) < 1e-9]
    print(f"  lr 3e-3 all-hardness: ns5 {nsolved([r for r in c3 if r['variant']=='ns5'])}/24 "
          f"vs baseline {nsolved([r for r in c3 if r['variant']=='baseline'])}/24")
    print("  (full cell table / interaction fit: run analyze_grid.py)")


def sec_frontier():
    print("== frontier v96_s768 (constant LR, 4k)")
    rows = load("frontier_v96s768.jsonl")
    for var in ("baseline", "ns5"):
        g = [r for r in rows if r["variant"] == var]
        mids = sum(1 for r in g if 0.2 <= r["final_val_acc"] < 0.8)
        print(f"  {var}: {nsolved(g)}/{len(g)} solved, {mids} mid-escape")


def sec_decode():
    print("== C-decode (regenerated numbers)")
    rows = load("c_decode_probe2k.jsonl")
    if not rows:
        print("  PENDING")
        return
    beh = {(r["variant"], r["seed"]): r["final_val_acc"] for r in load("probe2k_s512.jsonl")}
    groups = {}
    for r in rows:
        b = beh[(r["variant"], r["seed"])]
        groups.setdefault(("solved" if b >= 0.8 else "failed", r["variant"]), []).append((r, b))
    for k in sorted(groups):
        g = groups[k]
        print(f"  {k[0]} {k[1]:<9} n={len(g)}  beh {100*np.mean([b for _, b in g]):5.1f}%  "
              f"decode {100*np.mean([r['decode_o1'] for r, _ in g]):5.1f}%  "
              f"wrong-key {100*np.mean([r['wrongkey_o1'] for r, _ in g]):4.1f}%")
    orc = load("oracle_probe2k.jsonl")
    sol = [r for r in orc if beh[(r["variant"], r["seed"])] >= 0.8]
    print(f"  oracle-query positive control (solved models): native "
          f"{100*np.mean([r['native'] for r in sol]):.0f}% -> oracle "
          f"{100*np.mean([r['oracle+1'] for r in sol]):.0f}% (o+1)")


def _batch_files():
    return {bs: load(f"batch{bs}_grid.jsonl") for bs in (4, 8, 16, 32, 64)}


def sec_batch():
    print("== batch axis")
    files = _batch_files()
    missing = [bs for bs, rows in files.items() if not rows]
    if missing:
        print(f"  PENDING batch sizes: {missing}")
    print(f"  {'batch':<6} {'budget':<7}" + "".join(f"{f'lr={lr:g}':>16}" for lr in (1e-3, 3e-3, 1e-2)))
    for bs, rows in files.items():
        if not rows:
            continue
        line = f"  {bs:<6} {rows[0]['steps']:<7}"
        for lr in (1e-3, 3e-3, 1e-2):
            cell = [r for r in rows if abs(r["lr"] - lr) < 1e-12]
            dead = sum(1 for r in cell if collapse_step(r) is not None and not solved(r))
            line += f"{nsolved(cell)}/{len(cell)} ({dead} dead)".rjust(16)
        print(line)
    pooled = [r for rows in files.values() if rows for r in rows]
    if len(pooled) > 60:
        # hazard fit: log lr + log B (pure heat: c_B=0; token-heat: c_B=c_lr; noise lr^2/B: c_B>0)
        recs = []
        for r in pooled:
            esc, col = escape_step(r), collapse_step(r)
            end = min(x for x in (esc, col, r["steps"]) if x is not None)
            for j in range(1, end // 200 + 1):
                recs.append((r, j * 200, 1 if esc == j * 200 else 0))
        X = np.array([[1.0, math.log(t / 1000), math.log(r["lr"] / 3e-3),
                       math.log(r["batch_size"] / 16)] for r, t, _ in recs])
        y = np.array([e for _, _, e in recs], float)
        names = ["const", "log_t", "log_lr", "log_B"]
        b, s = fit_logistic(X, y, names)
        print(f"  hazard fit ({len(recs)} intervals, {int(y.sum())} escapes):")
        for n_, b_, s_ in zip(names, b, s):
            print(f"    {n_:<7} beta {b_: .3f}  z={b_/s_: .2f}")
        print("    interpretation: pure-heat predicts log_B=0; token-heat log_B==log_lr; "
              "noise-limited log_B>0")
        # same likelihood in the (heat, noise) basis: log lr and log(lr^2/B), centered
        # at (3e-3, 16). heat = lr effect at FIXED noise; noise = lr^2/B at fixed lr.
        Xh = np.array([[1.0, math.log(t / 1000), math.log(r["lr"] / 3e-3),
                        2 * math.log(r["lr"] / 3e-3) - math.log(r["batch_size"] / 16)]
                       for r, t, _ in recs])
        bh, sh = fit_logistic(Xh, y, None)
        for n_, b_, s_ in zip(["const", "log_t", "heat(log lr)", "noise(log lr2/B)"], bh, sh):
            print(f"    {n_:<17} beta {b_: .3f}  z={b_/s_: .2f}  "
                  f"CI [{b_-1.96*s_: .2f},{b_+1.96*s_: .2f}]")
    # ns5 multiplier by batch at lr 3e-3, matched budgets
    pairs = [(4, "batch4_ns5.jsonl", "batch4_grid.jsonl"),
             (8, "batch8_ns5.jsonl", "batch8_grid.jsonl"),
             (16, "ns5constlr2k_s512.jsonl", "constlr2k_s512.jsonl")]
    for bs, nf, bf in pairs:
        ns, ba = load(nf), load(bf)
        if not ns or not ba:
            print(f"  batch {bs} ns5-vs-baseline: PENDING")
            continue
        ba3 = [r for r in ba if abs(r["lr"] - 3e-3) < 1e-9]
        chi2, p = logrank(*surv(ba3), *surv(ns))
        print(f"  batch {bs}: ns5 {nsolved(ns)}/{len(ns)} vs baseline {nsolved(ba3)}/{len(ba3)} "
              f"log-rank p={p:.3f}")
    v96 = load("batch64_v96_cosine.jsonl")
    if v96:
        print(f"  v96_s768 batch64 cosine T2k (paper protocol): {nsolved(v96)}/{len(v96)} solved "
              f"(batch16 cosine: all-chance)")
    else:
        print("  v96 batch64 reconciliation: PENDING")


SECS = {"replication": sec_replication, "schedule": sec_schedule,
        "elimination": sec_elimination, "swap": sec_swap, "anneal": sec_anneal,
        "basin": sec_basin, "frontier": sec_frontier, "decode": sec_decode,
        "batch": sec_batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    names = [n for n in args.only.split(",") if n.strip()] or list(SECS)
    for n in names:
        SECS[n]()
        print()


if __name__ == "__main__":
    main()
