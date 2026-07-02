"""Paper figures (plotnine) — every figure regenerates from runs/*.jsonl.

  uv run python -m experiments.figures            # all figures -> paper/figs/
  uv run python -m experiments.figures --only survival,basin

Palette: CVD-validated 4-slot categorical (dataviz reference palette); series
identity is also carried by linetype + direct labels, never color alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from plotnine import (aes, annotate, element_blank, element_line, element_rect,
                      element_text, facet_grid, facet_wrap, geom_hline, geom_line,
                      geom_point, geom_step, geom_text, geom_tile, geom_vline, ggplot,
                      guides, labs, scale_color_gradient, scale_color_manual,
                      scale_fill_gradient, scale_linetype_manual, scale_x_continuous,
                      scale_y_continuous, scale_y_discrete, theme, theme_minimal)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "paper", "figs")

# variant identity (fixed assignment everywhere in the paper)
COL = {"baseline": "#2a78d6", "ns5": "#1baf7a", "constLR": "#eda100", "ns5+constLR": "#008300",
       "delta": "#4a3aa7"}
LTY = {"baseline": "solid", "ns5": "dashed", "constLR": "dashdot", "ns5+constLR": "dotted",
       "delta": "solid"}
INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#e1e0d9"
CRIT = "#d03b3b"

ESCAPE_ACC = 0.8
TRACE_RE = re.compile(r"step (\d+) acc ([\d.eE+-]+) loss")


def theme_paper(base_size=9):
    return (theme_minimal(base_size=base_size) +
            theme(text=element_text(color=INK2),
                  axis_text=element_text(color=INK2, size=base_size - 1),
                  axis_title=element_text(color=INK, size=base_size),
                  plot_title=element_text(color=INK, size=base_size + 1, weight="bold"),
                  panel_grid_major=element_line(color=GRID_C, size=0.4),
                  panel_grid_minor=element_blank(),
                  strip_text=element_text(color=INK, size=base_size, weight="bold"),
                  legend_position="none",
                  plot_background=element_rect(fill="white", color=None),
                  panel_background=element_rect(fill="white", color=None)))


def load(name):
    rows = []
    with open(os.path.join(RUNS, name)) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if not r.get("error")]


def parse_trace(row):
    """[(step, acc), ...] from the result's trace lines."""
    out = []
    for t in row.get("trace") or []:
        m = TRACE_RE.match(t)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def escape_step(row, thr=ESCAPE_ACC):
    for step, acc in parse_trace(row):
        if acc >= thr:
            return step
    return None


def km_points(rows, budget):
    """Cumulative escaped-fraction step function (KM with censoring only at budget)."""
    n = len(rows)
    events = sorted(e for e in (escape_step(r) for r in rows) if e is not None and e <= budget)
    xs, ys = [0], [0.0]
    for i, e in enumerate(events, start=1):
        xs.append(e)
        ys.append(i / n)
    xs.append(budget)
    ys.append(len(events) / n)
    return xs, ys


def km_frame(groups, budget):
    """groups: {label: rows}. Returns tidy df for geom_step."""
    parts = []
    for label, rows in groups.items():
        xs, ys = km_points(rows, budget)
        parts.append(pd.DataFrame({"step": xs, "frac": ys, "series": label}))
    return pd.concat(parts, ignore_index=True)


def save(p, name, width, height):
    os.makedirs(OUT, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 150)):
        p.save(os.path.join(OUT, f"{name}.{ext}"), width=width, height=height,
               dpi=dpi, verbose=False)
    print(f"wrote paper/figs/{name}.pdf/.png")


# ---------------------------------------------------------------- fig: survival
def fig_survival():
    """KM escape curves, v80_s512, paper schedule (cosine T=2000) + the two levers."""
    probe = load("probe2k_s512.jsonl")
    groups = {
        "baseline": [r for r in probe if r["variant"] == "baseline"],
        "ns5": [r for r in probe if r["variant"] == "ns5"],
        "constLR": load("constlr2k_s512.jsonl"),
        "ns5+constLR": load("ns5constlr2k_s512.jsonl"),
    }
    df = km_frame(groups, budget=2000)
    ends = df.groupby("series").last().reset_index()
    # end labels hang in the clear right margin at each curve's final height
    # (ns5 and constLR ends are 1/16 apart -> spread them vertically a touch)
    label_y = {"baseline": 0.1875, "ns5": 0.60, "constLR": 0.455, "ns5+constLR": 0.8125}
    p = (ggplot(df, aes("step", "frac", color="series", linetype="series"))
         + geom_step(size=0.9)
         + scale_color_manual(values=COL) + scale_linetype_manual(values=LTY)
         + scale_y_continuous(limits=(0, 1), breaks=[0, .25, .5, .75, 1],
                              labels=lambda l: [f"{v:.0%}" for v in l])
         + scale_x_continuous(limits=(0, 2700), breaks=[0, 500, 1000, 1500, 2000])
         + labs(x="training step", y="fraction of seeds escaped",
                title="Escape curves at the original budget (v80, seq 512, 16 seeds)")
         + theme_paper())
    for _, r in ends.iterrows():
        p += annotate("text", x=2060, y=label_y[r["series"]],
                      label=f'{r["series"]}  {int(round(r["frac"] * 16))}/16',
                      color=COL[r["series"]], size=8, ha="left")
    save(p, "fig_survival", 5.4, 3.2)


# ---------------------------------------------------------------- fig: schedule
def fig_schedule():
    """Schedule censoring: same seeds, baseline only, three schedules (heat-ordered ramp)."""
    groups = {
        "cosine, T=2000 (Tambde 2026)": [r for r in load("probe2k_s512.jsonl") if r["variant"] == "baseline"],
        "constant LR": load("constlr2k_s512.jsonl"),
        "cosine, T=4000": load("budget4k_baseline_s512.jsonl"),
    }
    ramp = {"cosine, T=2000 (Tambde 2026)": "#86b6ef", "constant LR": "#2a78d6",
            "cosine, T=4000": "#0d366b"}
    lty = {"cosine, T=2000 (Tambde 2026)": "solid", "constant LR": "dashdot",
           "cosine, T=4000": "dashed"}
    df = km_frame(groups, budget=4000)
    # 2k-budget series end at 2000, budget4k continues to 4000
    df = df[~((df["series"] != "cosine, T=4000") & (df["step"] > 2000))]
    p = (ggplot(df, aes("step", "frac", color="series", linetype="series"))
         + geom_step(size=0.9)
         + geom_vline(xintercept=2000, color=INK2, linetype="dotted", size=0.5)
         + scale_color_manual(values=ramp) + scale_linetype_manual(values=lty)
         + scale_y_continuous(limits=(0, 1), breaks=[0, .25, .5, .75, 1],
                              labels=lambda l: [f"{v:.0%}" for v in l])
         + scale_x_continuous(breaks=[0, 1000, 2000, 3000, 4000])
         + labs(x="training step", y="fraction of seeds escaped",
                title="The schedule alone flips 8 of 16 seeds (baseline, v80, seq 512)")
         + theme_paper())
    anno = [("cosine, T=2000 (Tambde 2026)", 2080, 0.15, "left"),
            ("constant LR", 2100, 0.50, "left"),
            ("cosine, T=4000", 3960, 0.87, "right")]
    for s, x, y, ha in anno:
        p += annotate("text", x=x, y=y, label=s, color=ramp[s], size=8, ha=ha)
    p += annotate("point", x=2000, y=11 / 16, color="#0d366b", size=2.2)
    p += annotate("text", x=2100, y=0.62, label="11/16 by the same step",
                  color="#0d366b", size=8, ha="left")
    p += annotate("text", x=1950, y=0.97, label="original budget", color=INK2, size=7, ha="right")
    save(p, "fig_schedule", 5.4, 3.2)


# ---------------------------------------------------------------- fig: basin map
def fig_basin():
    rows = load("grid_lr001.jsonl") + load("grid_lr003.jsonl") + load("grid_lr010.jsonl")
    recs = []
    for r in rows:
        tr = parse_trace(r)
        esc = escape_step(r)
        final = r["final_val_acc"]
        dead = final < 0.005 and max(a for _, a in tr) < 0.8  # collapsed below chance
        recs.append(dict(variant=r["variant"], vocab=r["vocab_size"], lr=r["lr"],
                         esc=esc, final=final, dead=dead))
    df = pd.DataFrame(recs)
    cells = []
    for (v, vc, lr), g in df.groupby(["variant", "vocab", "lr"]):
        solved = int((g["esc"].notna() & (g["final"] >= ESCAPE_ACC)).sum())
        dead = int(g["dead"].sum())
        esc = sorted(g.loc[g["esc"].notna(), "esc"])
        med = esc[len(esc) // 2] if esc else None  # same convention as analyze_grid.py
        lab = f"{solved}/8"
        if med is not None and solved:
            lab += f"\n@{int(med)}"
        if dead:
            lab += f"\n†{dead} dead"
        cells.append(dict(variant=v, vocab=str(vc), lr=f"{lr:g}", solved=solved,
                          dead=dead, label=lab))
    cdf = pd.DataFrame(cells)
    cdf["variant"] = pd.Categorical(cdf["variant"], ["baseline", "ns5"])
    cdf["vocab"] = pd.Categorical(cdf["vocab"], ["96", "88", "80"])  # hard at bottom? top->96
    cdf["lr"] = pd.Categorical(cdf["lr"], ["0.001", "0.003", "0.01"])
    cdf["ink"] = np.where(cdf["solved"] >= 5, "white", INK)
    p = (ggplot(cdf, aes("lr", "vocab", fill="solved"))
         + geom_tile(color="white", size=1.5)
         + geom_text(aes(label="label", color="ink"), size=7.5, lineheight=1.1)
         + scale_fill_gradient(low="#cde2fb", high="#0d366b", limits=(0, 8))
         + scale_color_manual(values={"white": "white", INK: INK}, guide=None)
         + facet_grid(". ~ variant")
         + labs(x="constant learning rate", y="vocabulary size",
                title="Basin map: solved/8 (3000 steps, seq 768); † = destabilized")
         + theme_paper()
         + theme(panel_grid_major=element_blank()))
    save(p, "fig_basin", 5.6, 2.9)


# ---------------------------------------------------------------- fig: rank
def fig_rank():
    """Raw-C effective rank trajectories; the delta panel is the dissociation."""
    probe = load("probe2k_s512.jsonl")
    srcs = {
        "baseline": [r for r in probe if r["variant"] == "baseline"],
        "ns5": [r for r in probe if r["variant"] == "ns5"],
        "delta (erase-write)": load("delta2k_s512.jsonl"),
    }
    parts = []
    for panel, rows in srcs.items():
        for r in rows:
            esc = escape_step(r) is not None and r["final_val_acc"] >= ESCAPE_ACC
            for pr in r.get("probe") or []:
                parts.append(dict(panel=panel, seed=r["seed"], step=pr["step"],
                                  erank=pr.get("c_erank"),
                                  outcome="escaped" if esc else "stuck"))
    df = pd.DataFrame(parts).dropna(subset=["erank"])
    df["panel"] = pd.Categorical(df["panel"], list(srcs))
    df["grp"] = df["panel"].astype(str) + df["seed"].astype(str)
    ocol = {"escaped": "#2a78d6", "stuck": "#b9b7ae"}
    p = (ggplot(df, aes("step", "erank", group="grp", color="outcome"))
         + geom_line(size=0.5, alpha=0.75)
         + scale_color_manual(values=ocol)
         + facet_wrap("~panel")
         + scale_x_continuous(breaks=[0, 1000, 2000])
         + labs(x="training step", y="effective rank of memory C",
                title="Rank collapse is a symptom, not the cause (16 seeds/panel)")
         + theme_paper())
    p += annotate("text", x=1150, y=13.3, label="escaped", color=ocol["escaped"], size=8, ha="left")
    p += annotate("text", x=1150, y=1.2, label="stuck", color="#8a887f", size=8, ha="left")
    save(p, "fig_rank", 6.2, 2.6)


# ---------------------------------------------------------------- fig: anneal
def fig_anneal():
    """Scaffold anneal: accuracy traces through the alpha ramp (fixed window 1400->2000)."""
    rows = load("anneal2k_s512.jsonl")
    parts = []
    for r in rows:
        solved = r["final_val_acc"] >= ESCAPE_ACC
        for step, acc in parse_trace(r):
            parts.append(dict(seed=r["seed"], step=step, acc=acc,
                              outcome="solved at α=0" if solved else "not solved"))
    df = pd.DataFrame(parts)
    ocol = {"solved at α=0": "#008300", "not solved": "#b9b7ae"}
    n_solved = sum(1 for r in rows if r["final_val_acc"] >= ESCAPE_ACC)
    p = (ggplot(df, aes("step", "acc", group="seed", color="outcome"))
         + annotate("rect", xmin=1400, xmax=2000, ymin=-0.02, ymax=1.02,
                    fill="#f0efec", alpha=0.6)
         + geom_line(size=0.5, alpha=0.8)
         + geom_hline(yintercept=ESCAPE_ACC, color=INK2, linetype="dotted", size=0.4)
         + scale_color_manual(values=ocol)
         + scale_y_continuous(limits=(-0.02, 1.02), breaks=[0, .5, .8, 1])
         + labs(x="training step", y="validation accuracy",
                title=f"Anneal to raw read (shaded): {n_solved}/16 finish as stock mLSTMs")
         + theme_paper())
    p += annotate("text", x=1700, y=0.08, label=r"$\alpha: 1 \rightarrow 0$", color=INK2, size=8)
    save(p, "fig_anneal", 5.4, 3.0)


# ---------------------------------------------------------------- fig: compute
TRACE_T_RE = re.compile(r"step (\d+) acc ([\d.eE+-]+) loss [\d.eE+-]+ t (\d+)s")


def wall_events(rows, thr=ESCAPE_ACC):
    """[(escape_wall_minutes or None, budget_wall_minutes), ...] per run."""
    out = []
    for r in rows:
        esc, cap = None, 0.0
        for line in r.get("trace") or []:
            m = TRACE_T_RE.match(line)
            if not m:
                continue
            t_min = int(m.group(3)) / 60
            cap = max(cap, t_min)
            if esc is None and float(m.group(2)) >= thr:
                esc = t_min
        out.append((esc, cap))
    return out


def km_wall(rows):
    """Cumulative escaped-fraction (events/n) over measured wall-clock. Plain incidence,
    not Kaplan-Meier: budgets are deterministic per run, and container-speed jitter in
    the caps would let KM's risk-set correction inflate the tail."""
    ev = wall_events(rows)
    n = len(ev)
    xs, ys = [0.0], [0.0]
    for i, e in enumerate(sorted(e for e, _ in ev if e is not None), start=1):
        xs.append(e)
        ys.append(i / n)
    xs.append(max(c for _, c in ev))
    ys.append(ys[-1])
    return xs, ys


def fig_compute():
    """Escape vs measured GPU wall-clock: the per-compute view of Fig. 1."""
    probe = load("probe2k_s512.jsonl")
    con = []
    if os.path.exists(os.path.join(RUNS, "constlr12k_s512.jsonl")):
        con = load("constlr12k_s512.jsonl")
    if len(con) < 16:  # wall-matched arm still draining -> fall back to the 2k-budget runs
        con = load("constlr2k_s512.jsonl")
    groups = {
        "baseline": load("budget4k_baseline_s512.jsonl"),
        "constLR": con,
        "ns5": [r for r in probe if r["variant"] == "ns5"],
        "ns5+constLR": load("ns5constlr2k_s512.jsonl"),
    }
    parts = []
    for label, rows in groups.items():
        xs, ys = km_wall(rows)
        parts.append(pd.DataFrame({"wall": xs, "frac": ys, "series": label}))
    df = pd.concat(parts, ignore_index=True)
    xmax = float(df["wall"].max())
    ends = df.groupby("series").last().reset_index()
    p = (ggplot(df, aes("wall", "frac", color="series", linetype="series"))
         + geom_step(size=0.9)
         + scale_color_manual(values=COL) + scale_linetype_manual(values=LTY)
         + scale_y_continuous(limits=(0, 1), breaks=[0, .25, .5, .75, 1],
                              labels=lambda l: [f"{v:.0%}" for v in l])
         + scale_x_continuous(limits=(0, xmax * 1.36))
         + labs(x="measured wall-clock per run (L4 minutes)", y="fraction of seeds escaped",
                title="The same escape curves, per unit compute (v80, seq 512)")
         + theme_paper())
    # explicit label heights: clear of every curve path, not just endpoints
    label_y = {"baseline": 0.90, "constLR": 0.875, "ns5": 0.5625, "ns5+constLR": 0.74}
    for _, r in ends.iterrows():
        p += annotate("text", x=r["wall"] + xmax * 0.02, y=label_y[r["series"]],
                      label=f'{r["series"]}  {int(round(r["frac"] * 16))}/16',
                      color=COL[r["series"]], size=8, ha="left")
    save(p, "fig_compute", 5.4, 3.2)


# ---------------------------------------------------------------- fig: batch
def fig_batch():
    """Batch x LR corridor map (baseline, constant LR): the corridor moves cold as B shrinks."""
    parts = []
    for bs in (4, 8, 16, 32, 64):
        try:
            rows = load(f"batch{bs}_grid.jsonl")
        except FileNotFoundError:
            continue
        for r in rows:
            tr = parse_trace(r)
            esc = escape_step(r)
            dead = r["final_val_acc"] < 0.005 and max(a for _, a in tr) < 0.8
            parts.append(dict(batch=str(bs), lr=f"{r['lr']:g}", budget=r["steps"],
                              solved=esc is not None and r["final_val_acc"] >= ESCAPE_ACC,
                              dead=dead))
    if not parts:
        print("fig_batch: no data yet")
        return
    df = pd.DataFrame(parts)
    cells = []
    for (b, lr), g in df.groupby(["batch", "lr"]):
        lab = f"{int(g['solved'].sum())}/{len(g)}"
        if g["dead"].sum():
            lab += f"\n†{int(g['dead'].sum())} dead"
        cells.append(dict(batch=b, lr=lr, solved=int(g["solved"].sum()), n=len(g),
                          budget=g["budget"].iloc[0], label=lab))
    cdf = pd.DataFrame(cells)
    cdf["batch"] = pd.Categorical(cdf["batch"], ["4", "8", "16", "32", "64"])
    cdf["lr"] = pd.Categorical(cdf["lr"], ["0.001", "0.003", "0.01"])
    cdf["frac"] = cdf["solved"] / cdf["n"]
    cdf["ink"] = np.where(cdf["frac"] >= 0.6, "white", INK)
    ylabs = {r["batch"]: f'{r["batch"]}  ({r["budget"]} steps)' for _, r in cdf.iterrows()}
    p = (ggplot(cdf, aes("lr", "batch", fill="frac"))
         + geom_tile(color="white", size=1.5)
         + geom_text(aes(label="label", color="ink"), size=7.5, lineheight=1.1)
         + scale_fill_gradient(low="#cde2fb", high="#0d366b", limits=(0, 1))
         + scale_color_manual(values={"white": "white", INK: INK}, guide=None)
         + scale_y_discrete(labels=lambda ls: [ylabs.get(l, l) for l in ls])
         + labs(x="constant learning rate", y="batch size (budget)",
                title="Baseline corridor across batch size (v80, seq 512)")
         + theme_paper()
         + theme(panel_grid_major=element_blank()))
    save(p, "fig_batch", 4.6, 3.2)


# ---------------------------------------------------------------- fig: dose
def fig_dose():
    """Solved fraction vs NS iteration count (whitening dose), cosine T=2000 protocol."""
    probe = load("probe2k_s512.jsonl")
    srcs = {0: [r for r in load("ns02k_s512.jsonl")],
            5: [r for r in probe if r["variant"] == "ns5"]}
    try:
        dose = load("nsdose2k_s512.jsonl")
        for k in (1, 3):
            g = [r for r in dose if r["variant"] == f"ns{k}"]
            if g:
                srcs[k] = g
    except FileNotFoundError:
        pass
    if 1 not in srcs:
        print("fig_dose: ns1/ns3 not landed yet")
    pts = [dict(k=k, frac=sum(r["final_val_acc"] >= ESCAPE_ACC for r in g) / len(g),
                n=len(g)) for k, g in sorted(srcs.items())]
    df = pd.DataFrame(pts)
    df["lab"] = df["frac"].map(lambda f: f"{f:.0%}")
    nb = [r for r in probe if r["variant"] == "baseline"]
    base = sum(r["final_val_acc"] >= ESCAPE_ACC for r in nb) / len(nb)
    p = (ggplot(df, aes("k", "frac"))
         + geom_hline(yintercept=base, color=COL["baseline"], linetype="dashed", size=0.6)
         + geom_line(color=COL["ns5"], size=0.9)
         + geom_point(color=COL["ns5"], size=2.5)
         + geom_text(aes(label="lab"), nudge_y=0.06, color=INK2, size=8)
         + scale_x_continuous(breaks=[0, 1, 3, 5])
         + scale_y_continuous(limits=(0, 1), breaks=[0, .25, .5, .75, 1],
                              labels=lambda l: [f"{v:.0%}" for v in l])
         + labs(x="Newton–Schulz iterations at read", y="fraction of seeds solved",
                title="Whitening dose response (v80, seq 512, cosine T=2000)")
         + theme_paper())
    p += annotate("text", x=4.4, y=base - 0.06, label="baseline", color=COL["baseline"], size=8)
    save(p, "fig_dose", 4.6, 3.0)


FIGS = {"survival": fig_survival, "schedule": fig_schedule, "basin": fig_basin,
        "rank": fig_rank, "anneal": fig_anneal, "batch": fig_batch, "dose": fig_dose,
        "compute": fig_compute}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(FIGS)
    for n in names:
        FIGS[n]()


if __name__ == "__main__":
    main()
