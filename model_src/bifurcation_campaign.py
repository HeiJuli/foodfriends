#!/usr/bin/env python3
"""
2D bifurcation campaign over the (beta, theta_gate_c) plane, twin ensemble N=2000.

Asks one question: does the model have a bistable / discontinuous regime anywhere,
or only a smooth crossover? beta is the temperature-like control (inverse
temperature of the Boltzmann choice rule) and theta_gate_c is the field-like
control (the p_opp threshold the gate has to clear). The OAT campaign
(sensitivity_campaign.py) stops at beta=30, where the response has already
saturated, and at theta_gate_c=0.45, where F_veg is still falling steeply, so the
region where a discontinuity could sit was never visited.

Order parameter is F_veg_final. Two things are already settled and are not
re-derived here:
  - amp_max variance is extreme-value sampling from a heavy tail (reproduced to
    within 1 pp by bootstrapping the max of the pooled per-agent multipliers), so
    it is not a bistability signal;
  - per-seed F_veg is unimodal at every existing OAT sweep point (Sarle BC 0.27
    to 0.51, all below the 0.555 uniform threshold, no gaps in the support).

The diagnostic is therefore distribution SHAPE across the seeds of a cell, not
their standard deviation: sd cannot tell "wide" from "split". Per cell:
  F_veg mean and sd
  gap   largest gap between consecutive sorted seed values, in F_veg units.
        THIS IS THE FLAG. Absolute, not normalised by the cell range: a split
        cell has a large range too, so gap/range does not grow with separation
        and has near-zero power (measured: 2% at a 0.10 split, n=12).
  BC    Sarle bimodality coefficient, (g1^2 + 1) / (g2 + 3(n-1)^2/((n-2)(n-3)))
        with g2 the excess kurtosis. Reported alongside, not flagged on: at
        n=12 it is the weaker test of the two (numbers below). It does clear
        5/9 on a clean split -- a perfect symmetric 12-point split gives
        g1=0, g2=-2.44, BC=0.63 -- but it loses that under realistic noise.

DETECTION LIMIT (simulated here, 20k replicates, Gaussian cells at sd=0.015 and
n=12, splits drawn by assigning each seed to either well with p=0.5):

  separation      gap > 0.05      BC > 5/9
  none (null)         0.01%          1.2%     <- false positive rate
  0.06                 2.5%          5.8%     <- the design floor
  0.10                  83%           38%
  0.20                 100%          98%

So a null result from this grid means "no bistability with attractors separated
by more than about 0.1 in F_veg". State that limit whenever the null is quoted;
a negative result without its detection limit is not a result.

The sd=0.015 above is pessimistic against what N=2000 actually does. Over the 35
OAT sweep points the within-cell sd is 0.0071 median, 0.0255 max, and the largest
gap ever seen there (n=30, all unimodal) is 0.0145, well under the 0.05 flag.
Power and false positives against the realised range, n=12:

  cell sd        false positive      power vs a 0.10 split
  0.007 (median)      0.00%                  99.9%
  0.015               0.03%                  82.9%
  0.025 (worst)       1.70%                  41.2%

DO NOT recalibrate the flag on the cell's own sd. It looks like the careful thing
to do and it destroys the test: a split inflates that cell's sd, so the gap stops
looking significant and power collapses to under 3% across the whole sd range.
A grid-median recalibration is no better than the fixed threshold either (31% vs
41% power at sd=0.025, same false-positive rate). Both measured. The fixed
threshold wins; leave it alone. This is the same trap as normalising the gap by
the cell range, which is why that was dropped too.

Grid is 7 beta x 7 theta_gate_c = 49 cells. The default configuration
(beta=13, c=0.35) is a grid point and is marked on both panels. The c axis is
deliberately finer than the OAT sweep's over 0.25-0.45: that is where F_veg is
still falling steeply and so where a discontinuity would sit, and 0.10 spacing
covers the whole steepening region with a single interior point.

Nine of the 49 cells are already measured by the OAT campaign and are loaded
from its pickle rather than rerun -- the beta sweep at the baseline gate and the
theta_gate_c sweep at the baseline beta, i.e. the cross through the default
configuration. Identical BASE_PARAMS and the same paired seeds, so they are on
the same footing; they carry 30 seeds against 12 elsewhere, which is where the
gap diagnostic most wants the power. Consequences: n varies across the grid, so
the figure annotates it per cell, and the 0.05 flag is calibrated at n=12, which
makes it conservative on the reused cells. --oat "" reruns everything instead.

Outputs are tagged <date>_N<N> -- a same-day rerun that reuses a filename
silently overwrites the previous ensemble:
  model_output/bifurcation_campaign_<tag>.pkl      raw per-run rows
  model_output/bifurcation_summary_<tag>.csv       per cell, mean/sd/BC/gap
  visualisations_output/bifurcation_diagram_<tag>.pdf

RUN THIS ON THE COMPUTE SERVER. The full campaign is 49 cells x 12 runs, less
the 9 reused from the OAT pickle: 480 new runs at 150k steps, about 20-24
core-hours at the measured 153 s per run.

    python bifurcation_campaign.py --runs 12 --cores <n>

Figures rebuild locally from the pickle without rerunning anything:

    python bifurcation_campaign.py --plot-only <tag>
"""
import sys, os, io, random, argparse, contextlib
from multiprocessing import Pool
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

sys.path.append('.')
sys.path.append('..')
sys.path.append('../plotting')
import model_main
import model_runner_mp
import sensitivity_campaign as sc
from plot_styles import (set_publication_style, apply_axis_style, COLORS,
                         ECO_CMAP, ECO_DIV_CMAP)

# --- grid ------------------------------------------------------------------
BETAS = [5, 9, 13, 20, 30, 50, 80]
GATES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.65]
DEFAULT = {"beta": 13, "theta_gate_c": 0.35}

BC_UNIFORM = 5.0 / 9.0     # 0.5556; reported only -- weaker than gap at n~12
GAP_FLAG = 0.05            # F_veg units; calibration and power in the module docstring

# Inherit from the runner that produced the reported ensemble, NOT from
# model_main.params -- the latter is for ad-hoc single runs and differs. Same
# override block as sensitivity_campaign.BASE_PARAMS, for the same reasons.
BASE_PARAMS = dict(model_runner_mp.DEFAULT_PARAMS)
BASE_PARAMS.update({
    "agent_ini": "twin",         # the reported ensemble; sample-max would force N to 385
    "N": 2000,
    "survey_file": "../data/hierarchical_agents.csv",
    "topology": "homophilic_emp",
    "decay": 0.7,                # absent from both runners' DEFAULT_PARAMS
    "snapshot_dense_start": 0,   # observables computed in-worker; dense snaps unused
})

# The campaigns inherit kappa from the runner above. An absent key falls through to
# model_main's .get("kappa", 1.0) and the whole campaign then sweeps at face value
# while everything else runs discounted -- same class as the tau = 11,700 incident.
assert "kappa" in BASE_PARAMS, "ERROR: no kappa in model_runner_mp.DEFAULT_PARAMS"


# --- reuse from the OAT campaign -------------------------------------------
# sensitivity_campaign.py swept beta at the baseline gate and theta_gate_c at
# the baseline beta, on the same BASE_PARAMS and the same paired seeds, so the
# cross through the default configuration is already measured and rerunning it
# would buy nothing but a smaller n.
OAT_PKL = "../model_output/sensitivity_campaign_20260820_N2000.pkl"
OAT_BASE = {"beta": 13, "theta_gate_c": 0.35}


def load_oat_cells(path):
    """Rows of the OAT pickle that land on this grid, relabelled from
    (param, value) to (beta, theta_gate_c). Returns the rows and the set of
    cells they cover."""
    oat = pd.read_pickle(path)
    parts = []
    for prm, other in (("beta", "theta_gate_c"), ("theta_gate_c", "beta")):
        d = oat[oat.param == prm].copy()
        d[prm] = d["value"]
        d[other] = OAT_BASE[other]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d[d.beta.isin(BETAS) & d.theta_gate_c.isin(GATES)]
    # expand_baseline relabels the shared baseline block under every parameter,
    # so the default cell arrives twice -- keep one copy
    d = d.drop_duplicates(subset=["beta", "theta_gate_c", "seed"])
    d = d.drop(columns=["param", "value"])
    d["beta"] = d["beta"].astype(int)
    return d, set(zip(d.beta, d.theta_gate_c))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _run_cell(job):
    beta, gate, seed, steps = job
    np.random.seed(seed); random.seed(seed)
    p = BASE_PARAMS.copy()
    p.update({"seed": seed, "steps": steps, "tau_persistence": None,
              "beta": beta, "theta_gate_c": gate})
    with contextlib.redirect_stdout(io.StringIO()):
        m = model_main.Model(p, pmf_tables=sc.pmf_tables())
        m.run()
    return {"beta": beta, "theta_gate_c": gate, "seed": seed, **sc._observables(m)}


def build_jobs(runs, steps, skip=()):
    """Seeds are paired across cells (42..42+runs-1), as in the OAT campaign:
    differences between cells are the parameters, not the draw. Cells in `skip`
    come from the OAT pickle instead (see load_oat_cells)."""
    skip = set(skip)
    return [(b, c, 42 + i, steps)
            for b in BETAS for c in GATES if (b, c) not in skip
            for i in range(runs)]


# ---------------------------------------------------------------------------
# Per-cell diagnostics
# ---------------------------------------------------------------------------
def bimodality(x):
    """Sarle's BC on sample-corrected moments. NaN below n=4, where the formula
    is undefined, and on a degenerate cell."""
    n = len(x)
    if n < 4 or np.ptp(x) == 0:
        return np.nan
    g1 = skew(x, bias=False)
    g2 = kurtosis(x, bias=False)          # excess
    return (g1 ** 2 + 1.0) / (g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def cell_stats(df):
    rows = []
    for b in BETAS:
        for c in GATES:
            f = df[(df.beta == b) & (df.theta_gate_c == c)]["F_veg_final"].values
            f = np.sort(np.asarray(f, float))
            f = f[np.isfinite(f)]
            n = len(f)
            rows.append({
                "beta": b, "theta_gate_c": c, "n": n,
                "F_mean": f.mean() if n else np.nan,
                "F_sd": f.std(ddof=1) if n > 1 else np.nan,
                "BC": bimodality(f),
                "gap": np.max(np.diff(f)) if n > 2 else np.nan,
            })
    return pd.DataFrame(rows)


def _mat(stats, col):
    return np.array([[stats[(stats.beta == b) & (stats.theta_gate_c == c)][col].values[0]
                      for c in GATES] for b in BETAS], float)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _beta_colour(i):
    return ECO_CMAP(0.25 + 0.75 * i / max(len(BETAS) - 1, 1))


def _panel_a(ax, df, stats, panel_label="A"):
    """F_veg_final against the gate threshold, one line per beta, with every
    seed overplotted -- a split ensemble is visible directly, not through a
    summary statistic. Shared by the two-panel figure and the SI standalone."""
    dx = 0.005                              # per-beta x offset, < gate spacing
                                            # (0.05 spacing over 0.25-0.45)
    off = (np.arange(len(BETAS)) - (len(BETAS) - 1) / 2) * dx
    ends = []
    for i, b in enumerate(BETAS):
        col = _beta_colour(i)
        d = df[df.beta == b]
        ax.scatter(d.theta_gate_c.values + off[i], d.F_veg_final.values,
                   s=7, color=col, alpha=0.35, lw=0, zorder=2)
        s = stats[stats.beta == b].sort_values("theta_gate_c")
        ax.plot(s.theta_gate_c.values + off[i], s.F_mean.values, 'o-',
                color=col, ms=3.5, lw=1.6, zorder=3)
        ends.append([s.F_mean.values[-1], b, col])
    # line-end labels, pushed apart where curves converge
    ends.sort(key=lambda e: e[0])
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.035 * span)
    for y, b, col in ends:
        ax.annotate(rf"$\beta={b:g}$", (GATES[-1] + off[-1], y),
                    textcoords="offset points", xytext=(7, 0), va='center',
                    fontsize=7, color=col)
    js = BETAS.index(DEFAULT["beta"])
    sub = stats[(stats.beta == DEFAULT["beta"]) &
                (stats.theta_gate_c == DEFAULT["theta_gate_c"])]
    ax.plot(DEFAULT["theta_gate_c"] + off[js], sub.F_mean.values[0], marker='s',
            ms=13, mfc='none', mec=COLORS['highlight'], mew=2.0, ls='none',
            zorder=4, label='default configuration')
    ax.set_xlim(GATES[0] - 0.05, GATES[-1] + 0.09)
    ax.set_xticks(GATES)
    ax.set_xlabel(r"$c$ (gate threshold)")
    ax.set_ylabel(r"$F_{veg}$ (final)")
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    apply_axis_style(ax)
    if panel_label:
        ax.text(-0.10, 1.02, panel_label, transform=ax.transAxes,
                fontsize=11, fontweight='bold')


def fig_panel_a(df, stats, out):
    """Panel A alone, for the SI -- the two-panel figure is kept as well."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    _panel_a(ax, df, stats, panel_label=None)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


def fig_bifurcation(df, stats, out):
    """A: F_veg_final against the gate threshold, one line per beta, with every
    seed overplotted -- a split ensemble is visible directly, not through a
    summary statistic. B: the bimodality diagnostic over the whole plane."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4),
                                   gridspec_kw={'width_ratios': [1.15, 1.0]})

    # --- A: bifurcation diagram -----------------------------------------
    _panel_a(axA, df, stats)

    # --- B: bimodality heatmap --------------------------------------------
    bc, gap, nn = _mat(stats, "BC"), _mat(stats, "gap"), _mat(stats, "n")
    im = axB.imshow(gap, cmap=ECO_CMAP, origin='lower', aspect='auto',
                    vmin=0.0, vmax=max(0.05, np.nanmax(gap)))
    axB.set_xticks(range(len(GATES))); axB.set_xticklabels([f"{c:g}" for c in GATES], fontsize=8)
    axB.set_yticks(range(len(BETAS))); axB.set_yticklabels([f"{b:g}" for b in BETAS], fontsize=8)
    axB.set_xlabel(r"$c$ (gate threshold)")
    axB.set_ylabel(r"$\beta$ (inverse temp.)")
    for i in range(len(BETAS)):
        for j in range(len(GATES)):
            if np.isfinite(gap[i, j]):
                hot = gap[i, j] > GAP_FLAG
                axB.text(j, i, f"{gap[i, j]:.3f}", ha='center', va='bottom',
                         fontsize=7.5, color=COLORS['meat'] if hot else '#222',
                         fontweight='bold' if hot else 'normal')
            if np.isfinite(bc[i, j]):
                # n varies: reused OAT cells carry 30 seeds against 12 elsewhere
                axB.text(j, i, f"BC {bc[i, j]:.2f}  n={nn[i, j]:.0f}",
                         ha='center', va='top', fontsize=4.8, color='#555')
    axB.plot(GATES.index(DEFAULT["theta_gate_c"]), BETAS.index(DEFAULT["beta"]),
             marker='s', ms=15, mfc='none', mec=COLORS['highlight'], mew=2.0)
    cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02)
    cb.set_label(r"largest gap between adjacent runs, $F_{veg}$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(GAP_FLAG, color='#222', lw=1.2)
    axB.text(-0.14, 1.02, "B", transform=axB.transAxes, fontsize=11, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


# ---------------------------------------------------------------------------
def print_report(stats):
    print(f"\n{'='*78}\n PER CELL (order parameter F_veg_final)\n{'='*78}")
    print(f"{'beta':>6}{'c':>7}{'n':>4}{'mean':>9}{'sd':>9}{'gap':>8}{'BC':>8}")
    for r in stats.itertuples():
        mark = ""
        if np.isfinite(r.gap) and r.gap > GAP_FLAG:
            mark += "  SPLIT"
        if r.beta == DEFAULT["beta"] and r.theta_gate_c == DEFAULT["theta_gate_c"]:
            mark += "  *default"
        print(f"{r.beta:>6g}{r.theta_gate_c:>7g}{r.n:>4d}{r.F_mean:>9.3f}"
              f"{r.F_sd:>9.3f}{r.gap:>8.3f}{r.BC:>8.3f}{mark}")
    flagged = stats[stats.gap > GAP_FLAG]
    print()
    if len(flagged):
        print(f"WARNING: {len(flagged)} of {len(stats)} cells show a split "
              f"(largest gap > {GAP_FLAG:g} in F_veg):")
        for r in flagged.itertuples():
            print(f"  beta={r.beta:g}, c={r.theta_gate_c:g}: "
                  f"gap={r.gap:.3f}, sd={r.F_sd:.3f}, BC={r.BC:.3f}")
    else:
        print(f"INFO: no cell splits -- smooth crossover across the whole grid, "
              f"down to the {GAP_FLAG:g} detection limit (see module docstring).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=12, help='runs (seeds) per grid cell')
    ap.add_argument('--steps', type=int, default=150000)
    ap.add_argument('--cores', type=int, default=max(1, int(0.75 * os.cpu_count())))
    ap.add_argument('--N', type=int, help='override population size (smoke tests only; '
                                          'the production campaign is N=2000)')
    ap.add_argument('--tag', help='override the output tag (use for smoke tests)')
    ap.add_argument('--oat', default=OAT_PKL,
                    help='OAT pickle to reuse overlapping cells from; '
                         'pass "" to rerun every cell')
    ap.add_argument('--plot-only', metavar='TAG',
                    help='regenerate figures from an existing campaign pkl; '
                         'TAG is <date>_N<N>, e.g. 20260821_N2000')
    args = ap.parse_args()
    if args.N:
        BASE_PARAMS["N"] = args.N
        args.oat = ""          # the OAT ensemble is N=2000; not comparable

    os.makedirs("../visualisations_output", exist_ok=True)
    os.makedirs("../model_output", exist_ok=True)
    set_publication_style()
    tag = args.plot_only or args.tag or \
        f"{date.today().strftime('%Y%m%d')}_N{BASE_PARAMS['N']}"
    pkl = f"../model_output/bifurcation_campaign_{tag}.pkl"

    if args.plot_only:
        df = pd.read_pickle(pkl)
    else:
        reuse, covered = (load_oat_cells(args.oat) if args.oat
                          else (None, set()))
        jobs = build_jobs(args.runs, args.steps, covered)
        print(f"INFO: {len(BETAS)}x{len(GATES)} = {len(BETAS)*len(GATES)} cells, "
              f"{len(covered)} reused from {args.oat or 'nothing'} "
              f"({0 if reuse is None else len(reuse)} runs), "
              f"{len(jobs)} new runs at {args.runs}/cell, "
              f"N={BASE_PARAMS['N']}, kappa={BASE_PARAMS['kappa']}, "
              f"steps={args.steps}, on {args.cores} cores")
        with Pool(args.cores) as pool:
            df = pd.DataFrame(pool.map(_run_cell, jobs))
        if reuse is not None and len(reuse):
            df = pd.concat([df, reuse], ignore_index=True)
        df.to_pickle(pkl)
        print(f"INFO: Saved -> {pkl}")

    stats = cell_stats(df)
    stats.to_csv(f"../model_output/bifurcation_summary_{tag}.csv", index=False)
    print_report(stats)
    fig_bifurcation(df, stats,
                    f"../visualisations_output/bifurcation_diagram_{tag}.pdf")
    fig_panel_a(df, stats,
                f"../visualisations_output/bifurcation_panelA_{tag}.pdf")
    print("\nINFO: done.")


if __name__ == "__main__":
    main()
