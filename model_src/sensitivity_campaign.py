#!/usr/bin/env python3
"""
One-at-a-time (OAT) sensitivity campaign for the sample-max ensemble.

Answers three reviewer comments in one pass:
  R1.3  -- which inputs is the model most sensitive to?
  R4.16 -- why is memory fixed at M=9?
  R4.21 -- a sensitivity analysis of the attenuation factor lambda is necessary.

Seven parameters are swept one at a time around the submitted configuration, each
against six observables. Every sweep point uses the SAME seed set (42..42+runs-1),
so arms are paired on network realisation and initial condition: differences are
the parameter, not the draw.

Parameters swept (baseline starred in SWEEPS below):
  decay (lambda), M, beta, gamma, immune_n, theta_gate_c, theta_gate_k

Observables:
  F_veg_final  steady-state vegetarian fraction
  F_c          F_veg at max acceleration, F<0.5   (same estimator as
               analysis/results_analysis.py:analysis_5_inflection)
  t_50         first crossing of F_veg = 0.5, in ksteps
  amp_mean     mean amplification multiplier over credited agents
  amp_p90      90th percentile of the same
  amp_max      maximum of the same (the "88x" ceiling)

NOTE on M: tau_persistence is defined as M*2*N (model_main.py:336), so the M sweep
moves the dwell-time window with the memory length. That is the model's definition
of M, not a confound introduced here, but it must be stated wherever the M sweep is
reported.

Outputs (per date):
  model_output/sensitivity_campaign_<date>.pkl        raw per-run rows
  model_output/sensitivity_summary_<date>.csv         per sweep point, mean/std
  visualisations_output/sensitivity_tornado_<date>.pdf
  visualisations_output/sensitivity_heatmap_<date>.pdf
  visualisations_output/sensitivity_response_curves_<date>.pdf
  visualisations_output/sensitivity_lambda_<date>.pdf
  visualisations_output/sensitivity_table_<date>.tex  SI table
  visualisations_output/sensitivity_interaction_<date>.pdf   (--interaction only)

RUN THIS ON THE COMPUTE SERVER, not the laptop. The full campaign is 870 model
runs (1370 with --interaction) at roughly 5 s each.

    python sensitivity_campaign.py --runs 30 --interaction --cores <n>

Figures and the SI table can then be rebuilt locally from the returned pickles
without rerunning anything:

    python sensitivity_campaign.py --plot-only <date>

See claude_stuff/Review/revision_triage.md, Wave 4 item 1.
"""
import sys, os, io, random, pickle, argparse, contextlib
from multiprocessing import Pool
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sys.path.append('.')
sys.path.append('..')
sys.path.append('../plotting')
import model_main
from plot_styles import (set_publication_style, apply_axis_style, COLORS,
                         ECO_CMAP, ECO_DIV_CMAP)

DIRECT_REDUCTION_KG = 2054 - 1390   # meat_CO2 - veg_CO2

# --- sweep design ----------------------------------------------------------
# value lists; the baseline value must appear in each list exactly once
BASELINE = {"decay": 0.7, "M": 9, "beta": 13, "gamma": 0.3,
            "immune_n": 0.10, "theta_gate_c": 0.35, "theta_gate_k": 35}

SWEEPS = {
    "decay":        [0.5, 0.6, 0.7, 0.8, 0.9],
    "M":            [3, 6, 9, 12, 15],
    "beta":         [5, 9, 13, 20, 30],
    "gamma":        [0.10, 0.30, 0.50, 0.70, 1.00],   # 1.0 = simple-contagion null
    "immune_n":     [0.0, 0.05, 0.10, 0.15, 0.20],
    "theta_gate_c": [0.25, 0.30, 0.35, 0.40, 0.45],
    "theta_gate_k": [15, 25, 35, 45, 55],
}

PLABEL = {"decay": r"$\lambda$ (attenuation)", "M": r"$M$ (memory)",
          "beta": r"$\beta$ (inverse temp.)", "gamma": r"$\gamma$ (dim. returns)",
          "immune_n": r"$f_{imm}$ (immune)", "theta_gate_c": r"$c$ (gate threshold)",
          "theta_gate_k": r"$k$ (gate steepness)"}

OBS = ["F_veg_final", "F_c", "t_50", "amp_mean", "amp_p90", "amp_max"]
OLABEL = {"F_veg_final": r"$F_{veg}$ (final)", "F_c": r"$F_c$ (max accel.)",
          "t_50": r"$t_{50}$ (ksteps)", "amp_mean": "mean amplification",
          "amp_p90": "p90 amplification", "amp_max": "max amplification"}
HEADLINE = ["F_veg_final", "F_c", "amp_mean", "amp_max"]

BASE_PARAMS = dict(model_main.params)
BASE_PARAMS.update({
    "agent_ini": "sample-max",
    "survey_file": "../data/hierarchical_agents.csv",
    "topology": "homophilic_emp",
    "snapshot_dense_start": 0,   # observables computed in-worker; dense snaps unused
})


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------
def _inflection(traj, win=10001, burnin=5000):
    """F_c (max d2F/dt2 below F=0.5) and inflection (max dF/dt). Mirrors
    analysis/results_analysis.py:analysis_5_inflection so the campaign and the
    main analysis report the same estimator."""
    traj = np.asarray(traj, float)
    if len(traj) < burnin + 2 * win:
        return np.nan, np.nan, np.nan
    sm = savgol_filter(traj, win, 3)
    d1 = savgol_filter(traj, win, 3, deriv=1); d1[:burnin] = 0
    d2 = savgol_filter(traj, win, 3, deriv=2); d2[:burnin] = 0
    i1 = np.argmax(d1)
    d2m = d2.copy(); d2m[sm > 0.5] = 0
    i2 = np.argmax(d2m)
    F_c = sm[i2] if d2m[i2] > 0 else np.nan
    return F_c, sm[i1], i1 / 1000.0


def _observables(m):
    traj = np.asarray(m.fraction_veg, float)
    reds = np.asarray(m.snapshots['final']['reductions'], float)
    mult = reds[reds > 0] / DIRECT_REDUCTION_KG
    cross = np.flatnonzero(traj >= 0.5)
    F_c, F_infl, t_infl = _inflection(traj)
    return {
        "mult": mult.astype(np.float32),   # pooled for CCDF panels
        "F_veg_final": traj[-1],
        "F_c": F_c, "F_infl": F_infl, "t_infl": t_infl,
        "t_50": cross[0] / 1000.0 if len(cross) else np.nan,
        "amp_mean": mult.mean() if len(mult) else 0.0,
        "amp_p90": np.percentile(mult, 90) if len(mult) else 0.0,
        "amp_max": mult.max() if len(mult) else 0.0,
        "n_credited": int(len(mult)),
        "total_credit_kg": float(reds.sum()),
        "steady_state_t": m.steady_state_t,
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _run_one(job):
    param, value, seed, steps = job
    np.random.seed(seed); random.seed(seed)
    p = BASE_PARAMS.copy()
    p.update({"seed": seed, "steps": steps, "tau_persistence": None})
    if param != "baseline":
        p[param] = value
    with contextlib.redirect_stdout(io.StringIO()):
        m = model_main.Model(p, pmf_tables=None)
        m.run()
    return {"param": param, "value": value, "seed": seed, **_observables(m)}


def build_jobs(runs, steps):
    """Baseline runs once and is reused as every parameter's baseline point."""
    jobs = [("baseline", np.nan, 42 + i, steps) for i in range(runs)]
    for prm, vals in SWEEPS.items():
        for v in vals:
            if v == BASELINE[prm]:
                continue
            jobs += [(prm, v, 42 + i, steps) for i in range(runs)]
    return jobs


def expand_baseline(df):
    """Relabel the shared baseline block as one sweep point per parameter."""
    base = df[df.param == "baseline"]
    out = [df[df.param != "baseline"]]
    for prm in SWEEPS:
        b = base.copy(); b["param"] = prm; b["value"] = BASELINE[prm]
        out.append(b)
    return pd.concat(out, ignore_index=True)


def summarise(df):
    g = df.groupby(["param", "value"])
    s = g[OBS + ["n_credited", "total_credit_kg"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_runs"] = g.size()
    s["F_c_n"] = g["F_c"].apply(lambda x: int(np.isfinite(x).sum()))
    s = s.reset_index()
    s["is_baseline"] = [v == BASELINE[p] for p, v in zip(s.param, s.value)]
    return s.sort_values(["param", "value"]).reset_index(drop=True)


def sensitivity_index(summary):
    """Signed relative range per (parameter, observable).

        S = sign * (max_v Ybar(v) - min_v Ybar(v)) / |Ybar(baseline)|

    sign is the sign of Spearman(parameter value, Ybar) over the sweep points, so
    the magnitude reads as "fraction of the baseline value spanned by the sweep"
    and the sign as the direction of the response. Monotonicity is not assumed;
    non-monotone responses are visible in the response-curve figure.
    """
    from scipy.stats import spearmanr
    rows = []
    for prm in SWEEPS:
        sub = summary[summary.param == prm].sort_values("value")
        base = sub[sub.is_baseline]
        for ob in OBS:
            y = sub[f"{ob}_mean"].values
            yb = base[f"{ob}_mean"].values[0]
            if np.all(np.isnan(y)) or not np.isfinite(yb) or yb == 0:
                rows.append({"param": prm, "obs": ob, "S": np.nan,
                             "lo": np.nan, "hi": np.nan, "base": yb})
                continue
            rho = spearmanr(sub.value.values, y, nan_policy="omit").statistic
            sign = 1.0 if not np.isfinite(rho) or rho >= 0 else -1.0
            rows.append({"param": prm, "obs": ob,
                         "S": sign * (np.nanmax(y) - np.nanmin(y)) / abs(yb),
                         "lo": np.nanmin(y), "hi": np.nanmax(y), "base": yb})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_tornado(sens, out, observables=HEADLINE):
    """Per observable: bar spanning [min, max] over the sweep, baseline marked.
    Sorted by span. Direction encoded by colour."""
    fig, axes = plt.subplots(1, len(observables),
                             figsize=(3.4 * len(observables), 4.4))
    for ax, ob in zip(np.atleast_1d(axes), observables):
        d = sens[sens.obs == ob].dropna(subset=["S"]).copy()
        d["span"] = d.hi - d.lo
        d = d.sort_values("span")
        y = np.arange(len(d))
        span_max = d.span.max()
        for yi, r in zip(y, d.itertuples()):
            c = COLORS['primary'] if r.S >= 0 else COLORS['meat']
            ax.barh(yi, r.hi - r.lo, left=r.lo, height=0.62, color=c, alpha=0.85)
            ax.plot([r.base], [yi], marker='|', ms=14, mew=2.0, color='#222')
            if r.span < 1e-9 * max(span_max, 1e-12) or r.span == 0:
                ax.annotate("no effect", (r.base, yi), textcoords="offset points",
                            xytext=(7, 0), va='center', fontsize=6.5, color='#666')
        ax.set_yticks(y)
        ax.set_yticklabels([PLABEL[p] for p in d.param], fontsize=8)
        ax.set_xlabel(OLABEL[ob], fontsize=9)
        apply_axis_style(ax)
    h = [plt.Line2D([], [], color=COLORS['primary'], lw=6, label='increases with parameter'),
         plt.Line2D([], [], color=COLORS['meat'], lw=6, label='decreases with parameter'),
         plt.Line2D([], [], color='#222', marker='|', ms=12, mew=2, ls='none',
                    label='submitted configuration')]
    fig.legend(handles=h, frameon=False, fontsize=8, ncol=3,
               loc='lower center', bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("One-at-a-time parameter sensitivity (sample-max, $N=385$)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


def fig_heatmap(sens, out):
    """Signed relative-range matrix: parameters x observables, one panel."""
    mat = sens.pivot(index="param", columns="obs", values="S")
    mat = mat.reindex(index=list(SWEEPS), columns=OBS)
    order = mat.abs().max(axis=1).sort_values(ascending=False).index
    mat = mat.loc[order]
    v = np.nanmax(np.abs(mat.values))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(mat.values, cmap=ECO_DIV_CMAP, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(OBS)))
    ax.set_xticklabels([OLABEL[o] for o in OBS], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([PLABEL[p] for p in mat.index], fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            s = mat.values[i, j]
            if np.isfinite(s):
                ax.text(j, i, f"{s:+.2f}", ha='center', va='center', fontsize=7.5,
                        color='white' if abs(s) > 0.55 * v else '#222')
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("signed relative range  $S$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Sensitivity index: sweep range as a fraction of the submitted value\n"
                 "(sign = direction of response; rows ordered by maximum $|S|$)",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


def fig_response_curves(summary, out, observables=HEADLINE):
    """Full OAT response surface: every parameter against every headline
    observable, mean +/- 1 sd over runs. This is where non-monotonicity shows."""
    prms = list(SWEEPS)
    fig, axes = plt.subplots(len(observables), len(prms),
                             figsize=(2.05 * len(prms), 2.05 * len(observables)),
                             sharey='row')
    for i, ob in enumerate(observables):
        for j, prm in enumerate(prms):
            ax = axes[i, j]
            d = summary[summary.param == prm].sort_values("value")
            x, y = d.value.values, d[f"{ob}_mean"].values
            sd = d[f"{ob}_std"].values
            ax.fill_between(x, y - sd, y + sd, color=COLORS['primary'], alpha=0.18)
            ax.plot(x, y, 'o-', color=COLORS['primary'], ms=4, lw=1.6)
            ax.axvline(BASELINE[prm], color=COLORS['highlight'], ls='--', lw=1.2, zorder=3)
            apply_axis_style(ax)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(PLABEL[prm], fontsize=8.5)
            if i == len(observables) - 1:
                ax.set_xlabel("value", fontsize=8)
            if j == 0:
                ax.set_ylabel(OLABEL[ob], fontsize=8)
    fig.suptitle("OAT response curves (mean $\\pm$ 1 sd over runs; dashed = submitted value)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


def fig_lambda(df, summary, out):
    """R4.21: the attenuation sweep, with the tail separated from the mean --
    the claim under test is that the representative multiplier is robust while
    the ceiling is lambda-dependent."""
    d = summary[summary.param == "decay"].sort_values("value")
    x = d.value.values
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    ax = axes[0]
    for ob, c, lbl in [("amp_mean", COLORS['primary'], "mean"),
                       ("amp_p90", COLORS['vegetation'], "90th pct."),
                       ("amp_max", COLORS['meat'], "maximum")]:
        y, sd = d[f"{ob}_mean"].values, d[f"{ob}_std"].values
        ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.16)
        ax.plot(x, y, 'o-', color=c, ms=5, lw=1.9)
        ax.annotate(lbl, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 0), va='center', fontsize=8, color=c)
    ax.axvline(BASELINE["decay"], color=COLORS['highlight'], ls='--', lw=1.3, zorder=3)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='y', which='minor', labelsize=6)
    ax.set_xlim(x[0] - 0.02, x[-1] + 0.12)
    ax.set_xlabel(r"attenuation factor $\lambda$")
    ax.set_ylabel("amplification multiplier over credited agents")
    apply_axis_style(ax)
    ax.set_title("Amplification vs. attenuation", fontsize=10)

    # right: pooled CCDF -- shows the body of the distribution barely moving
    # while the tail slides, which is the substance of the R4.21 answer
    ax = axes[1]
    cmap = plt.get_cmap('viridis')
    lo, hi = min(SWEEPS["decay"]), max(SWEEPS["decay"])
    for v in SWEEPS["decay"]:
        pool = np.sort(np.concatenate(
            df[(df.param == "decay") & (df.value == v)]["mult"].values))
        if not len(pool):
            continue
        ccdf = 1.0 - np.arange(1, len(pool) + 1) / len(pool)
        base = v == BASELINE["decay"]
        ax.plot(pool, ccdf, color=cmap(0.12 + 0.76 * (v - lo) / (hi - lo)),
                lw=2.6 if base else 1.6, ls='-' if base else '--',
                label=rf"$\lambda={v:g}$" + (" (submitted)" if base else ""))
    ax.axvline(1.0, color='#555', ls=':', lw=1.0)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(left=0.05)
    ax.set_xlabel("amplification multiplier"); ax.set_ylabel(r"CCDF  $P(X>x)$")
    ax.legend(frameon=False, fontsize=7.5)
    apply_axis_style(ax)
    ax.set_title("Pooled distribution over all runs", fontsize=10)

    fig.suptitle(r"Sensitivity to the attenuation factor $\lambda$ (R4.21)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


# ---------------------------------------------------------------------------
# 2D interaction grid (--interaction)
# ---------------------------------------------------------------------------
# OAT cannot see interactions. This grid crosses the memory length against the
# gate steepness, which is the pair the mean-field argument turns on: binomial
# memory noise (sd ~ 1/sqrt(M)) against the gate transition width (~1/k). If a
# sharp collective threshold is reachable anywhere in the model, it is here.
GRID = {"M": [3, 6, 9, 12, 15], "theta_gate_k": [15, 25, 35, 45, 55]}
GOBS = [("F_veg_final", "mean", r"$F_{veg}$ (final)"),
        ("F_c", "mean", r"$F_c$ (max accel.)"),
        ("F_c", "iqr", r"$F_c$ interquartile range"),
        ("amp_mean", "mean", "mean amplification")]


def _run_cell(job):
    (pa, va), (pb, vb), seed, steps = job
    np.random.seed(seed); random.seed(seed)
    p = BASE_PARAMS.copy()
    p.update({"seed": seed, "steps": steps, "tau_persistence": None, pa: va, pb: vb})
    with contextlib.redirect_stdout(io.StringIO()):
        m = model_main.Model(p, pmf_tables=None)
        m.run()
    o = _observables(m); o.pop("mult")
    return {pa: va, pb: vb, "seed": seed, **o}


def run_interaction(runs, steps, cores, today):
    (pa, va), (pb, vb) = list(GRID.items())
    jobs = [((pa, a), (pb, b), 42 + i, steps)
            for a in va for b in vb for i in range(runs)]
    print(f"INFO: interaction grid {pa} x {pb} = {len(va)}x{len(vb)} cells, "
          f"{len(jobs)} runs on {cores} cores")
    with Pool(cores) as pool:
        df = pd.DataFrame(pool.map(_run_cell, jobs))
    df.to_pickle(f"../model_output/sensitivity_interaction_{today}.pkl")
    return df


def fig_interaction(df, out):
    (pa, va), (pb, vb) = list(GRID.items())
    fig, axes = plt.subplots(1, len(GOBS), figsize=(3.3 * len(GOBS), 3.5))
    for ax, (ob, stat, lbl) in zip(axes, GOBS):
        agg = (lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25)) \
            if stat == "iqr" else (lambda x: np.nanmean(x))
        mat = np.array([[agg(df[(df[pa] == a) & (df[pb] == b)][ob].values)
                         for b in vb] for a in va])
        im = ax.imshow(mat, cmap=ECO_CMAP if stat == "mean" else 'magma_r',
                       aspect='auto', origin='lower')
        ax.set_xticks(range(len(vb))); ax.set_xticklabels(vb, fontsize=7)
        ax.set_yticks(range(len(va))); ax.set_yticklabels(va, fontsize=7)
        ax.set_xlabel(PLABEL[pb], fontsize=8)
        ax.set_ylabel(PLABEL[pa], fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3g}", ha='center', va='center',
                            fontsize=6.5, color='#222')
        # submitted configuration
        ax.plot(vb.index(BASELINE[pb]), va.index(BASELINE[pa]), marker='s',
                ms=15, mfc='none', mec=COLORS['highlight'], mew=2.0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(lbl, fontsize=9)
    fig.suptitle("Memory length against gate steepness: the two parameters that set "
                 "whether a sharp threshold can form\n"
                 "(gold square = submitted configuration)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


# ---------------------------------------------------------------------------
# SI table
# ---------------------------------------------------------------------------
def write_table(summary, sens, out):
    fmt = {"F_veg_final": "{:.3f}", "F_c": "{:.3f}", "t_50": "{:.1f}",
           "amp_mean": "{:.2f}", "amp_p90": "{:.2f}", "amp_max": "{:.1f}"}
    pname = {"decay": r"$\lambda$", "M": r"$M$", "beta": r"$\beta$",
             "gamma": r"$\gamma$", "immune_n": r"$f_{imm}$",
             "theta_gate_c": r"$c$", "theta_gate_k": r"$k$"}
    n = int(summary.n_runs.max())
    L = [r"% generated by model_src/sensitivity_campaign.py -- do not edit by hand",
         r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{One-at-a-time parameter sensitivity for the sample-max "
         rf"ensemble ($N=385$, {n} runs per sweep point, seeds paired across "
         r"points). Entries give the mean over runs with the standard deviation "
         r"in parentheses; bold rows are the submitted configuration. $S$ is the "
         r"signed relative range: the span of each observable across the sweep, "
         r"expressed as a fraction of its value at the submitted configuration "
         r"and signed by the direction of the response. $S$ is conditional on "
         r"the ranges swept here, so it ranks parameters within those ranges "
         r"rather than globally. Note that $\tau_{p}$ is defined as $2MN$, so "
         r"the $M$ sweep moves the memory length and the dwell-time window "
         r"together.}",
         r"\label{tab:sensitivity}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Parameter & Value & $F_{veg}$ & $F_c$ & $t_{50}$ (k) & "
         r"$\bar{A}$ & $A_{90}$ & $A_{max}$ \\", r"\midrule"]
    partial = []   # sweep points where the F_c estimator was undefined in some runs
    for prm in SWEEPS:
        sub = summary[summary.param == prm].sort_values("value")
        for i, r in enumerate(sub.itertuples()):
            cells = []
            for ob in OBS:
                mu, sd = getattr(r, f"{ob}_mean"), getattr(r, f"{ob}_std")
                cells.append("--" if not np.isfinite(mu) else
                             fmt[ob].format(mu) + ("" if not np.isfinite(sd)
                                                   else f" ({fmt[ob].format(sd)})"))
            if r.F_c_n < r.n_runs:
                partial.append(f"{pname[prm][:-1]}={r.value:g}$ "
                               f"({r.F_c_n}/{r.n_runs})")
            cells = [pname[prm] if i == 0 else "", f"{r.value:g}"] + cells
            if r.is_baseline:
                cells = [c if not c else r"\textbf{" + c + "}" for c in cells]
            L.append(" & ".join(cells) + r" \\")
        s = sens[sens.param == prm].set_index("obs")["S"]
        L.append(r"\addlinespace[1pt]")
        L.append(r"& $S$ & " + " & ".join(
            "--" if not np.isfinite(s[ob]) else f"{s[ob]:+.2f}" for ob in OBS) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}"]
    if partial:
        L += [r"\begin{minipage}{\textwidth}\vspace{2pt}\footnotesize",
              r"$F_c$ is defined only where the smoothed trajectory has positive "
              r"acceleration below $F_{veg}=0.5$; it was undefined in some runs at "
              + ", ".join(partial) + r" (runs with a defined $F_c$ shown in "
              r"parentheses), and the entry is the mean over those runs.",
              r"\end{minipage}"]
    L += [r"\end{table}"]
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"INFO: Saved -> {out}")


def print_report(summary, sens):
    print(f"\n{'='*78}\n SENSITIVITY RANKING (max |S| over observables)\n{'='*78}")
    rank = sens.dropna(subset=["S"]).groupby("param")["S"].apply(
        lambda x: x.abs().max()).sort_values(ascending=False)
    for prm, v in rank.items():
        row = sens[(sens.param == prm)].set_index("obs")["S"]
        worst = row.abs().idxmax()
        print(f"  {prm:<14} max|S| = {v:5.2f}   (on {worst})   " +
              "  ".join(f"{o}={row[o]:+.2f}" for o in HEADLINE if np.isfinite(row[o])))
    print(f"\n{'='*78}\n PER SWEEP POINT\n{'='*78}")
    for prm in SWEEPS:
        print(f"\n-- {prm} " + "-" * (74 - len(prm)))
        print(f"{'value':>8}" + "".join(f"{o:>16}" for o in HEADLINE))
        for r in summary[summary.param == prm].sort_values("value").itertuples():
            mark = " *" if r.is_baseline else "  "
            print(f"{r.value:>8g}" + "".join(
                f"{getattr(r, o + '_mean'):>16.3f}" for o in HEADLINE) + mark)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=30, help='runs per sweep point')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--cores', type=int, default=max(1, int(0.75 * os.cpu_count())))
    ap.add_argument('--quick', action='store_true', help='4 runs x 30000 steps smoke test')
    ap.add_argument('--plot-only', metavar='DATE',
                    help='regenerate figures/table from an existing campaign pkl')
    ap.add_argument('--interaction', action='store_true',
                    help='also run the M x theta_gate_k 2D grid')
    ap.add_argument('--interaction-runs', type=int, default=20)
    args = ap.parse_args()
    if args.quick:
        args.runs = 4

    os.makedirs("../visualisations_output", exist_ok=True)
    os.makedirs("../model_output", exist_ok=True)
    set_publication_style()
    today = args.plot_only or date.today().strftime('%Y%m%d')
    pkl = f"../model_output/sensitivity_campaign_{today}.pkl"

    if args.plot_only:
        df = pd.read_pickle(pkl)
    else:
        jobs = build_jobs(args.runs, args.steps)
        print(f"INFO: {len(jobs)} runs ({args.runs}/point, {len(SWEEPS)} parameters, "
              f"steps={args.steps}) on {args.cores} cores")
        with Pool(args.cores) as pool:
            rows = pool.map(_run_one, jobs)
        df = expand_baseline(pd.DataFrame(rows))
        df.to_pickle(pkl)
        print(f"INFO: Saved -> {pkl}")

    summary = summarise(df)
    summary.to_csv(f"../model_output/sensitivity_summary_{today}.csv", index=False)
    sens = sensitivity_index(summary)

    print_report(summary, sens)
    V = "../visualisations_output"
    fig_tornado(sens, f"{V}/sensitivity_tornado_{today}.pdf")
    fig_heatmap(sens, f"{V}/sensitivity_heatmap_{today}.pdf")
    fig_response_curves(summary, f"{V}/sensitivity_response_curves_{today}.pdf")
    fig_lambda(df, summary, f"{V}/sensitivity_lambda_{today}.pdf")
    write_table(summary, sens, f"{V}/sensitivity_table_{today}.tex")

    ipkl = f"../model_output/sensitivity_interaction_{today}.pkl"
    if args.interaction or (args.plot_only and os.path.exists(ipkl)):
        gdf = pd.read_pickle(ipkl) if args.plot_only and os.path.exists(ipkl) \
            else run_interaction(args.interaction_runs, args.steps, args.cores, today)
        fig_interaction(gdf, f"{V}/sensitivity_interaction_{today}.pdf")
    print("\nINFO: done.")


if __name__ == "__main__":
    main()
