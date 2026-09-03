#!/usr/bin/env python3
"""
One-at-a-time (OAT) sensitivity campaign for the twin ensemble (N=2000).

Swept around the configuration the reported results come from. Every headline
number in the manuscript is twin N=2000, so the sensitivity analysis has to
sweep that configuration. This is not a relabelled sample-max campaign: N=385
is not a choice of N but the complete-case count that sample-max mode forces
(model_main.py:415-422), and tau_persistence = M*2*N makes the M sweep span
tau 12,000-60,000 at N=2000 against 2,310-11,550 at N=385 -- a different
response, not a rescale of the same one.

Answers three reviewer comments in one pass:
  R1.3  -- which inputs is the model most sensitive to?
  R4.16 -- why is memory fixed at M=9?
  R4.21 -- a sensitivity analysis of the attenuation factor lambda is necessary.

Seven parameters are swept one at a time around the default configuration, each
against six observables. Every sweep point uses the SAME seed set (42..42+runs-1),
so arms are paired on network realisation and initial condition: differences are
the parameter, not the draw.

Parameters swept (baseline starred in SWEEPS below):
  decay (lambda), M, beta, gamma, immune_n, theta_gate_c, theta_gate_k, kappa

Observables:
  F_veg_final  steady-state vegetarian fraction
  F_c          F_veg at max acceleration, F<0.5   (same estimator as
               analysis/results_analysis.py:analysis_5_inflection)
  t_50         first crossing of F_veg = 0.5, in ksteps
  amp_mean     mean amplification multiplier over credited agents
  amp_p90      90th percentile of the same
  amp_max      maximum of the same (the "88x" ceiling)

The amplification observables are the PRIMARY credit convention fixed on 2026-09-02
(exposure-proportional parents, event count, no dwell weight, lambda from params),
replayed in-worker from the event log by analysis/attribution_ledger.py. The
in-simulation ledger (last-draw parent, dwell-weighted) is the submitted convention
and is carried alongside as amp_mean_sub / amp_max_sub / n_credited_sub so the two
can be compared without a rerun. See
claude_stuff/Review/model_changes_before_rerun_2026-09-02.md section 6.

NOTE on M: tau_persistence = M*2*N (model_main.py:336) enters the dwell weight only,
so under the primary convention the M row is a clean memory-length response; the
_sub columns still carry the M/tau coupling and must be labelled as such if quoted.
NOTE on gamma: exposure-proportional credit splits by the same n**gamma share that
h_soc uses, so the gamma sweep moves the dynamics and the credit split together --
the coupling the M row used to have, moved to gamma by the convention change.

Outputs are tagged <date>_N<N>, because a campaign is only interpretable against
the configuration it swept and sample-max (385) and twin (2000) runs are not
comparable on any amplification observable:
  model_output/sensitivity_campaign_<tag>.pkl        raw per-run rows
  model_output/sensitivity_summary_<tag>.csv         per sweep point, mean/std
  visualisations_output/sensitivity_tornado_<tag>.pdf
  visualisations_output/sensitivity_heatmap_<tag>.pdf
  visualisations_output/sensitivity_response_curves_<tag>.pdf
  visualisations_output/sensitivity_lambda_<tag>.pdf
  visualisations_output/sensitivity_table_<tag>.tex  SI table
  visualisations_output/sensitivity_interaction_<tag>.pdf   (--interaction only)

RUN THIS ON THE COMPUTE SERVER, not the laptop. The full campaign is 990 model
runs (1490 with --interaction) since kappa joined the sweep on 2026-09-03; the
interaction grid is a fixed pair and did not grow. Measured on this laptop at
N=2000: ~14 s fixed setup (network build + agent init) plus 0.62 ms/step, so
~106 s per 150k-step run -- about 30 core-hours for the OAT campaign, 44 with
--interaction. Scale that by the run length: kappa = 0.55 pushes saturation out
roughly 3.4x, so a campaign at the length the adopted configuration needs costs
proportionately more. Set --steps from the fitted t_end, not from habit.

    python sensitivity_campaign.py --runs 30 --interaction --cores <n>

Figures and the SI table can then be rebuilt locally from the returned pickles
without rerunning anything:

    python sensitivity_campaign.py --plot-only <tag>

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
sys.path.append('../analysis')
import model_main
import model_runner_mp
from attribution_ledger import replay
from t_end_logistic import estimate_t_end
from plot_styles import (set_publication_style, apply_axis_style, COLORS,
                         ECO_CMAP, ECO_DIV_CMAP)

DIRECT_REDUCTION_KG = 2054 - 1390   # meat_CO2 - veg_CO2

# --- sweep design ----------------------------------------------------------
# value lists; the baseline value must appear in each list exactly once
BASELINE = {"decay": 0.7, "M": 9, "beta": 13, "gamma": 0.3,
            "immune_n": 0.10, "theta_gate_c": 0.35, "theta_gate_k": 35,
            "kappa": 0.55}

SWEEPS = {
    "decay":        [0.5, 0.6, 0.7, 0.8, 0.9],
    "M":            [3, 6, 9, 12, 15],
    "beta":         [5, 9, 13, 20, 30],
    "gamma":        [0.10, 0.30, 0.50, 0.70, 1.00],   # 1.0 = simple-contagion null
    "immune_n":     [0.0, 0.05, 0.10, 0.15, 0.20],
    "theta_gate_c": [0.25, 0.30, 0.35, 0.40, 0.45],
    "theta_gate_k": [15, 25, 35, 45, 55],
    # kappa = 1.00 is the pre-2026-09-03 configuration, i.e. stated intention at face
    # value, so the sweep contains the undiscounted model as an endpoint and "what if
    # you had not discounted?" reads straight off the tornado.
    "kappa":        [0.40, 0.55, 0.70, 0.85, 1.00],
}

PLABEL = {"decay": r"$\lambda$ (attenuation)", "M": r"$M$ (memory)",
          "beta": r"$\beta$ (inverse temp.)", "gamma": r"$\gamma$ (dim. returns)",
          "immune_n": r"$f_{imm}$ (immune)", "theta_gate_c": r"$c$ (gate threshold)",
          "theta_gate_k": r"$k$ (gate steepness)",
          "kappa": r"$\kappa$ (intention discount)"}

OBS = ["F_veg_final", "F_c", "t_50", "amp_mean", "amp_p90", "amp_max"]
OLABEL = {"F_veg_final": r"$F_{veg}$ (final)", "F_c": r"$F_c$ (max accel.)",
          "t_50": r"$t_{50}$ (ksteps)", "amp_mean": "mean amplification",
          "amp_p90": "p90 amplification", "amp_max": "max amplification"}
HEADLINE = ["F_veg_final", "F_c", "amp_mean", "amp_max"]

# Inherit from the runner that produced the reported ensemble, NOT from
# model_main.params -- the latter is for ad-hoc single runs and differs. Starting
# from the wrong runner is what put the 2026-08-19 campaign on tau = 11,700
# (claude_stuff/Review/regeneration_results_2026-08-20.md section 6).
BASE_PARAMS = dict(model_runner_mp.DEFAULT_PARAMS)
BASE_PARAMS.update({
    "agent_ini": "twin",         # the reported ensemble; sample-max would force N to 385
    "N": 2000,
    "survey_file": "../data/hierarchical_agents.csv",
    "topology": "homophilic_emp",
    "decay": 0.7,                # absent from both runners' DEFAULT_PARAMS; the reported
                                 # ensemble got it from model_main's .get("decay", 0.7)
                                 # fallback (model_main.py:278). Pinned here so the
                                 # lambda sweep's baseline arm is explicit.
    "snapshot_dense_start": 0,   # observables computed in-worker; dense snaps unused
})

# The campaigns inherit kappa from the runner above. An absent key falls through to
# model_main's .get("kappa", 1.0) and the whole campaign then sweeps at face value
# while everything else runs discounted -- same class as the tau = 11,700 incident.
assert "kappa" in BASE_PARAMS, "ERROR: no kappa in model_runner_mp.DEFAULT_PARAMS"

# Population the loaded results were swept at. Set from the tag in main() so that
# --plot-only on an older pkl labels its figures with the N it actually ran at.
CFG_N = BASE_PARAMS["N"]

_PMF = None


def pmf_tables():
    """Twin mode imputes alpha/rho from the demographic PMFs. Loaded lazily so
    --plot-only does no I/O; fork gives each worker its own copy on first job."""
    global _PMF
    if _PMF is None:
        _PMF = model_runner_mp.load_pmf_tables()
        if _PMF is None:
            raise SystemExit("ERROR: ../data/demographic_pmfs.pkl missing. Twin mode "
                             "would silently fall back to synthetic alpha/rho for the "
                             "whole campaign. Run auxillary/create_pmf_tables.py first.")
    return _PMF


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
    """Primary amplification is replayed from the event log; the in-simulation
    ledger rides along as the _sub columns. The 'final' snapshot is taken at
    t = steps, which is replay's default t_end, so the two cover the same window."""
    traj = np.asarray(m.fraction_veg, float)
    reds = replay(m.events, m.snapshots[0]['diets'], m.params,
                  parent="exposure", weight="none", unit="event")
    mult = reds[reds > 0] / DIRECT_REDUCTION_KG
    sub = np.asarray(m.snapshots['final']['reductions'], float)
    mult_sub = sub[sub > 0] / DIRECT_REDUCTION_KG
    cross = np.flatnonzero(traj >= 0.5)
    F_c, F_infl, t_infl = _inflection(traj)
    # Amplification accumulates for as long as the run lasts, so the columns above
    # -- replayed to t = steps -- are only comparable across sweep points that
    # saturate at the same time. kappa moves t_end by 3.4x, so at a fixed 400k the
    # kappa = 1.00 point would spend most of the run accumulating credit past its
    # own saturation and the tornado would read that as a kappa effect. The _tend
    # columns replay the same convention at the run's own fitted t_end instead; the
    # event log does not leave the worker, so this cannot be recovered afterwards.
    t_end = estimate_t_end(traj)
    if t_end is None or t_end >= len(traj):
        t_end = len(traj) - 1
    reds_te = replay(m.events, m.snapshots[0]['diets'], m.params,
                     parent="exposure", weight="none", unit="event", t_end=t_end)
    mult_te = reds_te[reds_te > 0] / DIRECT_REDUCTION_KG
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
        "amp_mean_sub": mult_sub.mean() if len(mult_sub) else 0.0,
        "amp_max_sub": mult_sub.max() if len(mult_sub) else 0.0,
        "n_credited_sub": int(len(mult_sub)),
        "steady_state_t": m.steady_state_t,
        "t_end_fit": t_end,
        "F_veg_tend": traj[t_end],
        "amp_mean_tend": mult_te.mean() if len(mult_te) else 0.0,
        "amp_p90_tend": np.percentile(mult_te, 90) if len(mult_te) else 0.0,
        "amp_max_tend": mult_te.max() if len(mult_te) else 0.0,
        "n_credited_tend": int(len(mult_te)),
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
        m = model_main.Model(p, pmf_tables=pmf_tables())
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
                    label='default configuration')]
    fig.legend(handles=h, frameon=False, fontsize=8, ncol=3,
               loc='lower center', bbox_to_anchor=(0.5, -0.015))
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
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


def fig_lambda(df, summary, out):
    """R4.21: the attenuation sweep in one panel. The pooled distribution carries
    the argument -- the body sits on top of itself while the tail slides -- so the
    summary statistics go in an inset rather than a second panel of equal weight."""
    d = summary[summary.param == "decay"].sort_values("value")
    lo, hi = min(SWEEPS["decay"]), max(SWEEPS["decay"])
    cmap = plt.get_cmap('viridis')
    col = lambda v: cmap(0.12 + 0.76 * (v - lo) / (hi - lo))

    fig, ax = plt.subplots(figsize=(5.4, 4.2))

    # main: pooled CCDF, one curve per lambda
    for v in SWEEPS["decay"]:
        pool = np.sort(np.concatenate(
            df[(df.param == "decay") & (df.value == v)]["mult"].values))
        if not len(pool):
            continue
        ccdf = 1.0 - np.arange(1, len(pool) + 1) / len(pool)
        base = v == BASELINE["decay"]
        ax.plot(pool, ccdf, color=col(v), lw=2.4 if base else 1.4,
                ls='-' if base else '--', zorder=3 if base else 2,
                label=rf"$\lambda={v:g}$" + (" (default)" if base else ""))
    ax.axvline(1.0, color='#555', ls=':', lw=1.0, zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(left=0.05)
    ax.set_xlabel("amplification multiplier")
    ax.set_ylabel(r"CCDF  $P(X>x)$")
    ax.legend(frameon=False, fontsize=7, loc='lower left', handlelength=1.8,
              borderpad=0.2, labelspacing=0.35)
    apply_axis_style(ax)

    # inset: summary statistics against lambda
    ins = ax.inset_axes((0.60, 0.62, 0.38, 0.35))
    x = d.value.values
    for ob, c, lbl in [("amp_max", COLORS['meat'], "max"),
                       ("amp_p90", COLORS['vegetation'], "p90"),
                       ("amp_mean", COLORS['primary'], "mean")]:
        y, sd = d[f"{ob}_mean"].values, d[f"{ob}_std"].values
        ins.fill_between(x, y - sd, y + sd, color=c, alpha=0.16)
        ins.plot(x, y, 'o-', color=c, ms=3, lw=1.3)
        ins.annotate(lbl, (x[-1], y[-1]), textcoords="offset points",
                     xytext=(4, 0), va='center', fontsize=6.5, color=c)
    ins.axvline(BASELINE["decay"], color=COLORS['highlight'], ls='--', lw=1.0)
    ins.set_yscale('log')
    ins.set_xlim(x[0] - 0.02, x[-1] + 0.10)
    ins.set_xticks(x)
    ins.set_xlabel(r"$\lambda$", fontsize=7, labelpad=1)
    ins.tick_params(labelsize=6, pad=1.5)
    apply_axis_style(ins)

    fig.tight_layout()
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
        m = model_main.Model(p, pmf_tables=pmf_tables())
        m.run()
    o = _observables(m); o.pop("mult")
    return {pa: va, pb: vb, "seed": seed, **o}


def run_interaction(runs, steps, cores, tag):
    (pa, va), (pb, vb) = list(GRID.items())
    jobs = [((pa, a), (pb, b), 42 + i, steps)
            for a in va for b in vb for i in range(runs)]
    print(f"INFO: interaction grid {pa} x {pb} = {len(va)}x{len(vb)} cells, "
          f"{len(jobs)} runs on {cores} cores")
    with Pool(cores) as pool:
        df = pd.DataFrame(pool.map(_run_cell, jobs))
    df.to_pickle(f"../model_output/sensitivity_interaction_{tag}.pkl")
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
        # default configuration
        ax.plot(vb.index(BASELINE[pb]), va.index(BASELINE[pa]), marker='s',
                ms=15, mfc='none', mec=COLORS['highlight'], mew=2.0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(lbl, fontsize=9)
    fig.tight_layout()
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
         rf"\caption{{One-at-a-time parameter sensitivity for the $N={CFG_N}$ "
         rf"ensemble ({n} runs per sweep point, seeds paired across "
         r"points). Entries give the mean over runs with the standard deviation "
         r"in parentheses; bold rows are the default configuration. $S$ is the "
         r"signed relative range: the span of each observable across the sweep, "
         r"expressed as a fraction of its value at the default configuration "
         r"and signed by the direction of the response. $S$ is conditional on "
         r"the ranges swept here, so it ranks parameters within those ranges "
         r"rather than globally. Amplification uses the primary credit "
         r"convention (exposure-proportional parents, event count, no dwell "
         r"weight); the $\gamma$ sweep therefore moves the social-influence "
         r"kernel and the credit split together.}",
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
    ap.add_argument('--steps', type=int, default=150000)
    ap.add_argument('--cores', type=int, default=max(1, int(0.75 * os.cpu_count())))
    ap.add_argument('--quick', action='store_true', help='4 runs per sweep point, smoke test')
    ap.add_argument('--plot-only', metavar='TAG',
                    help='regenerate figures/table from an existing campaign pkl; '
                         'TAG is <date>_N<N>, e.g. 20260820_N2000')
    ap.add_argument('--interaction', action='store_true',
                    help='also run the M x theta_gate_k 2D grid')
    ap.add_argument('--interaction-runs', type=int, default=20)
    args = ap.parse_args()
    if args.quick:
        args.runs = 4

    os.makedirs("../visualisations_output", exist_ok=True)
    os.makedirs("../model_output", exist_ok=True)
    set_publication_style()
    # N goes in the filename: a campaign is only interpretable against the
    # configuration it swept, and sample-max (385) and twin (2000) runs are not
    # comparable on any amplification observable.
    tag = args.plot_only or f"{date.today().strftime('%Y%m%d')}_N{BASE_PARAMS['N']}"
    global CFG_N
    if "_N" in tag:
        CFG_N = int(tag.split("_N")[1].split("_")[0])
    pkl = f"../model_output/sensitivity_campaign_{tag}.pkl"

    if args.plot_only:
        df = pd.read_pickle(pkl)
    else:
        jobs = build_jobs(args.runs, args.steps)
        print(f"INFO: {len(jobs)} runs ({args.runs}/point, {len(SWEEPS)} parameters, "
              f"N={BASE_PARAMS['N']}, kappa={BASE_PARAMS['kappa']}, "
              f"steps={args.steps}) on {args.cores} cores")
        with Pool(args.cores) as pool:
            rows = pool.map(_run_one, jobs)
        df = expand_baseline(pd.DataFrame(rows))
        df.to_pickle(pkl)
        print(f"INFO: Saved -> {pkl}")

    summary = summarise(df)
    summary.to_csv(f"../model_output/sensitivity_summary_{tag}.csv", index=False)
    sens = sensitivity_index(summary)

    print_report(summary, sens)
    V = "../visualisations_output"
    fig_tornado(sens, f"{V}/sensitivity_tornado_{tag}.pdf")
    fig_heatmap(sens, f"{V}/sensitivity_heatmap_{tag}.pdf")
    fig_response_curves(summary, f"{V}/sensitivity_response_curves_{tag}.pdf")
    fig_lambda(df, summary, f"{V}/sensitivity_lambda_{tag}.pdf")
    write_table(summary, sens, f"{V}/sensitivity_table_{tag}.tex")

    ipkl = f"../model_output/sensitivity_interaction_{tag}.pkl"
    if args.interaction or (args.plot_only and os.path.exists(ipkl)):
        gdf = pd.read_pickle(ipkl) if args.plot_only and os.path.exists(ipkl) \
            else run_interaction(args.interaction_runs, args.steps, args.cores, tag)
        fig_interaction(gdf, f"{V}/sensitivity_interaction_{tag}.pdf")
    print("\nINFO: done.")


if __name__ == "__main__":
    main()
