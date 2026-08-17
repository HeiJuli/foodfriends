#!/usr/bin/env python3
"""
Topology comparison (sample-max, N=385): how much does an EXPERIMENTAL
theta-assortative network shift the in-review results?

Three arms, identical except network topology:
  - homophilic_emp    : PRODUCTION network (in-review; theta-assort ~0, mean-field)
  - homophilic_theta  : NEW theta-homophily (Newman theta-assort ~0.22, PA kept on)
  - ER                : Erdos-Renyi null (matched mean degree)

Settings mirror the in-review sample-max ensemble (steps=30000, dense snapshots
from t=12000, N_RUNS=30) so the homophilic_emp arm reproduces the in-review
setup and the other two differ ONLY by topology.

Outputs (per date):
  model_output/topology_comparison_smax_<date>.pkl          (combined raw)
  model_output/trajectory_analysis_sample-max_<topo>_<date>.pkl  (per-arm, reusable
      by plotting/agency_predictor_analysis.py and publication_plots_main.py)
  visualisations_output/topology_comparison_{trajectory,amplification,two_dvs}_<date>.pdf

Plus printed: network properties, degree-amplification scaling, and the two-DV
(adoption + amplification) ensemble for each arm.

Run from model_src/:
    python topology_comparison.py [--runs N] [--steps S] [--quick]

See claude_stuff/Infrastructure/homophily_network_audit_2026-05-29.md sections 6-7.
"""
import sys, os, random, pickle, argparse
from multiprocessing import Pool
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

sys.path.append('.')
sys.path.append('..')
sys.path.append('../plotting')
import model_main
import agency_predictor_analysis as agency  # two-DV machinery (reused, not duplicated)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOPOS  = ["homophilic_emp", "homophilic_theta", "ER", "CM"]
LABELS = {"homophilic_emp": "Production (demo-only, theta-assort ~0)",
          "homophilic_theta": "Theta-homophily (assort ~0.22)",
          "ER": "Erdos-Renyi (random)",
          "CM": "Configuration model (matched degree seq)"}
SHORT  = {"homophilic_emp": "prod", "homophilic_theta": "theta", "ER": "ER", "CM": "CM"}
COLORS = {"homophilic_emp": "#006d77", "homophilic_theta": "#9b2226", "ER": "#e29578",
          "CM": "#adb5bd"}
DIRECT_REDUCTION_KG = 664  # 2054 - 1390

# Start from canonical production defaults so the prod arm == in-review setup,
# then override only what the comparison needs.
BASE_PARAMS = dict(model_main.params)
BASE_PARAMS.update({
    "agent_ini": "sample-max",
    "survey_file": "../data/hierarchical_agents.csv",
    "snapshot_dense_start": 12000,   # match in-review dense snapshot schedule
    # theta-homophily frozen calibration (used only by homophilic_theta arm):
    "theta_w": 1.0, "sim_power": 4.0, "pa_power": 1.0, "tc_sim": True,
})


# ---------------------------------------------------------------------------
# Single-run worker
# ---------------------------------------------------------------------------
def _net_stats(g):
    degs = [d for _, d in g.degree()]
    return {"avg_degree": float(np.mean(degs)), "degree_std": float(np.std(degs)),
            "max_degree": int(max(degs)), "clustering": nx.average_clustering(g),
            "n_components": nx.number_connected_components(g),
            "degree_assort": nx.degree_assortativity_coefficient(g)}

def _theta_assort(g):
    try:
        return nx.numeric_assortativity_coefficient(g, 'theta')
    except Exception:
        return np.nan

def _run_one(args):
    topo, seed, steps = args
    np.random.seed(seed); random.seed(seed)
    p = BASE_PARAMS.copy()
    p.update({"topology": topo, "seed": seed, "steps": steps})
    m = model_main.Model(p, pmf_tables=None)
    m.run()
    return {
        "topology": topo, "seed": seed,
        "fraction_veg_trajectory": m.fraction_veg,
        "system_C_trajectory": m.system_C,
        "snapshots": m.snapshots,
        "steady_state_t": m.steady_state_t,
        "final_veg_f": m.fraction_veg[-1],
        "theta_assort_t0": _theta_assort(m.snapshots[0]['graph']),
        "net_t0": _net_stats(m.snapshots[0]["graph"]),
    }


# ---------------------------------------------------------------------------
# Assemble per-arm DataFrames (agency-compatible) + save
# ---------------------------------------------------------------------------
def _to_frame(rows):
    df = pd.DataFrame(rows)
    df['agent_ini'] = 'sample-max'
    df['run'] = range(len(df))
    fv = df['final_veg_f'].values
    med_idx = df.index[np.argmin(np.abs(fv - np.median(fv)))]
    df['is_median_twin'] = False
    df.loc[med_idx, 'is_median_twin'] = True
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_trajectory(frames, out):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for topo in TOPOS:
        mat = np.array([r[:min(len(x) for x in frames[topo]['fraction_veg_trajectory'])]
                        for r in frames[topo]['fraction_veg_trajectory']])
        t = np.arange(mat.shape[1])
        mu, lo, hi = mat.mean(0), np.percentile(mat, 5, 0), np.percentile(mat, 95, 0)
        ax.plot(t, mu, color=COLORS[topo], lw=2.2, label=LABELS[topo])
        ax.fill_between(t, lo, hi, color=COLORS[topo], alpha=0.18)
    ax.set_xlabel("Timestep"); ax.set_ylabel("Vegetarian fraction")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, ls="--", alpha=0.3, zorder=-10)
    ax.set_title("Adoption trajectory by network topology (sample-max)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")

def fig_amplification(amp_pools, out):
    """CCDF of amplification multiplier (reduction / direct) across pooled adopters."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for topo in TOPOS:
        m = np.sort(amp_pools[topo])
        if len(m) == 0:
            continue
        ccdf = 1.0 - np.arange(1, len(m) + 1) / len(m)
        ax.plot(m, ccdf, color=COLORS[topo], lw=2.0,
                label=f"{SHORT[topo]}: mean={m.mean():.1f}x, max={m.max():.0f}x")
    ax.axvline(1.0, color='#555', ls='--', lw=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("Amplification multiplier (cascade credit / direct reduction)")
    ax.set_ylabel("CCDF  P(X > x)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title("Amplification distribution by topology", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")

def fig_two_dvs(ens, out):
    """Grouped bars: adoption (r_pb) and amplification (rho_s) per predictor, 3 arms."""
    preds = agency.ALL_PREDS
    y = np.arange(len(preds)); h = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, dv, vcol, xlabel in [(axes[0], 'adoption', 'r_pb', r'$r_{pb}$'),
                                 (axes[1], 'amplification', 'rho_s', r'$\rho_s$')]:
        for j, topo in enumerate(TOPOS):
            d = ens[topo][dv].set_index('predictor').reindex(preds)
            vals = np.nan_to_num(d[vcol].values)
            sig = d['frac_sig'].fillna(0).values > 0.5
            bar_colors = [mcolors.to_rgba(COLORS[topo], 0.95 if s else 0.35) for s in sig]
            ax.barh(y + (1 - j) * h, vals, height=h, color=bar_colors,
                    label=SHORT[topo] if dv == 'adoption' else None)
        ax.axvline(0, color='#555', lw=0.7, ls='--')
        ax.set_yticks(y); ax.set_yticklabels(preds, fontsize=9)
        ax.set_xlabel(xlabel); ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(dv.capitalize(), fontsize=11)
    axes[0].legend(frameon=False, fontsize=9, title="topology", loc='lower left')
    fig.suptitle("Two-DV predictor comparison (full alpha = sig in >50% of runs)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--cores', type=int, default=max(1, int(0.6 * os.cpu_count())))
    ap.add_argument('--quick', action='store_true', help='3 runs x 8000 steps smoke test')
    args = ap.parse_args()
    if args.quick:
        args.runs, args.steps = 3, 8000

    os.makedirs("../visualisations_output", exist_ok=True)
    os.makedirs("../model_output", exist_ok=True)
    today = date.today().strftime('%Y%m%d')

    jobs = [(t, 42 + i, args.steps) for t in TOPOS for i in range(args.runs)]
    print(f"INFO: {len(jobs)} runs ({args.runs}/topology, steps={args.steps}) "
          f"on {args.cores} cores")
    with Pool(args.cores) as pool:
        results = pool.map(_run_one, jobs)

    with open(f"../model_output/topology_comparison_smax_{today}.pkl", "wb") as f:
        pickle.dump(results, f)

    frames = {t: _to_frame([r for r in results if r['topology'] == t]) for t in TOPOS}
    for topo, df in frames.items():
        path = f"../model_output/trajectory_analysis_sample-max_{SHORT[topo]}_{today}.pkl"
        df.to_pickle(path)
        print(f"INFO: Saved per-arm -> {path}")

    # -- network properties (t0, seed-averaged) ----------------------------
    print(f"\n{'='*78}\n NETWORK PROPERTIES (t=0, mean over runs)\n{'='*78}")
    print(f"{'topology':<18}{'theta_assort':>13}{'k':>7}{'deg_std':>9}"
          f"{'max_deg':>9}{'CC':>7}{'deg_assort':>12}{'comps':>7}")
    for topo in TOPOS:
        sub = [r for r in results if r['topology'] == topo]
        g = lambda key: np.mean([r['net_t0'][key] for r in sub])
        print(f"{SHORT[topo]:<18}{np.nanmean([r['theta_assort_t0'] for r in sub]):>+13.3f}"
              f"{g('avg_degree'):>7.1f}{g('degree_std'):>9.1f}{g('max_degree'):>9.0f}"
              f"{g('clustering'):>7.3f}{g('degree_assort'):>12.3f}{g('n_components'):>7.1f}")

    # -- final F_veg --------------------------------------------------------
    print(f"\n{'='*78}\n FINAL VEGETARIAN FRACTION (mean +/- std over runs)\n{'='*78}")
    for topo in TOPOS:
        fv = frames[topo]['final_veg_f'].values
        print(f"{SHORT[topo]:<18} F_veg = {fv.mean():.3f} +/- {fv.std():.3f}")

    # -- two-DV ensemble + degree scaling + amplification pools ------------
    ens, amp_pools = {}, {}
    for topo in TOPOS:
        print(f"\n{'#'*78}\n# TWO-DV ANALYSIS: {LABELS[topo]}\n{'#'*78}")
        ens[topo] = agency.run_ensemble(frames[topo], n_runs=len(frames[topo]))
        median_row = agency.get_median_row(frames[topo])
        df_feat = agency.extract_features(median_row)
        agency.analyze_degree_scaling(df_feat)
        agency.analyze_network_properties(agency._resolve_snap(median_row['snapshots'])['graph'])
        # pool amplification multipliers across all runs
        pool = []
        for _, row in frames[topo].iterrows():
            reds = np.array(agency._resolve_snap(row['snapshots'])['reductions'])
            pool.extend((reds[reds > 0] / DIRECT_REDUCTION_KG).tolist())
        amp_pools[topo] = np.array(pool)

    # -- figures ------------------------------------------------------------
    fig_trajectory(frames, f"../visualisations_output/topology_comparison_trajectory_{today}.pdf")
    fig_amplification(amp_pools, f"../visualisations_output/topology_comparison_amplification_{today}.pdf")
    fig_two_dvs(ens, f"../visualisations_output/topology_comparison_two_dvs_{today}.pdf")

    print(f"\nINFO: done. Per-arm pkls reusable by agency_predictor_analysis.py / "
          f"publication_plots_main.py")


if __name__ == "__main__":
    main()
