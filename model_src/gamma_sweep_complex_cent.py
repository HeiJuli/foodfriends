#!/usr/bin/env python3
"""Gamma sweep: complex centrality vs amplification across gamma values [0.3, 0.7].

Runs 3 sample-max simulations per gamma, extracts complex centrality from the
median run, and produces:
  - Row 1: binned complex_cent vs amplification line (one panel per gamma)
  - Row 2: OLS std coef + Spearman rho_s vs gamma summary chart

Usage: cd model_src && python gamma_sweep_complex_cent.py
"""
import sys, os, random
from collections import deque
from datetime import date
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
import statsmodels.api as sm
sys.path.insert(0, '../plotting')
from plot_styles import set_publication_style, COLORS
import model_main

GAMMAS = np.round(np.arange(0.3, 0.75, 0.1), 2)   # [0.3, 0.4, 0.5, 0.6, 0.7]
N_RUNS = 3
DIRECT_REDUCTION_KG = 664  # 2054 - 1390

BASE_PARAMS = model_main.params.copy()
BASE_PARAMS.update({
    "agent_ini": "sample-max",
    "survey_file": "../data/hierarchical_agents.csv",
    "adjust_veg_fraction": True,
    "target_veg_fraction": 0.06,
})


# --- Complex centrality (Guilbeault & Centola 2021) ---

def compute_complex_centrality(G, T=2):
    nodes = list(G.nodes())
    n = len(nodes)
    closed_nbr = {v: set(G.neighbors(v)) | {v} for v in nodes}
    adj_suf = {v: [] for v in nodes}
    for u, v in G.edges():
        cn = len(closed_nbr[u] & closed_nbr[v]) - 2
        if cn + 1 >= T:
            adj_suf[u].append(v)
            adj_suf[v].append(u)
    cc = {}
    for i in nodes:
        dist = {i: 0}
        q = deque([i])
        while q:
            u = q.popleft()
            for v in adj_suf[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        ni = closed_nbr[i]
        n_t = n - len(ni)
        cc[i] = sum(d for v, d in dist.items() if v not in ni) / n_t if n_t > 0 else 0.0
    return cc


def run_gamma(gamma):
    """3 sample-max runs for given gamma; returns list of result dicts."""
    p = BASE_PARAMS.copy()
    p["gamma"] = gamma
    results = []
    for seed in range(42, 42 + N_RUNS):
        np.random.seed(seed)
        random.seed(seed)
        mdl = model_main.Model(p)
        mdl.run()
        results.append({
            'gamma': gamma, 'seed': seed,
            'final_veg_f': mdl.fraction_veg[-1],
            'snap_final': mdl.snapshots['final'],
            'snap_init':  mdl.snapshots[0],
        })
        print(f"  gamma={gamma:.1f} seed={seed} final_veg={mdl.fraction_veg[-1]:.3f}")
    return results


def extract_stats(run_result):
    """Complex_cent stats + binned line from a single run result."""
    snap = run_result['snap_final']
    G = snap['graph']
    nodes = list(G.nodes())
    N = len(nodes)

    reductions = np.array(snap['reductions'])
    multipliers = reductions / DIRECT_REDUCTION_KG

    print(f"    Computing complex centrality (T=2) for N={N}...")
    cc = compute_complex_centrality(G, T=2)
    cc_vals = np.array([cc[nd] for nd in nodes])

    mask = reductions > 0
    x, y = cc_vals[mask], multipliers[mask]
    log_y = np.log(y)

    rho_s, p_val = spearmanr(x, log_y)

    X_std = (x - x.mean()) / (x.std() + 1e-9)
    X_sm = sm.add_constant(X_std)
    ols = sm.OLS(log_y, X_sm).fit()
    std_coef = ols.params[1]
    std_coef_se = ols.bse[1]

    # Binned means (7 quantile bins)
    bins = np.percentile(x, np.linspace(0, 100, 8))
    bc, bm = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (x >= lo) & (x < hi)
        if msk.sum() > 2:
            bc.append(x[msk].mean())
            bm.append(y[msk].mean())

    return {
        'gamma': run_result['gamma'],
        'rho_s': rho_s, 'p_val': p_val,
        'std_coef': std_coef, 'std_coef_se': std_coef_se,
        'bin_x': np.array(bc), 'bin_y': np.array(bm),
        'x': x, 'y': y,
    }


def make_figure(stats_list):
    set_publication_style()
    cm = 1 / 2.54
    n_g = len(GAMMAS)

    fig = plt.figure(figsize=(n_g * 4.5 * cm, 10 * cm))
    gs = gridspec.GridSpec(2, n_g, height_ratios=[2.2, 1.3], hspace=0.55, wspace=0.35)

    # --- Row 1: one scatter+line panel per gamma ---
    for i, s in enumerate(stats_list):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(s['x'], s['y'], s=3, alpha=0.25, c=COLORS['primary'],
                   edgecolors='none', rasterized=True)
        if len(s['bin_x']) > 1:
            ax.plot(s['bin_x'], s['bin_y'], 'o-', color=COLORS['secondary'],
                    lw=1.2, ms=3, zorder=5)
        ax.set_yscale('log')
        ax.set_xlabel('complex_cent', fontsize=6)
        if i == 0:
            ax.set_ylabel('Amplification', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        sig = '***' if s['p_val'] < 0.001 else '**' if s['p_val'] < 0.01 else '*' if s['p_val'] < 0.05 else 'ns'
        ax.set_title(
            f"$\\gamma$={s['gamma']:.1f}\n"
            f"$\\rho_s$={s['rho_s']:.3f}{sig}\n"
            f"$\\beta_{{std}}$={s['std_coef']:.3f}",
            fontsize=5.5, pad=2
        )

    # --- Row 2: summary line chart spanning full width ---
    ax_sum = fig.add_subplot(gs[1, :])
    g_vals = [s['gamma'] for s in stats_list]
    rho_vals = [s['rho_s'] for s in stats_list]
    coef_vals = [s['std_coef'] for s in stats_list]
    coef_se   = [s['std_coef_se'] for s in stats_list]

    ax_sum.plot(g_vals, rho_vals, 'o-', color=COLORS['primary'],
                label='Spearman $\\rho_s$', lw=1.2, ms=4)
    ax_sum.errorbar(g_vals, coef_vals, yerr=coef_se, fmt='s--',
                    color=COLORS['secondary'], label='OLS $\\beta_{std}$',
                    lw=1.2, ms=4, capsize=2)
    ax_sum.axhline(0, color='#888', lw=0.6, ls=':')
    ax_sum.set_xlabel('$\\gamma$', fontsize=7)
    ax_sum.set_ylabel('Coefficient', fontsize=7)
    ax_sum.set_xticks(g_vals)
    ax_sum.tick_params(labelsize=6)
    ax_sum.legend(fontsize=6, frameon=False, ncol=2)
    ax_sum.spines['top'].set_visible(False)
    ax_sum.spines['right'].set_visible(False)
    ax_sum.set_title('complex_cent: OLS $\\beta_{std}$ + Spearman $\\rho_s$ vs $\\gamma$', fontsize=7)

    os.makedirs('../visualisations_output', exist_ok=True)
    out = f'../visualisations_output/gamma_sweep_complex_cent_{date.today().strftime("%Y%m%d")}.pdf'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out}")
    return fig


def main():
    os.makedirs('../model_output', exist_ok=True)

    # --- Run sweep ---
    all_runs = []
    for g in GAMMAS:
        print(f"\n=== gamma={g:.1f} ===")
        all_runs.extend(run_gamma(g))

    # --- Pick median run per gamma, extract stats ---
    stats_list = []
    gamma_groups = {}
    for r in all_runs:
        gamma_groups.setdefault(r['gamma'], []).append(r)

    for g in GAMMAS:
        runs = gamma_groups[g]
        final_veg = np.array([r['final_veg_f'] for r in runs])
        med_idx = int(np.argmin(np.abs(final_veg - np.median(final_veg))))
        med_run = runs[med_idx]
        print(f"\nExtracting stats: gamma={g:.1f}, median seed={med_run['seed']}, "
              f"final_veg={med_run['final_veg_f']:.3f}")
        stats_list.append(extract_stats(med_run))

    # Save summary stats (no graph objects)
    rows = [{k: v for k, v in s.items() if k not in ('bin_x', 'bin_y', 'x', 'y')}
            for s in stats_list]
    pd.DataFrame(rows).to_pickle(
        f'../model_output/gamma_sweep_stats_{date.today().strftime("%Y%m%d")}.pkl')
    print("\nStats summary:")
    print(f"{'gamma':>6} {'rho_s':>8} {'p_val':>10} {'std_coef':>10} {'se':>8}")
    for s in stats_list:
        print(f"  {s['gamma']:.1f}   {s['rho_s']:>8.4f}  {s['p_val']:>10.3e}  "
              f"{s['std_coef']:>10.4f}  {s['std_coef_se']:>8.4f}")

    fig = make_figure(stats_list)
    plt.show()


if __name__ == '__main__':
    main()
