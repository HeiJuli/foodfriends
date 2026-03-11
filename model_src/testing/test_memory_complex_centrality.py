#!/usr/bin/env python3
"""Memory-length sweep: does shorter M increase complex centrality's predictive power?

Hypothesis: M=9 provides temporal reinforcement that collapses structural
requirements measured by complex centrality (Guilbeault & Centola 2021).
Shorter M should increase reliance on simultaneous neighbor adoption,
making complex centrality a stronger predictor of spreading effectiveness.

Sweep M in [1, 3, 5, 7, 9] x 10 trajectories each (sample-max, N=385).
For each M, compute Spearman correlations of centrality measures vs
log(amplification) and compare complex centrality's predictive power.

Usage: python test_memory_complex_centrality.py
"""
import sys, os, random, time
import numpy as np
import pandas as pd
from collections import deque, Counter
from scipy.stats import spearmanr
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# Change to model_src so relative paths (../data, ../model_output) work
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import model_main
from model_runn import DEFAULT_PARAMS, get_model, ensure_output_dir

import networkx as nx

DIRECT_REDUCTION_KG = 664
M_VALUES = [1, 3, 5, 7, 9]
N_RUNS = 10

# --- Complex centrality (from agency_predictor_analysis.py) ---
def compute_complex_centrality(G, T=2):
    nodes = list(G.nodes())
    n = len(nodes)
    closed_nbr = {v: set(G.neighbors(v)) | {v} for v in nodes}
    adj_sufficient = {v: [] for v in nodes}
    for u, v in G.edges():
        cn = len(closed_nbr[u] & closed_nbr[v]) - 2
        w = cn + 1
        if w >= T:
            adj_sufficient[u].append(v)
            adj_sufficient[v].append(u)
    cc = {}
    for i in nodes:
        dist = {i: 0}
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v in adj_sufficient[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        ni = closed_nbr[i]
        n_targets = n - len(ni)
        cc[i] = sum(d for v, d in dist.items() if v not in ni) / n_targets if n_targets > 0 else 0.0
    return cc


def run_single(params, seed):
    np.random.seed(seed)
    random.seed(seed)
    model = get_model(params)
    model.run()
    return model


def extract_centrality_correlations(model):
    """Extract features from final snapshot and compute Spearman correlations."""
    snap = model.snapshots.get('steady', model.snapshots.get('final'))
    snap_init = model.snapshots[0]
    G = snap['graph']
    nodes = list(G.nodes())
    N = len(nodes)

    reductions = np.array(snap['reductions'])
    multipliers = reductions / DIRECT_REDUCTION_KG

    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality_numpy(G)
    clustering = nx.clustering(G)
    complex_cent = compute_complex_centrality(G, T=2)

    df = pd.DataFrame({
        'node': nodes,
        'multiplier': multipliers,
        'reduction_kg': reductions,
        'degree': [G.degree(n) for n in nodes],
        'betweenness': [betweenness[n] for n in nodes],
        'eigenvector': [eigenvector[n] for n in nodes],
        'clustering': [clustering[n] for n in nodes],
        'complex_cent': [complex_cent[n] for n in nodes],
        'theta': [G.nodes[n].get('theta', 0) for n in nodes],
        'initial_diet': [snap_init['diets'][i] for i in range(N)],
    })
    df['init_veg'] = (df['initial_diet'] == 'veg').astype(int)

    # Spearman on positive reductions only
    pos = df[df['reduction_kg'] > 0].copy()
    if len(pos) < 10:
        return None
    pos['log_mult'] = np.log(pos['multiplier'])

    predictors = ['degree', 'betweenness', 'eigenvector', 'clustering',
                  'complex_cent', 'theta', 'init_veg']
    results = {}
    for p in predictors:
        rho, pval = spearmanr(pos[p], pos['log_mult'])
        results[p] = {'rho_s': rho, 'p': pval}
    results['n_pos'] = len(pos)
    results['n_total'] = N
    results['final_veg_f'] = sum(1 for d in snap['diets'] if d == 'veg') / N
    return results


def main():
    print("=" * 65)
    print("MEMORY-LENGTH SWEEP: Complex centrality vs temporal reinforcement")
    print("=" * 65)

    all_results = []
    t0 = time.time()

    for M in M_VALUES:
        print(f"\n{'─' * 60}")
        print(f"  M = {M}  ({N_RUNS} runs, sample-max N=385)")
        print(f"{'─' * 60}")

        params = DEFAULT_PARAMS.copy()
        params['M'] = M
        params['agent_ini'] = 'sample-max'

        for run_i in range(N_RUNS):
            seed = 42 + run_i
            t1 = time.time()
            model = run_single(params, seed)
            elapsed = time.time() - t1

            corrs = extract_centrality_correlations(model)
            if corrs is None:
                print(f"  run {run_i}: too few positive reductions, skipping")
                continue

            row = {'M': M, 'run': run_i, 'seed': seed,
                   'n_pos': corrs.pop('n_pos'), 'n_total': corrs.pop('n_total'),
                   'final_veg_f': corrs.pop('final_veg_f')}
            for pred, vals in corrs.items():
                row[f'{pred}_rho'] = vals['rho_s']
                row[f'{pred}_p'] = vals['p']
            all_results.append(row)
            print(f"  run {run_i}: veg_f={row['final_veg_f']:.3f}  "
                  f"CC_rho={row.get('complex_cent_rho', float('nan')):.3f}  "
                  f"deg_rho={row.get('degree_rho', float('nan')):.3f}  "
                  f"({elapsed:.0f}s)")

    df = pd.DataFrame(all_results)
    total = time.time() - t0
    print(f"\nTotal runtime: {int(total/60)}m {total%60:.0f}s")

    # --- Summary table ---
    predictors = ['degree', 'betweenness', 'eigenvector', 'clustering',
                  'complex_cent', 'theta', 'init_veg']

    print(f"\n{'=' * 75}")
    print(f"SUMMARY: Mean Spearman rho_s across {N_RUNS} runs per M")
    print(f"{'=' * 75}")
    header = f"{'M':>3}  " + "  ".join(f"{p:>14}" for p in predictors)
    print(header)
    print("-" * len(header))

    summary_rows = []
    for M in M_VALUES:
        sub = df[df['M'] == M]
        vals = [f"{sub[f'{p}_rho'].mean():>7.3f}({sub[f'{p}_rho'].std():>.3f})" for p in predictors]
        print(f"{M:>3}  " + "  ".join(f"{v:>14}" for v in vals))
        summary_rows.append({
            'M': M,
            **{f'{p}_rho_mean': sub[f'{p}_rho'].mean() for p in predictors},
            **{f'{p}_rho_std': sub[f'{p}_rho'].std() for p in predictors},
            **{f'{p}_sig_frac': (sub[f'{p}_p'] < 0.05).mean() for p in predictors},
        })

    print(f"\nFraction of runs with p < 0.05:")
    print(f"{'M':>3}  " + "  ".join(f"{p:>14}" for p in predictors))
    print("-" * len(header))
    for sr in summary_rows:
        vals = [f"{sr[f'{p}_sig_frac']:>14.0%}" for p in predictors]
        print(f"{sr['M']:>3}  " + "  ".join(vals))

    # --- Save results ---
    ensure_output_dir()
    outfile = '../model_output/memory_complex_centrality_sweep.pkl'
    df.to_pickle(outfile)
    print(f"\nSaved raw results: {outfile}")

    # --- Plot ---
    make_summary_plot(df, predictors, M_VALUES)


def make_summary_plot(df, predictors, M_values):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'plotting'))
    try:
        from plot_styles import set_publication_style, COLORS
        set_publication_style()
    except ImportError:
        COLORS = {'primary': '#2c3e50', 'secondary': '#e74c3c'}

    cm = 1/2.54
    fig, axes = plt.subplots(1, 2, figsize=(16*cm, 7*cm))

    # Panel A: mean rho_s vs M for key predictors
    ax = axes[0]
    key_preds = ['degree', 'betweenness', 'complex_cent', 'clustering']
    colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad']
    for pred, c in zip(key_preds, colors):
        means = [df[df['M'] == M][f'{pred}_rho'].mean() for M in M_values]
        sems = [df[df['M'] == M][f'{pred}_rho'].sem() for M in M_values]
        ax.errorbar(M_values, means, yerr=sems, marker='o', ms=4, lw=1.2,
                    color=c, label=pred, capsize=2)
    ax.set_xlabel('Memory length M', fontsize=8)
    ax.set_ylabel('Mean Spearman $\\rho_s$', fontsize=8)
    ax.set_xticks(M_values)
    ax.legend(fontsize=6, loc='best')
    ax.axhline(0, color='#999', lw=0.5, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_title('A  Correlation with log(amplification)', fontsize=8, loc='left', fontweight='bold')

    # Panel B: fraction significant (p<0.05) vs M
    ax = axes[1]
    for pred, c in zip(key_preds, colors):
        fracs = [(df[df['M'] == M][f'{pred}_p'] < 0.05).mean() for M in M_values]
        ax.plot(M_values, fracs, marker='s', ms=4, lw=1.2, color=c, label=pred)
    ax.set_xlabel('Memory length M', fontsize=8)
    ax.set_ylabel('Fraction runs p < 0.05', fontsize=8)
    ax.set_xticks(M_values)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, loc='best')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_title('B  Statistical significance', fontsize=8, loc='left', fontweight='bold')

    plt.tight_layout()
    outdir = '../../visualisations_output'
    os.makedirs(outdir, exist_ok=True)
    outpath = f'{outdir}/memory_complex_centrality_sweep.pdf'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {outpath}")


if __name__ == '__main__':
    main()
