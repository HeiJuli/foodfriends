#!/usr/bin/env python3
"""Alpha regime sweep: does complex centrality matter more when agents are socially driven?

Hypothesis: under high alpha (self-reliant), complex centrality is irrelevant because
agents switch on internal disposition. Under low alpha (socially driven), bridging
topology matters -> complex centrality should become a stronger predictor of amplification.

Sweeps alpha_min/alpha_max windows, runs 3 trajectories per regime, extracts
Spearman correlations of centrality measures vs log(amplification).

Output: summary table + interaction plot (mean_alpha vs rho_s per predictor).
"""
import sys, os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import spearmanr
import networkx as nx

# Run from model_src/ so model imports work
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import model_main
from model_runn import get_model, ensure_output_dir

sys.path.append('../plotting')
from plot_styles import set_publication_style, COLORS
from agency_predictor_analysis import compute_complex_centrality

DIRECT_REDUCTION_KG = 664  # 2054 - 1390

# Alpha regimes: (alpha_min, alpha_max, label)
ALPHA_REGIMES = [
    (0.05, 0.25, "very low"),
    (0.05, 0.45, "low"),
    (0.05, 0.65, "mid-low"),
    (0.05, 0.80, "baseline"),
    (0.30, 0.80, "mid-high"),
    (0.50, 0.90, "high"),
]

BASE_PARAMS = {
    "veg_CO2": 1390, "vegan_CO2": 1054, "meat_CO2": 2054,
    "N": 650, "erdos_p": 3, "steps": 35000, "k": 8,
    "immune_n": 0.10, "M": 9, "veg_f": 0, "meat_f": 0.95,
    "p_rewire": 0.01, "rewire_h": 0.1, "tc": 0.7,
    "topology": "homophilic_emp", "beta": 20,
    "alpha": 0.36, "rho": 0.45, "theta": 0.44,
    "agent_ini": "sample-max",
    "survey_file": "../data/hierarchical_agents.csv",
    "adjust_veg_fraction": True, "target_veg_fraction": 0.06,
    "tau": 0.015, "theta_gate_c": 0.35, "theta_gate_k": 25,
    "mu": 0.2, "gamma": 0.5, "tau_persistence": None,
}

RUNS_PER_REGIME = 3
PREDICTORS = ['degree', 'betweenness', 'eigenvector', 'clustering', 'complex_cent']


def extract_features(snapshots):
    """Compute centrality + amplification from final snapshot."""
    G = snapshots['final']['graph']
    reductions = np.array(snapshots['final']['reductions'])
    init_diets = snapshots[0]['diets']
    nodes = list(G.nodes())
    N = len(nodes)

    multipliers = reductions / DIRECT_REDUCTION_KG
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality_numpy(G)
    clustering = nx.clustering(G)
    complex_cent = compute_complex_centrality(G, T=2)

    df = pd.DataFrame({
        'node': nodes, 'multiplier': multipliers, 'reduction_kg': reductions,
        'degree': [G.degree(n) for n in nodes],
        'betweenness': [betweenness[n] for n in nodes],
        'eigenvector': [eigenvector[n] for n in nodes],
        'clustering': [clustering[n] for n in nodes],
        'complex_cent': [complex_cent[n] for n in nodes],
        'theta': [G.nodes[n].get('theta', 0) for n in nodes],
        'init_veg': [1 if init_diets[i] == 'veg' else 0 for i in range(N)],
    })
    return df


def run_regime(alpha_min, alpha_max, n_runs=RUNS_PER_REGIME):
    """Run n_runs trajectories for one alpha regime, return median-run features."""
    params = BASE_PARAMS.copy()
    params['alpha_min'] = alpha_min
    params['alpha_max'] = alpha_max

    runs = []
    for i in range(n_runs):
        seed = 42 + i
        np.random.seed(seed); random.seed(seed)
        model = get_model(params)
        model.run()
        mean_alpha = np.mean([a.alpha for a in model.agents])
        mean_wi = np.mean([a.w_i for a in model.agents])
        runs.append({
            'final_veg': model.fraction_veg[-1],
            'mean_alpha': mean_alpha, 'mean_wi': mean_wi,
            'snapshots': model.snapshots,
        })
        print(f"  run {i+1}/{n_runs}: final_veg={model.fraction_veg[-1]:.3f}, "
              f"mean_alpha={mean_alpha:.3f}, mean_w_i={mean_wi:.3f}")

    # Pick median trajectory
    finals = [r['final_veg'] for r in runs]
    median_idx = np.argmin(np.abs(np.array(finals) - np.median(finals)))
    return runs[median_idx]


def compute_correlations(df):
    """Spearman correlations of each predictor vs log(multiplier)."""
    pos = df[df['reduction_kg'] > 0].copy()
    if len(pos) < 10:
        return {p: (np.nan, np.nan) for p in PREDICTORS}, len(pos)
    pos['log_mult'] = np.log(pos['multiplier'])
    results = {}
    for p in PREDICTORS:
        rho, pval = spearmanr(pos[p], pos['log_mult'])
        results[p] = (rho, pval)
    return results, len(pos)


def main():
    print("=" * 65)
    print("ALPHA REGIME SWEEP: complex centrality interaction test")
    print("=" * 65)

    all_results = []
    for alpha_min, alpha_max, label in ALPHA_REGIMES:
        print(f"\n--- Regime: [{alpha_min:.2f}, {alpha_max:.2f}] ({label}) ---")
        run_data = run_regime(alpha_min, alpha_max)
        df = extract_features(run_data['snapshots'])
        corrs, n_pos = compute_correlations(df)

        row = {
            'alpha_min': alpha_min, 'alpha_max': alpha_max, 'label': label,
            'mean_alpha': run_data['mean_alpha'], 'mean_wi': run_data['mean_wi'],
            'final_veg': run_data['final_veg'], 'n_agents_with_reduction': n_pos,
        }
        for p in PREDICTORS:
            row[f'rho_{p}'] = corrs[p][0]
            row[f'p_{p}'] = corrs[p][1]
        all_results.append(row)

    results_df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "=" * 85)
    print(f"{'Regime':<12} {'mean_a':>7} {'mean_w':>7} {'veg_f':>6} {'N+':>4}"
          + "".join(f" {'rho_'+p:>14}" for p in PREDICTORS))
    print("-" * 85)
    for _, r in results_df.iterrows():
        line = f"{r['label']:<12} {r['mean_alpha']:>7.3f} {r['mean_wi']:>7.3f} {r['final_veg']:>6.3f} {r['n_agents_with_reduction']:>4.0f}"
        for p in PREDICTORS:
            rho, pval = r[f'rho_{p}'], r[f'p_{p}']
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            line += f" {rho:>8.3f}{sig:<4}"
            # padding
        print(line)

    # Plot
    set_publication_style()
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(12*cm, 8*cm))
    colors = {'degree': COLORS['primary'], 'betweenness': '#e67e22',
              'eigenvector': '#8e44ad', 'clustering': '#27ae60', 'complex_cent': '#c0392b'}
    markers = {'degree': 'o', 'betweenness': 's', 'eigenvector': '^',
               'clustering': 'D', 'complex_cent': 'P'}

    x = results_df['mean_alpha']
    for p in PREDICTORS:
        y = results_df[f'rho_{p}']
        sig_mask = results_df[f'p_{p}'] < 0.05
        ax.plot(x, y, f'-{markers[p]}', color=colors[p], label=p, ms=5, lw=1.2)
        # Mark non-significant as open
        if (~sig_mask).any():
            ax.scatter(x[~sig_mask], y[~sig_mask], marker=markers[p],
                       facecolors='none', edgecolors=colors[p], s=30, zorder=5)

    ax.axhline(0, color='#666', lw=0.8, ls='--')
    ax.set_xlabel('Mean alpha (self-reliance)', fontsize=8)
    ax.set_ylabel(r'$\rho_s$ (Spearman vs log amplification)', fontsize=8)
    ax.set_title('Centrality predictors across alpha regimes', fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs('../../visualisations_output', exist_ok=True)
    out = '../../visualisations_output/alpha_complex_centrality_sweep.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {out}")

    # Save data
    ensure_output_dir()
    results_df.to_pickle(f'../model_output/alpha_cc_sweep_{pd.Timestamp.now().strftime("%Y%m%d")}.pkl')
    plt.show()


if __name__ == '__main__':
    main()
