#!/usr/bin/env python3
"""Contagion transition diagnostic: is our model simple or complex?

Implements the Horsevad et al. (2022, Nat. Commun. 13:1442) diagnostic:
- Simple contagion: spreading correlates negatively with path length / Kirchhoff
  index, and is uncorrelated with clustering.
- Complex contagion: spreading correlates positively with clustering, and is
  uncorrelated with path length / Kirchhoff index.

We test this at two levels:
1. NODE-LEVEL: do individual topology metrics predict adoption/speed?
2. NETWORK-LEVEL: across parameter regimes, does the correlation structure
   between topology and spreading shift from simple -> complex signatures?

Also computes Kirchhoff index R_g = N * sum(1/lambda_i) from the graph
Laplacian (Horsevad Eq. 8), which we haven't used before.

Usage: cd model_src/testing && python test_contagion_transition.py
"""
import sys, os, random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import spearmanr, pointbiserialr

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import model_main
from model_runn import get_model, DEFAULT_PARAMS

sys.path.append('../plotting')
from plot_styles import set_publication_style, COLORS
from agency_predictor_analysis import compute_complex_centrality

RUNS = 5
SEED_BASE = 42

# ---------------------------------------------------------------------------
# Kirchhoff index (Horsevad Eq. 8)
# ---------------------------------------------------------------------------

def kirchhoff_index(G):
    """R_g = N * sum(1/lambda_i) for nonzero Laplacian eigenvalues."""
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigs = np.linalg.eigvalsh(L)
    # Skip lambda_0 = 0 (connected graph has exactly one zero eigenvalue)
    nonzero = eigs[eigs > 1e-10]
    N = len(G)
    return N * np.sum(1.0 / nonzero)


def local_resistance_distance(G, node, sample_size=50):
    """Approximate mean resistance distance from node to random sample of others.
    Uses pseudoinverse of Laplacian: r(i,j) = L+_ii + L+_jj - 2*L+_ij."""
    L = nx.laplacian_matrix(G).toarray().astype(float)
    L_pinv = np.linalg.pinv(L)
    nodes = list(G.nodes())
    idx = nodes.index(node)
    # Sample targets
    others = [n for n in nodes if n != node]
    if len(others) > sample_size:
        others = random.sample(others, sample_size)
    dists = []
    for t in others:
        j = nodes.index(t)
        r_ij = L_pinv[idx, idx] + L_pinv[j, j] - 2 * L_pinv[idx, j]
        dists.append(r_ij)
    return np.mean(dists)


# ---------------------------------------------------------------------------
# Polarization speed (Horsevad Eq. 3)
# ---------------------------------------------------------------------------

def polarization_speed(model, t_early=1000):
    """v = (P(t) - P(0)) / t, where P = fraction veg.
    Measured at t_early to capture early contagion dynamics."""
    t = min(t_early, len(model.fraction_veg) - 1)
    return (model.fraction_veg[t] - model.fraction_veg[0]) / t


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_node_features(model):
    """Per-node topology + adoption features from live model."""
    G = model.G1
    nodes = list(G.nodes())
    N = len(nodes)

    # Topology
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)
    cc = compute_complex_centrality(G, T=2)

    # Resistance distance per node (expensive but informative)
    print("  Computing resistance distances...")
    L = nx.laplacian_matrix(G).toarray().astype(float)
    L_pinv = np.linalg.pinv(L)
    node_idx = {n: i for i, n in enumerate(nodes)}

    rows = []
    for a in model.agents:
        init_diet = model.snapshots[0]['diets'][a.i]
        adopted = (init_diet == 'meat' and a.diet == 'veg')

        # Mean resistance distance to all other nodes
        idx = node_idx[a.i]
        r_dists = [L_pinv[idx, idx] + L_pinv[j, j] - 2 * L_pinv[idx, j]
                    for j in range(N) if j != idx]
        mean_r_dist = np.mean(r_dists)

        # Ego-network clustering (local bridge density)
        ego_nodes = list(G.neighbors(a.i)) + [a.i]
        ego = G.subgraph(ego_nodes)
        ego_density = nx.density(ego) if len(ego_nodes) > 2 else 0

        rows.append({
            'node': a.i,
            'adopted': int(adopted),
            'init_meat': int(init_diet == 'meat'),
            'change_time': a.change_time if adopted else np.nan,
            'immune': int(a.immune),
            'theta': a.theta, 'alpha': a.alpha, 'rho': a.rho, 'w_i': a.w_i,
            'degree': G.degree(a.i),
            'betweenness': betweenness[a.i],
            'clustering': clustering[a.i],
            'complex_cent': cc[a.i],
            'mean_resistance_dist': mean_r_dist,
            'ego_density': ego_density,
            'memory_veg_sources': len(set(src for d, src in a.memory if d == 'veg')),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transition diagnostic
# ---------------------------------------------------------------------------

def horsevad_diagnostic(df, outcome='adopted'):
    """Horsevad et al. transition diagnostic.
    Tests correlation of topology metrics with spreading outcome.
    Returns classification: 'simple', 'complex', or 'transition'."""

    meat = df[(df['init_meat'] == 1) & (df['immune'] == 0)].copy()
    if len(meat) < 20:
        return 'insufficient_data', {}

    metrics = {
        'clustering': 'C',
        'mean_resistance_dist': 'R_local',
        'degree': 'k',
        'complex_cent': 'CC',
        'ego_density': 'ego_C',
    }

    results = {}
    print(f"\n  {'='*60}")
    print(f"  HORSEVAD DIAGNOSTIC: topology ~ {outcome}")
    print(f"  {'='*60}")
    print(f"  {'Metric':<22} {'r':>8} {'p-value':>12} {'sig':>5}")
    print(f"  {'-'*50}")

    for col, label in metrics.items():
        valid = meat[[col, outcome]].dropna()
        if len(valid) < 10 or valid[col].std() == 0:
            print(f"  {label:<22} {'--':>8} {'--':>12}  (skip)")
            continue
        if meat[outcome].nunique() == 2:
            r, p = pointbiserialr(valid[outcome], valid[col])
        else:
            r, p = spearmanr(valid[col], valid[outcome])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {label:<22} {r:>8.3f} {p:>12.2e}  {sig:>4}")
        results[label] = (r, p)

    # Classification per Horsevad Fig 2b:
    # Simple: r_s(R_g) < 0 and r_s(C) ~ 0
    # Complex: r_s(C) > 0 and r_s(R_g) ~ 0
    r_C = results.get('C', (0, 1))[0]
    p_C = results.get('C', (0, 1))[1]
    r_R = results.get('R_local', (0, 1))[0]
    p_R = results.get('R_local', (0, 1))[1]

    if p_C < 0.05 and r_C > 0 and p_R > 0.05:
        classification = 'COMPLEX'
    elif p_R < 0.05 and r_R < 0 and p_C > 0.05:
        classification = 'SIMPLE'
    elif p_C < 0.05 and r_C > 0 and p_R < 0.05 and r_R < 0:
        classification = 'TRANSITION (both sig)'
    else:
        classification = 'INDETERMINATE'

    print(f"\n  >> Classification: {classification}")
    print(f"     r_s(C)={r_C:.3f} (p={p_C:.2e}), r_s(R_local)={r_R:.3f} (p={p_R:.2e})")
    return classification, results


def adoption_speed_diagnostic(df):
    """Among adopters: does topology predict WHEN they adopt?
    This is closer to the polarization speed concept."""
    adopters = df[(df['adopted'] == 1) & df['change_time'].notna()].copy()
    if len(adopters) < 15:
        print("  Too few adopters for speed diagnostic")
        return {}

    print(f"\n  {'='*60}")
    print(f"  SPEED DIAGNOSTIC: change_time ~ topology [N={len(adopters)}]")
    print(f"  (negative rho = higher metric -> earlier adoption)")
    print(f"  {'='*60}")

    metrics = ['clustering', 'mean_resistance_dist', 'degree',
               'complex_cent', 'ego_density', 'theta', 'alpha', 'rho']
    results = {}
    print(f"  {'Metric':<22} {'rho_s':>8} {'p-value':>12} {'sig':>5}")
    print(f"  {'-'*50}")
    for m in metrics:
        valid = adopters[[m, 'change_time']].dropna()
        if len(valid) < 10 or valid[m].std() == 0:
            continue
        r, p = spearmanr(valid[m], valid['change_time'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {m:<22} {r:>8.3f} {p:>12.2e}  {sig:>4}")
        results[m] = (r, p)
    return results


# ---------------------------------------------------------------------------
# Network-level analysis across regimes
# ---------------------------------------------------------------------------

def network_level_diagnostic(models_data):
    """Horsevad-style: across multiple runs/topologies, correlate
    network-level metrics with network-level spreading outcomes."""
    print(f"\n{'='*65}")
    print("NETWORK-LEVEL DIAGNOSTIC (across runs)")
    print(f"{'='*65}")

    rows = []
    for d in models_data:
        G = d['graph']
        rows.append({
            'clustering': nx.average_clustering(G),
            'avg_path': nx.average_shortest_path_length(G) if nx.is_connected(G)
                        else np.nan,
            'kirchhoff': kirchhoff_index(G) if nx.is_connected(G) else np.nan,
            'final_veg': d['final_veg'],
            'pol_speed': d['pol_speed'],
            'n_adopters': d['n_adopters'],
        })
    ndf = pd.DataFrame(rows)
    print(ndf.to_string(index=False))

    if len(ndf) < 5:
        print("  Too few data points for network-level correlations")
        return ndf

    for outcome in ['final_veg', 'pol_speed']:
        print(f"\n  Spearman: {outcome} ~ network metric")
        for metric in ['clustering', 'avg_path', 'kirchhoff']:
            valid = ndf[[metric, outcome]].dropna()
            if len(valid) < 4:
                continue
            r, p = spearmanr(valid[metric], valid[outcome])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {metric:<16} rho_s={r:>7.3f}  p={p:.2e} {sig}")
    return ndf


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_diagnostic(all_dfs, output_dir='../../visualisations_output'):
    """Horsevad-style diagnostic plot: topology metrics vs adoption."""
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    meat = pd.concat(all_dfs, ignore_index=True)
    meat = meat[(meat['init_meat'] == 1) & (meat['immune'] == 0)]

    metrics = [
        ('clustering', 'Clustering $C$'),
        ('mean_resistance_dist', 'Mean resistance dist'),
        ('degree', 'Degree $k$'),
        ('complex_cent', 'Complex centrality'),
        ('ego_density', 'Ego density'),
    ]

    cm_ = 1/2.54
    fig, axes = plt.subplots(2, len(metrics), figsize=(len(metrics)*4*cm_, 10*cm_))

    for i, (col, label) in enumerate(metrics):
        valid = meat[[col, 'adopted']].dropna()
        if valid[col].std() == 0:
            continue

        # Top row: adoption probability
        ax = axes[0, i]
        try:
            valid['bin'] = pd.qcut(valid[col], q=8, duplicates='drop')
        except ValueError:
            valid['bin'] = pd.cut(valid[col], bins=5)
        binned = valid.groupby('bin', observed=True).agg(
            x=(col, 'mean'), rate=('adopted', 'mean'), n=('adopted', 'count')
        ).dropna()
        ax.plot(binned['x'], binned['rate'], 'o-',
                color=COLORS['primary'], lw=1.5, ms=4)
        r, p = pointbiserialr(valid['adopted'], valid[col])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.set_title(f"$r_{{pb}}$={r:.3f}{sig}", fontsize=7)
        if i == 0:
            ax.set_ylabel('P(adopt)', fontsize=7)
        ax.set_xlabel(label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.97, chr(65+i), transform=ax.transAxes, fontsize=9,
                fontweight='bold', va='top')

        # Bottom row: time to adoption (among adopters)
        ax2 = axes[1, i]
        adopters = meat[(meat['adopted'] == 1) & meat['change_time'].notna()]
        valid2 = adopters[[col, 'change_time']].dropna()
        if len(valid2) > 10 and valid2[col].std() > 0:
            try:
                valid2['bin'] = pd.qcut(valid2[col], q=8, duplicates='drop')
            except ValueError:
                valid2['bin'] = pd.cut(valid2[col], bins=5)
            binned2 = valid2.groupby('bin', observed=True).agg(
                x=(col, 'mean'), t=('change_time', 'mean'), n=('change_time', 'count')
            ).dropna()
            ax2.plot(binned2['x'], binned2['t'], 's-',
                     color=COLORS.get('secondary', '#e67e22'), lw=1.5, ms=4)
            r2, p2 = spearmanr(valid2[col], valid2['change_time'])
            sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
            ax2.set_title(f"$\\rho_s$={r2:.3f}{sig2}", fontsize=7)
        if i == 0:
            ax2.set_ylabel('Time to adopt', fontsize=7)
        ax2.set_xlabel(label, fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        ax2.text(0.02, 0.97, chr(70+i), transform=ax2.transAxes, fontsize=9,
                 fontweight='bold', va='top')

    plt.tight_layout()
    out = f'{output_dir}/contagion_transition_diagnostic.pdf'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("CONTAGION TRANSITION DIAGNOSTIC")
    print("Horsevad et al. (2022, Nat. Commun. 13:1442)")
    print("Is our model operating as simple or complex contagion?")
    print("=" * 65)

    params = DEFAULT_PARAMS.copy()
    all_dfs = []
    models_data = []

    for run_i in range(RUNS):
        seed = SEED_BASE + run_i
        print(f"\n--- Run {run_i+1}/{RUNS} (seed={seed}) ---")
        np.random.seed(seed); random.seed(seed)
        model = get_model(params)
        model.run()

        v = polarization_speed(model, t_early=1000)
        print(f"  Final veg: {model.fraction_veg[-1]:.3f}, "
              f"pol_speed(t=1000): {v:.6f}")

        # Global network stats
        G = model.G1
        print(f"  C={nx.average_clustering(G):.3f}, "
              f"R_g={kirchhoff_index(G):.0f}")

        df = extract_node_features(model)
        classification, corrs = horsevad_diagnostic(df)
        speed_corrs = adoption_speed_diagnostic(df)
        df['run'] = run_i
        all_dfs.append(df)

        n_adopt = df[(df['init_meat']==1) & (df['immune']==0)]['adopted'].sum()
        models_data.append({
            'graph': G.copy(),
            'final_veg': model.fraction_veg[-1],
            'pol_speed': v,
            'n_adopters': n_adopt,
        })

    # Pooled node-level diagnostic
    print("\n" + "=" * 65)
    print("POOLED NODE-LEVEL DIAGNOSTIC (all runs)")
    print("=" * 65)
    pooled = pd.concat(all_dfs, ignore_index=True)
    horsevad_diagnostic(pooled)
    adoption_speed_diagnostic(pooled)

    # Network-level diagnostic
    network_level_diagnostic(models_data)

    # Kirchhoff context
    print("\n" + "=" * 65)
    print("KIRCHHOFF INDEX CONTEXT")
    print("=" * 65)
    for i, md in enumerate(models_data):
        G = md['graph']
        Rg = kirchhoff_index(G)
        N = len(G)
        # For comparison: complete graph R_g = N-1, ring R_g ~ N^2/6
        print(f"  Run {i}: R_g={Rg:.0f}, N={N}, "
              f"R_g/N={Rg/N:.1f} (complete={N-1}, ring~{N**2/6:.0f})")

    plot_diagnostic(all_dfs)
    plt.show()


if __name__ == '__main__':
    main()
