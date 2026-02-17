#!/usr/bin/env python3
"""Agency predictor analysis: what predicts high individual amplification?

Standalone exploratory script. Computes centrality measures (including complex
centrality per Guilbeault & Centola 2021) from stored graph, runs Spearman
correlations and OLS regression, saves diagnostic figure.

Usage: python agency_predictor_analysis.py [path_to_pkl]
"""
import sys, os, glob
from collections import deque
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from plot_styles import set_publication_style, COLORS

cm = 1/2.54
DIRECT_REDUCTION_KG = 664  # 2054 - 1390

DEFAULT_FILE = '/home/jpoveralls/Downloads/trajectory_analysis_twin_20260213.pkl'

def load_latest(file_path=None):
    if file_path:
        return pd.read_pickle(file_path)
    if os.path.exists(DEFAULT_FILE):
        print(f"Loading: {DEFAULT_FILE}")
        return pd.read_pickle(DEFAULT_FILE)
    files = sorted(glob.glob('../model_output/trajectory_analysis_*.pkl'))
    if not files:
        print("ERROR: No trajectory_analysis pkl files found"); sys.exit(1)
    print(f"Loading: {files[-1]}")
    return pd.read_pickle(files[-1])

def get_median_row(data):
    if 'is_median_twin' in data.columns and data['is_median_twin'].any():
        return data[data['is_median_twin']].iloc[0]
    return data.iloc[len(data) // 2]


# --- Complex centrality (Guilbeault & Centola 2021, Nat. Commun. 12:4430) ---
# Eqs 1-7 in Methods section. Bridge width captures whether sufficient
# reinforcing ties exist between neighborhoods for complex contagion to spread.

def compute_complex_centrality(G, T=2):
    """Compute complex centrality for all nodes.

    Per Guilbeault & Centola (2021): complex centrality CC_i is the average
    complex path length from node i to all other nodes. Complex path length
    counts hops through a "sufficient bridge graph" where edge (u,v) exists
    only if v has >= T neighbors in the closed neighborhood N[u].

    Args:
        G: networkx Graph
        T: adoption threshold (default 2 = simplest complex contagion)

    Returns:
        dict: {node: complex_centrality_value}
    """
    nodes = list(G.nodes())
    n = len(nodes)

    # Precompute closed neighborhoods (sets) for fast intersection
    closed_nbr = {v: set(G.neighbors(v)) | {v} for v in nodes}

    # Build sufficient bridge graph: directed edge u->v exists iff
    # bridge_width(u->v) >= T, i.e., |N(v) ∩ N[u]| >= T
    # For edge (u,v): N(v) ∩ N[u] = common_neighbors(u,v) ∪ {u}
    # so bridge_width = |common_neighbors(u,v)| + 1
    adj_sufficient = {v: [] for v in nodes}
    for u, v in G.edges():
        cn = len(closed_nbr[u] & closed_nbr[v]) - 2  # common neighbors (excl u,v)
        # Bridge u->v: v's neighbors in N[u] = common_neighbors + u itself
        w_uv = cn + 1
        # Bridge v->u: u's neighbors in N[v] = common_neighbors + v itself
        w_vu = cn + 1
        if w_uv >= T:
            adj_sufficient[u].append(v)
        if w_vu >= T:
            adj_sufficient[v].append(u)

    # BFS from each node in sufficient bridge graph
    cc = {}
    for i in nodes:
        # BFS
        dist = {i: 0}
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v in adj_sufficient[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)

        # Average complex path length to nodes outside N[i]
        # Unreachable nodes contribute 0 (Eq. 4: PL_C = 0 if no path)
        ni = closed_nbr[i]
        n_targets = n - len(ni)
        if n_targets > 0:
            total = sum(d for v, d in dist.items() if v not in ni)
            cc[i] = total / n_targets
        else:
            cc[i] = 0.0

    return cc


def extract_features(median_row):
    """Extract all predictor features and response variable from snapshot."""
    snap_final = median_row['snapshots']['final']
    snap_init = median_row['snapshots'][0]
    G = snap_final['graph']
    nodes = list(G.nodes())
    N = len(nodes)

    reductions_kg = np.array(snap_final['reductions'])
    multipliers = reductions_kg / DIRECT_REDUCTION_KG

    # Standard centrality measures
    print(f"Computing centrality measures for N={N} nodes...")
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality_numpy(G)
    clustering = nx.clustering(G)

    # Complex centrality (Guilbeault & Centola 2021)
    # NOTE: The homophilic_emp network has very low clustering (~0.007), so
    # almost no edges form triangles. At T=2 only ~6% of edges are sufficient
    # bridges (243/2000 nodes with CC>0). T>=3 is completely degenerate (0
    # sufficient bridges). This is a topology-dependent result: complex
    # centrality requires clustered networks to differentiate nodes.
    print(f"Computing complex centrality (T=2)...")
    complex_cent = compute_complex_centrality(G, T=2)

    # Initial diets from step-0 snapshot
    init_diets = snap_init['diets']

    df = pd.DataFrame({
        'node': nodes,
        'multiplier': multipliers,
        'reduction_kg': reductions_kg,
        'degree': [G.degree(n) for n in nodes],
        'betweenness': [betweenness[n] for n in nodes],
        'eigenvector': [eigenvector[n] for n in nodes],
        'clustering': [clustering[n] for n in nodes],
        'complex_cent': [complex_cent[n] for n in nodes],
        'theta': [G.nodes[n].get('theta', 0) for n in nodes],
        'initial_diet': [init_diets[i] for i in range(N)],
    })
    df['init_veg'] = (df['initial_diet'] == 'veg').astype(int)
    return df

def run_correlations(df):
    """Spearman correlations of each predictor vs log(multiplier)."""
    pos = df[df['reduction_kg'] > 0].copy()
    pos['log_mult'] = np.log(pos['multiplier'])
    predictors = ['degree', 'betweenness', 'eigenvector', 'clustering',
                  'complex_cent', 'theta', 'init_veg']

    print(f"\n{'='*60}")
    print(f"SPEARMAN CORRELATIONS vs log(multiplier)  [N={len(pos)}]")
    print(f"{'='*60}")
    print(f"{'Predictor':<16} {'rho_s':>8} {'p-value':>12} {'sig':>5}")
    print(f"{'-'*45}")
    results = []
    for p in predictors:
        rho, pval = spearmanr(pos[p], pos['log_mult'])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"{p:<16} {rho:>8.3f} {pval:>12.2e}  {sig:>4}")
        results.append({'predictor': p, 'rho_s': rho, 'p': pval})
    return pd.DataFrame(results), pos

def run_regression(pos):
    """OLS regression: log(multiplier) ~ all predictors."""
    predictors = ['degree', 'betweenness', 'eigenvector', 'clustering',
                  'complex_cent', 'theta', 'init_veg']
    X = pos[predictors].copy()
    y = pos['log_mult']

    # Standardize for comparable coefficients
    X_std = (X - X.mean()) / X.std()
    X_std = sm.add_constant(X_std)
    model = sm.OLS(y, X_std).fit()

    print(f"\n{'='*60}")
    print(f"OLS REGRESSION: log(multiplier) ~ predictors (standardized)")
    print(f"{'='*60}")
    print(model.summary2().tables[1].to_string())
    print(f"\nR-squared: {model.rsquared:.3f}   Adj R-squared: {model.rsquared_adj:.3f}")

    # VIF for collinearity
    X_raw = sm.add_constant(pos[predictors])
    print(f"\n{'='*60}")
    print(f"VARIANCE INFLATION FACTORS")
    print(f"{'='*60}")
    for i, p in enumerate(['const'] + predictors):
        vif = variance_inflation_factor(X_raw.values, i)
        flag = ' <-- HIGH' if vif > 5 else ''
        print(f"  {p:<16} VIF = {vif:.2f}{flag}")

    return model, predictors

def make_diagnostic_figure(pos, corr_df, model, predictors, output_dir='../visualisations_output'):
    """Multi-panel diagnostic figure: scatters for significant predictors + coefficient forest plot."""
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # Identify significant predictors (p < 0.05) sorted by |rho|
    sig = corr_df[corr_df['p'] < 0.05].sort_values('rho_s', key=abs, ascending=False)
    n_sig = len(sig)
    n_panels = min(n_sig, 5) + 1  # scatter panels + forest plot

    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4.5 * cm, 5.5 * cm))
    if n_panels == 1:
        axes = [axes]

    # Scatter panels for significant predictors
    for i, (_, row) in enumerate(sig.head(5).iterrows()):
        ax = axes[i]
        pred = row['predictor']
        x = pos[pred]
        y = pos['multiplier']
        ax.scatter(x, y, s=6, alpha=0.4, c=COLORS['primary'], edgecolors='none', rasterized=True)

        # Binned means
        bins = np.percentile(x, np.linspace(0, 100, 8))
        bc, bm = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (x >= lo) & (x < hi)
            if mask.sum() > 2:
                bc.append(x[mask].mean()); bm.append(y[mask].mean())
        if bc:
            ax.plot(bc, bm, 'o-', color=COLORS['secondary'], lw=1.2, ms=3, zorder=5)

        ax.set_yscale('log')
        ax.set_xlabel(pred, fontsize=7)
        if i == 0:
            ax.set_ylabel('Amplification', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_title(f"$\\rho_s$={row['rho_s']:.2f}", fontsize=7)
        ax.text(0.02, 0.97, chr(65+i), transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

    # Forest plot of standardized coefficients
    ax_f = axes[-1]
    coefs = model.params[1:]  # skip const
    cis = model.conf_int().iloc[1:]
    colors_bar = ['#c0392b' if model.pvalues[p] < 0.05 else '#aaa' for p in predictors]
    y_pos = np.arange(len(predictors))

    ax_f.barh(y_pos, coefs, color=colors_bar, height=0.6, alpha=0.7)
    ax_f.errorbar(coefs, y_pos, xerr=[(coefs - cis[0]).values, (cis[1] - coefs).values],
                  fmt='none', color='#333', lw=0.8, capsize=2)
    ax_f.axvline(0, color='#666', lw=0.8, ls='--')
    ax_f.set_yticks(y_pos); ax_f.set_yticklabels(predictors, fontsize=6)
    ax_f.set_xlabel('Std. coef.', fontsize=7)
    ax_f.tick_params(labelsize=6)
    ax_f.spines['top'].set_visible(False); ax_f.spines['right'].set_visible(False)
    ax_f.set_title(f"OLS (R$^2$={model.rsquared:.2f})", fontsize=7)
    ax_f.text(0.02, 0.97, chr(65+n_panels-1), transform=ax_f.transAxes, fontsize=9,
              fontweight='bold', va='top')

    plt.tight_layout()
    out = f'{output_dir}/agency_predictor_diagnostic.pdf'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out}")
    return fig

if __name__ == '__main__':
    fp = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_latest(fp)
    print(f"Loaded {len(data)} runs, N={len(data.iloc[0]['snapshots']['final']['reductions'])} agents")
    median_row = get_median_row(data)
    df = extract_features(median_row)
    corr_df, pos = run_correlations(df)
    model, predictors = run_regression(pos)
    make_diagnostic_figure(pos, corr_df, model, predictors)
    plt.show()
