#!/usr/bin/env python3
"""Adoption probability analysis: does complex centrality predict who adopts?

Tests the core complex contagion prediction (Guilbeault & Centola 2021):
nodes reachable via wide bridges (low complex path length) should be more
likely to adopt. This is distinct from amplification (cascade size), which
conflates topology with agent heterogeneity.

Runs model live to access agent-level attributes (alpha, rho, w_i, change_time)
not stored in snapshot pkl files. Tests adoption via logistic regression and
time-to-adoption via Cox proportional hazards.

Usage: cd model_src/testing && python test_adoption_complex_centrality.py
"""
import sys, os, random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, Counter
from scipy.stats import spearmanr, pointbiserialr
import statsmodels.api as sm

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
# Feature extraction from live model
# ---------------------------------------------------------------------------

def extract_agent_features(model):
    """Extract per-agent features from live model after run completes."""
    G = model.G1
    nodes = list(G.nodes())
    N = len(nodes)

    # Complex centrality
    print("  Computing complex centrality (T=2)...")
    cc = compute_complex_centrality(G, T=2)
    n_nonzero = sum(1 for v in cc.values() if v > 0)
    print(f"  {n_nonzero}/{N} nodes with CC>0 ({100*n_nonzero/N:.0f}%)")

    # Standard centralities
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)

    # Per-agent data from live objects
    rows = []
    for a in model.agents:
        init_diet = model.snapshots[0]['diets'][a.i]
        adopted = (init_diet == 'meat' and a.diet == 'veg')
        rows.append({
            'node': a.i,
            'adopted': int(adopted),
            'init_meat': int(init_diet == 'meat'),
            'change_time': a.change_time if adopted else np.nan,
            'theta': a.theta,
            'alpha': a.alpha,
            'rho': a.rho,
            'w_i': a.w_i,
            'immune': int(a.immune),
            'degree': G.degree(a.i),
            'betweenness': betweenness[a.i],
            'clustering': clustering[a.i],
            'complex_cent': cc[a.i],
            # Source diversity: unique neighbors who contributed to memory at switch
            'memory_sources': len(set(src for _, src in a.memory)),
            'memory_veg_sources': len(set(src for d, src in a.memory if d == 'veg')),
        })
    return pd.DataFrame(rows)


def source_diversity_at_switch(model):
    """For agents that switched meat->veg, count unique veg sources in memory
    at the time of switch. Proxy for reinforcement breadth."""
    # We can't reconstruct exact memory-at-switch from post-hoc data,
    # but current memory_veg_sources is a reasonable proxy for agents
    # that switched and stayed switched.
    pass  # handled in extract_agent_features via memory fields


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_adoption_analysis(df):
    """Logistic regression: P(adopt) ~ complex_cent + controls."""
    # Only initial meat-eaters (adoption is meat->veg transition)
    meat = df[df['init_meat'] == 1].copy()
    # Exclude immune agents (they can't switch by design)
    meat = meat[meat['immune'] == 0]
    n_adopt = meat['adopted'].sum()
    print(f"\n  Initial meat-eaters (non-immune): {len(meat)}, adopted: {n_adopt} "
          f"({100*n_adopt/len(meat):.1f}%)")

    if n_adopt < 5 or n_adopt == len(meat):
        print("  WARNING: Too few/many adoptions for logistic regression")
        return None, meat

    # Point-biserial correlations (adopted is binary)
    predictors = ['complex_cent', 'degree', 'betweenness', 'clustering',
                  'theta', 'alpha', 'w_i', 'rho', 'memory_veg_sources']
    print(f"\n  {'='*55}")
    print(f"  POINT-BISERIAL CORRELATIONS: adopted ~ predictor")
    print(f"  {'='*55}")
    print(f"  {'Predictor':<20} {'r_pb':>8} {'p-value':>12} {'sig':>5}")
    print(f"  {'-'*50}")
    corr_results = []
    for p in predictors:
        valid = meat[[p, 'adopted']].dropna()
        if valid[p].std() == 0:
            print(f"  {p:<20} {'--':>8} {'--':>12}  (no variance)")
            continue
        r, pval = pointbiserialr(valid['adopted'], valid[p])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"  {p:<20} {r:>8.3f} {pval:>12.2e}  {sig:>4}")
        corr_results.append({'predictor': p, 'r_pb': r, 'p': pval})

    # Logistic regression with standardized predictors
    log_preds = ['complex_cent', 'degree', 'clustering', 'theta', 'alpha', 'rho']
    X = meat[log_preds].copy()
    # Drop columns with zero variance
    X = X.loc[:, X.std() > 0]
    active_preds = list(X.columns)
    y = meat['adopted']

    X_std = (X - X.mean()) / X.std()
    X_std = sm.add_constant(X_std)
    try:
        logit = sm.Logit(y, X_std).fit(disp=0)
        print(f"\n  {'='*55}")
        print(f"  LOGISTIC REGRESSION: P(adopt) ~ standardized predictors")
        print(f"  {'='*55}")
        print(logit.summary2().tables[1].to_string())
        print(f"\n  Pseudo R-squared: {logit.prsquared:.3f}")
    except Exception as e:
        print(f"  Logit failed: {e}")
        logit = None

    return pd.DataFrame(corr_results), meat


def run_time_to_adoption(df):
    """Among adopters: does complex centrality predict earlier adoption?"""
    adopters = df[(df['adopted'] == 1) & df['change_time'].notna()].copy()
    if len(adopters) < 10:
        print("  Too few adopters for time-to-adoption analysis")
        return

    predictors = ['complex_cent', 'degree', 'clustering', 'theta', 'alpha', 'w_i']
    print(f"\n  {'='*55}")
    print(f"  SPEARMAN: change_time ~ predictor [N={len(adopters)} adopters]")
    print(f"  {'='*55}")
    print(f"  {'Predictor':<20} {'rho_s':>8} {'p-value':>12} {'sig':>5}")
    print(f"  {'-'*50}")
    for p in predictors:
        valid = adopters[[p, 'change_time']].dropna()
        if len(valid) < 5 or valid[p].std() == 0:
            continue
        rho, pval = spearmanr(valid[p], valid['change_time'])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        # Negative rho = higher predictor -> earlier adoption
        print(f"  {p:<20} {rho:>8.3f} {pval:>12.2e}  {sig:>4}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_adoption_diagnostics(all_meat_dfs, output_dir='../../visualisations_output'):
    """Multi-panel figure: adoption probability vs complex centrality + controls."""
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)
    meat = pd.concat(all_meat_dfs, ignore_index=True)

    # Only non-immune initial meat-eaters
    meat = meat[(meat['init_meat'] == 1) & (meat['immune'] == 0)]

    predictors = ['complex_cent', 'degree', 'clustering', 'theta', 'w_i']
    labels = ['Complex centrality', 'Degree', 'Clustering', r'$\theta$', r'$w_i$']
    n = len(predictors)
    cm_ = 1/2.54
    fig, axes = plt.subplots(1, n, figsize=(n * 4.5 * cm_, 5.5 * cm_))

    for i, (pred, lab) in enumerate(zip(predictors, labels)):
        ax = axes[i]
        valid = meat[[pred, 'adopted']].dropna()
        if valid[pred].std() == 0:
            ax.text(0.5, 0.5, 'No variance', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            continue

        # Binned adoption rates
        try:
            valid['bin'] = pd.qcut(valid[pred], q=8, duplicates='drop')
        except ValueError:
            valid['bin'] = pd.cut(valid[pred], bins=5)
        binned = valid.groupby('bin', observed=True).agg(
            x=(pred, 'mean'), adopt_rate=('adopted', 'mean'),
            n=('adopted', 'count')
        ).dropna()

        ax.scatter(valid[pred], valid['adopted'] + np.random.normal(0, 0.02, len(valid)),
                   s=4, alpha=0.15, c='#999', edgecolors='none', rasterized=True)
        ax.plot(binned['x'], binned['adopt_rate'], 'o-',
                color=COLORS['primary'], lw=1.5, ms=4, zorder=5)

        r, pval = pointbiserialr(valid['adopted'], valid[pred])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        ax.set_title(f"$r_{{pb}}$={r:.2f}{sig}", fontsize=7)
        ax.set_xlabel(lab, fontsize=7)
        if i == 0:
            ax.set_ylabel('P(adopt)', fontsize=7)
        ax.set_ylim(-0.05, 1.15)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.97, chr(65+i), transform=ax.transAxes, fontsize=9,
                fontweight='bold', va='top')

    plt.tight_layout()
    out = f'{output_dir}/adoption_complex_centrality.pdf'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("ADOPTION PROBABILITY ANALYSIS: complex centrality as predictor")
    print("Guilbeault & Centola (2021) prediction: nodes reachable via")
    print("wide bridges should be more likely to adopt")
    print("=" * 65)

    params = DEFAULT_PARAMS.copy()
    all_meat_dfs = []

    for run_i in range(RUNS):
        seed = SEED_BASE + run_i
        print(f"\n--- Run {run_i+1}/{RUNS} (seed={seed}) ---")
        np.random.seed(seed); random.seed(seed)
        model = get_model(params)
        model.run()
        print(f"  Final veg fraction: {model.fraction_veg[-1]:.3f}")

        df = extract_agent_features(model)
        corr_df, meat_df = run_adoption_analysis(df)
        run_time_to_adoption(df)
        meat_df['run'] = run_i
        all_meat_dfs.append(meat_df)

    # Pooled analysis across runs
    print("\n" + "=" * 65)
    print("POOLED ANALYSIS (all runs)")
    print("=" * 65)
    pooled = pd.concat(all_meat_dfs, ignore_index=True)
    pooled_meat = pooled[(pooled['init_meat'] == 1) & (pooled['immune'] == 0)]
    print(f"Total observations: {len(pooled_meat)}")

    # Pooled point-biserial
    predictors = ['complex_cent', 'degree', 'betweenness', 'clustering',
                  'theta', 'alpha', 'w_i', 'rho', 'memory_veg_sources']
    print(f"\n{'Predictor':<20} {'r_pb':>8} {'p-value':>12} {'sig':>5}")
    print(f"{'-'*50}")
    for p in predictors:
        valid = pooled_meat[[p, 'adopted']].dropna()
        if len(valid) < 10 or valid[p].std() == 0:
            continue
        r, pval = pointbiserialr(valid['adopted'], valid[p])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"{p:<20} {r:>8.3f} {pval:>12.2e}  {sig:>4}")

    # Pooled logistic regression
    log_preds = ['complex_cent', 'degree', 'clustering', 'theta', 'alpha', 'rho']
    X = pooled_meat[log_preds].copy()
    X = X.loc[:, X.std() > 0]
    y = pooled_meat['adopted']
    X_std = (X - X.mean()) / X.std()
    X_std = sm.add_constant(X_std)
    try:
        logit = sm.Logit(y, X_std).fit(disp=0)
        print(f"\nPOOLED LOGISTIC REGRESSION (N={len(pooled_meat)}):")
        print(logit.summary2().tables[1].to_string())
        print(f"Pseudo R-squared: {logit.prsquared:.3f}")
    except Exception as e:
        print(f"Pooled logit failed: {e}")

    plot_adoption_diagnostics(all_meat_dfs)
    plt.show()


if __name__ == '__main__':
    main()
