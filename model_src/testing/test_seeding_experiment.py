#!/usr/bin/env python3
"""Seeding experiment: does complex centrality predict cascade size?

Matches Guilbeault & Centola (2021) methodology: activate ONE node per
simulation, measure cascade size, repeat for all/sample of nodes, correlate
with complex centrality and other centrality measures.

Design:
1. Build network + assign heterogeneous agent parameters ONCE
2. For each seed node i (sample of N_SEEDS nodes):
   - Reset all diets to meat
   - Set node i to veg + immune (stays veg)
   - Run model for SEED_STEPS timesteps
   - Record number of adoptions (cascade size)
3. Compute CC at T=2,3,4 + standard centralities
4. Correlate cascade_size ~ CC + degree + betweenness + clustering

Usage: cd model_src/testing && python test_seeding_experiment.py
"""
import sys, os, random, time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import spearmanr
import statsmodels.api as sm

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import model_main
from model_runn import get_model, DEFAULT_PARAMS

sys.path.append('../plotting')
from plot_styles import set_publication_style, COLORS
from agency_predictor_analysis import compute_complex_centrality

N_SEEDS = 200       # nodes to test (random sample)
SEED_STEPS = [1000, 5000, 20000]  # multiple timescales
BASE_SEED = 42

# ---------------------------------------------------------------------------
# Seeding experiment
# ---------------------------------------------------------------------------

def build_baseline_model(params):
    """Build model once to get network + agent parameters.
    Initializes agents and network without running the simulation."""
    np.random.seed(BASE_SEED); random.seed(BASE_SEED)
    model = get_model(params)
    # Initialize agents + network (normally done at start of run())
    model.agent_ini()
    model.harmonise_netIn()
    model.record_fraction()
    model.record_snapshot(0)
    return model


def run_single_seed(model, seed_node_id, steps=20000):
    """Reset all agents to meat, seed one node, run, return cascade size.

    Matches model_main.run() loop: one random agent activated per step,
    coin-flip whether they interact. No rewiring (frozen topology)."""
    agents = model.agents
    G = model.G1
    N = len(agents)

    # Store original state
    orig = [(a.diet, a.immune, a.influence_parent, a.change_time,
             a.reduction_out, a.influenced_agents.copy(), list(a.memory))
            for a in agents]

    # Reset all agents to meat, no immune
    for a in agents:
        a.diet = "meat"
        a.immune = False
        a.influence_parent = None
        a.change_time = None
        a.reduction_out = 0
        a.influenced_agents = set()
        a.C = a.diet_emissions("meat")
        neigh_ids = list(G.neighbors(a.i))
        if neigh_ids:
            a.memory = [("meat", random.choice(neigh_ids)) for _ in range(model.params["M"])]
        else:
            a.memory = [("meat", a.i)] * model.params["M"]

    # Activate seed node (immune so it stays veg)
    seed_agent = agents[seed_node_id]
    seed_agent.diet = "veg"
    seed_agent.immune = True
    seed_agent.C = seed_agent.diet_emissions("veg")

    # Main loop — matches model_main.run() structure
    # Record cascade size at multiple checkpoints
    checkpoints = sorted(set(SEED_STEPS)) if isinstance(SEED_STEPS, list) else [steps]
    max_steps = max(checkpoints)
    cascade_at = {}
    for t in range(1, max_steps + 1):
        i = np.random.choice(N)
        if np.random.random() < 0.50:
            agents[i].step(G, agents, t)
        if t in checkpoints:
            cascade_at[t] = sum(1 for a in agents if a.diet == "veg" and a.i != seed_node_id)

    n_adopted = cascade_at[max_steps]
    adopter_times = [a.change_time for a in agents
                     if a.diet == "veg" and a.i != seed_node_id and a.change_time is not None]

    # Restore original state
    for a, (diet, immune, parent, ct, red, infl, mem) in zip(agents, orig):
        a.diet = diet
        a.immune = immune
        a.influence_parent = parent
        a.change_time = ct
        a.reduction_out = red
        a.influenced_agents = infl
        a.memory = mem
        a.C = a.diet_emissions(a.diet)

    return n_adopted, adopter_times, cascade_at


def compute_centralities(G, nodes):
    """Compute all centrality measures for the network."""
    print("Computing centralities...")
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality_numpy(G)
    clustering = nx.clustering(G)

    print("Computing complex centrality T=2...")
    cc2 = compute_complex_centrality(G, T=2)
    print("Computing complex centrality T=3...")
    cc3 = compute_complex_centrality(G, T=3)
    print("Computing complex centrality T=4...")
    cc4 = compute_complex_centrality(G, T=4)

    return {n: {
        'degree': G.degree(n),
        'betweenness': betweenness[n],
        'eigenvector': eigenvector[n],
        'clustering': clustering[n],
        'cc_T2': cc2[n],
        'cc_T3': cc3[n],
        'cc_T4': cc4[n],
    } for n in nodes}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(df):
    """Spearman correlations and partial correlations."""
    predictors = ['degree', 'betweenness', 'eigenvector', 'clustering',
                  'cc_T2', 'cc_T3', 'cc_T4', 'theta', 'alpha', 'rho']

    print(f"\n{'='*65}")
    print(f"BIVARIATE SPEARMAN: cascade_size ~ predictor [N={len(df)}]")
    print(f"{'='*65}")
    print(f"{'Predictor':<18} {'rho_s':>8} {'p-value':>12} {'sig':>5}")
    print(f"{'-'*48}")
    for p in predictors:
        if p not in df.columns or df[p].std() == 0:
            continue
        rho, pval = spearmanr(df['cascade_size'], df[p])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"{p:<18} {rho:>+8.3f} {pval:>12.2e}  {sig:>4}")

    # Partial correlations controlling for degree
    print(f"\n{'='*65}")
    print(f"PARTIAL SPEARMAN (controlling for degree): cascade_size ~ predictor")
    print(f"{'='*65}")
    from scipy.stats import rankdata
    rank_y = rankdata(df['cascade_size'])
    rank_deg = rankdata(df['degree'])
    # Residualize y on degree
    resid_y = sm.OLS(rank_y, sm.add_constant(rank_deg)).fit().resid

    print(f"{'Predictor':<18} {'partial_rho':>12} {'p-value':>12} {'sig':>5}")
    print(f"{'-'*48}")
    for p in predictors:
        if p == 'degree' or p not in df.columns or df[p].std() == 0:
            continue
        rank_x = rankdata(df[p])
        resid_x = sm.OLS(rank_x, sm.add_constant(rank_deg)).fit().resid
        rho, pval = spearmanr(resid_y, resid_x)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"{p:<18} {rho:>+12.3f} {pval:>12.2e}  {sig:>4}")

    # OLS regression
    ols_preds = [p for p in ['degree', 'cc_T2', 'cc_T3', 'clustering',
                              'betweenness', 'theta', 'rho']
                 if p in df.columns and df[p].std() > 0]
    X = df[ols_preds].copy()
    X_std = (X - X.mean()) / X.std()
    X_std = sm.add_constant(X_std)
    y = df['cascade_size']
    ols = sm.OLS(y, X_std).fit()
    print(f"\n{'='*65}")
    print(f"OLS REGRESSION: cascade_size ~ standardized predictors")
    print(f"{'='*65}")
    print(f"R² = {ols.rsquared:.4f}, Adj R² = {ols.rsquared_adj:.4f}")
    print(ols.summary2().tables[1].to_string())


def plot_results(df, output_dir='../../visualisations_output'):
    """Scatter plots: cascade size vs key predictors."""
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    predictors = ['degree', 'cc_T2', 'cc_T3', 'clustering', 'betweenness']
    labels = ['Degree', 'CC (T=2)', 'CC (T=3)', 'Clustering', 'Betweenness']
    n = len(predictors)
    cm_ = 1/2.54
    fig, axes = plt.subplots(1, n, figsize=(n * 4.5 * cm_, 5.5 * cm_))

    for i, (pred, lab) in enumerate(zip(predictors, labels)):
        ax = axes[i]
        if pred not in df.columns or df[pred].std() == 0:
            ax.text(0.5, 0.5, 'No variance', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            continue

        ax.scatter(df[pred], df['cascade_size'], s=8, alpha=0.5,
                   c=COLORS['primary'], edgecolors='none', rasterized=True)

        rho, pval = spearmanr(df['cascade_size'], df[pred])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        ax.set_title(f"$\\rho_s$={rho:+.2f}{sig}", fontsize=7)
        ax.set_xlabel(lab, fontsize=7)
        if i == 0:
            ax.set_ylabel('Cascade size', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.97, chr(65+i), transform=ax.transAxes, fontsize=9,
                fontweight='bold', va='top')

    plt.tight_layout()
    out = f'{output_dir}/seeding_experiment.pdf'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(params, beta_val, seed_nodes=None):
    """Run seeding experiment for a given beta (inverse temperature)."""
    params = params.copy()
    params['beta'] = beta_val
    print(f"\n{'#'*65}")
    print(f"# BETA = {beta_val} (inverse temperature)")
    print(f"{'#'*65}")

    model = build_baseline_model(params)
    G = model.G1
    nodes = list(G.nodes())
    N = len(nodes)
    print(f"Network: N={N}, E={G.number_of_edges()}, "
          f"avg_degree={2*G.number_of_edges()/N:.1f}, "
          f"clustering={nx.average_clustering(G):.3f}")

    cent = compute_centralities(G, nodes)

    # Sample seed nodes (stratified by degree)
    if seed_nodes is None:
        degrees = np.array([G.degree(n) for n in nodes])
        n_seeds = min(N_SEEDS, N)
        seed_nodes = []
        for q_lo, q_hi in [(0, 25), (25, 50), (50, 75), (75, 100)]:
            lo, hi = np.percentile(degrees, q_lo), np.percentile(degrees, q_hi)
            candidates = [n for n in nodes if lo <= G.degree(n) <= hi]
            k = min(n_seeds // 4, len(candidates))
            seed_nodes.extend(random.sample(candidates, k))
        seed_nodes = seed_nodes[:n_seeds]

    print(f"Sampled {len(seed_nodes)} seed nodes "
          f"(degree range: {min(G.degree(n) for n in seed_nodes)}-"
          f"{max(G.degree(n) for n in seed_nodes)})")

    results = []
    t0 = time.time()
    for idx, seed_id in enumerate(seed_nodes):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(seed_nodes) - idx - 1) if idx > 0 else 0
            print(f"  Seed {idx+1}/{len(seed_nodes)} "
                  f"(node={seed_id}, deg={G.degree(seed_id)}) "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        np.random.seed(BASE_SEED + seed_id)
        random.seed(BASE_SEED + seed_id)
        n_adopted, adopt_times, cascade_at = run_single_seed(model, seed_id)

        a = model.agents[seed_id]
        row = {
            'seed_node': seed_id, 'beta': beta_val,
            'cascade_size': n_adopted,
            'median_adopt_time': np.median(adopt_times) if adopt_times else np.nan,
            'theta': a.theta, 'alpha': a.alpha, 'rho': a.rho,
            **cent[seed_id],
        }
        for t_check, n_at in cascade_at.items():
            row[f'cascade_{t_check}'] = n_at
        results.append(row)

    df = pd.DataFrame(results)
    elapsed = time.time() - t0
    print(f"\nCompleted {len(df)} seeds in {elapsed:.0f}s")

    # Analyze at each timescale
    for steps in sorted(set(SEED_STEPS)):
        col = f'cascade_{steps}'
        if col in df.columns:
            print(f"\n--- Timescale: {steps} steps ---")
            tmp = df.copy()
            tmp['cascade_size'] = tmp[col]
            print(f"Cascade: mean={tmp['cascade_size'].mean():.1f}, "
                  f"std={tmp['cascade_size'].std():.1f}, "
                  f"range=[{tmp['cascade_size'].min()}, {tmp['cascade_size'].max()}]")
            analyze_results(tmp)

    return df, seed_nodes


def main():
    print("=" * 65)
    print("SEEDING EXPERIMENT: Guilbeault & Centola (2021) replication")
    print(f"Activate 1 node per run, measure cascade size, N_SEEDS={N_SEEDS}")
    print(f"Timescales: {SEED_STEPS}")
    print("=" * 65)

    params = DEFAULT_PARAMS.copy()
    beta_values = [5, 20, 40]  # low (noisy), default, high (sharp)

    all_dfs = []
    seed_nodes = None
    for beta_val in beta_values:
        df, seed_nodes = run_experiment(params, beta_val, seed_nodes)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    os.makedirs('../../model_output', exist_ok=True)
    outfile = f'../../model_output/seeding_experiment_{pd.Timestamp.now().strftime("%Y%m%d")}.pkl'
    combined.to_pickle(outfile)
    print(f"\nSaved: {outfile}")

    # Summary comparison across betas
    print(f"\n{'='*65}")
    print("BETA COMPARISON SUMMARY (cascade at t=5000)")
    print(f"{'='*65}")
    for beta_val in beta_values:
        sub = combined[combined['beta'] == beta_val]
        col = 'cascade_5000'
        if col in sub.columns:
            rho_deg, p_deg = spearmanr(sub[col], sub['degree'])
            rho_cc2, p_cc2 = spearmanr(sub[col], sub['cc_T2'])
            rho_cc3, p_cc3 = spearmanr(sub[col], sub['cc_T3'])
            print(f"  beta={beta_val:>3}: cascade mean={sub[col].mean():.1f} std={sub[col].std():.1f} | "
                  f"degree rho={rho_deg:+.3f} (p={p_deg:.3f}) | "
                  f"CC_T2 rho={rho_cc2:+.3f} (p={p_cc2:.3f}) | "
                  f"CC_T3 rho={rho_cc3:+.3f} (p={p_cc3:.3f})")

    plot_results(combined[combined['beta'] == 20])
    plt.show()

    return combined


if __name__ == '__main__':
    main()
