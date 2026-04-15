#!/usr/bin/env python3
"""Agency predictor analysis: adoption, contagion, and amplification.

Three dependent variables from the spreader's perspective:
  1. Adoption:      did agent i switch? (binary, logistic)
  2. Contagion:      how many agents did i directly convert? (count)
  3. Amplification:  total cascade credit in kg CO2 (continuous)
Plus: degree-amplification scaling (linear vs super-linear) and
network degree assortativity.

Usage: python agency_predictor_analysis.py [path_to_pkl]
"""
import sys, os, glob
from collections import deque
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pointbiserialr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from plot_styles import set_publication_style, COLORS

cm = 1/2.54
DIRECT_REDUCTION_KG = 664  # 2054 - 1390

DEFAULT_FILE = '../model_output/trajectory_analysis_twin_20260402.pkl'
CACHE_FILE  = '../model_output/agency_two_dvs_cache.pkl'

TOPO_PREDS = ['degree', 'betweenness', 'eigenvector', 'clustering', 'complex_cent']
PSYCH_PREDS = ['rho', 'alpha', 'theta']
ALL_PREDS = TOPO_PREDS + PSYCH_PREDS


# --- I/O ---

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

ANALYSIS_T = 139000  # analysis cutoff — ensemble-median logistic 95% t_end

def get_median_row(data):
    if 'is_median_twin' in data.columns and data['is_median_twin'].any():
        return data[data['is_median_twin']].iloc[0]
    return data.iloc[len(data) // 2]

def _resolve_snap(snapshots, t_cutoff=ANALYSIS_T):
    """Return snapshot at t_cutoff (nearest int key), else 'final'."""
    if t_cutoff is not None:
        int_ts = sorted(t for t in snapshots if isinstance(t, int) and t > 0)
        if int_ts:
            best = min(int_ts, key=lambda t: abs(t - t_cutoff))
            return snapshots[best]
    return snapshots.get('final', snapshots[max(t for t in snapshots if isinstance(t, int))])


# --- Complex centrality (Guilbeault & Centola 2021) ---

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


# --- Feature extraction ---

def extract_features(row):
    """Build agent-level DataFrame from a single run's snapshots."""
    snap_final = _resolve_snap(row['snapshots'])
    snap_init = row['snapshots'][0]
    G = snap_final['graph']
    nodes = list(G.nodes())
    N = len(nodes)

    reductions_kg = np.array(snap_final['reductions'])
    init_diets = snap_init['diets']
    final_diets = snap_final['diets']

    # Direct conversions: from snapshot if available, else from influence_parents
    if 'direct_conversions' in snap_final:
        direct_conv = np.array(snap_final['direct_conversions'])
    elif 'influence_parents' in snap_final:
        parents = snap_final['influence_parents']
        direct_conv = np.zeros(N, dtype=int)
        for p in parents:
            if p is not None:
                direct_conv[p] += 1
    else:
        direct_conv = None
        print("WARNING: no direct_conversions or influence_parents in snapshot")

    # Centrality measures
    print(f"Computing centrality measures for N={N}...")
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality_numpy(G)
    clustering = nx.clustering(G)
    print(f"Computing complex centrality (T=2)...")
    complex_cent = compute_complex_centrality(G, T=2)

    df = pd.DataFrame({
        'node': nodes,
        'reduction_kg': reductions_kg,
        'multiplier': reductions_kg / DIRECT_REDUCTION_KG,
        'degree': [G.degree(n) for n in nodes],
        'betweenness': [betweenness[n] for n in nodes],
        'eigenvector': [eigenvector[n] for n in nodes],
        'clustering': [clustering[n] for n in nodes],
        'complex_cent': [complex_cent[n] for n in nodes],
        'theta': [G.nodes[n].get('theta', 0) for n in nodes],
        'alpha': np.array(snap_final['alphas']),
        'rho': np.array(snap_final['rhos']),
        'initial_diet': [init_diets[i] for i in range(N)],
        'final_diet': [final_diets[i] for i in range(N)],
    })
    if direct_conv is not None:
        df['direct_conversions'] = direct_conv
    df['init_meat'] = (df['initial_diet'] == 'meat').astype(int)
    df['adopted'] = ((df['initial_diet'] == 'meat') & (df['final_diet'] == 'veg')).astype(int)
    return df


# --- Analysis: three dependent variables ---

def _sig_stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def analyze_adoption(df):
    """DV1: binary adoption among initial meat-eaters."""
    meat = df[df['init_meat'] == 1].copy()
    print(f"\n{'='*65}")
    print(f"DV1: ADOPTION (binary)  [N_meat={len(meat)}, switched={meat['adopted'].sum()}]")
    print(f"{'='*65}")

    # Point-biserial correlations
    print(f"\n--- Point-biserial correlations vs adopted ---")
    print(f"{'Predictor':<16} {'r_pb':>8} {'p-value':>12} {'sig':>5}")
    print(f"{'-'*45}")
    results = []
    for p in ALL_PREDS:
        r, pval = pointbiserialr(meat['adopted'], meat[p])
        print(f"{p:<16} {r:>8.3f} {pval:>12.2e}  {_sig_stars(pval):>4}")
        results.append({'predictor': p, 'r_pb': r, 'p': pval})

    # Logistic regression: psychology-only vs topology-only vs full
    for label, preds in [('psychology', PSYCH_PREDS), ('topology', TOPO_PREDS), ('full', ALL_PREDS)]:
        X = sm.add_constant((meat[preds] - meat[preds].mean()) / meat[preds].std())
        model = sm.Logit(meat['adopted'], X).fit(disp=0)
        print(f"\n  Logistic [{label}]: pseudo-R2={model.prsquared:.4f}  AIC={model.aic:.0f}")
    return pd.DataFrame(results)


def analyze_contagion(df):
    """DV2: direct conversion count (from spreader's perspective)."""
    if 'direct_conversions' not in df.columns:
        print("\nWARNING: skipping contagion analysis (no direct_conversions data)")
        return None

    # All agents who ever switched to veg (have any cascade credit OR direct conversions)
    spreaders = df[(df['reduction_kg'] > 0) | (df['direct_conversions'] > 0)].copy()
    print(f"\n{'='*65}")
    print(f"DV2: CONTAGION (direct conversions)  [N_spreaders={len(spreaders)}]")
    print(f"{'='*65}")
    print(f"  mean={spreaders['direct_conversions'].mean():.2f}  "
          f"median={spreaders['direct_conversions'].median():.0f}  "
          f"max={spreaders['direct_conversions'].max():.0f}")

    # Spearman correlations
    print(f"\n--- Spearman correlations vs direct_conversions ---")
    print(f"{'Predictor':<16} {'rho_s':>8} {'p-value':>12} {'sig':>5}")
    print(f"{'-'*45}")
    results = []
    for p in ALL_PREDS:
        rho, pval = spearmanr(spreaders[p], spreaders['direct_conversions'])
        print(f"{p:<16} {rho:>8.3f} {pval:>12.2e}  {_sig_stars(pval):>4}")
        results.append({'predictor': p, 'rho_s': rho, 'p': pval})

    # OLS: direct_conversions ~ predictors (Poisson would be better but OLS is fine for exploration)
    for label, preds in [('psychology', PSYCH_PREDS), ('topology', TOPO_PREDS), ('full', ALL_PREDS)]:
        X = sm.add_constant((spreaders[preds] - spreaders[preds].mean()) / spreaders[preds].std())
        model = sm.OLS(spreaders['direct_conversions'], X).fit()
        print(f"\n  OLS [{label}]: R2={model.rsquared:.4f}  adj-R2={model.rsquared_adj:.4f}")

    return pd.DataFrame(results)


def analyze_amplification(df):
    """DV3: log(amplification multiplier) for agents with positive cascade credit."""
    pos = df[df['reduction_kg'] > 0].copy()
    pos['log_mult'] = np.log(pos['multiplier'])
    print(f"\n{'='*65}")
    print(f"DV3: AMPLIFICATION (log multiplier)  [N_positive={len(pos)}]")
    print(f"{'='*65}")

    # Spearman correlations
    print(f"\n--- Spearman correlations vs log(multiplier) ---")
    print(f"{'Predictor':<16} {'rho_s':>8} {'p-value':>12} {'sig':>5}")
    print(f"{'-'*45}")
    results = []
    for p in ALL_PREDS:
        rho, pval = spearmanr(pos[p], pos['log_mult'])
        print(f"{p:<16} {rho:>8.3f} {pval:>12.2e}  {_sig_stars(pval):>4}")
        results.append({'predictor': p, 'rho_s': rho, 'p': pval})

    # OLS comparisons
    for label, preds in [('psychology', PSYCH_PREDS), ('topology', TOPO_PREDS), ('full', ALL_PREDS)]:
        X = sm.add_constant((pos[preds] - pos[preds].mean()) / pos[preds].std())
        model = sm.OLS(pos['log_mult'], X).fit()
        print(f"\n  OLS [{label}]: R2={model.rsquared:.4f}  adj-R2={model.rsquared_adj:.4f}")

    # Full model details
    X_full = sm.add_constant((pos[ALL_PREDS] - pos[ALL_PREDS].mean()) / pos[ALL_PREDS].std())
    full_model = sm.OLS(pos['log_mult'], X_full).fit()
    print(f"\n--- Full OLS coefficients (standardized) ---")
    print(full_model.summary2().tables[1].to_string())

    return pd.DataFrame(results), pos, full_model


# --- Degree scaling and network diagnostics ---

def analyze_degree_scaling(df):
    """Check if amplification scales linearly or super-linearly with degree."""
    pos = df[df['reduction_kg'] > 0].copy()
    print(f"\n{'='*65}")
    print(f"DEGREE-AMPLIFICATION SCALING")
    print(f"{'='*65}")

    # Bin by degree deciles and compute mean amplification
    pos['deg_bin'] = pd.qcut(pos['degree'], 10, duplicates='drop')
    binned = pos.groupby('deg_bin', observed=True).agg(
        mean_deg=('degree', 'mean'), mean_red=('reduction_kg', 'mean'),
        median_red=('reduction_kg', 'median'), n=('degree', 'size')
    ).reset_index()
    print("\n  Degree decile bins:")
    print(f"  {'mean_deg':>8} {'mean_red':>10} {'median_red':>10} {'n':>5}")
    for _, r in binned.iterrows():
        print(f"  {r['mean_deg']:>8.1f} {r['mean_red']:>10.1f} {r['median_red']:>10.1f} {r['n']:>5.0f}")

    # Log-log regression: log(reduction) ~ log(degree) -> slope > 1 means super-linear
    log_deg = np.log(pos['degree'])
    log_red = np.log(pos['reduction_kg'])
    X = sm.add_constant(log_deg)
    model = sm.OLS(log_red, X).fit()
    slope = model.params.iloc[1]
    ci = model.conf_int().iloc[1]
    print(f"\n  log-log slope: {slope:.3f}  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  Interpretation: {'SUPER-LINEAR' if ci[0] > 1 else 'LINEAR' if ci[1] > 1 else 'SUB-LINEAR'}")
    print(f"  R2={model.rsquared:.3f}")

    # Also check contagion scaling if available
    if 'direct_conversions' in pos.columns:
        dc_pos = pos[pos['direct_conversions'] > 0]
        if len(dc_pos) > 10:
            log_dc = np.log(dc_pos['direct_conversions'])
            log_d = np.log(dc_pos['degree'])
            X2 = sm.add_constant(log_d)
            m2 = sm.OLS(log_dc, X2).fit()
            s2 = m2.params.iloc[1]
            ci2 = m2.conf_int().iloc[1]
            print(f"\n  Contagion log-log slope: {s2:.3f}  95% CI: [{ci2[0]:.3f}, {ci2[1]:.3f}]")

    return slope, ci


def analyze_network_properties(G):
    """Degree assortativity and basic network diagnostics."""
    print(f"\n{'='*65}")
    print(f"NETWORK PROPERTIES")
    print(f"{'='*65}")
    r = nx.degree_assortativity_coefficient(G)
    print(f"  Degree assortativity: {r:.4f}")
    print(f"  {'ASSORTATIVE (high-deg connect to high-deg)' if r > 0 else 'DISASSORTATIVE'}")
    degs = [G.degree(n) for n in G.nodes()]
    print(f"  Degree: mean={np.mean(degs):.1f}  std={np.std(degs):.1f}  "
          f"min={min(degs)}  max={max(degs)}")
    print(f"  Clustering: {nx.average_clustering(G):.3f}")
    print(f"  Transitivity: {nx.transitivity(G):.3f}")
    return r


# --- Ensemble analysis (all runs) ---

def run_ensemble(data, n_runs=None):
    """Run correlations across multiple runs. Returns ensemble summary DataFrames."""
    if n_runs is None:
        n_runs = len(data)
    n_runs = min(n_runs, len(data))
    print(f"\n{'='*65}")
    print(f"ENSEMBLE ANALYSIS ({n_runs} runs)")
    print(f"{'='*65}")

    dvs = {
        'adoption': {'results': [], 'stat': 'r_pb'},
        'amplification': {'results': [], 'stat': 'rho_s'},
    }
    snap0 = _resolve_snap(data.iloc[0]['snapshots'])
    has_contagion = 'direct_conversions' in snap0 or 'influence_parents' in snap0
    if has_contagion:
        dvs['contagion'] = {'results': [], 'stat': 'rho_s'}

    for i in range(n_runs):
        row = data.iloc[i]
        df = extract_features_fast(row)

        # Adoption
        meat = df[df['init_meat'] == 1]
        adoption_r = {}
        for p in ALL_PREDS:
            r, pval = pointbiserialr(meat['adopted'], meat[p])
            adoption_r[p] = {'val': r, 'sig': pval < 0.05}
        dvs['adoption']['results'].append(adoption_r)

        # Amplification
        pos = df[df['reduction_kg'] > 0]
        amp_r = {}
        if len(pos) > 10:
            log_m = np.log(pos['multiplier'])
            for p in ALL_PREDS:
                rho, pval = spearmanr(pos[p], log_m)
                amp_r[p] = {'val': rho, 'sig': pval < 0.05}
        dvs['amplification']['results'].append(amp_r)

        # Contagion
        if has_contagion and 'direct_conversions' in df.columns:
            spreaders = df[(df['reduction_kg'] > 0) | (df['direct_conversions'] > 0)]
            con_r = {}
            if len(spreaders) > 10:
                for p in ALL_PREDS:
                    rho, pval = spearmanr(spreaders[p], spreaders['direct_conversions'])
                    con_r[p] = {'val': rho, 'sig': pval < 0.05}
            dvs['contagion']['results'].append(con_r)

    # Build summary DataFrames and print
    ensemble_dfs = {}
    for dv_name, dv in dvs.items():
        stat_name = dv['stat']
        print(f"\n  --- {dv_name.upper()} ({stat_name}) across {n_runs} runs ---")
        print(f"  {'Predictor':<16} {'median':>8} {'IQR_lo':>8} {'IQR_hi':>8} {'frac_sig':>8}")
        rows = []
        for p in ALL_PREDS:
            vals = [r[p]['val'] for r in dv['results'] if p in r]
            sigs = [r[p]['sig'] for r in dv['results'] if p in r]
            if vals:
                med = np.median(vals)
                q25, q75 = np.percentile(vals, [25, 75])
                frac_sig = np.mean(sigs)
                print(f"  {p:<16} {med:>8.3f} {q25:>8.3f} {q75:>8.3f} {frac_sig:>7.0%}")
                rows.append({'predictor': p, stat_name: med,
                             'q25': q25, 'q75': q75, 'frac_sig': frac_sig,
                             'p': 0.001 if frac_sig > 0.5 else 0.1})
        ensemble_dfs[dv_name] = pd.DataFrame(rows)

    return ensemble_dfs


def extract_features_fast(row):
    """Fast feature extraction (no complex centrality) for ensemble runs."""
    snap_final = _resolve_snap(row['snapshots'])
    snap_init = row['snapshots'][0]
    G = snap_final['graph']
    nodes = list(G.nodes())
    N = len(nodes)

    reductions_kg = np.array(snap_final['reductions'])
    init_diets, final_diets = snap_init['diets'], snap_final['diets']

    # Direct conversions
    direct_conv = None
    if 'direct_conversions' in snap_final:
        direct_conv = np.array(snap_final['direct_conversions'])
    elif 'influence_parents' in snap_final:
        parents = snap_final['influence_parents']
        direct_conv = np.zeros(N, dtype=int)
        for p in parents:
            if p is not None:
                direct_conv[p] += 1

    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality_numpy(G)
    clustering = nx.clustering(G)
    cc = compute_complex_centrality(G, T=2)

    df = pd.DataFrame({
        'node': nodes,
        'reduction_kg': reductions_kg,
        'multiplier': reductions_kg / DIRECT_REDUCTION_KG,
        'degree': [G.degree(n) for n in nodes],
        'betweenness': [betweenness[n] for n in nodes],
        'eigenvector': [eigenvector[n] for n in nodes],
        'clustering': [clustering[n] for n in nodes],
        'complex_cent': [cc[n] for n in nodes],
        'theta': [G.nodes[n].get('theta', 0) for n in nodes],
        'alpha': np.array(snap_final['alphas']),
        'rho': np.array(snap_final['rhos']),
        'initial_diet': [init_diets[i] for i in range(N)],
        'final_diet': [final_diets[i] for i in range(N)],
    })
    if direct_conv is not None:
        df['direct_conversions'] = direct_conv
    df['init_meat'] = (df['initial_diet'] == 'meat').astype(int)
    df['adopted'] = ((df['initial_diet'] == 'meat') & (df['final_diet'] == 'veg')).astype(int)
    return df


# --- Figures ---

def make_three_panel_figure(df, adopt_corr, contag_corr, amp_corr,
                            output_dir='../visualisations_output'):
    """Side-by-side coefficient comparison for all three DVs."""
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18*cm, 7*cm), sharey=True)
    titles = ['Adoption (r_pb)', 'Contagion (rho_s)', 'Amplification (rho_s)']
    corr_dfs = [adopt_corr, contag_corr, amp_corr]
    val_cols = ['r_pb', 'rho_s', 'rho_s']

    for ax, title, cdf, vcol, letter in zip(axes, titles, corr_dfs, val_cols,
                                              'ABC'):
        if cdf is None:
            ax.set_visible(False); continue
        preds = cdf['predictor'].values
        vals = cdf[vcol].values
        sigs = cdf['p'].values < 0.05
        colors = [COLORS['primary'] if s else '#aaa' for s in sigs]
        y_pos = np.arange(len(preds))

        ax.barh(y_pos, vals, color=colors, height=0.6, alpha=0.8)
        ax.axvline(0, color='#666', lw=0.8, ls='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(preds, fontsize=7)
        ax.set_xlabel(vcol, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.97, letter, transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top')

    plt.tight_layout()
    out = f'{output_dir}/agency_three_dvs.pdf'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out}")
    return fig


def make_two_panel_figure(adopt_corr, amp_corr, output_dir='../visualisations_output',
                          ensemble=False, n_runs=None):
    """Publication two-panel: adoption vs amplification predictor comparison.

    If ensemble=True, adopt_corr/amp_corr should have 'q25'/'q75' columns
    for IQR error bars, and 'frac_sig' for significance (>50% of runs).
    """
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    SIG_COLOR = '#5e4fa2'
    NS_COLOR = '#bbb'
    PRED_LABELS = {
        'degree': 'Degree', 'betweenness': 'Betweenness', 'eigenvector': 'Eigenvector',
        'clustering': 'Clustering', 'complex_cent': 'Complex cent.',
        'rho': r'Beh. intention ($\rho$)', 'alpha': r'Self-reliance ($\alpha$)',
        'theta': r'Dietary pref. ($\theta$)',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14*cm, 8.5*cm), sharey=True)
    panels = [
        (adopt_corr, 'r_pb', r'$r_{pb}$', 'A'),
        (amp_corr, 'rho_s', r'$\rho_s$', 'B'),
    ]
    for ax, (cdf, vcol, xlabel, letter) in zip(axes, panels):
        preds, vals = cdf['predictor'].values, cdf[vcol].values
        if ensemble and 'frac_sig' in cdf.columns:
            sigs = cdf['frac_sig'].values > 0.5
        else:
            sigs = cdf['p'].values < 0.05
        colors = [SIG_COLOR if s else NS_COLOR for s in sigs]
        y_pos = np.arange(len(preds))
        labels = [PRED_LABELS.get(p, p) for p in preds]

        ax.barh(y_pos, vals, color=colors, height=0.6, alpha=0.85)
        # IQR error bars for ensemble
        if ensemble and 'q25' in cdf.columns:
            xerr_lo = vals - cdf['q25'].values
            xerr_hi = cdf['q75'].values - vals
            ax.errorbar(vals, y_pos, xerr=[xerr_lo, xerr_hi], fmt='none',
                        ecolor='#333', elinewidth=0.7, capsize=2, capthick=0.7)
        ax.axvline(0, color='#555', lw=0.7, ls='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top')

    axes[0].set_title('Adoption', fontsize=9, fontstyle='italic', pad=4)
    axes[1].set_title('Amplification', fontsize=9, fontstyle='italic', pad=4)
    plt.tight_layout()
    out = f'{output_dir}/agency_two_dvs.pdf'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    return fig


def make_degree_scaling_figure(df, output_dir='../visualisations_output'):
    """Degree vs amplification and degree vs contagion scatter + binned means."""
    set_publication_style()
    pos = df[df['reduction_kg'] > 0].copy()
    has_dc = 'direct_conversions' in pos.columns

    n_panels = 2 if has_dc else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels*6*cm, 5.5*cm))
    if n_panels == 1:
        axes = [axes]

    def _scatter_binned(ax, x, y, xlabel, ylabel, letter):
        ax.scatter(x, y, s=4, alpha=0.3, c=COLORS['primary'], edgecolors='none', rasterized=True)
        bins = np.percentile(x, np.linspace(0, 100, 12))
        bc, bm = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (x >= lo) & (x < hi)
            if mask.sum() > 2:
                bc.append(x[mask].mean()); bm.append(y[mask].mean())
        if bc:
            ax.plot(bc, bm, 'o-', color=COLORS['secondary'], lw=1.5, ms=4, zorder=5)
        ax.set_xlabel(xlabel, fontsize=7); ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.97, letter, transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top')

    _scatter_binned(axes[0], pos['degree'], pos['reduction_kg'],
                    'Degree', 'Amplification (kg)', 'A')
    if has_dc:
        dc_pos = pos[pos['direct_conversions'] > 0]
        _scatter_binned(axes[1], dc_pos['degree'], dc_pos['direct_conversions'],
                        'Degree', 'Direct conversions', 'B')

    plt.tight_layout()
    out = f'{output_dir}/degree_scaling.pdf'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    return fig


# --- Main ---

if __name__ == '__main__':
    import pickle, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl', nargs='?', help='path to trajectory pkl')
    parser.add_argument('--plot-only', action='store_true',
                        help='skip analysis; load cached results and regenerate figure')
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(CACHE_FILE):
            print(f"ERROR: cache file not found: {CACHE_FILE}")
            print("Run without --plot-only first to generate the cache.")
            sys.exit(1)
        with open(CACHE_FILE, 'rb') as fh:
            cache = pickle.load(fh)
        print(f"Loaded cache: {CACHE_FILE}  (n_runs={cache['n_runs']})")
        make_two_panel_figure(cache['adoption'], cache['amplification'],
                              ensemble=True, n_runs=cache['n_runs'])
        plt.show()
        sys.exit(0)

    data = load_latest(args.pkl)
    N_agents = len(_resolve_snap(data.iloc[0]['snapshots'])['reductions'])
    print(f"Loaded {len(data)} runs, N={N_agents} agents")

    # Detailed analysis on median run (for degree scaling, network props, three-panel)
    median_row = get_median_row(data)
    df = extract_features(median_row)

    # Three DVs (single-run, for detailed printout)
    adopt_corr = analyze_adoption(df)
    contag_corr = analyze_contagion(df)
    amp_corr, pos, amp_model = analyze_amplification(df)

    # Degree scaling
    slope, ci = analyze_degree_scaling(df)

    # Network properties
    G = _resolve_snap(median_row['snapshots'])['graph']
    r_assort = analyze_network_properties(G)

    # Ensemble analysis — used for the two-panel figure
    n_ens = min(len(data), 50)
    print(f"\nRunning ensemble analysis ({n_ens} runs, this may take a while)...")
    ensemble_dfs = run_ensemble(data, n_runs=n_ens)

    # Save cache so --plot-only can regenerate the figure without re-running analysis
    cache = {'adoption': ensemble_dfs['adoption'], 'amplification': ensemble_dfs['amplification'],
             'n_runs': n_ens}
    with open(CACHE_FILE, 'wb') as fh:
        pickle.dump(cache, fh)
    print(f"Cache saved: {CACHE_FILE}")

    # Figures
    make_three_panel_figure(df, adopt_corr, contag_corr, amp_corr)
    make_two_panel_figure(ensemble_dfs['adoption'], ensemble_dfs['amplification'],
                          ensemble=True, n_runs=n_ens)
    make_degree_scaling_figure(df)

    plt.show()
