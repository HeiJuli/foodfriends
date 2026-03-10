#!/usr/bin/env python3
"""Analytical results for the results section.

1. Power-law / heavy-tail fit on CCDF of emission reductions
2. Gini coefficient + concentration ratios from Lorenz curve
3. Network assortativity and clustering by diet group (t0 vs t_end)
4. Scaling exponent: amplification ~ k^gamma
5. Inflection point / critical mass from trajectory

Primary data: twin (N=2000, n=50 runs)
Robustness:   sample-max (N=385, n=10 runs)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

TWIN = '../model_output/trajectory_analysis_twin_20260306.pkl'
SMAX = '../model_output/trajectory_analysis_sample-max_20260306.pkl'

def _apply_diets(G, diets):
    """Return graph copy with node diet attributes updated from diets list."""
    G2 = G.copy()
    for i, n in enumerate(G2.nodes()):
        G2.nodes[n]['diet'] = diets[i]
    return G2
DIRECT_REDUCTION_KG = 664  # 2054 - 1390

def _resolve_snapshot(snapshots, t_cutoff=None):
    """Return the snapshot key to use as 'final'.
    Priority: explicit t_cutoff > auto-detected 'steady' > true 'final'."""
    if t_cutoff is not None:
        int_times = sorted(t for t in snapshots if isinstance(t, int) and t > 0)
        return min(int_times, key=lambda t: abs(t - t_cutoff)) if int_times else 'final'
    return 'steady' if 'steady' in snapshots else 'final'

def load(path):
    d = pd.read_pickle(path)
    median = d[d['is_median_twin']].iloc[0] if d['is_median_twin'].any() else d.iloc[0]
    return d, median

def analysis_1_powerlaw(median_row, label='twin', t_cutoff=None):
    """Fit heavy-tail distribution to emission reductions CCDF."""
    snap_key = _resolve_snapshot(median_row['snapshots'], t_cutoff)
    print(f"\n{'='*60}")
    print(f"  1. HEAVY-TAIL CHARACTERIZATION  [{label}, N={len(median_row['snapshots'][snap_key]['reductions'])}, t={snap_key}]")
    print(f"{'='*60}")

    reds = np.array(median_row['snapshots'][snap_key]['reductions'])
    pos = reds[reds > 0] / 1000  # tonnes

    try:
        import powerlaw
        # Use explicit xmin (median) to avoid numpy compat bug in auto xmin search
        xmin_val = float(np.median(pos))
        fit = powerlaw.Fit(pos, xmin=xmin_val)
        print(f"  Power-law fit:  alpha = {fit.alpha:.2f},  xmin = {fit.xmin:.3f} t CO2e")

        # Compare distributions
        for alt in ['lognormal', 'exponential', 'truncated_power_law']:
            R, p = fit.distribution_compare('power_law', alt)
            winner = 'power_law' if R > 0 else alt
            sig = '*' if p < 0.05 else ''
            print(f"  vs {alt:25s}: R = {R:+.3f}, p = {p:.3f} {sig}  -> {winner}")

    except (ImportError, Exception) as e:
        print(f"  powerlaw package issue ({e}), using manual fit")

    # Always do manual CCDF slope as well (transparent, reproducible)
    sorted_x = np.sort(pos)
    ccdf_y = 1.0 - np.arange(1, len(sorted_x) + 1) / len(sorted_x)
    mask = ccdf_y > 0
    lx, ly = np.log10(sorted_x[mask]), np.log10(ccdf_y[mask])
    # Fit on upper tail (above median)
    tail = lx > np.median(lx)
    coeffs = np.polyfit(lx[tail], ly[tail], 1)
    alpha_ccdf = -coeffs[0]
    print(f"  CCDF tail slope (OLS, above median): alpha_ccdf ~ {alpha_ccdf:.2f}")

    # Summary stats
    print(f"\n  Summary: n={len(pos)} agents with positive reductions")
    print(f"  Mean = {np.mean(pos):.2f} t, Median = {np.median(pos):.2f} t, Max = {np.max(pos):.1f} t")
    print(f"  Skewness = {pd.Series(pos).skew():.2f}")


def analysis_2_gini(median_row, all_data, label='twin', t_cutoff=None):
    """Gini coefficient and concentration ratios across snapshots."""
    print(f"\n{'='*60}")
    print(f"  2. GINI COEFFICIENT & CONCENTRATION  [{label}]")
    print(f"{'='*60}")

    snapshots = median_row['snapshots']
    snap_key = _resolve_snapshot(snapshots, t_cutoff)
    int_times = sorted(t for t in snapshots if isinstance(t, int) and t > 0)
    if t_cutoff is not None:
        int_times = [t for t in int_times if t <= t_cutoff]
    times = int_times + [snap_key]

    for t in times:
        reds = np.array(snapshots[t]['reductions'])
        pos = reds[reds > 0]
        if len(pos) == 0:
            continue

        # Gini: mean absolute difference / (2 * mean)
        n = len(pos)
        sorted_r = np.sort(pos)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_r) / (n * np.sum(sorted_r))) - (n + 1) / n

        # Concentration ratios
        sorted_desc = np.sort(reds[reds > 0])[::-1]
        total = sorted_desc.sum()
        top10_frac = sorted_desc[:max(1, int(0.1 * len(sorted_desc)))].sum() / total
        top1_frac = sorted_desc[:max(1, int(0.01 * len(sorted_desc)))].sum() / total
        median_kg = np.median(pos)

        # Lorenz curve: top X% share = 1 - L(1-X) where sorted_r is ascending
        top10_lorenz = sorted_r[int(0.90 * n):].sum() / sorted_r.sum()
        top20_lorenz = sorted_r[int(0.80 * n):].sum() / sorted_r.sum()

        t_label = f't_end' if t == 'final' else f't={t//1000}k'
        print(f"\n  {t_label}:  Gini = {gini:.3f}")
        print(f"    Top  1% accounts for {top1_frac*100:.1f}% of total reductions")
        print(f"    Top 10% accounts for {top10_frac*100:.1f}% of total reductions")
        print(f"    Median agent: {median_kg:.0f} kg CO2")
        print(f"    Lorenz (f_pop=0.90): top 10% -> {top10_lorenz*100:.2f}% of total reductions")
        print(f"    Lorenz (f_pop=0.80): top 20% -> {top20_lorenz*100:.2f}% of total reductions")

    # Ensemble Gini (resolved snapshot across all runs)
    ginis = []
    for _, row in all_data.iterrows():
        sk = _resolve_snapshot(row['snapshots'], t_cutoff)
        reds = np.array(row['snapshots'][sk]['reductions'])
        pos = reds[reds > 0]
        if len(pos) < 2:
            continue
        n = len(pos)
        sorted_r = np.sort(pos)
        g = (2 * np.sum(np.arange(1, n+1) * sorted_r) / (n * np.sum(sorted_r))) - (n + 1) / n
        ginis.append(g)

    ginis = np.array(ginis)
    print(f"\n  Ensemble (n={len(ginis)} runs):")
    print(f"    Gini median = {np.median(ginis):.3f}, IQR = [{np.percentile(ginis,25):.3f}, {np.percentile(ginis,75):.3f}]")


def analysis_3_network_assortativity(median_row, label='twin', t_cutoff=None):
    """Assortativity and clustering by diet group at t0 vs t_end."""
    print(f"\n{'='*60}")
    print(f"  3. NETWORK ASSORTATIVITY & CLUSTERING  [{label}]")
    print(f"{'='*60}")

    snaps = median_row['snapshots']
    snap_key = _resolve_snapshot(snaps, t_cutoff)

    for t_key, tlabel in [(0, 't0'), (snap_key, 't_end')]:
        snap = snaps[t_key]
        G = _apply_diets(snap['graph'], snap['diets'])
        diets = snap['diets']
        n_veg = sum(1 for x in diets if x == 'veg')
        n_meat = len(diets) - n_veg
        fveg = snap['veg_fraction']

        r = nx.attribute_assortativity_coefficient(G, 'diet')

        rows = []
        for group in ('meat', 'veg'):
            nodes = [n for n, a in G.nodes(data=True) if a.get('diet') == group]
            if not nodes:
                rows.append((group, None, None))
                continue
            homophily = np.mean([
                sum(1 for nb in G.neighbors(n) if G.nodes[nb].get('diet') == group) / max(1, G.degree(n))
                for n in nodes
            ])
            clust = np.mean([nx.clustering(G, n) for n in nodes])
            rows.append((group, homophily, clust))

        print(f"\n  {tlabel}  (F_veg={fveg:.3f}, N_veg={n_veg}, N_meat={n_meat})")
        print(f"    Overall assortativity r = {r:.4f}")
        for group, h, c in rows:
            if h is not None:
                print(f"    {group:4s}: neighbor homophily = {h:.4f},  clustering = {c:.4f}")


def analysis_4_degree_scaling(median_row, label='twin', t_cutoff=None):
    """Fit amplification ~ k^gamma scaling."""
    snap_key = _resolve_snapshot(median_row['snapshots'], t_cutoff)
    print(f"\n{'='*60}")
    print(f"  4. DEGREE-AMPLIFICATION SCALING  [{label}, t={snap_key}]")
    print(f"{'='*60}")

    snap = median_row['snapshots'][snap_key]
    G = snap['graph']
    nodes = list(G.nodes())
    reds = np.array(snap['reductions'])
    degrees = np.array([G.degree(n) for n in nodes])
    multipliers = reds / DIRECT_REDUCTION_KG

    # Positive reductions only
    mask = reds > 0
    k, A = degrees[mask].astype(float), multipliers[mask]

    # Filter k > 0 for log
    valid = k > 0
    k, A = k[valid], A[valid]

    # OLS in log-log
    lk, lA = np.log10(k), np.log10(A)
    coeffs = np.polyfit(lk, lA, 1)
    gamma, intercept = coeffs[0], coeffs[1]
    A_pred = 10**(gamma * lk + intercept)
    ss_res = np.sum((lA - (gamma * lk + intercept))**2)
    ss_tot = np.sum((lA - np.mean(lA))**2)
    r2 = 1 - ss_res / ss_tot

    rho_s, p_s = spearmanr(k, A)

    print(f"  A ~ k^gamma")
    print(f"  gamma = {gamma:.2f}  (R^2 = {r2:.3f})")
    print(f"  Spearman rho_s = {rho_s:.3f}  (p = {p_s:.1e})")
    if gamma > 1:
        print(f"  -> SUPERLINEAR: doubling degree more than doubles amplification")
    elif gamma > 0:
        print(f"  -> SUBLINEAR: diminishing returns with degree")

    # Binned medians for robustness
    print(f"\n  Binned medians (degree percentile bins):")
    for lo, hi in [(0,25), (25,50), (50,75), (75,100)]:
        plo, phi = np.percentile(k, lo), np.percentile(k, hi)
        bm = (k >= plo) & (k < phi) if hi < 100 else (k >= plo) & (k <= phi)
        if bm.sum() > 0:
            print(f"    k in [{plo:.0f}, {phi:.0f}]: median A = {np.median(A[bm]):.1f}x  (n={bm.sum()})")

    return gamma, r2


def analysis_5_inflection(all_data, label='twin'):
    """Find inflection point (max dF/dt) across ensemble."""
    print(f"\n{'='*60}")
    print(f"  5. INFLECTION POINT / CRITICAL MASS  [{label}]")
    print(f"{'='*60}")

    inflection_fveg = []
    inflection_times = []
    window = 2000  # smoothing window
    burnin = 5000  # skip initial equilibration jump

    for _, row in all_data.iterrows():
        traj = np.array(row['fraction_veg_trajectory'], dtype=float)
        if len(traj) < burnin + window * 2:
            continue

        # Smooth, then 2nd derivative (max acceleration of adoption)
        smoothed = uniform_filter1d(traj, size=window)
        d2Fdt2 = np.gradient(np.gradient(smoothed))
        d2Fdt2[:burnin] = 0  # mask burn-in
        # Mask post-50% regime (late-stage dynamics confound)
        d2Fdt2[smoothed > 0.5] = 0

        idx_max = np.argmax(d2Fdt2)
        inflection_fveg.append(smoothed[idx_max])
        inflection_times.append(idx_max)

    fv = np.array(inflection_fveg)
    tv = np.array(inflection_times) / 1000  # to thousands

    print(f"  Across n={len(fv)} runs:")
    print(f"  F_veg at max d2F/dt2:  median = {np.median(fv):.3f}  ({np.median(fv)*100:.1f}%)")
    print(f"                       IQR = [{np.percentile(fv,25):.3f}, {np.percentile(fv,75):.3f}]")
    print(f"  Time of inflection:  median = {np.median(tv):.1f}k steps")
    print(f"                       IQR = [{np.percentile(tv,25):.1f}k, {np.percentile(tv,75):.1f}k]")

    # Also report "tipping thresholds" at 25% and 50%
    for threshold in [0.25, 0.50]:
        crossing_times = []
        for _, row in all_data.iterrows():
            traj = row['fraction_veg_trajectory']
            crossed = [i for i, v in enumerate(traj) if v >= threshold]
            if crossed:
                crossing_times.append(crossed[0] / 1000)
        if crossing_times:
            ct = np.array(crossing_times)
            print(f"\n  Time to F_veg >= {threshold:.0%}:  median = {np.median(ct):.1f}k  "
                  f"IQR = [{np.percentile(ct,25):.1f}k, {np.percentile(ct,75):.1f}k]  "
                  f"({len(ct)}/{len(all_data)} runs reached)")


def main():
    import sys
    twin_cutoff = int(sys.argv[1]) if len(sys.argv) > 1 else None
    smax_cutoff = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if twin_cutoff:
        print(f"twin t_cutoff={twin_cutoff}, sample-max t_cutoff={smax_cutoff}")

    print("Loading data...")
    twin_all, twin_med = load(TWIN)
    smax_all, smax_med = load(SMAX)

    # --- Run all analyses on twin (primary) ---
    analysis_1_powerlaw(twin_med, 'twin', twin_cutoff)
    analysis_2_gini(twin_med, twin_all, 'twin', twin_cutoff)
    analysis_3_network_assortativity(twin_med, 'twin', twin_cutoff)
    analysis_4_degree_scaling(twin_med, 'twin', twin_cutoff)
    analysis_5_inflection(twin_all, 'twin')

    # --- Robustness: sample-max ---
    print(f"\n\n{'#'*60}")
    print(f"  ROBUSTNESS: sample-max (N=385, n=10)")
    print(f"{'#'*60}")
    analysis_1_powerlaw(smax_med, 'sample-max', smax_cutoff)
    analysis_2_gini(smax_med, smax_all, 'sample-max', smax_cutoff)
    analysis_3_network_assortativity(smax_med, 'sample-max', smax_cutoff)
    analysis_4_degree_scaling(smax_med, 'sample-max', smax_cutoff)
    analysis_5_inflection(smax_all, 'sample-max')


if __name__ == '__main__':
    main()
