#!/usr/bin/env python3
"""Analytical results for the results section.

1. Power-law / heavy-tail fit on CCDF of emission reductions
2. Gini coefficient + concentration ratios from Lorenz curve
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
DIRECT_REDUCTION_KG = 664  # 2054 - 1390

def load(path):
    d = pd.read_pickle(path)
    median = d[d['is_median_twin']].iloc[0] if d['is_median_twin'].any() else d.iloc[0]
    return d, median

def analysis_1_powerlaw(median_row, label='twin'):
    """Fit heavy-tail distribution to emission reductions CCDF."""
    print(f"\n{'='*60}")
    print(f"  1. HEAVY-TAIL CHARACTERIZATION  [{label}, N={len(median_row['snapshots']['final']['reductions'])}]")
    print(f"{'='*60}")

    reds = np.array(median_row['snapshots']['final']['reductions'])
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


def analysis_2_gini(median_row, all_data, label='twin'):
    """Gini coefficient and concentration ratios across snapshots."""
    print(f"\n{'='*60}")
    print(f"  2. GINI COEFFICIENT & CONCENTRATION  [{label}]")
    print(f"{'='*60}")

    snapshots = median_row['snapshots']
    times = sorted([t for t in snapshots if isinstance(t, int) and t > 0]) + ['final']

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

        t_label = f't_end' if t == 'final' else f't={t//1000}k'
        print(f"\n  {t_label}:  Gini = {gini:.3f}")
        print(f"    Top  1% accounts for {top1_frac*100:.1f}% of total reductions")
        print(f"    Top 10% accounts for {top10_frac*100:.1f}% of total reductions")
        print(f"    Median agent: {median_kg:.0f} kg CO2")

    # Ensemble Gini (final snapshot across all runs)
    ginis = []
    for _, row in all_data.iterrows():
        reds = np.array(row['snapshots']['final']['reductions'])
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


def analysis_4_degree_scaling(median_row, label='twin'):
    """Fit amplification ~ k^gamma scaling."""
    print(f"\n{'='*60}")
    print(f"  4. DEGREE-AMPLIFICATION SCALING  [{label}]")
    print(f"{'='*60}")

    snap = median_row['snapshots']['final']
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
    print("Loading data...")
    twin_all, twin_med = load(TWIN)
    smax_all, smax_med = load(SMAX)

    # --- Run all analyses on twin (primary) ---
    analysis_1_powerlaw(twin_med, 'twin')
    analysis_2_gini(twin_med, twin_all, 'twin')
    analysis_4_degree_scaling(twin_med, 'twin')
    analysis_5_inflection(twin_all, 'twin')

    # --- Robustness: sample-max ---
    print(f"\n\n{'#'*60}")
    print(f"  ROBUSTNESS: sample-max (N=385, n=10)")
    print(f"{'#'*60}")
    analysis_1_powerlaw(smax_med, 'sample-max')
    analysis_2_gini(smax_med, smax_all, 'sample-max')
    analysis_4_degree_scaling(smax_med, 'sample-max')
    analysis_5_inflection(smax_all, 'sample-max')


if __name__ == '__main__':
    main()
