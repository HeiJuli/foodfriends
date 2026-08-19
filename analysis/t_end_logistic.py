#!/usr/bin/env python3
"""Logistic-fit t_end estimation for adoption trajectories.

Fits F_veg(t) = K / (1 + exp(-r*(t - t0))) + b and returns the time at which
a given fraction (default 95%) of the fitted asymptote K is reached.

Usage:
    from t_end_logistic import estimate_t_end, estimate_t_end_ensemble

    t = estimate_t_end(fraction_veg_trajectory)           # single run
    t, ci = estimate_t_end_ensemble(dataframe, pct=0.95)  # ensemble median + IQR
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def _logistic(t, L, k, t0, b):
    return L / (1 + np.exp(-k * (t - t0))) + b


def estimate_t_end(traj, pct=0.95, smooth_window=5001):
    """Fit logistic to trajectory, return t at pct of asymptote.

    Parameters
    ----------
    traj : array-like
        F_veg trajectory (one value per timestep).
    pct : float
        Fraction of fitted asymptote (0.90, 0.95, 0.99).
    smooth_window : int
        Savitzky-Golay window for pre-smoothing (odd, >= 3).

    Returns
    -------
    int or None
        Estimated t_end (None if fit fails).
    """
    traj = np.asarray(traj, dtype=float)
    n = len(traj)
    if n < 1000:
        return None
    win = min(smooth_window, n // 2 * 2 - 1)
    if win < 5:
        return None
    smooth = savgol_filter(traj, win, 3)
    tt = np.arange(n)
    p0 = [traj[-1] - traj[0], 1e-4, n * 0.1, traj[0]]
    bounds = ([0, 0, 0, 0], [1, 1e-2, n * 2, 0.5])
    try:
        popt, _ = curve_fit(_logistic, tt, smooth, p0=p0, bounds=bounds, maxfev=50000)
        L, k, t0, b = popt
        # t at pct of asymptotic change: F(t) = b + pct*L
        t_end = t0 - np.log((1 - pct) / pct) / k
        return max(0, int(round(t_end)))
    except (RuntimeError, ValueError):
        return None


def estimate_t_end_ensemble(df, pct=0.95, traj_key='fraction_veg_trajectory'):
    """Fit logistic to every run, return median t_end and IQR.

    Parameters
    ----------
    df : DataFrame
        Ensemble output with trajectory column.
    pct : float
        Fraction of fitted asymptote.
    traj_key : str
        Column name for trajectory data.

    Returns
    -------
    (median, iqr_low, iqr_high) or (None, None, None)
    """
    vals = []
    for _, row in df.iterrows():
        traj = row.get(traj_key, row.get('fraction_veg', []))
        t = estimate_t_end(traj, pct=pct)
        if t is not None:
            vals.append(t)
    if not vals:
        return None, None, None
    arr = np.array(vals)
    return int(np.median(arr)), int(np.percentile(arr, 25)), int(np.percentile(arr, 75))


if __name__ == '__main__':
    import sys, pandas as pd
    path = sys.argv[1] if len(sys.argv) > 1 else '../model_output/trajectory_analysis_twin_20260402.pkl'
    pct = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
    df = pd.read_pickle(path)
    print(f"Loaded {len(df)} runs from {path}")

    # Ensemble
    med, q25, q75 = estimate_t_end_ensemble(df, pct=pct)
    print(f"Ensemble t_end ({pct*100:.0f}% asymptote): median={med}, IQR=[{q25}, {q75}]")

    # Median run
    final_vals = [np.array(row.get('fraction_veg_trajectory',
                                    row.get('fraction_veg', [])))[-1]
                  for _, row in df.iterrows()]
    median_idx = np.argsort(final_vals)[len(final_vals) // 2]
    traj = np.array(df.iloc[median_idx]['fraction_veg_trajectory'], dtype=float)
    t_single = estimate_t_end(traj, pct=pct)
    print(f"Median run t_end: {t_single}")

    # Empirical check
    final_mean = np.mean(traj[-5000:])
    change = final_mean - traj[0]
    if t_single and t_single < len(traj):
        achieved = (traj[t_single] - traj[0]) / change
        print(f"Empirical fraction of total change at t_end: {achieved*100:.1f}%")
