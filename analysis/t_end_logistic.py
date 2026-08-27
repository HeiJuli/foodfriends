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


def _ic(y, yhat, n_par):
    """(AIC, BIC, R2) for a Gaussian-error fit."""
    n = len(y)
    rss = np.sum((y - yhat) ** 2)
    ll = -n / 2 * (np.log(2 * np.pi) + np.log(rss / n) + 1)
    return 2 * n_par - 2 * ll, n_par * np.log(n) - 2 * ll, 1 - rss / np.sum((y - y.mean()) ** 2)


def compare_logistic_linear(df, traj_key='fraction_veg_trajectory',
                            smooth_window=5001, burnin=5000, lo_hi=(0.1, 0.9)):
    """Logistic vs linear over the growth phase, per run (analysis A3).

    Growth phase = the span between lo_hi fractions of the total change, so the
    saturating tail cannot hand the logistic an automatic win. Returns a DataFrame
    with per-run R2, AIC, BIC and the fitted r, t0 with standard errors.
    """
    import pandas as pd
    from scipy.stats import linregress
    stride = 10   # decimation; the fits are unaffected and savgol gets ~100x cheaper
    rows = []
    for _, row in df.iterrows():
        full = np.asarray(row.get(traj_key, row.get('fraction_veg', [])), dtype=float)
        if len(full) < burnin + smooth_window * 2:
            continue
        traj = full[::stride]
        tix = np.arange(len(traj)) * stride
        sm = savgol_filter(traj, max(5, int(smooth_window // stride) | 1), 3)
        lo, hi = sm[burnin // stride], np.mean(sm[-5000 // stride:])
        f_lo, f_hi = lo + lo_hi[0] * (hi - lo), lo + lo_hi[1] * (hi - lo)
        idx = np.where((sm >= f_lo) & (sm <= f_hi) & (tix >= burnin))[0]
        if len(idx) < 100:
            continue
        t, y = tix[idx].astype(float), sm[idx]
        lr = linregress(t, y)
        aic_l, bic_l, r2_l = _ic(y, lr.slope * t + lr.intercept, 2)
        try:
            popt, pcov = curve_fit(_logistic, t, y,
                                   p0=[hi - lo, 1e-4, t.mean(), lo],
                                   bounds=([0, 0, 0, 0], [1, 1e-2, len(full) * 2, 0.5]),
                                   maxfev=60000)
        except (RuntimeError, ValueError):
            continue
        aic_g, bic_g, r2_g = _ic(y, _logistic(t, *popt), 4)
        se = np.sqrt(np.diag(pcov))
        rows.append(dict(r2_lin=r2_l, r2_log=r2_g, d_aic=aic_g - aic_l, d_bic=bic_g - bic_l,
                         r=popt[1], r_se=se[1], t0=popt[2], t0_se=se[2],
                         span=len(idx), f_lo=y[0], f_hi=y[-1]))
    return pd.DataFrame(rows)


def report_logistic_linear(d):
    """Print the A3 comparison. Negative dAIC/dBIC favours the logistic."""
    q = lambda c: (f"median = {d[c].median():.4g}  "
                   f"IQR = [{d[c].quantile(.25):.4g}, {d[c].quantile(.75):.4g}]")
    print(f"\n  Logistic vs linear over the growth phase (n={len(d)} runs)")
    print(f"    growth window: F_veg {d.f_lo.median():.3f} -> {d.f_hi.median():.3f}, "
          f"{d.span.median()/1000:.0f}k steps")
    print(f"    R2 linear    : {q('r2_lin')}")
    print(f"    R2 logistic  : {q('r2_log')}")
    print(f"    R2 gain      : median = {(d.r2_log - d.r2_lin).median():.4f}")
    print(f"    dAIC (log-lin): {q('d_aic')}")
    print(f"    dBIC (log-lin): {q('d_bic')}")
    print(f"    logistic preferred: AIC {(d.d_aic < 0).sum()}/{len(d)}, "
          f"BIC {(d.d_bic < 0).sum()}/{len(d)}")
    print(f"    fitted r     : {q('r')}  (median SE {d.r_se.median():.2g})")
    print(f"    fitted t0    : {q('t0')}  (median SE {d.t0_se.median():.2g})")


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

    # A3: is the sigmoid actually earning its two extra parameters?
    report_logistic_linear(compare_logistic_linear(df))

    # Empirical check
    final_mean = np.mean(traj[-5000:])
    change = final_mean - traj[0]
    if t_single and t_single < len(traj):
        achieved = (traj[t_single] - traj[0]) / change
        print(f"Empirical fraction of total change at t_end: {achieved*100:.1f}%")
