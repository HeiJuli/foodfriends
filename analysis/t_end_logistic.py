#!/usr/bin/env python3
"""Logistic-fit t_end estimation for adoption trajectories.

Fits F_veg(t) = K / (1 + exp(-r*(t - t0))) + b and returns the time at which
a given fraction (default 95%) of the fitted asymptote K is reached.

Also the one home for the analysis-window conventions, so the campaigns cannot
drift apart on them (see fc_window below).

Usage:
    from t_end_logistic import estimate_t_end, estimate_t_end_ensemble, fc_window

    t = estimate_t_end(fraction_veg_trajectory)           # single run
    t, ci = estimate_t_end_ensemble(dataframe, pct=0.95)  # ensemble median + IQR
    w = fc_window(len(traj))                              # savgol window for F_c
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


# Savitzky-Golay window for derivative-based quantities (F_c, inflection), as a
# FRACTION of run length rather than an absolute constant. 10001 was chosen
# against a 139k trajectory; at 400k it is a different fraction of the transient,
# and kappa=0.55 moves t_end by 3.4x, so an absolute window is not comparable
# across configurations. 20% is where F_c's per-seed IQR is tightest on the
# kappa=0.55 ensemble (0.086 against 0.121 at 10001, 0.107-0.143 at 2-10%) and
# where its median is window-stable (0.352 against 0.352-0.382 over 2-20%).
# It also makes N=385 (100k steps) and N=2000 (400k) comparable by construction,
# which the old absolute window could not do: at win=10001 a third of the short
# run was masked and at 15001 the estimator failed outright.
FC_WIN_FRAC = 0.20


def fc_window(n, frac=FC_WIN_FRAC):
    """Odd Savitzky-Golay window for an n-step run, as a fraction of the run.

    The caller must also mask the first `max(burnin, win)` samples: a kernel of
    width w straddles the initial equilibration jump for its first w/2 steps.
    """
    return max(5, int(frac * n) | 1)


def _logistic(t, L, k, t0, b):
    return L / (1 + np.exp(-k * (t - t0))) + b


def _p0(smooth, tt):
    """Initial guess read off the smoothed trajectory instead of from constants.

    The least-squares surface is multi-modal and curve_fit returns whichever
    optimum lies nearest the start, so the guess decides the answer. The old
    constants (t0 = 0.1n, k = 1e-4) were never right -- measured t0/n is 0.39
    pre-kappa and 0.47 at kappa = 0.55 -- but what breaks the fit is the
    absolute distance in steps, not the ratio. At 30k steps the guess is 8.8k
    short and every run still converges (checked: identical t_end on the
    20260820 sample-max ensemble, so this change is backwards-compatible).
    At 400k it is 147k short, and 8 of the 50 headline runs converge instead on
    a spurious low-k mode (R2 0.87-0.93 against 0.995, t_end 10-40k short) --
    silently, because the fit does converge. Longer runs stop converging at all;
    that is what this module reports as "fit_failed". Reading L, b, t0 and k off
    the data removes both, and scales with run length by construction.

    k comes from the 25-75% crossing span, which a logistic covers in 2 ln 3 / k.
    """
    b = smooth[0]
    L = smooth[-1] - b
    if L <= 0:
        return None

    def cross(f):
        i = np.where(smooth >= b + f * L)[0]
        return tt[i[0]] if len(i) else tt[-1]

    span = max(cross(0.75) - cross(0.25), tt[-1] * 1e-3)
    return [L, 2 * np.log(3) / span, cross(0.5), max(b, 0.0)]


def _fit(traj, smooth_window):
    """(popt, R2) for the logistic fit; (None, nan) if there is no usable fit.

    R2 comes back because a converged fit is not necessarily a good one -- see
    _p0 -- and every caller used to have no way of telling the two apart.
    """
    traj = np.asarray(traj, dtype=float)
    n = len(traj)
    win = min(smooth_window, n // 2 * 2 - 1)
    if n < 1000 or win < 5:
        return None, np.nan
    smooth = savgol_filter(traj, win, 3)
    tt = np.arange(n, dtype=float)
    p0 = _p0(smooth, tt)
    if p0 is None:
        return None, np.nan
    lo, hi = [0, 0, 0, 0], [1, 1e-2, n * 2, 0.5]
    p0 = [min(max(v, a), b) for v, a, b in zip(p0, lo, hi)]
    try:
        popt, _ = curve_fit(_logistic, tt, smooth, p0=p0, bounds=(lo, hi),
                            maxfev=50000)
    except (RuntimeError, ValueError):
        return None, np.nan
    resid = smooth - _logistic(tt, *popt)
    ss_tot = np.sum((smooth - smooth.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan
    # A sigmoid that fits worse than the trajectory's own mean is not a fit. This
    # is the degenerate case, not a quality threshold: a trajectory that never
    # moves leaves L at the savgol noise floor and curve_fit converges happily on
    # it (R2 ~ -5000 on a constant). No number is tuned here.
    if not np.isfinite(r2) or r2 <= 0:
        return None, np.nan
    return popt, r2


def fit_params(traj, smooth_window=5001):
    """Fitted logistic parameters, percentile times and R2, or None.

    The scripts that need L, k, t0 and b rather than just t_end each had their
    own copy of this fit -- kappa_ensemble_measures, fc_viability_kappa,
    trajectory_t_end_facet -- all three seeded from the constants _p0 replaces.
    One home, one guess.
    """
    popt, r2 = _fit(traj, smooth_window)
    if popt is None:
        return None
    L, k, t0, b = popt
    t_at = lambda pct: max(0.0, t0 - np.log((1 - pct) / pct) / k)
    return dict(L=float(L), k=float(k), t0=float(t0), b=float(b),
                asymptote=float(b + L), r2=float(r2), t_50=t_at(0.50),
                t_90=t_at(0.90), t_end=t_at(0.95))


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
        Estimated t_end (None if fit fails). May exceed len(traj); use
        t_end_with_status when that distinction matters.
    """
    popt, _ = _fit(traj, smooth_window)
    if popt is None:
        return None
    L, k, t0, b = popt
    # t at pct of asymptotic change: F(t) = b + pct*L
    return max(0, int(round(t0 - np.log((1 - pct) / pct) / k)))


def t_end_with_status(traj, pct=0.95, smooth_window=5001):
    """(t_end, status, r2) with status in {"ok", "beyond_run", "fit_failed"}.

    `estimate_t_end` returns None when curve_fit does not converge, which is NOT
    the same thing as a run that has not saturated -- but every caller used to
    collapse both into "t_end == len-1", so a fit failure was silently reported as
    censoring. Measured 2026-09-07: the fit fails on 60-70% of system-size scaling
    runs and ~20% of sensitivity rows, in both cases on runs whose F_veg says they
    had saturated. Only "beyond_run" is evidence of a short run.

    R2 rides along because "ok" is not the same as "good": before the 2026-09-07
    p0 fix, 8 of the 50 headline runs converged on a spurious optimum at R2 0.87
    and reported it as ok. Record it; a healthy kappa = 0.55 run sits above 0.99.
    No threshold is imposed here -- that is a reporting decision, not a fit one.
    """
    n = len(traj)
    popt, r2 = _fit(traj, smooth_window)
    if popt is None:
        return n - 1, "fit_failed", np.nan
    L, k, t0, b = popt
    t = max(0, int(round(t0 - np.log((1 - pct) / pct) / k)))
    if t >= n:
        return n - 1, "beyond_run", r2
    return t, "ok", r2


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
