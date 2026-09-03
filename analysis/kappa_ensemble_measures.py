#!/usr/bin/env python3
"""Section-5 re-measurements on the kappa=0.55 headline ensemble.

Runs off the per-run artefacts written by reduce_ensemble.py (the raw 1.77 GB
pickle does not fit in this machine's memory). Covers the plan's section 5
items other than F_c and its window sensitivity, which are measured separately
by fc_viability_kappa.py:

  trajectory   t_50 / t_90 / t_end (logistic 95% of asymptotic change), fitted
               asymptote, endpoint, and the burn-in jump F_veg(1000) - F_veg(0)
               -- the artefact the whole change exists to remove (0.120 at
               kappa=1)
  mediation    share of conversions with h_soc = 0, i.e. no veg entry in the
               M-entry memory buffer the switch was drawn against. Whole run
               and first 1000 steps (5.8% / 39.1% at kappa=1)
  churn        conversions, reversions, net adopters, switches per converter --
               the factor that must be quoted alongside any amplification
               number

Usage:
    python kappa_ensemble_measures.py <reduced_dir> [-o out.csv]
"""
import os, sys, glob, pickle, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t_end_logistic import estimate_t_end, _logistic
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

BURN_IN_T = 1000


def _fit(traj, smooth_window=5001):
    """Logistic fit returning the parameters as well as the percentile times."""
    traj = np.asarray(traj, dtype=float)
    n = len(traj)
    win = min(smooth_window, n // 2 * 2 - 1)
    smooth = savgol_filter(traj, win, 3)
    p0 = [traj[-1] - traj[0], 1e-4, n * 0.1, traj[0]]
    bounds = ([0, 0, 0, 0], [1, 1e-2, n * 2, 0.5])
    popt, _ = curve_fit(_logistic, np.arange(n), smooth, p0=p0,
                        bounds=bounds, maxfev=50000)
    L, k, t0, b = popt
    t_at = lambda pct: max(0.0, t0 - np.log((1 - pct) / pct) / k)
    return dict(L=L, k=k, t0=t0, b=b, asymptote=b + L,
                t_50=t_at(0.50), t_90=t_at(0.90), t_end=t_at(0.95))


def mediation(events, t_cut=None):
    """h_soc = 0 exactly when the buffer carries no veg entry (model_main.py:88).
    Buffer entries are (diet, source, t_sampled) since 2026-09-02; older logs
    carry two fields."""
    n_conv = n_unmediated = 0
    for e in events:
        if e[0] != 'conv':
            continue
        if t_cut is not None and e[1] > t_cut:
            continue
        n_conv += 1
        if not any(entry[0] == 'veg' for entry in e[5]):
            n_unmediated += 1
    return n_conv, n_unmediated


def measure(run):
    traj = np.asarray(run['fraction_veg'], dtype=float)
    ev = run['events']
    p = run['params']

    fit = _fit(traj)
    conv = [e for e in ev if e[0] == 'conv']
    rev = [e for e in ev if e[0] == 'rev']
    converters = {e[2] for e in conv}
    d0 = np.asarray([d == 'veg' for d in run['initial_diets']])
    final = run['snapshots']['final']['diets'].astype(bool)
    net = int((~d0 & final).sum())

    n_c_all, n_u_all = mediation(ev)
    n_c_burn, n_u_burn = mediation(ev, BURN_IN_T)

    return {
        'run': run['run'], 'seed': p.get('seed'), 'kappa': p['kappa'],
        'N': p['N'], 'steps': p['steps'], 'tau_persistence': p['tau_persistence'],
        't_50': fit['t_50'], 't_90': fit['t_90'], 't_end': fit['t_end'],
        'asymptote': fit['asymptote'], 'F_end': float(traj[-1]),
        'F_0': float(traj[0]), 'burnin_jump': float(traj[BURN_IN_T] - traj[0]),
        'n_conv': len(conv), 'n_rev': len(rev), 'n_converters': len(converters),
        'net_adopters': net,
        'churn_per_converter': len(conv) / len(converters) if converters else np.nan,
        'switches_per_net_adopter': (len(conv) + len(rev)) / net if net else np.nan,
        'unmediated_frac': n_u_all / n_c_all if n_c_all else np.nan,
        'unmediated_frac_burnin': n_u_burn / n_c_burn if n_c_burn else np.nan,
        'n_conv_burnin': n_c_burn,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reduced_dir')
    ap.add_argument('-o', '--out', default=None)
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl'))):
        with open(path, 'rb') as f:
            rows.append(measure(pickle.load(f)))
        print(f"INFO: {os.path.basename(path)} done", flush=True)
    df = pd.DataFrame(rows)

    out = a.out or os.path.join(a.reduced_dir, 'ensemble_measures.csv')
    df.to_csv(out, index=False)

    def q(col, fmt='{:.4g}'):
        v = df[col].dropna()
        return (f"{fmt.format(np.median(v))}  IQR [{fmt.format(np.percentile(v, 25))}, "
                f"{fmt.format(np.percentile(v, 75))}]  range [{fmt.format(v.min())}, "
                f"{fmt.format(v.max())}]")

    print(f"\n{len(df)} runs, kappa={df['kappa'].unique()}, N={df['N'].unique()}, "
          f"steps={df['steps'].unique()}, tau={df['tau_persistence'].unique()}")
    for col in ['t_50', 't_90', 't_end', 'asymptote', 'F_end', 'burnin_jump',
                'unmediated_frac', 'unmediated_frac_burnin', 'churn_per_converter',
                'switches_per_net_adopter', 'n_conv', 'n_rev', 'net_adopters']:
        print(f"{col:26s} {q(col)}")
    print(f"\nINFO: wrote {out}")


if __name__ == '__main__':
    main()
