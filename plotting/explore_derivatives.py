"""Ensemble 1st and 2nd derivatives of F_veg(t) under several smoothing windows.

Analysis A1 of claude_stuff/Review/tipping_reframe_plan_2026-08-18.md: shows whether
acceleration has a peak or a broad plateau. Median across runs with the IQR band; the
smoothing window is stated on every panel because F_c depends on it.

Usage: python explore_derivatives.py [pkl|npz] [win1,win2,...] [out_tag]

An .npz input reads the fc_viability trajectory cache (50 x 400001 float64,
keys `trajectories` and `seeds`) instead of a pickle -- the kappa=0.55 headline
ensemble does not fit in this machine's memory as a DataFrame.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from plot_styles import set_publication_style, COLORS

set_publication_style()
cm = 1/2.54

SRC = sys.argv[1] if len(sys.argv) > 1 else '../model_output/trajectory_analysis_twin_20260820.pkl'
WINDOWS = ([int(w) for w in sys.argv[2].split(',')] if len(sys.argv) > 2
           else [2001, 5001, 10001, 15001])
OUT_TAG = sys.argv[3] if len(sys.argv) > 3 else 'derivatives_ensemble'
BURNIN = 5000
POLY = 3
STRIDE = 10   # F_veg moves by 1/N per step; decimating cuts savgol cost ~100x

if SRC.endswith('.npz'):
    T = np.asarray(np.load(SRC)['trajectories'], dtype=float)[:, ::STRIDE]
    print(f"Loaded {len(T)} runs from {SRC}")
else:
    data = pd.read_pickle(SRC)
    T = np.array([np.asarray(r['fraction_veg_trajectory'], dtype=float)[::STRIDE]
                  for _, r in data.iterrows()])
    print(f"Loaded {len(T)} runs from {SRC}")

t_idx = np.arange(T.shape[1]) * STRIDE
t_k = t_idx / 1000
COL = COLORS['primary']
COL_D1 = '#e76f51'
COL_D2 = '#9b59b6'

fig, axes = plt.subplots(3, len(WINDOWS), figsize=(28*cm, 18*cm), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1, 1]})

def band(ax, y, color):
    ax.fill_between(t_k, np.percentile(y, 25, axis=0), np.percentile(y, 75, axis=0),
                    color=color, alpha=0.2, lw=0)
    ax.plot(t_k, np.median(y, axis=0), color=color, lw=1.0)

for col, win in enumerate(WINDOWS):
    w = max(5, int(win // STRIDE) | 1)
    sm = np.array([savgol_filter(t, w, POLY) for t in T])
    d1 = np.array([savgol_filter(t, w, POLY, deriv=1, delta=STRIDE) for t in T])
    d2 = np.array([savgol_filter(t, w, POLY, deriv=2, delta=STRIDE) for t in T])
    # widen the burn-in with the window: a width-w kernel still straddles the
    # initial equilibration jump for its first w/2 steps
    burn = t_idx < max(BURNIN, win)
    d1[:, burn] = 0
    d2[:, burn] = 0

    m1, m2, msm = np.median(d1, axis=0), np.median(d2, axis=0), np.median(sm, axis=0)
    peak = m1.max()
    for frac in (0.9, 0.5):
        ix = np.where(m1 >= frac * peak)[0]
        print(f"win={win}: dF/dt >= {frac:.0%} of peak over t=[{t_k[ix.min()]:.1f}k, "
              f"{t_k[ix.max()]:.1f}k] = {100*len(ix)/len(m1):.0f}% of the run, "
              f"F_veg in [{msm[ix.min()]:.3f}, {msm[ix.max()]:.3f}]")
    i1 = int(np.argmax(m1))
    m2m = m2.copy(); m2m[msm > 0.5] = 0
    i2 = int(np.argmax(m2m))
    print(f"win={win}: max dF/dt @ t={t_k[i1]:.1f}k F={msm[i1]:.3f} | "
          f"max d2F/dt2 (F<0.5) @ t={t_k[i2]:.1f}k F={msm[i2]:.3f}")

    ax = axes[0, col]
    band(ax, sm, COL)
    ax.axvline(t_k[i1], color=COL_D1, ls='--', lw=1, alpha=0.8)
    ax.axvline(t_k[i2], color=COL_D2, ls='--', lw=1, alpha=0.8)
    ax.set_title(f'window = {win}', fontsize=7, loc='left')
    if col == 0: ax.set_ylabel(r'$F_{\rm veg}$')

    ax = axes[1, col]
    band(ax, d1, COL_D1)
    ax.axvline(t_k[i1], color=COL_D1, ls='--', lw=1, alpha=0.8)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    if col == 0: ax.set_ylabel(r'$dF/dt$')

    ax = axes[2, col]
    band(ax, d2, COL_D2)
    ax.axvline(t_k[i2], color=COL_D2, ls='--', lw=1, alpha=0.8)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    if col == 0: ax.set_ylabel(r'$d^2F/dt^2$')
    ax.set_xlabel(r'$t$ [thousands]')

plt.tight_layout()
for ext in ('png', 'pdf'):
    out = f'../visualisations_output/{OUT_TAG}.{ext}'
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")
plt.close()

# Second view: d2 against F_veg (not t), per run. The F_c question is where on
# the F axis each run's maximum acceleration sits, so this is the view that
# shows the per-run peaks aligning or not. Thin per-run lines, median overlaid,
# and a rug of the per-run masked-argmax locations (the per-run F_c values).
fig2, axes2 = plt.subplots(1, len(WINDOWS), figsize=(28*cm, 7*cm), sharey=True)
for col, win in enumerate(WINDOWS):
    w = max(5, int(win // STRIDE) | 1)
    burn = t_idx < max(BURNIN, win)
    ax = axes2[col]
    fc_runs = []
    # common F grid for the median: interpolate each run's d2 onto it
    fgrid = np.linspace(0.06, 0.75, 400)
    d2_on_f = []
    for t in T:
        sm = savgol_filter(t, w, POLY)
        d2 = savgol_filter(t, w, POLY, deriv=2, delta=STRIDE)
        d2[burn] = 0
        d2m = d2.copy(); d2m[sm > 0.5] = 0
        i2 = int(np.argmax(d2m))
        fc_runs.append(sm[i2] if d2m[i2] > 0 else np.nan)
        # monotone smoothed F -> d2 as a function of F
        order = np.argsort(sm)
        d2_on_f.append(np.interp(fgrid, sm[order], d2[order]))
        ax.plot(sm[::20], d2[::20], color=COL_D2, alpha=0.08, lw=0.4, rasterized=True)
    ax.plot(fgrid, np.median(d2_on_f, axis=0), color=COL_D2, lw=1.4)
    fc_runs = np.asarray(fc_runs)
    ax.plot(fc_runs, np.full_like(fc_runs, ax.get_ylim()[0]), '|', color='k',
            ms=6, alpha=0.6)
    ax.axvline(np.nanmedian(fc_runs), color='k', ls='--', lw=0.8)
    ax.set_title(f'window = {win}\n'
                 f'median $F_c$ = {np.nanmedian(fc_runs):.3f}, '
                 f'IQR [{np.nanpercentile(fc_runs,25):.3f}, '
                 f'{np.nanpercentile(fc_runs,75):.3f}]', fontsize=7, loc='left')
    ax.set_xlabel(r'$F_{\rm veg}$')
    if col == 0: ax.set_ylabel(r'$d^2F/dt^2$')
plt.tight_layout()
for ext in ('png', 'pdf'):
    out = f'../visualisations_output/{OUT_TAG}_vs_F.{ext}'
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")
plt.close()
