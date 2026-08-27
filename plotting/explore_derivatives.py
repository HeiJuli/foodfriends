"""Ensemble 1st and 2nd derivatives of F_veg(t) under several smoothing windows.

Analysis A1 of claude_stuff/Review/tipping_reframe_plan_2026-08-18.md: shows whether
acceleration has a peak or a broad plateau. Median across runs with the IQR band; the
smoothing window is stated on every panel because F_c depends on it.

Usage: python explore_derivatives.py [pkl]
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

PKL = sys.argv[1] if len(sys.argv) > 1 else '../model_output/trajectory_analysis_twin_20260820.pkl'
WINDOWS = [2001, 5001, 10001, 15001]
BURNIN = 5000
POLY = 3
STRIDE = 10   # F_veg moves by 1/N per step; decimating cuts savgol cost ~100x

data = pd.read_pickle(PKL)
T = np.array([np.asarray(r['fraction_veg_trajectory'], dtype=float)[::STRIDE]
              for _, r in data.iterrows()])
print(f"Loaded {len(T)} runs from {PKL}")

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
    out = f'../visualisations_output/derivatives_ensemble.{ext}'
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")
plt.close()
