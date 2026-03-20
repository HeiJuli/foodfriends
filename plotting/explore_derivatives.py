"""Exploratory: overlay 1st and 2nd derivatives on the F_veg trajectory."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from plot_styles import set_publication_style, COLORS

set_publication_style()
cm = 1/2.54

# Load median twin trajectory
pkl = '../model_output/trajectory_analysis_twin_20260317.pkl'
data = pd.read_pickle(pkl)
row = data[data['is_median_twin']].iloc[0]
traj = np.array(row['fraction_veg_trajectory'], dtype=float)

burnin = 5000
t = np.arange(len(traj))
t_k = t / 1000

# Try multiple smoothing windows
windows = [2001, 5001, 10001, 15001]
COL = COLORS['primary']
COL_D1 = '#e76f51'
COL_D2 = '#9b59b6'

fig, axes = plt.subplots(3, len(windows), figsize=(28*cm, 18*cm), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1, 1]})

for col, win in enumerate(windows):
    smoothed = savgol_filter(traj, win, polyorder=3)
    d1 = savgol_filter(traj, win, polyorder=3, deriv=1)
    d2 = savgol_filter(traj, win, polyorder=3, deriv=2)

    d1m = d1.copy(); d1m[:burnin] = 0
    d2m = d2.copy(); d2m[:burnin] = 0; d2m[smoothed > 0.5] = 0

    i1 = np.argmax(d1m)
    i2 = np.argmax(d2m)
    print(f"win={win}: max dF/dt @ t={i1/1000:.1f}k F={smoothed[i1]:.3f} | "
          f"max d2F/dt2 @ t={i2/1000:.1f}k F={smoothed[i2]:.3f}")

    ax = axes[0, col]
    ax.plot(t_k, traj, color=COL, alpha=0.2, lw=0.3)
    ax.plot(t_k, smoothed, color=COL, lw=1.2)
    ax.axvline(i1/1000, color=COL_D1, ls='--', lw=1, alpha=0.8)
    ax.axvline(i2/1000, color=COL_D2, ls='--', lw=1, alpha=0.8)
    ax.set_title(f'win={win}', fontsize=7, loc='left')
    if col == 0: ax.set_ylabel(r'$F_{\rm veg}$')

    ax = axes[1, col]
    ax.plot(t_k, d1, color=COL_D1, lw=0.8)
    ax.axvline(i1/1000, color=COL_D1, ls='--', lw=1, alpha=0.8)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    if col == 0: ax.set_ylabel(r'$dF/dt$')

    ax = axes[2, col]
    ax.plot(t_k, d2, color=COL_D2, lw=0.8)
    ax.axvline(i2/1000, color=COL_D2, ls='--', lw=1, alpha=0.8)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    if col == 0: ax.set_ylabel(r'$d^2F/dt^2$')
    ax.set_xlabel(r'$t$ [thousands]')

plt.tight_layout()
out = '../visualisations_output/explore_derivatives.png'
plt.savefig(out, dpi=200)
print(f"Saved: {out}")
plt.close()
