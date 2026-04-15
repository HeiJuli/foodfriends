#!/usr/bin/env python3
"""Facet plot: every trajectory with its logistic-fit t_end marker.

Usage:
    python trajectory_t_end_facet.py [pkl_file]
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
from t_end_logistic import estimate_t_end, _logistic
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'model_output')


def _fit_params(traj, smooth_window=5001):
    """Return (popt, smooth_traj) or (None, None)."""
    traj = np.asarray(traj, dtype=float)
    n = len(traj)
    win = min(smooth_window, n // 2 * 2 - 1)
    smooth = savgol_filter(traj, win, 3)
    tt = np.arange(n)
    p0 = [traj[-1] - traj[0], 1e-4, n * 0.1, traj[0]]
    bounds = ([0, 0, 0, 0], [1, 1e-2, n * 2, 0.5])
    try:
        popt, _ = curve_fit(_logistic, tt, smooth, p0=p0, bounds=bounds, maxfev=50000)
        return popt, smooth
    except Exception:
        return None, smooth


def facet_plot(pkl_file, pct=0.95, out='../visualisations_output/t_end_facet.png'):
    df = pd.read_pickle(pkl_file)
    n_runs = len(df)
    print(f"INFO: {n_runs} runs in {pkl_file}")

    ncols = 8
    nrows = int(np.ceil(n_runs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 1.2),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    anomalies = []
    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        traj = np.array(row.get('fraction_veg_trajectory',
                                row.get('fraction_veg', [])), dtype=float)
        t_end = estimate_t_end(traj, pct=pct)
        popt, smooth = _fit_params(traj)
        tt = np.arange(len(traj))

        ax.plot(tt / 1000, traj, color='#888', lw=0.3, alpha=0.6)
        if smooth is not None:
            ax.plot(tt / 1000, smooth, color='#2a9d8f', lw=0.6, alpha=0.9)
        if popt is not None:
            ax.plot(tt / 1000, _logistic(tt, *popt), color='#e76f51',
                    lw=0.5, alpha=0.8, linestyle='--')
        if t_end is not None:
            ax.axvline(t_end / 1000, color='#d4a029', lw=0.8, alpha=0.9)
            # flag anomalies: t_end in first 20% of run
            if t_end < 0.2 * len(traj):
                anomalies.append((i, t_end, popt, traj))

        ax.set_title(f'#{i} t={t_end/1000 if t_end else np.nan:.0f}k',
                     fontsize=6, pad=1)
        ax.tick_params(axis='both', labelsize=4)
        ax.set_ylim(0, 1)

    for j in range(n_runs, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{os.path.basename(pkl_file)} — logistic t_end ({int(pct*100)}% asymptote)',
                 fontsize=9)
    fig.supxlabel('t [thousands]', fontsize=7)
    fig.supylabel('$F_{veg}$', fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"INFO: saved {out}")

    # Diagnose anomalies
    print(f"\n=== ANOMALIES (t_end < 20% of run length) ===")
    for i, t_end, popt, traj in anomalies:
        L, k, t0, b = popt if popt is not None else (np.nan,)*4
        print(f"  Run {i}: t_end={t_end}, L={L:.3f}, k={k:.2e}, t0={t0:.0f}, b={b:.3f}, "
              f"F_start={traj[0]:.3f}, F_end={traj[-1]:.3f}, F_max={traj.max():.3f}")
    return fig


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        MODEL_OUTPUT, 'trajectory_analysis_twin_20260317.pkl')
    facet_plot(path)
