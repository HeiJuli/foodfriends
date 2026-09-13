"""Diet assortativity on the network through the run, against its two nulls (topology_local_effects.py).

Usage (repo root): python analysis/topology_assortativity_plot.py
Writes visualisations_output/topology_diet_assortativity.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = 'model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced'
a = pd.read_csv(f'{DIR}/topology_local_assortativity.csv')
a = a[a.level.isna()]                                   # fixed time grid only
g = a.groupby('t')
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for col, lab, c in (('r_obs', 'observed', '#006d77'), ('r_cross_mean', 'cross-run null', '#e29578'),
                    ('r_perm_mean', 'degree-decile permutation null', '#adb5bd')):
    med, lo, hi = g[col].median(), g[col].quantile(0.25), g[col].quantile(0.75)
    ax[0].plot(med.index / 1000, med, color=c, label=lab)
    ax[0].fill_between(med.index / 1000, lo, hi, color=c, alpha=0.25, lw=0)
F = g.F.median()
ax2 = ax[0].twinx(); ax2.plot(F.index / 1000, F, color='k', lw=0.8, ls=':'); ax2.set_ylabel('F_veg (median, dotted)')
ax[0].axvline(310, color='#555', lw=0.6, ls='--')
ax[0].set_xlabel('t (thousand steps)'); ax[0].set_ylabel('diet assortativity r'); ax[0].legend(frameon=False, fontsize=7)
ax[1].scatter(a.F, a.r_obs - a.r_cross_mean, s=3, color='#006d77', alpha=0.4)
ax[1].axhline(0, color='#555', lw=0.6)
ax[1].set_xlabel('F_veg'); ax[1].set_ylabel('r observed - cross-run null (per run)')
for x in ax:
    x.spines[['top']].set_visible(False)
fig.tight_layout()
fig.savefig('visualisations_output/topology_diet_assortativity.pdf')
print('INFO: saved visualisations_output/topology_diet_assortativity.pdf')
