"""SI figure: diet clustering on the network through the run, and which group carries it.

(a) Diet assortativity r(t) against its two nulls (topology_local_effects.py).
(b) Excess same-diet neighbour share over the degree-decile null, per diet group, against
    adoption (topology_group_assortativity.py).

Usage (repo root): python analysis/topology_assortativity_plot.py
Writes visualisations_output/topology_diet_assortativity.pdf
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'plotting')
from plot_styles import set_publication_style, COLORS

DIR = 'model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced'
T_END = 310000

set_publication_style()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))

a = pd.read_csv(f'{DIR}/topology_local_assortativity.csv')
g = a[a.level.isna()].groupby('t')
for col, lab, c in (('r_obs', 'observed', COLORS['primary']),
                    ('r_cross_mean', 'cross-run null', COLORS['secondary']),
                    ('r_perm_mean', 'degree-decile null', '#adb5bd')):
    med = g[col].median()
    ax[0].plot(med.index / 1000, med, color=c, label=lab)
    ax[0].fill_between(med.index / 1000, g[col].quantile(0.25), g[col].quantile(0.75),
                       color=c, alpha=0.2, lw=0)
F = g.F.median()
tw = ax[0].twinx()
tw.plot(F.index / 1000, F, color='k', lw=0.9, ls=':')
tw.set_ylabel('$F_{veg}$ (dotted)')
tw.spines['top'].set_visible(False)
ax[0].axvline(T_END / 1000, color='#555', lw=0.7, ls='--')
ax[0].set_xlabel('t (thousand steps)')
ax[0].set_ylabel('diet assortativity $r$')
ax[0].legend(frameon=False, loc='lower right')

b = pd.read_csv(f'{DIR}/topology_group_assortativity.csv')
b['ex_veg'] = b.s_veg - b.s_veg_null
b['ex_omni'] = b.s_omni - b.s_omni_null
h = b.groupby('t')
x = h.F.median()
for col, lab, c in (('ex_veg', 'vegetarians', COLORS['vegetation']),
                    ('ex_omni', 'omnivores', COLORS['meat'])):
    ax[1].plot(x, h[col].median(), color=c, label=lab)
    ax[1].fill_between(x, h[col].quantile(0.25), h[col].quantile(0.75), color=c, alpha=0.2, lw=0)
ax[1].axhline(0, color='#555', lw=0.7)
ax[1].set_xlabel('$F_{veg}$')
ax[1].set_ylabel('excess same-diet neighbours')
ax[1].legend(frameon=False, loc='upper right')

for p in ax:
    p.spines[['top', 'right']].set_visible(False)
for lab, p in zip('ab', ax):
    p.text(-0.16, 1.04, lab, transform=p.transAxes, fontweight='bold', fontsize=10)
fig.tight_layout()
fig.savefig('visualisations_output/topology_diet_assortativity.pdf')
print('INFO: saved visualisations_output/topology_diet_assortativity.pdf')
