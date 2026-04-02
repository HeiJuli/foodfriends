"""
Demographic breakdown of theta, rho, and alpha.

Data sources:
- theta, rho: hierarchical_agents.csv (merged with theta survey)
- alpha: alpha_demographics.xlsx (full original survey, n~4944)
  Alpha is NOT loaded from hierarchical_agents.csv because ~2,328 alpha
  respondents have no theta match and would be excluded from the merged file.

Saves summary table to demographic_parameter_summary.csv
and figure to demographic_parameter_distributions.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT_DIR = Path(__file__).parent
DATA_FILE  = Path(__file__).parent.parent / "hierarchical_agents.csv"
ALPHA_FILE = Path(__file__).parent.parent / "alpha_demographics.xlsx"

# theta and rho from merged file
df = pd.read_csv(DATA_FILE)
rho_df = df[df['has_rho']].copy()

# alpha from full original survey
_alpha_raw = pd.read_excel(ALPHA_FILE)
alpha_df = pd.DataFrame({
    'alpha':     pd.to_numeric(_alpha_raw['Self-identity weight (alpha)'], errors='coerce'),
    'gender':    _alpha_raw['Gender'],
    'age':       pd.to_numeric(_alpha_raw['Age of the household member'], errors='coerce'),
    'incquart':  _alpha_raw['Income Quartile'],
    'educlevel': _alpha_raw['Education Level'],
}).dropna()
alpha_df['age_group'] = pd.cut(alpha_df['age'], bins=[17,29,39,49,59,69,120],
                                labels=['18-29','30-39','40-49','50-59','60-69','70+'])
alpha_df = alpha_df.dropna(subset=['age_group'])

# ── Labels ────────────────────────────────────────────────────────────────────
DEMO_VARS = {
    'gender':    {'order': ['Female', 'Male'],
                  'labels': {'Female': 'Female', 'Male': 'Male'}},
    'age_group': {'order': ['18-29','30-39','40-49','50-59','60-69','70+'],
                  'labels': None},
    'incquart':  {'order': [1, 2, 3, 4],
                  'labels': {1:'Q1 (lowest)', 2:'Q2', 3:'Q3', 4:'Q4 (highest)'}},
    'educlevel': {'order': [1, 2, 3, 4],
                  'labels': {1:'Low', 2:'Mid-low', 3:'Mid-high', 4:'High'}},
}

PARAM_LABELS = {
    'theta': 'θ — veg food preference\n(−1 = strongly meat, +1 = strongly veg)',
    'rho':   'ρ — behavioral intention\n(0 = no intention, 1 = strong intention)',
    'alpha': 'α — personal identity strength\n("My personal identity is very important to me")',
}

# ── Summary table ─────────────────────────────────────────────────────────────
rows = []
for var, cfg in DEMO_VARS.items():
    alpha_var = 'age_group' if var == 'age_group' else var
    for lvl in cfg['order']:
        label = cfg['labels'][lvl] if cfg['labels'] else str(lvl)
        t = df[df[var] == lvl]['theta']
        r = rho_df[rho_df[var] == lvl]['rho']
        a = alpha_df[alpha_df[alpha_var] == lvl]['alpha']
        rows.append({
            'variable': var, 'group': label,
            'theta_mean': round(t.mean(), 3), 'theta_sd': round(t.std(), 3), 'theta_n': len(t),
            'rho_mean':   round(r.mean(), 3), 'rho_sd':   round(r.std(), 3), 'rho_n':   len(r),
            'alpha_mean': round(a.mean(), 3), 'alpha_sd': round(a.std(), 3), 'alpha_n': len(a),
        })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "demographic_parameter_summary.csv", index=False)
print(f"Saved: demographic_parameter_summary.csv")

# ── Figure ────────────────────────────────────────────────────────────────────
PARAMS = ['theta', 'rho', 'alpha']
SOURCE = {'theta': df, 'rho': rho_df, 'alpha': alpha_df}
# alpha uses its own demographic column names (age_group instead of age)
ALPHA_VAR_MAP = {'age_group': 'age_group'}  # all others match
COLORS = {'theta': '#2a9d8f', 'rho': '#e76f51', 'alpha': '#457b9d'}

fig = plt.figure(figsize=(15, 14))
gs = gridspec.GridSpec(len(DEMO_VARS), len(PARAMS), hspace=0.55, wspace=0.35)

for row_i, (var, cfg) in enumerate(DEMO_VARS.items()):
    order  = cfg['order']
    xlabs  = [cfg['labels'][l] if cfg['labels'] else str(l) for l in order]

    for col_i, param in enumerate(PARAMS):
        ax = fig.add_subplot(gs[row_i, col_i])
        src = SOURCE[param]

        src_var = var  # alpha_df uses same column names except age is age_group
        means = [src[src[src_var] == lvl][param].mean() for lvl in order]
        sds   = [src[src[src_var] == lvl][param].std()  for lvl in order]
        ns    = [len(src[src[src_var] == lvl][param])    for lvl in order]
        ses   = [s / np.sqrt(n) for s, n in zip(sds, ns)]

        x = np.arange(len(order))
        bars = ax.bar(x, means, yerr=ses, capsize=4,
                      color=COLORS[param], alpha=0.8, edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(xlabs, fontsize=8,
                           rotation=30 if var in ('age_group',) else 0, ha='right')
        ax.set_ylabel('Mean ± SE', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

        if row_i == 0:
            ax.set_title(PARAM_LABELS[param], fontsize=9, pad=8)
        if col_i == 0:
            var_label = {'gender': 'Gender', 'age_group': 'Age group',
                         'incquart': 'Income quartile', 'educlevel': 'Education level'}[var]
            ax.set_ylabel(f'{var_label}\n\nMean ± SE', fontsize=8)

fig.suptitle('Distribution of θ, ρ, α across demographic groups\n(error bars = standard error)',
             fontsize=12, y=0.98)

fig.savefig(OUT_DIR / "demographic_parameter_distributions.png", dpi=150, bbox_inches='tight')
print(f"Saved: demographic_parameter_distributions.png")
plt.close()
