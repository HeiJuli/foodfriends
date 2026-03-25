"""
Compares alpha values between respondents who also answered the theta survey
("matched") and those who did not ("unmatched").

Statistical testing:
- Independent samples t-test per demographic cell
- Cohen's d as effect size (small=0.2, medium=0.5, large=0.8)
- Benjamini-Hochberg FDR correction for multiple comparisons (14 tests)

Saves:
- alpha_differences_summary.csv  — mean, SD, n, t, p, p_fdr, cohen_d per group
- alpha_differences.png          — bar chart comparison
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT_DIR   = Path(__file__).parent
ALPHA_FILE = Path(__file__).parent.parent / "alpha_demographics.xlsx"
DATA_FILE  = Path(__file__).parent.parent / "hierarchical_agents.csv"

# ── Load data ─────────────────────────────────────────────────────────────────
alpha_raw = pd.read_excel(ALPHA_FILE)
alpha_full = pd.DataFrame({
    'id':        alpha_raw['id'],
    'alpha':     pd.to_numeric(alpha_raw['Self-identity weight (alpha)'], errors='coerce'),
    'gender':    alpha_raw['Gender'],
    'age':       pd.to_numeric(alpha_raw['Age of the household member'], errors='coerce'),
    'incquart':  alpha_raw['Income Quartile'],
    'educlevel': alpha_raw['Education Level'],
}).dropna()
alpha_full['age_group'] = pd.cut(alpha_full['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                  labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
alpha_full = alpha_full.dropna(subset=['age_group'])

matched_ids = set(pd.read_csv(DATA_FILE)['nomem_encr'])
alpha_full['has_theta'] = alpha_full['id'].isin(matched_ids)
matched   = alpha_full[alpha_full['has_theta']]
unmatched = alpha_full[~alpha_full['has_theta']]

print(f"Full alpha sample:      n={len(alpha_full)}")
print(f"Matched (has theta):    n={len(matched)}")
print(f"Unmatched (no theta):   n={len(unmatched)}")

# ── Demo config ───────────────────────────────────────────────────────────────
DEMO_VARS = {
    'gender':    {'order': ['Female', 'Male'],
                  'labels': {'Female': 'Female', 'Male': 'Male'}},
    'age_group': {'order': ['18-29', '30-39', '40-49', '50-59', '60-69', '70+'],
                  'labels': None},
    'incquart':  {'order': [1, 2, 3, 4],
                  'labels': {1: 'Q1 (lowest)', 2: 'Q2', 3: 'Q3', 4: 'Q4 (highest)'}},
    'educlevel': {'order': [1, 2, 3, 4],
                  'labels': {1: 'Low', 2: 'Mid-low', 3: 'Mid-high', 4: 'High'}},
}

# ── Summary with significance tests ──────────────────────────────────────────
def cohens_d(a, b):
    pooled_sd = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan

rows = []
for var, cfg in DEMO_VARS.items():
    for lvl in cfg['order']:
        label = cfg['labels'][lvl] if cfg['labels'] else str(lvl)
        m = matched[matched[var] == lvl]['alpha']
        u = unmatched[unmatched[var] == lvl]['alpha']
        t_stat, p_val = stats.ttest_ind(m, u)
        d = cohens_d(m, u)
        rows.append({
            'variable': var, 'group': label,
            'matched_mean':   round(m.mean(), 3), 'matched_sd':   round(m.std(), 3), 'matched_n':   len(m),
            'unmatched_mean': round(u.mean(), 3), 'unmatched_sd': round(u.std(), 3), 'unmatched_n': len(u),
            'diff':    round(m.mean() - u.mean(), 3),
            't_stat':  round(t_stat, 3),
            'p_raw':   round(p_val, 4),
            'cohens_d': round(d, 3),
        })

summary = pd.DataFrame(rows)

# Benjamini-Hochberg FDR correction across all tests
rejected, p_fdr, _, _ = multipletests(summary['p_raw'], method='fdr_bh')
summary['p_fdr']       = p_fdr.round(4)
summary['sig_fdr']     = rejected  # True = significant after correction

# Effect size label
def d_label(d):
    d = abs(d)
    if d < 0.2:  return 'negligible'
    if d < 0.5:  return 'small'
    if d < 0.8:  return 'medium'
    return 'large'
summary['effect_size'] = summary['cohens_d'].apply(d_label)

summary.to_csv(OUT_DIR / "alpha_differences_summary.csv", index=False)
print("Saved: alpha_differences_summary.csv")

# Print results to console
print(f"\n{'Variable':<12} {'Group':<14} {'Diff':>7} {'t':>7} {'p_raw':>8} {'p_fdr':>8} {'sig':>5} {'d':>7} {'effect'}")
print("-" * 85)
for _, row in summary.iterrows():
    sig = '***' if row['p_fdr'] < 0.001 else '**' if row['p_fdr'] < 0.01 else '*' if row['p_fdr'] < 0.05 else 'ns'
    print(f"{row['variable']:<12} {row['group']:<14} {row['diff']:>+7.3f} {row['t_stat']:>7.3f} "
          f"{row['p_raw']:>8.4f} {row['p_fdr']:>8.4f} {sig:>5}  {row['cohens_d']:>6.3f}  {row['effect_size']}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

COLORS = {'Matched (has θ)': '#2a9d8f', 'Unmatched (no θ)': '#e76f51'}
VAR_TITLES = {
    'gender': 'Gender',
    'age_group': 'Age group',
    'incquart': 'Income quartile',
    'educlevel': 'Education level',
}

for ax, (var, cfg) in zip(axes, DEMO_VARS.items()):
    order = cfg['order']
    xlabs = [cfg['labels'][l] if cfg['labels'] else str(l) for l in order]
    x = np.arange(len(order))
    width = 0.35

    for i, (grp, src, color) in enumerate([
        ('Matched (has θ)',  matched,   '#2a9d8f'),
        ('Unmatched (no θ)', unmatched, '#e76f51'),
    ]):
        means = [src[src[var] == lvl]['alpha'].mean() for lvl in order]
        sds   = [src[src[var] == lvl]['alpha'].std()  for lvl in order]
        ns    = [len(src[src[var] == lvl]['alpha'])    for lvl in order]
        ses   = [s / np.sqrt(n) for s, n in zip(sds, ns)]
        ax.bar(x + i * width - width / 2, means, width, yerr=ses, capsize=3,
               label=grp, color=color, alpha=0.85, edgecolor='white')

    # annotate significant differences
    for j, lvl in enumerate(order):
        label = cfg['labels'][lvl] if cfg['labels'] else str(lvl)
        row = summary[(summary['variable'] == var) & (summary['group'] == label)].iloc[0]
        m_mean = matched[matched[var] == lvl]['alpha'].mean()
        u_mean = unmatched[unmatched[var] == lvl]['alpha'].mean()
        if row['sig_fdr']:
            sig_str = '***' if row['p_fdr'] < 0.001 else '**' if row['p_fdr'] < 0.01 else '*'
            ax.text(j, max(m_mean, u_mean) + 0.025,
                    f'Δ{row["diff"]:+.3f}\nd={row["cohens_d"]:.2f}{sig_str}',
                    ha='center', fontsize=7, color='#333333', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, fontsize=9,
                        rotation=30 if var == 'age_group' else 0, ha='right')
    ax.set_ylabel('Mean α ± SE', fontsize=9)
    ax.set_ylim(0.55, 0.80)
    ax.set_title(VAR_TITLES[var], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle(
    'Alpha: matched (has θ) vs unmatched (no θ) respondents\n'
    'Annotations show FDR-corrected significant differences only (Δ, Cohen\'s d, * p<.05, ** p<.01, *** p<.001)',
    fontsize=10, y=1.01
)
plt.tight_layout()
fig.savefig(OUT_DIR / "alpha_differences.png", dpi=150, bbox_inches='tight')
print("Saved: alpha_differences.png")
plt.close()
