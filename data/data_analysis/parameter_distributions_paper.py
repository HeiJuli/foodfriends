"""
Publication-quality distribution plots for theta, rho, and alpha.
Uses original LISS survey data (full empirical populations, pre-ABM filtering).

- theta: continuous — histogram + best-fit parametric curve (skewnorm vs norm by AIC)
- rho:   discrete 4-point Likert scale → bar chart only
- alpha: discrete 7-point Likert scale → bar chart only

Data sources (original Stata files):
- theta: theta_diet.dta  (N~5742)
- alpha: alpha.dta        (N~5095)
- rho:   rho.dta          (N~2505)

Saves:
- parameter_distributions_paper.pdf  — vector, editable in Illustrator
- parameter_distributions_paper.png  — raster preview
"""
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "plotting"))
from plot_styles import set_publication_style, apply_axis_style, COLORS

OUT_DIR    = Path(__file__).parent
DATA_DIR   = Path("C:/Users/emma.thill/Dropbox/Projects/Foodfriends/Data/Final_paper_data")

# ── Load original survey data ──────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    theta_raw = pd.read_stata(DATA_DIR / "theta_diet.dta")
    alpha_raw = pd.read_stata(DATA_DIR / "alpha.dta")
    rho_raw   = pd.read_stata(DATA_DIR / "rho.dta")

theta_vals = pd.to_numeric(theta_raw['theta'], errors='coerce').dropna().values
alpha_vals = pd.to_numeric(alpha_raw['alpha'], errors='coerce').dropna().values
rho_vals   = pd.to_numeric(rho_raw['rho'],   errors='coerce').dropna().values

print(f"theta: N={len(theta_vals)}, mean={theta_vals.mean():.2f}, SD={theta_vals.std():.2f}")
print(f"alpha: N={len(alpha_vals)}, mean={alpha_vals.mean():.2f}, SD={alpha_vals.std():.2f}")
print(f"rho:   N={len(rho_vals)},   mean={rho_vals.mean():.2f}, SD={rho_vals.std():.2f}")

# ── Distribution fitting for theta only ───────────────────────────────────────
def fit_best(data, candidates):
    best_aic, best_dist, best_params = np.inf, None, None
    for name in candidates:
        dist = getattr(stats, name)
        try:
            params = dist.fit(data)
            ll  = np.sum(dist.logpdf(data, *params))
            a   = 2 * len(params) - 2 * ll
            if a < best_aic:
                best_aic, best_dist, best_params = a, dist, params
        except Exception:
            continue
    return best_dist, best_params

theta_dist, theta_params = fit_best(theta_vals, ['skewnorm', 'norm'])
print(f"theta best fit: {theta_dist.name}  params={np.round(theta_params, 3)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
BAR_COLOR = COLORS['vegetation']
FIT_COLOR = COLORS['neutral']

set_publication_style()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
plt.subplots_adjust(wspace=0.38)

# ── Panel 1: theta — continuous histogram + fitted curve ─────────────────────
ax = axes[0]
ax.hist(theta_vals, bins=40, density=True, color=BAR_COLOR, alpha=0.5,
        edgecolor='white', linewidth=0.4, zorder=2)
x_grid = np.linspace(-1.05, 1.05, 500)
ax.plot(x_grid, theta_dist.pdf(x_grid, *theta_params),
        color=FIT_COLOR, linewidth=2.0, zorder=3)
ax.axvline(theta_vals.mean(), color=BAR_COLOR, linestyle='--', linewidth=1.2, alpha=0.9, zorder=4)
ax.text(0.03, 0.97,
        f"mean = {theta_vals.mean():.2f}\nSD = {theta_vals.std():.2f}\nN = {len(theta_vals):,}",
        transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))
ax.set_xlabel(r'Dietary preference $\theta$', fontsize=9)
ax.set_ylabel('Density', fontsize=8)
ax.set_xlim(-1.05, 1.05)
apply_axis_style(ax)

# ── Panels 2 & 3: rho and alpha — discrete bar charts ────────────────────────
for ax, data, xlabel in [
    (axes[1], rho_vals,   r'Behavioural Intention $\rho$'),
    (axes[2], alpha_vals, r'Self-reliance $\alpha$'),
]:
    unique_vals, counts = np.unique(data, return_counts=True)
    proportions = counts / counts.sum()
    bar_width = np.min(np.diff(unique_vals)) * 0.6 if len(unique_vals) > 1 else 0.1
    ax.bar(unique_vals, proportions, width=bar_width, color=BAR_COLOR, alpha=0.8,
           edgecolor='white', linewidth=0.5, zorder=2)
    ax.axvline(data.mean(), color=BAR_COLOR, linestyle='--', linewidth=1.2, alpha=0.9, zorder=3)
    ha = 'left' if xlabel == r'Self-reliance $\alpha$' else 'right'
    x_pos = 0.03 if ha == 'left' else 0.97
    ax.text(x_pos, 0.97,
            f"mean = {data.mean():.2f}\nSD = {data.std():.2f}\nN = {len(data):,}",
            transform=ax.transAxes, fontsize=7.5, va='top', ha=ha,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Proportion', fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    apply_axis_style(ax)

fig.savefig(OUT_DIR / "parameter_distributions_paper.pdf", bbox_inches='tight')
fig.savefig(OUT_DIR / "parameter_distributions_paper.png", dpi=200, bbox_inches='tight')
print("Saved: parameter_distributions_paper.pdf and .png")
plt.close()
