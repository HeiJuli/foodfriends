"""
Analyze optimal sample size for twin mode considering trade-off between
finite-size effects and parameter imputation accuracy.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load hierarchical agents
df = pd.read_csv('../data/hierarchical_agents.csv')

print("=" * 80)
print("HIERARCHICAL AGENT DATASET COMPOSITION")
print("=" * 80)
print(f"\nTotal survey participants: {len(df)}")

# Analyze completeness
complete_cases = df[df['has_rho'] & df['has_alpha']]
has_rho_only = df[df['has_rho'] & ~df['has_alpha']]
has_alpha_only = df[~df['has_rho'] & df['has_alpha']]
has_neither = df[~df['has_rho'] & ~df['has_alpha']]

print(f"\nComplete cases (theta, rho, alpha): {len(complete_cases)} ({100*len(complete_cases)/len(df):.1f}%)")
print(f"Has rho only (impute alpha): {len(has_rho_only)} ({100*len(has_rho_only)/len(df):.1f}%)")
print(f"Has alpha only (impute rho): {len(has_alpha_only)} ({100*len(has_alpha_only)/len(df):.1f}%)")
print(f"Has neither (impute both): {len(has_neither)} ({100*len(has_neither)/len(df):.1f}%)")

# Analyze parameter distributions for complete cases
print("\n" + "=" * 80)
print("COMPLETE CASES PARAMETER DISTRIBUTIONS")
print("=" * 80)
print(f"\nTheta: mean={complete_cases['theta'].mean():.3f}, std={complete_cases['theta'].std():.3f}")
print(f"Rho:   mean={complete_cases['rho'].mean():.3f}, std={complete_cases['rho'].std():.3f}")
print(f"Alpha: mean={complete_cases['alpha'].mean():.3f}, std={complete_cases['alpha'].std():.3f}")

# Calculate effective threshold for complete cases
complete_cases['beta'] = 1 - complete_cases['alpha']
complete_cases['dissonance'] = np.where(complete_cases['diet'] == 'meat',
                                        complete_cases['theta'],
                                        1 - complete_cases['theta'])
complete_cases['eff_threshold'] = complete_cases['rho'] - complete_cases['alpha'] * complete_cases['dissonance']

meat_eaters = complete_cases[complete_cases['diet'] == 'meat']
print(f"\nEffective threshold (meat eaters): mean={meat_eaters['eff_threshold'].mean():.3f}, std={meat_eaters['eff_threshold'].std():.3f}")
print(f"Proportion with eff_threshold < 0.20: {(meat_eaters['eff_threshold'] < 0.20).mean():.1%}")

# Correlations
print("\n" + "=" * 80)
print("PARAMETER CORRELATIONS (complete cases)")
print("=" * 80)
print(f"theta-rho:   {complete_cases['theta'].corr(complete_cases['rho']):.3f}")
print(f"theta-alpha: {complete_cases['theta'].corr(complete_cases['alpha']):.3f}")
print(f"rho-alpha:   {complete_cases['rho'].corr(complete_cases['alpha']):.3f}")

# Demographic representativeness
print("\n" + "=" * 80)
print("DEMOGRAPHIC REPRESENTATIVENESS")
print("=" * 80)

print("\nGender distribution:")
print("All cases:")
print(df['gender'].value_counts(normalize=True))
print("\nComplete cases:")
print(complete_cases['gender'].value_counts(normalize=True))

print("\nAge distribution:")
print("All cases:")
print(df['age_group'].value_counts(normalize=True).sort_index())
print("\nComplete cases:")
print(complete_cases['age_group'].value_counts(normalize=True).sort_index())

# Finite-size effect analysis
print("\n" + "=" * 80)
print("FINITE-SIZE EFFECTS ANALYSIS")
print("=" * 80)

# For network models, finite-size effects typically scale as 1/sqrt(N)
# We want coefficient of variation (std/mean) to be small

candidate_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 5602]
print(f"\n{'N':>6} | {'Complete':>8} | {'Imputed':>8} | {'%Imputed':>9} | {'CV_finite':>9}")
print("-" * 60)

for N in candidate_sizes:
    n_complete = min(N, len(complete_cases))
    n_imputed = max(0, N - len(complete_cases))
    pct_imputed = 100 * n_imputed / N if N > 0 else 0
    cv_finite = 1 / np.sqrt(N)  # Coefficient of variation from finite-size
    print(f"{N:6d} | {n_complete:8d} | {n_imputed:8d} | {pct_imputed:8.1f}% | {cv_finite:9.4f}")

# Recommendation
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("""
FINITE-SIZE EFFECTS:
- For network ABMs, CV ~ 1/sqrt(N) represents statistical noise from finite population
- N=1000: CV=3.2%  (marginal for publication)
- N=2000: CV=2.2%  (acceptable for most network models)
- N=3000: CV=1.8%  (very good)
- N=5000: CV=1.4%  (excellent)

IMPUTATION ACCURACY:
- Complete cases: {n_complete} agents with all empirical parameters
- PMF imputation preserves correlations but introduces sampling uncertainty
- Theta-stratified PMFs maintain theta-rho and theta-alpha relationships

TRADE-OFF ANALYSIS:
- Up to N={n_complete}: 0% imputation, pure empirical data
- N=2000: ~{pct_2000:.0f}% imputation, CV=2.2%
- N=3000: ~{pct_3000:.0f}% imputation, CV=1.8%
- N=5602: ~{pct_5602:.0f}% imputation, CV=1.3%

RECOMMENDATION: N = 2000-3000
Rationale:
1. Finite-size effects: CV < 2% (publication standard for network models)
2. Imputation fraction: ~{mid_pct:.0f}% allows majority of agents to use empirical parameters
3. Statistical power: Sufficient for detecting phase transitions and critical phenomena
4. Computational efficiency: Faster iteration cycles during model development

For final publication runs: Consider ensemble of N=2000 with multiple seeds
vs single large run at N=5602. Ensemble approach provides better uncertainty
quantification while maintaining high empirical accuracy.
""".format(
    n_complete=len(complete_cases),
    pct_2000=100*(2000-len(complete_cases))/2000 if 2000 > len(complete_cases) else 0,
    pct_3000=100*(3000-len(complete_cases))/3000 if 3000 > len(complete_cases) else 0,
    pct_5602=100*(5602-len(complete_cases))/5602,
    mid_pct=100*(2500-len(complete_cases))/2500 if 2500 > len(complete_cases) else 0
))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Complete vs imputed agents
ax = axes[0, 0]
Ns = np.array(candidate_sizes)
complete_counts = np.minimum(Ns, len(complete_cases))
imputed_counts = np.maximum(0, Ns - len(complete_cases))
ax.plot(Ns, complete_counts, 'o-', label='Complete (empirical)', linewidth=2)
ax.plot(Ns, imputed_counts, 's-', label='Imputed (PMF)', linewidth=2)
ax.axvline(len(complete_cases), color='red', linestyle='--', alpha=0.5, label=f'Max complete ({len(complete_cases)})')
ax.set_xlabel('Total population size N')
ax.set_ylabel('Number of agents')
ax.legend()
ax.grid(alpha=0.3)
ax.set_title('Agent composition vs population size')

# Plot 2: Finite-size CV
ax = axes[0, 1]
Ns_smooth = np.linspace(500, 5602, 100)
cv = 1 / np.sqrt(Ns_smooth)
ax.plot(Ns_smooth, cv * 100, linewidth=2)
ax.axhline(2.0, color='orange', linestyle='--', alpha=0.5, label='2% threshold (good)')
ax.axhline(1.5, color='green', linestyle='--', alpha=0.5, label='1.5% threshold (excellent)')
for N in [1000, 2000, 3000, 5000]:
    cv_n = 100 / np.sqrt(N)
    ax.plot(N, cv_n, 'ro', markersize=8)
    ax.text(N, cv_n + 0.1, f'{N}', ha='center', fontsize=9)
ax.set_xlabel('Population size N')
ax.set_ylabel('Coefficient of variation (%)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_title('Finite-size statistical noise (CV ~ 1/√N)')

# Plot 3: Imputation fraction
ax = axes[1, 0]
impute_frac = np.maximum(0, Ns - len(complete_cases)) / Ns * 100
ax.plot(Ns, impute_frac, 'o-', linewidth=2, color='purple')
ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% imputed')
ax.axhline(30, color='orange', linestyle='--', alpha=0.5, label='30% imputed')
ax.set_xlabel('Population size N')
ax.set_ylabel('Imputed fraction (%)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_title('Parameter imputation fraction')

# Plot 4: Effective threshold distribution
ax = axes[1, 1]
ax.hist(meat_eaters['eff_threshold'], bins=50, alpha=0.7, edgecolor='black')
ax.axvline(meat_eaters['eff_threshold'].mean(), color='red', linestyle='--',
           linewidth=2, label=f"Mean = {meat_eaters['eff_threshold'].mean():.3f}")
ax.axvline(0.20, color='orange', linestyle='--', linewidth=2,
           label='Stability threshold (0.20)')
ax.set_xlabel('Effective threshold (rho - alpha*dissonance)')
ax.set_ylabel('Count (meat eaters)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_title('Effective threshold distribution (complete cases)')

plt.tight_layout()
plt.savefig('../visualisations_output/optimal_sample_size_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: visualisations_output/optimal_sample_size_analysis.png")
