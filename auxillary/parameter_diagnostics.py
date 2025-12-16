#!/usr/bin/env python3
"""
Consolidated parameter diagnostics script

Combines functionality from:
- diagnose_parameter_sampling.py
- analyze_conditional_thresholds.py
- check_complete_cases_representativeness.py

Run this script to diagnose parameter sampling issues, check threshold distributions,
and assess complete cases representativeness.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('../model_src')
from model_main_single import sample_from_pmf

# ============================================================================
# HIERARCHICAL AGENTS ANALYSIS
# ============================================================================

def analyze_hierarchical_csv():
    """Analyze the hierarchical agents CSV"""
    print("="*80)
    print("HIERARCHICAL AGENTS CSV ANALYSIS")
    print("="*80)

    df = pd.read_csv("../data/hierarchical_agents.csv")

    print(f"\nTotal agents: {len(df)}")
    print(f"\nParameter availability:")
    print(f"  Has theta: {df['theta'].notna().sum()} ({df['theta'].notna().mean()*100:.1f}%)")
    print(f"  Has rho: {df['has_rho'].sum()} ({df['has_rho'].mean()*100:.1f}%)")
    print(f"  Has alpha: {df['has_alpha'].sum()} ({df['has_alpha'].mean()*100:.1f}%)")
    print(f"  Has both rho & alpha: {(df['has_rho'] & df['has_alpha']).sum()} ({(df['has_rho'] & df['has_alpha']).mean()*100:.1f}%)")

    print(f"\nDiet distribution:")
    print(df['diet'].value_counts())
    print(f"Veg fraction: {(df['diet']=='veg').mean():.3f}")

    print(f"\nParameter statistics (empirical values only):")
    for param in ['theta', 'rho', 'alpha']:
        has_col = f'has_{param}' if param != 'theta' else None
        if has_col and has_col in df.columns:
            subset = df[df[has_col]][param]
        else:
            subset = df[param].dropna()

        print(f"\n  {param} (n={len(subset)}):")
        print(f"    mean: {subset.mean():.3f}, std: {subset.std():.3f}")
        print(f"    median: {subset.median():.3f}")
        print(f"    range: [{subset.min():.3f}, {subset.max():.3f}]")

    return df

# ============================================================================
# PMF TABLES ANALYSIS
# ============================================================================

def analyze_pmf_tables():
    """Analyze PMF tables used for filling missing parameters"""
    print("\n" + "="*80)
    print("PMF TABLES ANALYSIS")
    print("="*80)

    with open("../data/demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)

    for param in ['alpha', 'rho', 'theta']:
        print(f"\n{param.upper()}:")
        cells = pmf_tables[param]
        print(f"  Demographic cells: {len(cells)}")

        # Calculate statistics across all cells
        all_means, all_stds = [], []
        for cell_data in cells.values():
            vals, probs = np.array(cell_data['values']), np.array(cell_data['probabilities'])
            mean = np.sum(vals * probs)
            variance = np.sum((vals - mean)**2 * probs)
            all_means.append(mean)
            all_stds.append(np.sqrt(variance))

        print(f"  Mean across cells: {np.mean(all_means):.3f} ± {np.std(all_means):.3f}")
        print(f"  Mean std within cells: {np.mean(all_stds):.3f}")
        print(f"  Range of cell means: [{min(all_means):.3f}, {max(all_means):.3f}]")

    return pmf_tables

# ============================================================================
# PARAMETER CORRELATIONS
# ============================================================================

def analyze_parameter_correlations():
    """Check for correlations between parameters in original data"""
    from scipy import stats

    print("\n" + "="*80)
    print("PARAMETER CORRELATION ANALYSIS")
    print("="*80)

    # Load raw surveys
    theta_df = pd.read_excel("../data/theta_diet_demographics.xlsx")
    rho_df = pd.read_excel("../data/rho_demographics.xlsx")
    alpha_df = pd.read_excel("../data/alpha_demographics.xlsx")

    # Clean
    theta_clean = pd.DataFrame({
        'id': theta_df['id'],
        'theta': pd.to_numeric(theta_df['Personal Preference for Veg Diet'], errors='coerce')
    }).dropna()

    rho_clean = pd.DataFrame({
        'id': rho_df['id'],
        'rho': pd.to_numeric(rho_df['Cost parameter (rho)'], errors='coerce')
    }).dropna()

    alpha_clean = pd.DataFrame({
        'id': alpha_df['id'],
        'alpha': pd.to_numeric(alpha_df['Self-identity weight (alpha)'], errors='coerce')
    }).dropna()

    # Find complete cases
    common_ids = set(theta_clean['id']) & set(rho_clean['id']) & set(alpha_clean['id'])
    print(f"\nAgents with all 3 parameters: {len(common_ids)}")

    if len(common_ids) > 0:
        merged = theta_clean[theta_clean['id'].isin(common_ids)].copy()
        merged = merged.merge(rho_clean[rho_clean['id'].isin(common_ids)], on='id')
        merged = merged.merge(alpha_clean[alpha_clean['id'].isin(common_ids)], on='id')

        print(f"\nCorrelation matrix (n={len(merged)}):")
        corr_matrix = merged[['theta', 'rho', 'alpha']].corr()
        print(corr_matrix.round(3))

        print(f"\nPairwise correlations with p-values:")
        for p1, p2 in [('theta', 'rho'), ('theta', 'alpha'), ('rho', 'alpha')]:
            r, p = stats.pearsonr(merged[p1], merged[p2])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {p1} vs {p2}: r={r:.3f}, p={p:.4f} {sig}")

        return merged
    else:
        print("\nWARNING: No agents with all three parameters!")
        return None

# ============================================================================
# CONDITIONAL THRESHOLD ANALYSIS
# ============================================================================

def simulate_single_population():
    """Simulate one agent population"""
    df = pd.read_csv("../data/hierarchical_agents.csv")

    with open("../data/demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)

    results = {'theta': [], 'alpha': [], 'rho': [], 'diet': [],
               'alpha_source': [], 'rho_source': []}

    for idx, row in df.iterrows():
        theta = row['theta']

        # Sample alpha
        if row['has_alpha'] and not pd.isna(row['alpha']):
            alpha, alpha_source = row['alpha'], 'empirical'
        else:
            demo_key = tuple([row['gender'], row['age_group'], row['incquart'], row['educlevel']])
            alpha = sample_from_pmf(demo_key, pmf_tables, 'alpha', theta=theta)
            alpha_source = 'sampled'

        # Sample rho
        if row['has_rho'] and not pd.isna(row['rho']):
            rho, rho_source = row['rho'], 'empirical'
        else:
            demo_key = tuple([row['gender'], row['age_group'], row['incquart'], row['educlevel']])
            rho = sample_from_pmf(demo_key, pmf_tables, 'rho', theta=theta)
            rho_source = 'sampled'

        results['theta'].append(theta)
        results['alpha'].append(alpha)
        results['rho'].append(rho)
        results['diet'].append(row['diet'])
        results['alpha_source'].append(alpha_source)
        results['rho_source'].append(rho_source)

    return pd.DataFrame(results)

def analyze_conditional_thresholds(df):
    """Analyze effective thresholds stratified by theta bins"""
    print("\n" + "="*80)
    print("CONDITIONAL EFFECTIVE THRESHOLD ANALYSIS")
    print("="*80)

    # Define theta bins
    theta_bins = [-1.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    theta_labels = ['(-1.0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1.0]']

    df['theta_bin'] = pd.cut(df['theta'], bins=theta_bins, labels=theta_labels, include_lowest=True)
    df['dissonance'] = np.where(df['diet'] == 'meat', df['theta'], 1 - df['theta'])
    df['eff_threshold'] = df['rho'] - df['alpha'] * df['dissonance']

    print("\nEffective thresholds for MEAT EATERS by theta bin:")
    print(f"{'Theta Bin':<15} {'N':>6} {'Mean θ':>8} {'Mean ρ':>8} {'Mean α':>8} {'Mean Eff.Thresh':>16}")
    print("-"*80)

    meat_eaters = df[df['diet'] == 'meat']
    for bin_label in theta_labels:
        bin_data = meat_eaters[meat_eaters['theta_bin'] == bin_label]
        if len(bin_data) > 0:
            print(f"{bin_label:<15} {len(bin_data):>6} {bin_data['theta'].mean():>8.3f} "
                  f"{bin_data['rho'].mean():>8.3f} {bin_data['alpha'].mean():>8.3f} "
                  f"{bin_data['eff_threshold'].mean():>16.3f}")

    # Check if low-theta meat eaters have higher thresholds
    print("\n" + "="*80)
    print("KEY FINDING: Do meat eaters with low theta have higher rho?")
    print("="*80)

    low_theta_meat = meat_eaters[meat_eaters['theta'] < 0.4]
    high_theta_meat = meat_eaters[meat_eaters['theta'] >= 0.6]

    if len(low_theta_meat) > 0 and len(high_theta_meat) > 0:
        print(f"\nLow theta meat eaters (θ < 0.4, n={len(low_theta_meat)}):")
        print(f"  Mean theta: {low_theta_meat['theta'].mean():.3f}")
        print(f"  Mean rho:   {low_theta_meat['rho'].mean():.3f}")
        print(f"  Mean eff. threshold: {low_theta_meat['eff_threshold'].mean():.3f}")

        print(f"\nHigh theta meat eaters (θ >= 0.6, n={len(high_theta_meat)}):")
        print(f"  Mean theta: {high_theta_meat['theta'].mean():.3f}")
        print(f"  Mean rho:   {high_theta_meat['rho'].mean():.3f}")
        print(f"  Mean eff. threshold: {high_theta_meat['eff_threshold'].mean():.3f}")

        rho_diff = low_theta_meat['rho'].mean() - high_theta_meat['rho'].mean()
        thresh_diff = low_theta_meat['eff_threshold'].mean() - high_theta_meat['eff_threshold'].mean()

        print(f"\nDifference (low - high):")
        print(f"  Rho difference: {rho_diff:+.3f}")
        print(f"  Eff. threshold difference: {thresh_diff:+.3f}")

        if rho_diff > 0.05:
            print(f"\n  ✓ SUCCESS: Low-theta meat eaters have {rho_diff:.3f} higher rho!")
        else:
            print(f"\n  ✗ CONCERN: Low-theta meat eaters don't have significantly higher rho")

    return df

# ============================================================================
# COMPLETE CASES REPRESENTATIVENESS
# ============================================================================

def check_complete_cases_representativeness():
    """Check if complete cases are demographically representative"""
    print("\n" + "="*80)
    print("COMPLETE CASES DEMOGRAPHIC REPRESENTATIVENESS")
    print("="*80)

    # Load surveys
    theta_df = pd.read_excel('../data/theta_diet_demographics.xlsx')
    rho_df = pd.read_excel('../data/rho_demographics.xlsx')
    alpha_df = pd.read_excel('../data/alpha_demographics.xlsx')

    # Clean theta
    theta_clean = pd.DataFrame({
        'id': theta_df['id'],
        'theta': pd.to_numeric(theta_df['Personal Preference for Veg Diet'], errors='coerce'),
        'age': pd.to_numeric(theta_df['Age of the household member'], errors='coerce'),
        'gender': theta_df['Gender'],
        'incquart': theta_df['Income Quartile'],
        'educlevel': theta_df['Education Level']
    }).dropna()

    rho_clean = pd.DataFrame({
        'id': rho_df['id'],
        'rho': pd.to_numeric(rho_df['Cost parameter (rho)'], errors='coerce'),
    }).dropna()

    alpha_clean = pd.DataFrame({
        'id': alpha_df['id'],
        'alpha': pd.to_numeric(alpha_df['Self-identity weight (alpha)'], errors='coerce'),
    }).dropna()

    # Find complete cases
    common_ids = set(theta_clean['id']) & set(rho_clean['id']) & set(alpha_clean['id'])
    complete_cases = theta_clean[theta_clean['id'].isin(common_ids)].copy()

    print(f'\nSample sizes:')
    print(f'  Full theta survey: {len(theta_clean)}')
    print(f'  Complete cases (theta + rho + alpha): {len(complete_cases)}')
    print(f'  Percentage retained: {len(complete_cases)/len(theta_clean)*100:.1f}%')

    # Create age groups
    theta_clean['age_group'] = pd.cut(theta_clean['age'], bins=[17,29,39,49,59,69,120],
                                       labels=['18-29','30-39','40-49','50-59','60-69','70+'])
    complete_cases['age_group'] = pd.cut(complete_cases['age'], bins=[17,29,39,49,59,69,120],
                                          labels=['18-29','30-39','40-49','50-59','60-69','70+'])

    # Check age distribution
    print(f'\n{"="*80}')
    print('AGE GROUP DISTRIBUTION')
    print(f'{"="*80}')
    print(f"{'Category':<15} {'Full Survey':>15} {'Complete Cases':>15} {'Difference':>15}")
    print('-'*80)

    issues = []
    for ag in ['18-29','30-39','40-49','50-59','60-69','70+']:
        full_pct = (theta_clean['age_group'] == ag).mean() * 100
        comp_pct = (complete_cases['age_group'] == ag).mean() * 100
        diff = comp_pct - full_pct
        print(f'{ag:<15} {full_pct:>14.1f}% {comp_pct:>14.1f}% {diff:>+14.1f}%')

        if abs(diff) > 5:
            issues.append(f'Age {ag}: {abs(diff):.1f}% difference')

    # Assessment
    print(f'\n{"="*80}')
    print('REPRESENTATIVENESS ASSESSMENT')
    print(f'{"="*80}')
    print(f'\nUsing +/- 5% as acceptable deviation threshold:\n')

    if not issues:
        print('✓ ALL DEMOGRAPHIC CATEGORIES WITHIN +/-5% THRESHOLD')
        print('✓ COMPLETE CASES SAMPLE IS DEMOGRAPHICALLY REPRESENTATIVE')
    else:
        print('✗ SOME DEMOGRAPHIC CATEGORIES EXCEED +/-5% THRESHOLD:')
        for issue in issues:
            print(f'  - {issue}')
        print(f'\n  Sample size: n={len(complete_cases)}')
        print('\n==> Complete cases are NOT demographically representative')

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run all diagnostic analyses"""
    print("\n" + "#"*80)
    print("# PARAMETER DIAGNOSTICS SUITE")
    print("# Consolidated analysis of parameter sampling and representativeness")
    print("#"*80 + "\n")

    # Analysis 1: Hierarchical agents
    df = analyze_hierarchical_csv()

    # Analysis 2: PMF tables
    pmf_tables = analyze_pmf_tables()

    # Analysis 3: Parameter correlations
    merged_data = analyze_parameter_correlations()

    # Analysis 4: Conditional thresholds
    print("\n\nSimulating population for threshold analysis...")
    sim_df = simulate_single_population()
    analyze_conditional_thresholds(sim_df)

    # Analysis 5: Complete cases representativeness
    check_complete_cases_representativeness()

    print("\n" + "#"*80)
    print("# DIAGNOSTICS COMPLETE")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
