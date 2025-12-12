#!/usr/bin/env python3
"""
Diagnostic script to analyze parameter sampling strategy
Investigates how alpha, theta, rho are assigned to agents and identifies
potential issues causing rapid vegetarian uptake
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
        all_means = []
        all_stds = []
        for cell_data in cells.values():
            vals = np.array(cell_data['values'])
            probs = np.array(cell_data['probabilities'])
            mean = np.sum(vals * probs)
            variance = np.sum((vals - mean)**2 * probs)
            all_means.append(mean)
            all_stds.append(np.sqrt(variance))

        print(f"  Mean across cells: {np.mean(all_means):.3f} ± {np.std(all_means):.3f}")
        print(f"  Mean std within cells: {np.mean(all_stds):.3f}")
        print(f"  Range of cell means: [{min(all_means):.3f}, {max(all_means):.3f}]")

    return pmf_tables

def analyze_parameter_correlations():
    """Check for correlations between parameters in original data"""
    print("\n" + "="*80)
    print("PARAMETER CORRELATION ANALYSIS")
    print("="*80)

    # Load raw surveys
    theta_df = pd.read_excel("../data/theta_diet_demographics.xlsx")
    rho_df = pd.read_excel("../data/rho_demographics.xlsx")
    alpha_df = pd.read_excel("../data/alpha_demographics.xlsx")

    # Clean and prepare
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

    # Find agents with all three parameters
    common_ids = set(theta_clean['id']) & set(rho_clean['id']) & set(alpha_clean['id'])
    print(f"\nAgents with all 3 parameters: {len(common_ids)}")

    if len(common_ids) > 0:
        # Merge for correlation analysis
        merged = theta_clean[theta_clean['id'].isin(common_ids)].copy()
        merged = merged.merge(rho_clean[rho_clean['id'].isin(common_ids)], on='id')
        merged = merged.merge(alpha_clean[alpha_clean['id'].isin(common_ids)], on='id')

        print(f"\nCorrelation matrix (n={len(merged)}):")
        corr_matrix = merged[['theta', 'rho', 'alpha']].corr()
        print(corr_matrix.round(3))

        # Statistical significance
        print(f"\nPearwise correlations with p-values:")
        for p1, p2 in [('theta', 'rho'), ('theta', 'alpha'), ('rho', 'alpha')]:
            r, p = stats.pearsonr(merged[p1], merged[p2])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {p1} vs {p2}: r={r:.3f}, p={p:.4f} {sig}")

        return merged
    else:
        print("\nWARNING: No agents with all three parameters!")
        return None

def simulate_agent_initialization(n_runs=5):
    """Simulate the agent initialization process to see parameter distributions"""
    print("\n" + "="*80)
    print("AGENT INITIALIZATION SIMULATION")
    print("="*80)

    df = pd.read_csv("../data/hierarchical_agents.csv")

    with open("../data/demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)

    print(f"\nSimulating parameter assignment for {len(df)} agents...")

    # Simulate the filling process
    results = []
    for run in range(n_runs):
        alpha_values = []
        rho_values = []
        sources = []  # Track if param was empirical or sampled

        for idx, row in df.iterrows():
            # Simulate alpha assignment
            if row['has_alpha'] and not pd.isna(row['alpha']):
                alpha = row['alpha']
                alpha_source = 'empirical'
            else:
                # Sample from PMF
                demo_key = tuple([row['gender'], row['age_group'], row['incquart'], row['educlevel']])
                if demo_key in pmf_tables['alpha']:
                    pmf = pmf_tables['alpha'][demo_key]
                    vals, probs = pmf['values'], pmf['probabilities']
                    alpha = np.random.choice(vals, p=np.array(probs)/sum(probs))
                else:
                    # Fallback
                    all_vals = []
                    for cell in pmf_tables['alpha'].values():
                        all_vals.extend(cell['values'])
                    alpha = np.random.choice(all_vals) if all_vals else 0.5
                alpha_source = 'sampled'

            # Simulate rho assignment
            if row['has_rho'] and not pd.isna(row['rho']):
                rho = row['rho']
                rho_source = 'empirical'
            else:
                demo_key = tuple([row['gender'], row['age_group'], row['incquart'], row['educlevel']])
                if demo_key in pmf_tables['rho']:
                    pmf = pmf_tables['rho'][demo_key]
                    vals, probs = pmf['values'], pmf['probabilities']
                    rho = np.random.choice(vals, p=np.array(probs)/sum(probs))
                else:
                    all_vals = []
                    for cell in pmf_tables['rho'].values():
                        all_vals.extend(cell['values'])
                    rho = np.random.choice(all_vals) if all_vals else 0.5
                rho_source = 'sampled'

            alpha_values.append(alpha)
            rho_values.append(rho)
            sources.append((alpha_source, rho_source))

        results.append({
            'alpha': alpha_values,
            'rho': rho_values,
            'sources': sources
        })

    # Analyze results
    print(f"\nResults across {n_runs} simulation runs:")
    for param in ['alpha', 'rho']:
        all_vals = np.concatenate([r[param] for r in results])
        print(f"\n{param}:")
        print(f"  mean: {np.mean(all_vals):.3f} ± {np.std(all_vals):.3f}")
        print(f"  median: {np.median(all_vals):.3f}")
        print(f"  range: [{np.min(all_vals):.3f}, {np.max(all_vals):.3f}]")

    # Source breakdown
    sources_flat = [s for r in results for s in r['sources']]
    alpha_empirical = sum(1 for s in sources_flat if s[0] == 'empirical')
    rho_empirical = sum(1 for s in sources_flat if s[1] == 'empirical')
    total = len(sources_flat)

    print(f"\nParameter sources:")
    print(f"  Alpha: {alpha_empirical/total*100:.1f}% empirical, {(1-alpha_empirical/total)*100:.1f}% sampled")
    print(f"  Rho: {rho_empirical/total*100:.1f}% empirical, {(1-rho_empirical/total)*100:.1f}% sampled")

    return results

def check_rapid_uptake_indicators():
    """Check for parameter combinations that might cause rapid uptake"""
    print("\n" + "="*80)
    print("RAPID UPTAKE DIAGNOSTIC")
    print("="*80)

    df = pd.read_csv("../data/hierarchical_agents.csv")

    # Calculate effective switching probabilities for meat eaters
    # In the model: U_i = theta * alpha - rho
    # Positive U_i favors vegetarian, negative favors meat
    # For meat eaters with low theta, high alpha creates instability

    print("\nAnalyzing initial conditions that favor rapid switching...")

    # Get empirical parameter distributions
    theta_emp = df[df['theta'].notna()]['theta']
    rho_emp = df[df['has_rho']]['rho']
    alpha_emp = df[df['has_alpha']]['alpha']

    print(f"\nKey parameter combinations:")
    print(f"  Average theta (preference): {theta_emp.mean():.3f}")
    print(f"  Average alpha (self-weight): {alpha_emp.mean():.3f}")
    print(f"  Average rho (threshold): {rho_emp.mean():.3f}")
    print(f"  Average beta (social-weight): {1-alpha_emp.mean():.3f}")

    # For meat eaters (theta < 0.5), check utility
    meat_eaters = df[df['diet'] == 'meat']
    print(f"\nMeat eaters (n={len(meat_eaters)}):")
    print(f"  Average theta: {meat_eaters['theta'].mean():.3f}")

    # Estimate typical dissonance for meat eaters
    # Fixed: Dissonance = theta for meat eaters (distance from meat end = 0)
    avg_dissonance = meat_eaters['theta'].mean()  # Fixed: was 1-theta
    print(f"  Average dissonance (theta): {avg_dissonance:.3f}")
    print(f"  Dissonance * alpha: {avg_dissonance * alpha_emp.mean():.3f}")
    print(f"  Effective threshold (rho - alpha*dissonance): {rho_emp.mean() - alpha_emp.mean()*avg_dissonance:.3f}")

    if (rho_emp.mean() - alpha_emp.mean()*avg_dissonance) < 0.15:
        print("\n  WARNING: Effective threshold is LOW - agents highly susceptible to switching!")

    print(f"\nVegetarian fraction: {(df['diet']=='veg').mean():.3f}")

def main():
    """Run all diagnostic analyses"""
    print("\n" + "#"*80)
    print("# PARAMETER SAMPLING DIAGNOSTIC REPORT")
    print("# Investigating rapid vegetarian uptake issue")
    print("#"*80 + "\n")

    # Run analyses
    df = analyze_hierarchical_csv()
    pmf_tables = analyze_pmf_tables()
    merged_data = analyze_parameter_correlations()
    results = simulate_agent_initialization(n_runs=5)
    check_rapid_uptake_indicators()

    print("\n" + "#"*80)
    print("# DIAGNOSTIC COMPLETE")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
