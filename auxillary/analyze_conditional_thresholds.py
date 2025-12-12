#!/usr/bin/env python3
"""
Conditional threshold analysis

The average effective threshold doesn't tell the full story. What matters is
the CONDITIONAL effective threshold - do meat eaters (low theta) have higher
effective thresholds than vegetarians (high theta)?

This script analyzes effective thresholds stratified by theta bins.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../model_src')
from model_main_single import sample_from_pmf

def simulate_single_population():
    """Simulate one agent population"""
    df = pd.read_csv("../data/hierarchical_agents.csv")

    with open("../data/demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)

    results = {
        'theta': [],
        'alpha': [],
        'rho': [],
        'diet': [],
        'alpha_source': [],
        'rho_source': []
    }

    for idx, row in df.iterrows():
        theta = row['theta']

        # Sample alpha
        if row['has_alpha'] and not pd.isna(row['alpha']):
            alpha = row['alpha']
            alpha_source = 'empirical'
        else:
            demo_key = tuple([row['gender'], row['age_group'], row['incquart'], row['educlevel']])
            alpha = sample_from_pmf(demo_key, pmf_tables, 'alpha', theta=theta)
            alpha_source = 'sampled'

        # Sample rho
        if row['has_rho'] and not pd.isna(row['rho']):
            rho = row['rho']
            rho_source = 'empirical'
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
    print("="*80)
    print("CONDITIONAL EFFECTIVE THRESHOLD ANALYSIS")
    print("="*80)

    # Define theta bins
    theta_bins = [-1.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    theta_labels = ['(-1.0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1.0]']

    df['theta_bin'] = pd.cut(df['theta'], bins=theta_bins, labels=theta_labels, include_lowest=True)

    # Calculate effective thresholds
    df['dissonance'] = np.where(df['diet'] == 'meat', 1 - df['theta'], df['theta'])
    df['eff_threshold'] = df['rho'] - df['alpha'] * df['dissonance']

    print("\nEffective thresholds by theta bin (all agents):")
    print(f"{'Theta Bin':<15} {'N':>6} {'Mean θ':>8} {'Mean ρ':>8} {'Mean α':>8} {'Mean Eff.Thresh':>16}")
    print("-"*80)

    for bin_label in theta_labels:
        bin_data = df[df['theta_bin'] == bin_label]
        if len(bin_data) > 0:
            print(f"{bin_label:<15} {len(bin_data):>6} {bin_data['theta'].mean():>8.3f} "
                  f"{bin_data['rho'].mean():>8.3f} {bin_data['alpha'].mean():>8.3f} "
                  f"{bin_data['eff_threshold'].mean():>16.3f}")

    # Analyze meat eaters specifically
    meat_eaters = df[df['diet'] == 'meat']

    print(f"\n\nEffective thresholds for MEAT EATERS by theta bin:")
    print(f"{'Theta Bin':<15} {'N':>6} {'Mean θ':>8} {'Mean ρ':>8} {'Mean α':>8} {'Mean Eff.Thresh':>16}")
    print("-"*80)

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
            print(f"  ✓ Theta-rho correlation is WORKING!")
        else:
            print(f"\n  ✗ CONCERN: Low-theta meat eaters don't have significantly higher rho")

    return df

def create_threshold_distribution_plot(df):
    """Plot effective threshold distributions by theta bin"""
    print("\n" + "="*80)
    print("CREATING THRESHOLD DISTRIBUTION PLOTS")
    print("="*80)

    meat_eaters = df[df['diet'] == 'meat']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Rho vs theta scatter
    axes[0, 0].scatter(df['theta'], df['rho'], alpha=0.3, s=10)
    axes[0, 0].set_xlabel('theta (veg preference)', fontsize=10)
    axes[0, 0].set_ylabel('rho (threshold)', fontsize=10)
    axes[0, 0].set_title('Rho vs Theta (all agents)', fontsize=11)
    axes[0, 0].grid(alpha=0.3)

    # Add regression line
    z = np.polyfit(df['theta'], df['rho'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['theta'].min(), df['theta'].max(), 100)
    axes[0, 0].plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2,
                    label=f'y={z[0]:.3f}x+{z[1]:.3f}')
    axes[0, 0].legend(fontsize=9)

    # Plot 2: Effective threshold vs theta
    axes[0, 1].scatter(meat_eaters['theta'], meat_eaters['eff_threshold'], alpha=0.3, s=10)
    axes[0, 1].axhline(y=0.20, color='r', linestyle='--', label='Stability threshold (0.20)')
    axes[0, 1].set_xlabel('theta (veg preference)', fontsize=10)
    axes[0, 1].set_ylabel('Effective threshold', fontsize=10)
    axes[0, 1].set_title('Effective Threshold vs Theta (meat eaters)', fontsize=11)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(fontsize=9)

    # Plot 3: Box plot of rho by theta bin
    theta_labels = ['(-1.0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1.0]']
    box_data_rho = [df[df['theta_bin'] == label]['rho'].dropna() for label in theta_labels]
    axes[1, 0].boxplot(box_data_rho, labels=theta_labels)
    axes[1, 0].set_xlabel('theta bin', fontsize=10)
    axes[1, 0].set_ylabel('rho', fontsize=10)
    axes[1, 0].set_title('Rho Distribution by Theta Bin', fontsize=11)
    axes[1, 0].grid(alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=8)

    # Plot 4: Box plot of effective threshold by theta bin (meat eaters)
    box_data_eff = [meat_eaters[meat_eaters['theta_bin'] == label]['eff_threshold'].dropna()
                    for label in theta_labels]
    axes[1, 1].boxplot(box_data_eff, labels=theta_labels)
    axes[1, 1].axhline(y=0.20, color='r', linestyle='--', label='Stability threshold')
    axes[1, 1].set_xlabel('theta bin', fontsize=10)
    axes[1, 1].set_ylabel('Effective threshold', fontsize=10)
    axes[1, 1].set_title('Eff. Threshold Distribution by Theta Bin (meat eaters)', fontsize=11)
    axes[1, 1].grid(alpha=0.3, axis='y')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()
    plt.savefig('../visualisations_output/conditional_threshold_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to ../visualisations_output/conditional_threshold_analysis.png")
    plt.close()

def main():
    print("\n" + "#"*80)
    print("# CONDITIONAL THRESHOLD ANALYSIS")
    print("# Understanding threshold heterogeneity across theta bins")
    print("#"*80 + "\n")

    # Simulate population
    print("Simulating agent population with theta-stratified PMFs...")
    df = simulate_single_population()
    print(f"✓ Simulated {len(df)} agents")

    # Analyze conditional thresholds
    df = analyze_conditional_thresholds(df)

    # Create plots
    create_threshold_distribution_plot(df)

    print("\n" + "#"*80)
    print("# ANALYSIS COMPLETE")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
