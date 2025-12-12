#!/usr/bin/env python3
"""
Validation script to verify theta-stratified PMF sampling preserves correlations

This script simulates the agent initialization process with the new theta-stratified
PMF tables and validates that the theta-rho and theta-alpha correlations are preserved.
"""
import pandas as pd
import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../model_src')
from model_main_single import sample_from_pmf

def load_empirical_correlations():
    """Calculate empirical correlations from complete cases"""
    print("="*80)
    print("LOADING EMPIRICAL CORRELATIONS (Ground Truth)")
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

    merged = theta_clean[theta_clean['id'].isin(common_ids)].copy()
    merged = merged.merge(rho_clean[rho_clean['id'].isin(common_ids)], on='id')
    merged = merged.merge(alpha_clean[alpha_clean['id'].isin(common_ids)], on='id')

    print(f"\nComplete cases: n={len(merged)}")
    print(f"\nEmpirical correlations:")
    corr_matrix = merged[['theta', 'rho', 'alpha']].corr()
    print(corr_matrix.round(3))

    # Statistical tests
    print(f"\nStatistical significance:")
    empirical_corrs = {}
    for p1, p2 in [('theta', 'rho'), ('theta', 'alpha'), ('rho', 'alpha')]:
        r, p = stats.pearsonr(merged[p1], merged[p2])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {p1} vs {p2}: r={r:.3f}, p={p:.4f} {sig}")
        empirical_corrs[f"{p1}_{p2}"] = r

    return empirical_corrs, merged

def simulate_agent_population(n_simulations=10):
    """Simulate full agent population with theta-stratified PMF sampling"""
    print("\n" + "="*80)
    print("SIMULATING AGENT POPULATION WITH THETA-STRATIFIED PMFs")
    print("="*80)

    # Load hierarchical agents CSV
    df = pd.read_csv("../data/hierarchical_agents.csv")

    # Load PMF tables
    with open("../data/demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)

    print(f"\nHierarchical agent breakdown:")
    print(f"  Total agents: {len(df)}")
    print(f"  Has alpha (empirical): {df['has_alpha'].sum()} ({df['has_alpha'].mean()*100:.1f}%)")
    print(f"  Has rho (empirical): {df['has_rho'].sum()} ({df['has_rho'].mean()*100:.1f}%)")
    print(f"  Has both: {(df['has_rho'] & df['has_alpha']).sum()} ({(df['has_rho'] & df['has_alpha']).mean()*100:.1f}%)")

    print(f"\nRunning {n_simulations} simulations...")

    all_results = []

    for sim in range(n_simulations):
        theta_vals = []
        alpha_vals = []
        rho_vals = []
        alpha_sources = []
        rho_sources = []

        for idx, row in df.iterrows():
            theta = row['theta']
            theta_vals.append(theta)

            # Simulate alpha assignment
            if row['has_alpha'] and not pd.isna(row['alpha']):
                alpha = row['alpha']
                alpha_source = 'empirical'
            else:
                # Sample from theta-stratified PMF
                demo_key = tuple([row['gender'], row['age_group'], row['incquart'], row['educlevel']])
                alpha = sample_from_pmf(demo_key, pmf_tables, 'alpha', theta=theta)
                alpha_source = 'sampled'

            alpha_vals.append(alpha)
            alpha_sources.append(alpha_source)

            # Simulate rho assignment
            if row['has_rho'] and not pd.isna(row['rho']):
                rho = row['rho']
                rho_source = 'empirical'
            else:
                # Sample from theta-stratified PMF
                demo_key = tuple([row['gender'], row['age_group'], row['incquart'], row['educlevel']])
                rho = sample_from_pmf(demo_key, pmf_tables, 'rho', theta=theta)
                rho_source = 'sampled'

            rho_vals.append(rho)
            rho_sources.append(rho_source)

        all_results.append({
            'theta': theta_vals,
            'alpha': alpha_vals,
            'rho': rho_vals,
            'alpha_source': alpha_sources,
            'rho_source': rho_sources
        })

    return all_results, df

def analyze_simulated_correlations(simulation_results, empirical_corrs):
    """Analyze correlations from simulated populations"""
    print("\n" + "="*80)
    print("SIMULATED POPULATION CORRELATIONS")
    print("="*80)

    sim_corrs_theta_rho = []
    sim_corrs_theta_alpha = []
    sim_corrs_rho_alpha = []

    for result in simulation_results:
        df_sim = pd.DataFrame({
            'theta': result['theta'],
            'rho': result['rho'],
            'alpha': result['alpha']
        })

        r_theta_rho, _ = stats.pearsonr(df_sim['theta'], df_sim['rho'])
        r_theta_alpha, _ = stats.pearsonr(df_sim['theta'], df_sim['alpha'])
        r_rho_alpha, _ = stats.pearsonr(df_sim['rho'], df_sim['alpha'])

        sim_corrs_theta_rho.append(r_theta_rho)
        sim_corrs_theta_alpha.append(r_theta_alpha)
        sim_corrs_rho_alpha.append(r_rho_alpha)

    print(f"\nCorrelation preservation (mean ± std across {len(simulation_results)} simulations):")
    print(f"\n  theta vs rho:")
    print(f"    Empirical:  {empirical_corrs['theta_rho']:+.3f}")
    print(f"    Simulated:  {np.mean(sim_corrs_theta_rho):+.3f} ± {np.std(sim_corrs_theta_rho):.3f}")
    print(f"    Difference: {abs(np.mean(sim_corrs_theta_rho) - empirical_corrs['theta_rho']):.3f}")

    print(f"\n  theta vs alpha:")
    print(f"    Empirical:  {empirical_corrs['theta_alpha']:+.3f}")
    print(f"    Simulated:  {np.mean(sim_corrs_theta_alpha):+.3f} ± {np.std(sim_corrs_theta_alpha):.3f}")
    print(f"    Difference: {abs(np.mean(sim_corrs_theta_alpha) - empirical_corrs['theta_alpha']):.3f}")

    print(f"\n  rho vs alpha:")
    print(f"    Empirical:  {empirical_corrs['rho_alpha']:+.3f}")
    print(f"    Simulated:  {np.mean(sim_corrs_rho_alpha):+.3f} ± {np.std(sim_corrs_rho_alpha):.3f}")
    print(f"    Difference: {abs(np.mean(sim_corrs_rho_alpha) - empirical_corrs['rho_alpha']):.3f}")

    return {
        'theta_rho': (np.mean(sim_corrs_theta_rho), np.std(sim_corrs_theta_rho)),
        'theta_alpha': (np.mean(sim_corrs_theta_alpha), np.std(sim_corrs_theta_alpha)),
        'rho_alpha': (np.mean(sim_corrs_rho_alpha), np.std(sim_corrs_rho_alpha))
    }

def check_effective_thresholds(simulation_results):
    """Calculate effective thresholds for meat eaters"""
    print("\n" + "="*80)
    print("EFFECTIVE THRESHOLD ANALYSIS")
    print("="*80)

    # Use first simulation for detailed analysis
    result = simulation_results[0]
    df_sim = pd.DataFrame({
        'theta': result['theta'],
        'rho': result['rho'],
        'alpha': result['alpha']
    })

    # Load diet info
    df_hier = pd.read_csv("../data/hierarchical_agents.csv")
    df_sim['diet'] = df_hier['diet']

    # Analyze meat eaters
    meat_eaters = df_sim[df_sim['diet'] == 'meat']

    print(f"\nMeat eaters (n={len(meat_eaters)}):")
    print(f"  Average theta: {meat_eaters['theta'].mean():.3f}")
    print(f"  Average alpha: {meat_eaters['alpha'].mean():.3f}")
    print(f"  Average rho:   {meat_eaters['rho'].mean():.3f}")

    # Calculate effective thresholds
    avg_dissonance = 1 - meat_eaters['theta'].mean()
    eff_threshold = meat_eaters['rho'].mean() - meat_eaters['alpha'].mean() * avg_dissonance

    print(f"\n  Average dissonance (1-theta): {avg_dissonance:.3f}")
    print(f"  Dissonance adjustment (alpha*dissonance): {meat_eaters['alpha'].mean() * avg_dissonance:.3f}")
    print(f"  Effective threshold (rho - alpha*dissonance): {eff_threshold:.3f}")

    if eff_threshold > 0.20:
        print(f"\n  ✓ PASS: Effective threshold > 0.20 (stable initial conditions)")
    else:
        print(f"\n  ✗ FAIL: Effective threshold < 0.20 (unstable, rapid uptake expected)")

    return eff_threshold

def create_validation_plots(simulation_results, empirical_data):
    """Create scatter plots comparing empirical vs simulated correlations"""
    print("\n" + "="*80)
    print("GENERATING VALIDATION PLOTS")
    print("="*80)

    # Use first simulation
    result = simulation_results[0]
    df_sim = pd.DataFrame({
        'theta': result['theta'],
        'rho': result['rho'],
        'alpha': result['alpha']
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # theta vs rho
    axes[0].scatter(empirical_data['theta'], empirical_data['rho'], alpha=0.3, label='Empirical', s=20)
    axes[0].scatter(df_sim['theta'], df_sim['rho'], alpha=0.1, label='Simulated', s=10)
    axes[0].set_xlabel('theta (veg preference)')
    axes[0].set_ylabel('rho (threshold)')
    axes[0].set_title(f"theta vs rho\nEmpirical: r={stats.pearsonr(empirical_data['theta'], empirical_data['rho'])[0]:.3f}\nSimulated: r={stats.pearsonr(df_sim['theta'], df_sim['rho'])[0]:.3f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # theta vs alpha
    axes[1].scatter(empirical_data['theta'], empirical_data['alpha'], alpha=0.3, label='Empirical', s=20)
    axes[1].scatter(df_sim['theta'], df_sim['alpha'], alpha=0.1, label='Simulated', s=10)
    axes[1].set_xlabel('theta (veg preference)')
    axes[1].set_ylabel('alpha (self-identity weight)')
    axes[1].set_title(f"theta vs alpha\nEmpirical: r={stats.pearsonr(empirical_data['theta'], empirical_data['alpha'])[0]:.3f}\nSimulated: r={stats.pearsonr(df_sim['theta'], df_sim['alpha'])[0]:.3f}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # rho vs alpha
    axes[2].scatter(empirical_data['rho'], empirical_data['alpha'], alpha=0.3, label='Empirical', s=20)
    axes[2].scatter(df_sim['rho'], df_sim['alpha'], alpha=0.1, label='Simulated', s=10)
    axes[2].set_xlabel('rho (threshold)')
    axes[2].set_ylabel('alpha (self-identity weight)')
    axes[2].set_title(f"rho vs alpha\nEmpirical: r={stats.pearsonr(empirical_data['rho'], empirical_data['alpha'])[0]:.3f}\nSimulated: r={stats.pearsonr(df_sim['rho'], df_sim['alpha'])[0]:.3f}")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../visualisations_output/theta_stratification_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved validation plot to ../visualisations_output/theta_stratification_validation.png")
    plt.close()

def main():
    """Run complete validation pipeline"""
    print("\n" + "#"*80)
    print("# THETA-STRATIFIED PMF VALIDATION")
    print("# Verifying correlation preservation in agent population")
    print("#"*80 + "\n")

    # Step 1: Load empirical correlations
    empirical_corrs, empirical_data = load_empirical_correlations()

    # Step 2: Simulate agent populations
    simulation_results, hierarchical_df = simulate_agent_population(n_simulations=10)

    # Step 3: Analyze correlations
    sim_corrs = analyze_simulated_correlations(simulation_results, empirical_corrs)

    # Step 4: Check effective thresholds
    eff_threshold = check_effective_thresholds(simulation_results)

    # Step 5: Create plots
    create_validation_plots(simulation_results, empirical_data)

    # Final validation report
    print("\n" + "#"*80)
    print("# VALIDATION SUMMARY")
    print("#"*80)

    theta_rho_preserved = abs(sim_corrs['theta_rho'][0] - empirical_corrs['theta_rho']) < 0.05
    theta_alpha_preserved = abs(sim_corrs['theta_alpha'][0] - empirical_corrs['theta_alpha']) < 0.05
    threshold_stable = eff_threshold > 0.20

    print(f"\nValidation criteria:")
    print(f"  [{'✓' if theta_rho_preserved else '✗'}] theta-rho correlation preserved (diff < 0.05)")
    print(f"  [{'✓' if theta_alpha_preserved else '✗'}] theta-alpha correlation preserved (diff < 0.05)")
    print(f"  [{'✓' if threshold_stable else '✗'}] Effective threshold > 0.20 (stable)")

    if theta_rho_preserved and theta_alpha_preserved and threshold_stable:
        print(f"\n{'='*80}")
        print("✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓")
        print(f"{'='*80}")
        print("\nTheta-stratified PMF sampling successfully preserves correlations!")
        print("Model should now exhibit proper S-curve dynamics with slow initial uptake.")
    else:
        print(f"\n{'='*80}")
        print("✗✗✗ VALIDATION FAILED ✗✗✗")
        print(f"{'='*80}")
        print("\nSome validation criteria not met. Review PMF stratification approach.")

    print()

if __name__ == "__main__":
    main()
