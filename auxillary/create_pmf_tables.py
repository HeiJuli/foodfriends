#!/usr/bin/env python3
import pandas as pd
import pickle
import numpy as np

# OUTPUT FORMAT: 'pkl' (default) or 'csv'
# Change to 'csv' to output separate CSV files like original script
OUTPUT_FORMAT = 'pkl'

# Theta bins for conditional PMF stratification
THETA_BINS = [-1.0, 0.2, 0.4, 0.6, 0.8, 1.0]
THETA_LABELS = ['(-1.0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1.0]']

def create_pmf_tables():
    """Create theta-stratified PMF tables to preserve parameter correlations"""

    # Load theta data first (needed for merging with rho/alpha)
    theta_data = pd.read_excel("../data/theta_diet_demographics.xlsx")
    theta_base = pd.DataFrame({
        'id': theta_data['id'],
        'theta': pd.to_numeric(theta_data['Personal Preference for Veg Diet'], errors='coerce'),
        'age': pd.to_numeric(theta_data['Age of the household member'], errors='coerce'),
        'incquart': theta_data['Income Quartile'],
        'educlevel': theta_data['Education Level'],
        'gender': theta_data['Gender']
    })
    theta_base['age_group'] = pd.cut(theta_base['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                     labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    theta_base = theta_base.dropna(subset=['theta', 'gender', 'age_group', 'incquart', 'educlevel'])

    # Load and process alpha data, merging with theta
    alpha_data = pd.read_excel("../data/alpha_demographics.xlsx")
    alpha_raw = pd.DataFrame({
        'id': alpha_data['id'],
        'alpha': pd.to_numeric(alpha_data['Self-identity weight (alpha)'], errors='coerce'),
        'age': pd.to_numeric(alpha_data['Age of the household member'], errors='coerce'),
        'incquart': alpha_data['Income Quartile'],
        'educlevel': alpha_data['Education Level'],
        'gender': alpha_data['Gender']
    })
    alpha_raw['age_group'] = pd.cut(alpha_raw['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                    labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])

    # Merge with theta to get theta values for alpha respondents
    alpha_clean = alpha_raw.merge(theta_base[['id', 'theta']], on='id', how='inner')
    alpha_clean = alpha_clean.dropna(subset=['alpha', 'theta', 'gender', 'age_group', 'incquart', 'educlevel'])
    alpha_clean['theta_bin'] = pd.cut(alpha_clean['theta'], bins=THETA_BINS, labels=THETA_LABELS, include_lowest=True)
    alpha_clean = alpha_clean.dropna(subset=['theta_bin'])

    # Load and process rho data, merging with theta
    rho_data = pd.read_excel("../data/rho_demographics.xlsx")
    rho_raw = pd.DataFrame({
        'id': rho_data['id'],
        'rho': pd.to_numeric(rho_data['Cost parameter (rho)'], errors='coerce'),
        'age': pd.to_numeric(rho_data['Age of the household member'], errors='coerce'),
        'incquart': rho_data['Income Quartile'],
        'educlevel': rho_data['Education Level'],
        'gender': rho_data['Gender']
    })
    rho_raw['age_group'] = pd.cut(rho_raw['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                  labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])

    # Merge with theta to get theta values for rho respondents
    rho_clean = rho_raw.merge(theta_base[['id', 'theta']], on='id', how='inner')
    rho_clean = rho_clean.dropna(subset=['rho', 'theta', 'gender', 'age_group', 'incquart', 'educlevel'])
    rho_clean['theta_bin'] = pd.cut(rho_clean['theta'], bins=THETA_BINS, labels=THETA_LABELS, include_lowest=True)
    rho_clean = rho_clean.dropna(subset=['theta_bin'])

    # Process theta data (no theta binning for theta itself)
    theta_clean = theta_base.copy()
    
    demo_vars = ['gender', 'age_group', 'incquart', 'educlevel']

    print(f"\n{'='*80}")
    print("THETA-STRATIFIED PMF TABLE GENERATION")
    print(f"{'='*80}\n")
    print(f"Data loaded:")
    print(f"  Alpha: {len(alpha_clean)} observations with theta")
    print(f"  Rho:   {len(rho_clean)} observations with theta")
    print(f"  Theta: {len(theta_clean)} observations")

    # PKL output format with theta stratification
    pmf_tables = {}

    # Process alpha with theta stratification
    print(f"\nProcessing alpha (theta-stratified)...")
    alpha_stratified_vars = demo_vars + ['theta_bin']
    grouped_pmf_alpha = (
        alpha_clean.groupby(alpha_stratified_vars)['alpha']
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    pmf_dict_alpha = {}
    for combo in grouped_pmf_alpha.index:
        probs = grouped_pmf_alpha.loc[combo]
        nonzero_mask = probs > 0

        if nonzero_mask.any():
            vals = list(probs.index[nonzero_mask])
            probs_list = list(probs[nonzero_mask])
            pmf_dict_alpha[combo] = {
                'values': vals,
                'probabilities': probs_list
            }

    pmf_tables['alpha'] = pmf_dict_alpha
    print(f"  Created {len(pmf_dict_alpha)} theta-stratified cells")

    # Process rho with theta stratification
    print(f"Processing rho (theta-stratified)...")
    rho_stratified_vars = demo_vars + ['theta_bin']
    grouped_pmf_rho = (
        rho_clean.groupby(rho_stratified_vars)['rho']
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    pmf_dict_rho = {}
    for combo in grouped_pmf_rho.index:
        probs = grouped_pmf_rho.loc[combo]
        nonzero_mask = probs > 0

        if nonzero_mask.any():
            vals = list(probs.index[nonzero_mask])
            probs_list = list(probs[nonzero_mask])
            pmf_dict_rho[combo] = {
                'values': vals,
                'probabilities': probs_list
            }

    pmf_tables['rho'] = pmf_dict_rho
    print(f"  Created {len(pmf_dict_rho)} theta-stratified cells")

    # Process theta without theta stratification (demographics only)
    print(f"Processing theta (demographics only)...")
    grouped_pmf_theta = (
        theta_clean.groupby(demo_vars)['theta']
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    pmf_dict_theta = {}
    for combo in grouped_pmf_theta.index:
        probs = grouped_pmf_theta.loc[combo]
        nonzero_mask = probs > 0

        if nonzero_mask.any():
            vals = list(probs.index[nonzero_mask])
            probs_list = list(probs[nonzero_mask])
            pmf_dict_theta[combo] = {
                'values': vals,
                'probabilities': probs_list
            }

    pmf_tables['theta'] = pmf_dict_theta
    print(f"  Created {len(pmf_dict_theta)} demographic cells")

    # Save PMF metadata
    pmf_tables['_metadata'] = {
        'theta_bins': THETA_BINS,
        'theta_labels': THETA_LABELS,
        'stratified_params': ['alpha', 'rho'],
        'unstratified_params': ['theta']
    }

    # Save tables
    with open("../data/demographic_pmfs.pkl", 'wb') as f:
        pickle.dump(pmf_tables, f)

    print(f"\n✅ Saved theta-stratified PMF tables to ../data/demographic_pmfs.pkl")

    # Validation
    print(f"\n{'='*80}")
    print("VALIDATION")
    print(f"{'='*80}\n")

    for param in ['alpha', 'rho', 'theta']:
        cells = pmf_tables[param]
        means = [sum(v * p for v, p in zip(cell['values'], cell['probabilities']))
                for cell in cells.values()]
        print(f"{param}:")
        print(f"  Cells: {len(cells)}")
        print(f"  Mean range: [{min(means):.3f}, {max(means):.3f}]")
        print(f"  Avg cell mean: {np.mean(means):.3f} ± {np.std(means):.3f}")

    # Check theta bin distribution for alpha/rho
    print(f"\nTheta bin coverage:")
    for param in ['alpha', 'rho']:
        theta_bins_present = set(key[-1] for key in pmf_tables[param].keys())
        print(f"  {param}: {len(theta_bins_present)}/{len(THETA_LABELS)} bins present")
        print(f"    Bins: {sorted(theta_bins_present)}")

if __name__ == "__main__":
    # To output CSV files instead of pkl, change OUTPUT_FORMAT = 'csv' at top of script
    create_pmf_tables()