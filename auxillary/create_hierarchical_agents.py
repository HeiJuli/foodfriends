#!/usr/bin/env python3
import pandas as pd
import pickle
from scipy import stats
import numpy as np

def load_and_match_surveys():
    """Load raw Excel files and create matched datasets"""
    
    # Load theta (base population)
    theta_raw = pd.read_excel("../data/theta_diet_demographics.xlsx")
    print(f"Theta raw shape: {theta_raw.shape}")
    print(f"Theta columns: {theta_raw.columns.tolist()}")
    
    print(f"Diet column values: {theta_raw['Diet - Vegan or Not'].unique()}")
    
    theta_df = pd.DataFrame({
        'id': theta_raw['id'],
        'theta': pd.to_numeric(theta_raw['Personal Preference for Veg Diet'], errors='coerce'),
        'diet': theta_raw['Diet - Vegan or Not'],  # Already has 'meat'/'veg' values
        'gender': theta_raw['Gender'],
        'age': theta_raw['Age of the household member'],
        'incquart': theta_raw['Income Quartile'], 
        'educlevel': theta_raw['Education Level']
    })
    
    print(f"Theta before dropna: {len(theta_df)}")
    print(f"NaN counts: {theta_df.isnull().sum().to_dict()}")
    theta_df = theta_df.dropna()
    print(f"Theta after dropna: {len(theta_df)}")
    
    theta_df['age_group'] = pd.cut(theta_df['age'], bins=[17,29,39,49,59,69,120],
                                  labels=['18-29','30-39','40-49','50-59','60-69','70+'])
    
    # Load rho
    rho_raw = pd.read_excel("../data/rho_demographics.xlsx")
    print(f"Rho raw shape: {rho_raw.shape}")
    
    rho_df = pd.DataFrame({
        'id': rho_raw['id'],
        'rho': pd.to_numeric(rho_raw['Cost parameter (rho)'], errors='coerce')
    }).dropna()
    
    # Load alpha
    alpha_raw = pd.read_excel("../data/alpha_demographics.xlsx")
    print(f"Alpha raw shape: {alpha_raw.shape}")
    
    alpha_df = pd.DataFrame({
        'id': alpha_raw['id'],
        'alpha': pd.to_numeric(alpha_raw['Self-identity weight (alpha)'], errors='coerce')
    }).dropna()
    
    # Create match flags
    theta_df['has_rho'] = theta_df['id'].isin(rho_df['id'])
    theta_df['has_alpha'] = theta_df['id'].isin(alpha_df['id'])
    
    # Merge empirical values
    theta_df = theta_df.merge(rho_df[['id','rho']], on='id', how='left')
    theta_df = theta_df.merge(alpha_df[['id','alpha']], on='id', how='left')
    
    print(f"Survey matching completed:")
    print(f"  Base theta: {len(theta_df)}")
    print(f"  Has rho: {theta_df['has_rho'].sum()}")
    print(f"  Has alpha: {theta_df['has_alpha'].sum()}")
    print(f"  Has both: {(theta_df['has_rho'] & theta_df['has_alpha']).sum()}")
    
    return theta_df

def create_hierarchical_sample(theta_df, target_size=None):
    """Create demographically stratified sample with parameter hierarchy"""
    
    demo_vars = ['gender','age_group','incquart','educlevel']
    
    # Calculate target demographics
    if target_size and target_size < len(theta_df):
        scale = target_size / len(theta_df)
        target_demo = (theta_df.groupby(demo_vars).size() * scale).round().astype(int)
    else:
        target_demo = theta_df.groupby(demo_vars).size()
    
    selected = []
    stats = {'triple': 0, 'double': 0, 'single': 0}
    
    for demo_combo, target_n in target_demo.items():
        if target_n == 0: continue
        
        # Get cell
        cell_mask = (theta_df[demo_vars] == demo_combo).all(axis=1)
        cell = theta_df[cell_mask]
        
        # Priority groups
        triple = cell[cell['has_rho'] & cell['has_alpha']]
        double = cell[cell['has_rho'] & ~cell['has_alpha']] 
        single = cell[~cell['has_rho']]
        
        # Fill quota hierarchically
        n_triple = min(len(triple), target_n)
        n_double = min(len(double), target_n - n_triple)
        n_single = target_n - n_triple - n_double
        
        for df, n, typ in [(triple, n_triple, 'triple'), (double, n_double, 'double'), (single, n_single, 'single')]:
            if n > 0:
                batch = df.sample(n, random_state=42)
                selected.append(batch)
                stats[typ] += n
    
    final_df = pd.concat(selected, ignore_index=True)
    final_df['nomem_encr'] = final_df['id']  # Match existing model expectation
    
    print(f"Hierarchical sample: {len(final_df)} agents")
    print(f"  Triple: {stats['triple']} | Double: {stats['double']} | Single: {stats['single']}")
    
    return final_df

def save_hierarchical_csv():
    """Main function - create and save hierarchical agents CSV"""
    theta_df = load_and_match_surveys()
    agents_df = create_hierarchical_sample(theta_df)
    
    # Select final columns for model
    output_cols = ['nomem_encr','theta','diet','has_rho','has_alpha','rho','alpha',
                   'gender','age_group','incquart','educlevel']
    agents_df[output_cols].to_csv("../data/hierarchical_agents.csv", index=False)
    print("Saved hierarchical_agents.csv")

if __name__ == "__main__":
    save_hierarchical_csv()