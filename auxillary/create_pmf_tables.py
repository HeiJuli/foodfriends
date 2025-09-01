#!/usr/bin/env python3
import pandas as pd
import pickle

# OUTPUT FORMAT: 'pkl' (default) or 'csv' 
# Change to 'csv' to output separate CSV files like original script
OUTPUT_FORMAT = 'pkl'

def create_pmf_tables():
    """Create PMF tables from raw Excel survey data using original methodology"""
    
    # Load and process alpha data
    alpha_data = pd.read_excel("../data/alpha_demographics.xlsx")
    alpha_clean = pd.DataFrame({
        'alpha': alpha_data['Self-identity weight (alpha)'].dropna(),
        'age': alpha_data['Age of the household member'].dropna(),
        'incquart': alpha_data['Income Quartile'].dropna(),
        'educlevel': alpha_data['Education Level'].dropna(),
        'gender': alpha_data['Gender'].dropna()
    })
    alpha_clean['age_group'] = pd.cut(alpha_clean['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                     labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    alpha_clean = alpha_clean.dropna(subset=['alpha', 'gender', 'age_group', 'incquart', 'educlevel'])
    
    # Load and process rho data
    rho_data = pd.read_excel("../data/rho_demographics.xlsx")
    rho_clean = pd.DataFrame({
        'rho': rho_data['Cost parameter (rho)'].dropna(),
        'age': rho_data['Age of the household member'].dropna(),
        'incquart': rho_data['Income Quartile'].dropna(),
        'educlevel': rho_data['Education Level'].dropna(),
        'gender': rho_data['Gender'].dropna()
    })
    rho_clean['age_group'] = pd.cut(rho_clean['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                   labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    rho_clean = rho_clean.dropna(subset=['rho', 'gender', 'age_group', 'incquart', 'educlevel'])
    
    # Load and process theta data
    theta_data = pd.read_excel("../data/theta_diet_demographics.xlsx")
    theta_clean = pd.DataFrame({
        'theta': theta_data['Personal Preference for Veg Diet'].dropna(),
        'age': theta_data['Age of the household member'].dropna(),
        'incquart': theta_data['Income Quartile'].dropna(),
        'educlevel': theta_data['Education Level'].dropna(),
        'gender': theta_data['Gender'].dropna()
    })
    theta_clean['age_group'] = pd.cut(theta_clean['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                     labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    theta_clean = theta_clean.dropna(subset=['theta', 'gender', 'age_group', 'incquart', 'educlevel'])
    
    demo_vars = ['gender', 'age_group', 'incquart', 'educlevel']
    
    if OUTPUT_FORMAT == 'csv':
        # Original CSV output format
        for param, data in [('alpha', alpha_clean), ('rho', rho_clean), ('theta', theta_clean)]:
            print(f"Processing {param}...")
            
            # Compute empirical PMFs using original methodology
            grouped_pmf = (
                data.groupby(demo_vars)[param]
                .value_counts(normalize=True)
                .unstack()
                .fillna(0)
            )
            
            # Rename columns with "pmf for ..." labels
            grouped_pmf.columns = [f"pmf for {round(val, 2)}" for val in grouped_pmf.columns]
            
            # Save as CSV
            filename = f"{param}_empirical_pmfs_by_group.csv"
            grouped_pmf.reset_index().to_csv(filename, index=False)
            print(f"  Saved {filename}")
        
        print(f"\n✅ Saved CSV files for all parameters")
        
    else:
        # PKL output format
        pmf_tables = {}
        
        for param, data in [('alpha', alpha_clean), ('rho', rho_clean), ('theta', theta_clean)]:
            print(f"Processing {param}...")
            
            # Compute empirical PMFs using original methodology
            grouped_pmf = (
                data.groupby(demo_vars)[param]
                .value_counts(normalize=True)
                .unstack()
                .fillna(0)
            )
            
            # Convert to PMF dictionary format
            pmf_dict = {}
            for demo_combo in grouped_pmf.index:
                # Get non-zero probabilities
                probs = grouped_pmf.loc[demo_combo]
                nonzero_mask = probs > 0
                
                if nonzero_mask.any():
                    vals = list(probs.index[nonzero_mask])
                    probs_list = list(probs[nonzero_mask])
                    pmf_dict[demo_combo] = {
                        'values': vals,
                        'probabilities': probs_list
                    }
            
            pmf_tables[param] = pmf_dict
            print(f"  Created {len(pmf_dict)} demographic cells")
        
        # Save tables
        with open("demographic_pmfs.pkl", 'wb') as f:
            pickle.dump(pmf_tables, f)
        
        print("\n✅ Saved demographic_pmfs.pkl")
        
        # Validation
        for param in ['alpha', 'rho', 'theta']:
            cells = pmf_tables[param]
            means = [sum(v * p for v, p in zip(cell['values'], cell['probabilities'])) 
                    for cell in cells.values()]
            print(f"{param}: {len(cells)} cells, mean range [{min(means):.3f}, {max(means):.3f}]")

if __name__ == "__main__":
    # To output CSV files instead of pkl, change OUTPUT_FORMAT = 'csv' at top of script
    create_pmf_tables()