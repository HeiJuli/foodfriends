# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 11:57:19 2025

@author: emma.thill
@author: jordan.veverall
"""


import pandas as pd
import pickle

def create_pmf_tables():
    """Create PMF tables from survey data"""
    
    # Load and clean surveys
    def load_survey(file, param_col, param_name):
        data = pd.read_excel(file) if file.endswith('.xlsx') else pd.read_csv(file)
        clean = data[[param_col, 'Gender', 'Age of the household member', 
                     'Income Quartile', 'Education Level']].copy()
        clean.columns = [param_name, 'gender', 'age', 'incquart', 'educlevel']
        clean[param_name] = pd.to_numeric(clean[param_name], errors='coerce')
        clean = clean.dropna()
        clean['age_group'] = pd.cut(clean['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                   labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
        return clean
    
    alpha_data = load_survey("alpha_demographics.xlsx", 'Self-identity weight (alpha)', 'alpha')
    rho_data = load_survey("rho_demographics.xlsx", 'Cost parameter (rho)', 'rho')
    theta_data = load_survey("theta_diet_demographics.xlsx", 'Personal Preference for Veg Diet', 'theta')
    
    demo_vars = ['gender', 'age_group', 'incquart', 'educlevel']
    pmf_tables = {}
    
    # Create PMF for each parameter
    for param, data in [('alpha', alpha_data), ('rho', rho_data), ('theta', theta_data)]:
        pmf_dict = {}
        grouped = data.groupby(demo_vars)[param].value_counts(normalize=True).unstack().fillna(0)
        
        for demo_combo in grouped.index:
            vals = list(grouped.columns)
            probs = list(grouped.loc[demo_combo])
            pmf_dict[demo_combo] = {'values': vals, 'probabilities': probs}
        
        pmf_tables[param] = pmf_dict
        print(f"Created PMF for {param}: {len(pmf_dict)} cells")
    
    # Save tables
    with open("demographic_pmfs.pkl", 'wb') as f:
        pickle.dump(pmf_tables, f)
    
    print("Saved demographic_pmfs.pkl")

if __name__ == "__main__":
    create_pmf_tables()