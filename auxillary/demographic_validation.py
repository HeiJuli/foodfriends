#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:12:49 2025

@author: jpoveralls
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def analyze_rho_theta_bias_severity():
    """Analyze how severe the Rho-Theta bias really is"""
    
    print("=== RHO-THETA BIAS SEVERITY ANALYSIS ===")
    
    # Target (theta baseline) vs observed (rho-theta matched)
    target_age = {'18-29': 0.145, '30-39': 0.140, '40-49': 0.145, '50-59': 0.176, '60-69': 0.208, '70+': 0.185}
    observed_age = {'18-29': 0.092, '30-39': 0.115, '40-49': 0.126, '50-59': 0.158, '60-69': 0.240, '70+': 0.269}
    
    target_inc = {1: 0.229, 2: 0.271, 3: 0.237, 4: 0.264}
    observed_inc = {1: 0.168, 2: 0.282, 3: 0.258, 4: 0.292}
    
    print("Age bias analysis:")
    for age in target_age:
        target = target_age[age]
        observed = observed_age[age] 
        bias_factor = observed / target
        print(f"  {age:>6}: {observed:.3f} vs {target:.3f} (factor: {bias_factor:.2f})")
    
    print("\nIncome bias analysis:")  
    for inc in target_inc:
        target = target_inc[inc]
        observed = observed_inc[inc]
        bias_factor = observed / target
        print(f"  Q{inc}: {observed:.3f} vs {target:.3f} (factor: {bias_factor:.2f})")
    
    # Calculate overall bias magnitude
    age_bias_magnitude = np.mean([abs(observed_age[k] - target_age[k]) for k in target_age])
    inc_bias_magnitude = np.mean([abs(observed_inc[k] - target_inc[k]) for k in target_inc])
    
    print(f"\nBias magnitude (mean absolute deviation):")
    print(f"  Age: {age_bias_magnitude:.3f}")
    print(f"  Income: {inc_bias_magnitude:.3f}")
    
    return age_bias_magnitude, inc_bias_magnitude

def create_dual_correction_weights():
    """Create weights to correct both age and income bias"""
    
    print("\n=== DUAL BIAS CORRECTION ===")
    
    # Age weights
    target_age = {'18-29': 0.145, '30-39': 0.140, '40-49': 0.145, '50-59': 0.176, '60-69': 0.208, '70+': 0.185}
    observed_age = {'18-29': 0.092, '30-39': 0.115, '40-49': 0.126, '50-59': 0.158, '60-69': 0.240, '70+': 0.269}
    age_weights = {k: target_age[k]/observed_age[k] for k in target_age}
    
    # Income weights  
    target_inc = {1: 0.229, 2: 0.271, 3: 0.237, 4: 0.264}
    observed_inc = {1: 0.168, 2: 0.282, 3: 0.258, 4: 0.292}
    inc_weights = {k: target_inc[k]/observed_inc[k] for k in target_inc}
    
    print("Age weights:")
    for age, w in age_weights.items():
        print(f"  {age}: {w:.2f}")
        
    print("Income weights:")
    for inc, w in inc_weights.items():
        print(f"  Q{inc}: {w:.2f}")
    
    return age_weights, inc_weights

def test_scientific_importance():
    """Test the scientific importance of rho-theta vs alpha-theta relationships"""
    
    print("\n=== SCIENTIFIC IMPORTANCE ANALYSIS ===")
    
    # Load matched datasets
    alpha_data = pd.read_excel("alpha_demographics.xlsx")
    rho_data = pd.read_excel("rho_demographics.xlsx")  
    theta_data = pd.read_excel("theta_diet_demographics.xlsx")
    
    # Create matches
    at_matched = pd.merge(alpha_data[['id', 'Self-identity weight (alpha)']], 
                         theta_data[['id', 'Personal Preference for Veg Diet']], on='id')
    at_matched.columns = ['id', 'alpha', 'theta']
    
    rt_matched = pd.merge(rho_data[['id', 'Cost parameter (rho)']], 
                         theta_data[['id', 'Personal Preference for Veg Diet']], on='id')
    rt_matched.columns = ['id', 'rho', 'theta']
    
    # Calculate correlations
    at_corr = np.corrcoef(at_matched['alpha'], at_matched['theta'])[0,1]
    rt_corr = np.corrcoef(rt_matched['rho'], rt_matched['theta'])[0,1]
    
    print(f"Relationship strength:")
    print(f"  Alpha-Theta correlation: r = {at_corr:.3f} (n={len(at_matched)})")
    print(f"  Rho-Theta correlation: r = {rt_corr:.3f} (n={len(rt_matched)})")
    print(f"  Rho-Theta is {abs(rt_corr)/abs(at_corr):.1f}x stronger relationship")
    
    
    return at_corr, rt_corr

def compare_bias_correction_complexity():
    """Compare complexity of correcting each strategy's bias"""
    
    print("\n=== BIAS CORRECTION COMPLEXITY ===")
    
    
    age_bias_mag, inc_bias_mag = analyze_rho_theta_bias_severity()
    
    if inc_bias_mag < 0.02:  # Small bias
        print(f"\n✓ Income bias is small ({inc_bias_mag:.3f}) - easily correctable")
        print(f"✓ Recommendation: Use Rho-Theta with dual correction")
    else:
        print(f"\n⚠ Income bias is substantial ({inc_bias_mag:.3f})")
    

def implement_rho_theta_strategy():
    """Implement the scientifically-prioritized Rho-Theta strategy"""
    
    
    age_weights, inc_weights = create_dual_correction_weights()
    

if __name__ == "__main__":
    age_bias_mag, inc_bias_mag = analyze_rho_theta_bias_severity()
    at_corr, rt_corr = test_scientific_importance()
    compare_bias_correction_complexity()
    implement_rho_theta_strategy()