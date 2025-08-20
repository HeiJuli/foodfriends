import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def load_surveys_direct(alpha_file, rho_file, theta_file):
    """Load surveys using known column structure"""
    
    def safe_load(filepath):
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            return pd.read_excel(filepath)
        else:
            return pd.read_csv(filepath)
    
    # Load raw data
    alpha_raw = safe_load(alpha_file)
    rho_raw = safe_load(rho_file) 
    theta_raw = safe_load(theta_file)
    
    print("Raw survey sizes:")
    print(f"  Alpha: {len(alpha_raw)}")
    print(f"  Rho: {len(rho_raw)}")
    print(f"  Theta: {len(theta_raw)}")
    
    # Clean alpha survey
    alpha_clean = alpha_raw[['id', 'Self-identity weight (alpha)', 'Age of the household member', 
                            'Income Quartile', 'Education Level']].copy()
    alpha_clean.columns = ['id', 'alpha', 'age', 'incquart', 'educlevel']
    alpha_clean['alpha'] = pd.to_numeric(alpha_clean['alpha'], errors='coerce')
    alpha_clean = alpha_clean.dropna(subset=['id', 'alpha'])
    alpha_clean['age_group'] = pd.cut(alpha_clean['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                     labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    
    # Clean rho survey  
    rho_clean = rho_raw[['id', 'Cost parameter (rho)', 'Age of the household member',
                        'Income Quartile', 'Education Level']].copy()
    rho_clean.columns = ['id', 'rho', 'age', 'incquart', 'educlevel']
    rho_clean['rho'] = pd.to_numeric(rho_clean['rho'], errors='coerce')
    rho_clean = rho_clean.dropna(subset=['id', 'rho'])
    rho_clean['age_group'] = pd.cut(rho_clean['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                   labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    
    # Clean theta survey
    theta_clean = theta_raw[['id', 'Personal Preference for Veg Diet', 'Age of the household member',
                            'Income Quartile', 'Education Level']].copy()
    theta_clean.columns = ['id', 'theta', 'age', 'incquart', 'educlevel']
    theta_clean['theta'] = pd.to_numeric(theta_clean['theta'], errors='coerce')
    theta_clean = theta_clean.dropna(subset=['id', 'theta'])
    theta_clean['age_group'] = pd.cut(theta_clean['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                     labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    
    print("Cleaned survey sizes:")
    print(f"  Alpha: {len(alpha_clean)} records")
    print(f"  Rho: {len(rho_clean)} records") 
    print(f"  Theta: {len(theta_clean)} records")
    
    return alpha_clean, rho_clean, theta_clean

def analyze_id_overlap(alpha_df, rho_df, theta_df):
    """Analyze actual ID overlap between surveys"""
    
    # Convert all IDs to string for consistent comparison
    alpha_ids = set(alpha_df['id'].astype(str))
    rho_ids = set(rho_df['id'].astype(str)) 
    theta_ids = set(theta_df['id'].astype(str))
    
    print("\n=== ID Overlap Analysis ===")
    print(f"Alpha IDs: {len(alpha_ids)}")
    print(f"Rho IDs: {len(rho_ids)}")
    print(f"Theta IDs: {len(theta_ids)}")
    
    # Pairwise overlaps
    alpha_theta_overlap = alpha_ids & theta_ids
    alpha_rho_overlap = alpha_ids & rho_ids
    rho_theta_overlap = rho_ids & theta_ids
    
    print(f"\nPairwise overlaps:")
    print(f"  Alpha-Theta: {len(alpha_theta_overlap)} ({100*len(alpha_theta_overlap)/len(alpha_ids):.1f}% of alpha)")
    print(f"  Alpha-Rho: {len(alpha_rho_overlap)} ({100*len(alpha_rho_overlap)/len(alpha_ids):.1f}% of alpha)")
    print(f"  Rho-Theta: {len(rho_theta_overlap)} ({100*len(rho_theta_overlap)/len(rho_ids):.1f}% of rho)")
    
    # Three-way overlap
    all_three_overlap = alpha_ids & rho_ids & theta_ids
    print(f"  All three: {len(all_three_overlap)} IDs")
    
    return {
        'alpha_theta': alpha_theta_overlap,
        'alpha_rho': alpha_rho_overlap, 
        'rho_theta': rho_theta_overlap,
        'all_three': all_three_overlap
    }

def create_matched_dataset(alpha_df, rho_df, theta_df, overlaps):
    """Create dataset with ID-matched records and analyze correlations"""
    
    # Convert IDs to string consistently
    alpha_df = alpha_df.copy()
    theta_df = theta_df.copy()
    rho_df = rho_df.copy()
    
    alpha_df['id'] = alpha_df['id'].astype(str)
    theta_df['id'] = theta_df['id'].astype(str)
    rho_df['id'] = rho_df['id'].astype(str)
    
    # Alpha-Theta matches
    alpha_theta_ids = list(overlaps['alpha_theta'])
    alpha_subset = alpha_df[alpha_df['id'].isin(alpha_theta_ids)]
    theta_subset = theta_df[theta_df['id'].isin(alpha_theta_ids)]
    
    matched_alpha_theta = pd.merge(alpha_subset, theta_subset, on='id', 
                                  suffixes=('_alpha', '_theta'), how='inner')
    
    print(f"\n=== Alpha-Theta Matched Dataset ===")
    print(f"Overlapping IDs: {len(alpha_theta_ids)}")
    print(f"Alpha records for these IDs: {len(alpha_subset)}")
    print(f"Theta records for these IDs: {len(theta_subset)}")
    print(f"Successfully matched: {len(matched_alpha_theta)}")
    
    if len(matched_alpha_theta) > 0:
        corr = matched_alpha_theta['alpha'].corr(matched_alpha_theta['theta'])
        print(f"Alpha-Theta correlation: {corr:.3f}")
        
        # Use alpha survey demographics
        matched_alpha_theta['age_group'] = matched_alpha_theta['age_group_alpha']
        matched_alpha_theta['incquart'] = matched_alpha_theta['incquart_alpha']
        matched_alpha_theta['educlevel'] = matched_alpha_theta['educlevel_alpha']
    
    # Try to add rho data
    matched_all_three = None
    if len(overlaps['all_three']) > 0:
        all_three_ids = list(overlaps['all_three'])
        rho_subset = rho_df[rho_df['id'].isin(all_three_ids)]
        
        matched_all_three = pd.merge(
            matched_alpha_theta[['id', 'alpha', 'theta', 'age_group', 'incquart', 'educlevel']], 
            rho_subset[['id', 'rho']], 
            on='id', how='inner'
        )
        
        print(f"\n=== All Three Parameters Matched ===")
        print(f"Three-way matched records: {len(matched_all_three)}")
        
        if len(matched_all_three) > 0:
            corr_matrix = matched_all_three[['alpha', 'rho', 'theta']].corr()
            print(f"Correlation matrix:")
            print(corr_matrix.round(3))
    
    return matched_all_three, matched_alpha_theta

def validate_hybrid_approach(alpha_df, rho_df, theta_df, matched_data, overlaps):
    """Validate hybrid ID-matching + demographic sampling approach"""
    
    print(f"\n=== Hybrid Approach Validation ===")
    
    # Calculate coverage
    total_agents_needed = len(theta_df)
    direct_matches = len(matched_data) if matched_data is not None and not matched_data.empty else 0
    need_demographic_sampling = total_agents_needed - direct_matches
    
    print(f"Population size (theta survey): {total_agents_needed}")
    print(f"Direct ID matches available: {direct_matches} ({100*direct_matches/total_agents_needed:.1f}%)")
    print(f"Need demographic sampling: {need_demographic_sampling} ({100*need_demographic_sampling/total_agents_needed:.1f}%)")
    
    # Test demographic cell coverage for sampling
    demographic_cells = alpha_df.groupby(['age_group', 'incquart', 'educlevel']).size()
    small_cells = (demographic_cells < 5).sum()
    total_cells = len(demographic_cells)
    
    print(f"\nDemographic sampling feasibility:")
    print(f"  Total cells: {total_cells}")
    print(f"  Small cells (<5 obs): {small_cells} ({100*small_cells/total_cells:.1f}%)")
    print(f"  Sampling approach: {'Feasible' if small_cells/total_cells < 0.4 else 'Challenging'}")
    
    return need_demographic_sampling

def recommend_approach(overlaps, matched_data, need_demographic):
    """Provide recommendation for agent parameterization approach"""
    
    print(f"\n=== RECOMMENDATION ===")
    
    # Calculate coverage
    direct_matches = len(matched_data) if matched_data is not None and not matched_data.empty else 0
    total_population = direct_matches + need_demographic
    coverage = direct_matches / max(total_population, 1)
    
    print(f"ID matching coverage: {100*coverage:.1f}%")
    
    if coverage > 0.8:
        print("🎯 RECOMMENDED: ID-matching primary approach")
        print("   - Use direct ID matches for majority of agents")
        print("   - Demographic sampling for remaining agents")
        print("   - Preserves real correlations from survey data")
        
    elif coverage > 0.4:
        print("🎯 RECOMMENDED: Hybrid approach")
        print("   - ID matching for available pairs")
        print("   - Demographic sampling with correlation preservation")
        print("   - Validate correlation structure in matched subset")
        
    else:
        print("🎯 RECOMMENDED: Enhanced demographic approach")
        print("   - Primary: demographic-based PMF sampling")
        print("   - Validation: use matched subset to verify correlations")
        print("   - Consider correlation adjustment based on matched data")
    
    if direct_matches > 50:
        print(f"\n📊 Use matched data (n={direct_matches}) to:")
        print("   - Validate demographic sampling correlations")
        print("   - Estimate bias correction factors")
        print("   - Report empirical correlation bounds")

def main():
    # File paths
    alpha_file = "alpha_demographics.xlsx"
    rho_file = "rho_demographics.xlsx"
    theta_file = "theta_diet_demographics.xlsx"
    
    print("=== Survey ID Matching Analysis ===")
    
    # Load surveys
    alpha_df, rho_df, theta_df = load_surveys_direct(alpha_file, rho_file, theta_file)
    
    # Analyze overlaps
    overlaps = analyze_id_overlap(alpha_df, rho_df, theta_df)
    
    # Create matched datasets
    matched_all, matched_alpha_theta = create_matched_dataset(alpha_df, rho_df, theta_df, overlaps)
    
    # Validate hybrid approach
    matched_data = matched_all if matched_all is not None else matched_alpha_theta
    need_demographic = validate_hybrid_approach(alpha_df, rho_df, theta_df, matched_data, overlaps)
    
    # Provide recommendation
    recommend_approach(overlaps, matched_data, need_demographic)
    
    return {
        'surveys': {'alpha': alpha_df, 'rho': rho_df, 'theta': theta_df},
        'overlaps': overlaps,
        'matched_data': matched_data
    }

if __name__ == "__main__":
    results = main()