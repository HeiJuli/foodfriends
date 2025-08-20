import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt

def load_surveys(alpha_file, rho_file, theta_file):
    """Load and clean all three survey files"""
    
    def safe_load(filepath):
        if filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
        return pd.read_csv(filepath)
    
    # Load files
    alpha_data = safe_load(alpha_file)
    rho_data = safe_load(rho_file) 
    theta_data = safe_load(theta_file)
    
    # Clean alpha survey
    alpha_clean = alpha_data[['Self-identity weight (alpha)', 'Gender', 
                             'Age of the household member', 'Income Quartile', 'Education Level']].copy()
    alpha_clean.columns = ['alpha', 'gender', 'age', 'incquart', 'educlevel']
    alpha_clean['alpha'] = pd.to_numeric(alpha_clean['alpha'], errors='coerce')
    alpha_clean = alpha_clean.dropna()
    
    # Clean rho survey
    rho_clean = rho_data[['Cost parameter (rho)', 'Gender',
                         'Age of the household member', 'Income Quartile', 'Education Level']].copy()
    rho_clean.columns = ['rho', 'gender', 'age', 'incquart', 'educlevel']
    rho_clean['rho'] = pd.to_numeric(rho_clean['rho'], errors='coerce')
    rho_clean = rho_clean.dropna()
    
    # Clean theta survey  
    theta_clean = theta_data[['Personal Preference for Veg Diet', 'Gender',
                             'Age of the household member', 'Income Quartile', 'Education Level']].copy()
    theta_clean.columns = ['theta', 'gender', 'age', 'incquart', 'educlevel']
    theta_clean['theta'] = pd.to_numeric(theta_clean['theta'], errors='coerce')
    theta_clean = theta_clean.dropna()
    
    # Add age groups
    for df in [alpha_clean, rho_clean, theta_clean]:
        df['age_group'] = pd.cut(df['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                                labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    
    surveys = {'alpha': alpha_clean, 'rho': rho_clean, 'theta': theta_clean}
    
    for name, df in surveys.items():
        print(f"{name}: {len(df)} records, {name} range: [{df[name].min():.2f}, {df[name].max():.2f}]")
    
    return surveys

def validate_demographic_consistency(surveys):
    """Test if demographic distributions are consistent across surveys"""
    
    print("\n=== Demographic Distribution Consistency ===")
    demos = ['gender', 'age_group', 'incquart', 'educlevel']
    
    consistency_results = {}
    
    for demo in demos:
        print(f"\n{demo.upper()}:")
        
        # Get value counts for each survey
        alpha_dist = surveys['alpha'][demo].value_counts(normalize=True).sort_index()
        rho_dist = surveys['rho'][demo].value_counts(normalize=True).sort_index()
        theta_dist = surveys['theta'][demo].value_counts(normalize=True).sort_index()
        
        # Align indices (categories)
        all_cats = sorted(set(alpha_dist.index) | set(rho_dist.index) | set(theta_dist.index))
        alpha_aligned = alpha_dist.reindex(all_cats, fill_value=0)
        rho_aligned = rho_dist.reindex(all_cats, fill_value=0)
        theta_aligned = theta_dist.reindex(all_cats, fill_value=0)
        
        # Chi-square test for distribution similarity
        contingency = np.array([alpha_aligned, rho_aligned, theta_aligned])
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency)
            consistency_results[demo] = p_val > 0.05
            
            print(f"  Chi2={chi2:.2f}, p={p_val:.3f} ({'✓ Consistent' if p_val > 0.05 else '⚠ Different'})")
            
            # Show distributions
            for cat in all_cats:
                print(f"    {cat}: α={alpha_aligned[cat]:.2f}, ρ={rho_aligned[cat]:.2f}, θ={theta_aligned[cat]:.2f}")
        except:
            print(f"  Could not test {demo} consistency")
            consistency_results[demo] = False
    
    return consistency_results

def validate_pmf_coverage(surveys):
    """Test demographic cell coverage for PMF approach"""
    
    print("\n=== PMF Cell Coverage Analysis ===")
    
    demo_vars = ['gender', 'age_group', 'incquart', 'educlevel']
    
    for param, df in surveys.items():
        print(f"\n{param.upper()} survey:")
        
        # Group by demographics and check cell sizes
        grouped = df.groupby(demo_vars).size()
        
        n_total_cells = len(grouped)
        n_small_cells = (grouped < 5).sum()
        n_empty_cells = (grouped == 0).sum() 
        avg_cell_size = grouped.mean()
        
        print(f"  Total cells: {n_total_cells}")
        print(f"  Empty cells: {n_empty_cells} ({100*n_empty_cells/n_total_cells:.1f}%)")
        print(f"  Small cells (<5): {n_small_cells} ({100*n_small_cells/n_total_cells:.1f}%)")
        print(f"  Avg cell size: {avg_cell_size:.1f}")
        
        # Show parameter variation within demographics
        for demo in demo_vars:
            demo_means = df.groupby(demo)[param].mean()
            demo_var = df.groupby(demo)[param].var()
            effect_size = demo_var.mean() / df[param].var() if df[param].var() > 0 else 0
            print(f"  {demo} effect size (η²): {effect_size:.3f}")

def simulate_pmf_population(surveys, n_agents=1000):
    """Simulate agent population using PMF approach and test correlations"""
    
    print(f"\n=== PMF Population Simulation (n={n_agents}) ===")
    
    np.random.seed(42)
    demo_vars = ['gender', 'age_group', 'incquart', 'educlevel']
    
    # Use theta survey as reference population (largest)
    ref_pop = surveys['theta'].sample(n_agents, replace=True).reset_index(drop=True)
    
    # For each agent, sample alpha and rho from demographic-matched cells
    sampled_data = {'alpha': [], 'rho': [], 'theta': ref_pop['theta'].tolist()}
    
    for _, agent in ref_pop.iterrows():
        
        for param in ['alpha', 'rho']:
            param_df = surveys[param]
            
            # Find exact demographic matches
            mask = True
            for demo in demo_vars:
                mask &= (param_df[demo] == agent[demo])
            
            matches = param_df[mask]
            
            if len(matches) > 0:
                # Sample from exact matches
                sampled_val = np.random.choice(matches[param])
            else:
                # Fallback to broader matches (remove education requirement)
                broader_mask = True
                for demo in ['gender', 'age_group', 'incquart']:
                    broader_mask &= (param_df[demo] == agent[demo])
                broader_matches = param_df[broader_mask]
                
                if len(broader_matches) > 0:
                    sampled_val = np.random.choice(broader_matches[param])
                else:
                    # Final fallback: random sample from whole survey
                    sampled_val = np.random.choice(param_df[param])
            
            sampled_data[param].append(sampled_val)
    
    # Create simulated population dataframe
    sim_pop = pd.DataFrame(sampled_data)
    
    # Calculate correlations
    corr_matrix = sim_pop.corr()
    
    print("Simulated population statistics:")
    print(f"  α: mean={sim_pop['alpha'].mean():.3f}, std={sim_pop['alpha'].std():.3f}")
    print(f"  ρ: mean={sim_pop['rho'].mean():.3f}, std={sim_pop['rho'].std():.3f}")  
    print(f"  θ: mean={sim_pop['theta'].mean():.3f}, std={sim_pop['theta'].std():.3f}")
    
    print("\nCorrelation matrix:")
    print(f"  α-ρ: {corr_matrix.loc['alpha', 'rho']:.3f}")
    print(f"  α-θ: {corr_matrix.loc['alpha', 'theta']:.3f}")
    print(f"  ρ-θ: {corr_matrix.loc['rho', 'theta']:.3f}")
    
    return sim_pop, corr_matrix

def compare_with_matched_data(surveys, sim_corrs):
    """Compare simulated correlations with ID-matched data"""
    
    print("\n=== Validation Against ID-Matched Data ===")
    
    # This would require the actual ID matching logic from your validation script
    # For now, report the correlations you found
    print("From your ID-matched data (n=1298):")
    print("  α-ρ: -0.062")
    print("  α-θ:  0.136") 
    print("  ρ-θ: -0.341")
    
    print(f"\nPMF simulation correlations:")
    print(f"  α-ρ: {sim_corrs.loc['alpha', 'rho']:.3f}")
    print(f"  α-θ: {sim_corrs.loc['alpha', 'theta']:.3f}")
    print(f"  ρ-θ: {sim_corrs.loc['rho', 'theta']:.3f}")
    
    # Calculate differences
    matched_corrs = {'alpha-rho': -0.062, 'alpha-theta': 0.136, 'rho-theta': -0.341}
    sim_corrs_dict = {
        'alpha-rho': sim_corrs.loc['alpha', 'rho'],
        'alpha-theta': sim_corrs.loc['alpha', 'theta'], 
        'rho-theta': sim_corrs.loc['rho', 'theta']
    }
    
    print("\nCorrelation differences (PMF - Matched):")
    for pair in matched_corrs:
        diff = sim_corrs_dict[pair] - matched_corrs[pair]
        print(f"  {pair}: {diff:+.3f}")

def overall_assessment(consistency, sim_pop):
    """Provide overall assessment of PMF approach"""
    
    print("\n=== OVERALL ASSESSMENT ===")
    
    consistent_demos = sum(consistency.values())
    total_demos = len(consistency)
    
    max_corr = max(abs(sim_pop.corr().loc['alpha', 'rho']),
                   abs(sim_pop.corr().loc['alpha', 'theta']),
                   abs(sim_pop.corr().loc['rho', 'theta']))
    
    print(f"Demographic consistency: {consistent_demos}/{total_demos} ({100*consistent_demos/total_demos:.0f}%)")
    print(f"Max simulated |correlation|: {max_corr:.3f}")
    
    if consistent_demos >= 3 and max_corr < 0.4:
        verdict = "SUITABLE"
        icon = "✓"
    elif consistent_demos >= 2 and max_corr < 0.5:
        verdict = "ACCEPTABLE with caveats"
        icon = "⚠"
    else:
        verdict = "QUESTIONABLE"
        icon = "⚠"
    
    print(f"\n{icon} PMF demographic approach: {verdict}")
    
    if verdict != "SUITABLE":
        print("\nRecommendations:")
        if consistent_demos < 3:
            print("- Consider survey weighting to align demographic distributions")
        if max_corr >= 0.4:
            print("- Apply correlation adjustment to simulated population")
            print("- Report correlation uncertainty in results")

# Main execution
if __name__ == "__main__":
    
    # File paths - adjust as needed
    alpha_file = "alpha_demographics.xlsx"
    rho_file = "rho_demographics.xlsx"
    theta_file = "theta_diet_demographics.xlsx"
    
    surveys = load_surveys(alpha_file, rho_file, theta_file)
    consistency = validate_demographic_consistency(surveys)
    validate_pmf_coverage(surveys)
    sim_pop, sim_corrs = simulate_pmf_population(surveys, n_agents=1000)
    compare_with_matched_data(surveys, sim_corrs)
    overall_assessment(consistency, sim_pop)