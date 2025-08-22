import pandas as pd
import numpy as np
import pickle
from scipy.stats import chi2_contingency
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
        df['id'] = df.index
    
    surveys = {'alpha': alpha_clean, 'rho': rho_clean, 'theta': theta_clean}
    
    for name, df in surveys.items():
        print(f"{name}: {len(df)} records, range: [{df[name].min():.2f}, {df[name].max():.2f}]")
    
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
        
        # Align indices
        all_cats = sorted(set(alpha_dist.index) | set(rho_dist.index) | set(theta_dist.index))
        alpha_aligned = alpha_dist.reindex(all_cats, fill_value=0)
        rho_aligned = rho_dist.reindex(all_cats, fill_value=0)
        theta_aligned = theta_dist.reindex(all_cats, fill_value=0)
        
        # Chi-square test
        contingency = np.array([alpha_aligned, rho_aligned, theta_aligned])
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency)
            consistency_results[demo] = p_val > 0.05
            
            print(f"  Chi2={chi2:.2f}, p={p_val:.3f} ({'Consistent' if p_val > 0.05 else 'Different'})")
            
            for cat in all_cats:
                print(f"    {cat}: α={alpha_aligned[cat]:.2f}, ρ={rho_aligned[cat]:.2f}, θ={theta_aligned[cat]:.2f}")
        except:
            print(f"  Could not test {demo} consistency")
            consistency_results[demo] = False
    
    return consistency_results

def validate_pmf_estimates(surveys):
    """Validate PMF sampling approach as actually used in the model"""
    
    print("\n=== PMF Sampling Validation (Model Simulation) ===")
    
    # Load existing PMF tables
    try:
        with open("demographic_pmfs.pkl", 'rb') as f:
            pmf_tables = pickle.load(f)
        print("Loaded existing PMF tables")
    except FileNotFoundError:
        print("PMF tables not found - run create_pmf_tables.py first")
        return None
    
    demo_vars = ['gender', 'age_group', 'incquart', 'educlevel']
    
    def sample_from_pmf(demo_key, param_name):
        """Simulate exact PMF sampling as used in model"""
        if demo_key in pmf_tables[param_name]:
            pmf = pmf_tables[param_name][demo_key]
            vals, probs = pmf['values'], pmf['probabilities']
            nz = [(v,p) for v,p in zip(vals, probs) if p > 0]
            if nz:
                v, p = zip(*nz)
                return np.random.choice(v, p=np.array(p)/sum(p))
        
        # Fallback: sample from all values
        all_vals = []
        for cell in pmf_tables[param_name].values():
            all_vals.extend(cell['values'])
        return np.random.choice(all_vals) if all_vals else 0.5
    
    def validate_pmf_sampling(param_data, param_name, n_samples=50):
        """Test PMF sampling with multiple draws per person"""
        actual_vals, sampled_vals = [], []
        
        for _, row in param_data.iterrows():
            demo_key = tuple(row[col] for col in demo_vars)
            actual_val = row[param_name]
            
            # Sample multiple times from PMF for this person's demographics
            samples = [sample_from_pmf(demo_key, param_name) for _ in range(n_samples)]
            
            # Use all samples vs actual value
            actual_vals.extend([actual_val] * n_samples)
            sampled_vals.extend(samples)
        
        return np.array(actual_vals), np.array(sampled_vals)
    
    # Run validation with multiple samples per person
    np.random.seed(42)
    alpha_actual, alpha_sampled = validate_pmf_sampling(surveys['alpha'], 'alpha')
    rho_actual, rho_sampled = validate_pmf_sampling(surveys['rho'], 'rho')
    
    # Statistics
    def calc_stats(actual, predicted):
        if len(set(predicted)) == 1:  # No variance
            return 0.0, np.mean(predicted - actual), np.sqrt(np.mean((predicted - actual)**2))
        r2 = np.corrcoef(actual, predicted)[0,1]**2
        bias = np.mean(predicted - actual)
        rmse = np.sqrt(np.mean((predicted - actual)**2))
        return r2, bias, rmse
    
    alpha_r2, alpha_bias, alpha_rmse = calc_stats(alpha_actual, alpha_sampled)
    rho_r2, rho_bias, rho_rmse = calc_stats(rho_actual, rho_sampled)
    
    print(f"\nPMF sampling performance (50 samples per person):")
    print(f"Alpha (n={len(alpha_actual)}): R²={alpha_r2:.3f}, Bias={alpha_bias:+.3f}, RMSE={alpha_rmse:.3f}")
    print(f"Rho (n={len(rho_actual)}): R²={rho_r2:.3f}, Bias={rho_bias:+.3f}, RMSE={rho_rmse:.3f}")
    
    # Get overlaps for hybrid approach planning
    alpha_ids = set(surveys['alpha']['id'])
    rho_ids = set(surveys['rho']['id'])
    theta_ids = set(surveys['theta']['id'])
    
    alpha_theta_overlap = len(alpha_ids & theta_ids)
    rho_theta_overlap = len(rho_ids & theta_ids)
    all_three_overlap = len(alpha_ids & rho_ids & theta_ids)
    
    print(f"\nHybrid approach potential:")
    print(f"Alpha-Theta matches: {alpha_theta_overlap} ({100*alpha_theta_overlap/len(theta_ids):.1f}% of theta)")
    print(f"Rho-Theta matches: {rho_theta_overlap} ({100*rho_theta_overlap/len(theta_ids):.1f}% of theta)")
    print(f"All three matches: {all_three_overlap} ({100*all_three_overlap/len(theta_ids):.1f}% of theta)")
    
    # Plot subset for visualization (sample 2000 points to avoid overcrowding)
    n_plot = min(2000, len(alpha_actual))
    alpha_plot_idx = np.random.choice(len(alpha_actual), n_plot, replace=False)
    rho_plot_idx = np.random.choice(len(rho_actual), min(2000, len(rho_actual)), replace=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(alpha_actual[alpha_plot_idx], alpha_sampled[alpha_plot_idx], alpha=0.3, s=10)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.7)
    ax1.set_xlabel('Actual Alpha')
    ax1.set_ylabel('PMF Sampled Alpha')
    ax1.set_title(f'Alpha: R²={alpha_r2:.3f}')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(rho_actual[rho_plot_idx], rho_sampled[rho_plot_idx], alpha=0.3, s=10)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.7)
    ax2.set_xlabel('Actual Rho')
    ax2.set_ylabel('PMF Sampled Rho')
    ax2.set_title(f'Rho: R²={rho_r2:.3f}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {'alpha': alpha_r2, 'rho': rho_r2, 'overlaps': {
        'alpha_theta': alpha_theta_overlap, 'rho_theta': rho_theta_overlap, 'all_three': all_three_overlap
    }}

def overall_assessment(consistency, pmf_performance):
    """Provide overall assessment"""
    
    print("\n=== OVERALL ASSESSMENT ===")
    
    consistent_demos = sum(consistency.values())
    total_demos = len(consistency)
    
    max_r2 = max(pmf_performance['alpha'], pmf_performance['rho'])
    
    print(f"Demographic consistency: {consistent_demos}/{total_demos} ({100*consistent_demos/total_demos:.0f}%)")
    print(f"Max demographic prediction R²: {max_r2:.3f}")
    
    if consistent_demos >= 3 and max_r2 > 0.1:
        verdict = "SUITABLE"
        icon = "✓"
    elif consistent_demos >= 2 and max_r2 > 0.05:
        verdict = "ACCEPTABLE with caveats"
        icon = "⚠"
    else:
        verdict = "QUESTIONABLE"
        icon = "⚠"
    
    print(f"\n{icon} PMF demographic approach: {verdict}")
    
    # Recommendation based on overlaps
    overlaps = pmf_performance['overlaps']
    if overlaps['rho_theta'] > 0.9 * len(surveys['theta']):
        print(f"\nRecommendation: Hybrid approach")
        print(f"- Use theta (survey) + rho (direct match) for {overlaps['rho_theta']} agents")
        print(f"- Use alpha via demographic PMF for all agents")
        print(f"- Strong rho coverage: {100*overlaps['rho_theta']/len(surveys['theta']):.1f}%")


def diagnose_pmf_tables():
    """Investigate what's actually in the PMF tables"""
    
    with open("demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)
    
    for param in ['alpha', 'rho']:
        print(f"\n=== {param.upper()} PMF Analysis ===")
        
        # Collect all unique values across all cells
        all_values = set()
        cell_sizes = []
        
        for demo_key, pmf_data in pmf_tables[param].items():
            values = pmf_data['values']
            all_values.update(values)
            cell_sizes.append(len(values))
        
        print(f"Unique values in PMF: {sorted(all_values)}")
        print(f"Number of unique values: {len(all_values)}")
        print(f"Avg values per cell: {np.mean(cell_sizes):.1f}")
        print(f"Cells with <5 values: {sum(1 for s in cell_sizes if s < 5)}/{len(cell_sizes)}")
        
        # Check a few example cells
        print(f"\nExample demographic cells:")
        for i, (demo_key, pmf_data) in enumerate(list(pmf_tables[param].items())[:3]):
            values = pmf_data['values']
            print(f"  {demo_key}: {len(values)} values = {values[:10]}{'...' if len(values) > 10 else ''}")


if __name__ == "__main__":
    
    # File paths
    alpha_file = "alpha_demographics.xlsx"
    rho_file = "rho_demographics.xlsx"
    theta_file = "theta_diet_demographics.xlsx"
    
    surveys = load_surveys(alpha_file, rho_file, theta_file)
    consistency = validate_demographic_consistency(surveys)
    pmf_performance = validate_pmf_estimates(surveys)
    
    if pmf_performance:
        overall_assessment(consistency, pmf_performance)
    diagnose_pmf_tables()