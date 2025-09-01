#!/usr/bin/env python3
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def test_demographic_predictive_power():
    """Test if demographics actually predict parameters in raw survey data"""
    
    print("=== 1. DEMOGRAPHIC PREDICTIVE POWER IN RAW DATA ===")
    
    # Load raw survey data
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
    alpha_clean = alpha_clean.dropna()
    
    print(f"Raw survey data: {len(alpha_clean)} records")
    
    # Test each demographic variable individually
    demo_vars = ['gender', 'age_group', 'incquart', 'educlevel']
    
    for var in demo_vars:
        group_means = alpha_clean.groupby(var)['alpha'].mean()
        print(f"\n{var}:")
        print(f"  Groups: {len(group_means)}")
        print(f"  Mean range: [{group_means.min():.3f}, {group_means.max():.3f}]")
        print(f"  Between-group std: {group_means.std():.3f}")
        
        # ANOVA F-test
        groups = [group['alpha'].values for name, group in alpha_clean.groupby(var)]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"  ANOVA F={f_stat:.2f}, p={p_val:.3f}")
    
    # Test combined demographics
    demo_combos = alpha_clean.groupby(demo_vars)['alpha'].mean()
    print(f"\nCombined demographics:")
    print(f"  Unique combinations: {len(demo_combos)}")
    print(f"  Mean range: [{demo_combos.min():.3f}, {demo_combos.max():.3f}]")
    print(f"  Between-group std: {demo_combos.std():.3f}")
    print(f"  Overall alpha std: {alpha_clean['alpha'].std():.3f}")
    print(f"  Explained variance ratio: {(demo_combos.std() / alpha_clean['alpha'].std())**2:.3f}")

def test_pmf_cell_differences():
    """Test if PMF cells actually differ from each other"""
    
    print("\n=== 2. PMF CELL DIFFERENCES ===")
    
    with open("demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)
    
    for param in ['alpha', 'rho']:
        print(f"\n{param.upper()}:")
        cells = pmf_tables[param]
        
        # Calculate mean for each cell
        cell_means = {}
        for demo_key, pmf_data in cells.items():
            vals, probs = pmf_data['values'], pmf_data['probabilities']
            mean_val = sum(v * p for v, p in zip(vals, probs))
            cell_means[demo_key] = mean_val
        
        means_array = np.array(list(cell_means.values()))
        print(f"  PMF cell means range: [{means_array.min():.3f}, {means_array.max():.3f}]")
        print(f"  PMF cell means std: {means_array.std():.3f}")
        
        # Test if all cells are basically the same
        unique_means = len(set(np.round(means_array, 2)))
        print(f"  Unique means (2 decimal): {unique_means}/{len(means_array)}")
        
        if unique_means < 10:
            print(f"  WARNING: Very few unique means - cells may be too similar!")
        
        # Show most different cells
        sorted_means = sorted(cell_means.items(), key=lambda x: x[1])
        print(f"  Lowest mean: {sorted_means[0][0]} = {sorted_means[0][1]:.3f}")
        print(f"  Highest mean: {sorted_means[-1][0]} = {sorted_means[-1][1]:.3f}")

def test_sampling_behavior():
    """Test if sampling is actually demographic-specific vs uniform"""
    
    print("\n=== 3. SAMPLING BEHAVIOR TEST ===")
    
    with open("demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)
    
    def sample_from_pmf(demo_key, param):
        if demo_key in pmf_tables[param]:
            pmf = pmf_tables[param][demo_key]
            vals, probs = pmf['values'], pmf['probabilities']
            return np.random.choice(vals, p=np.array(probs)/sum(probs))
        else:
            # Fallback
            all_vals = []
            for cell in pmf_tables[param].values():
                all_vals.extend(cell['values'])
            return np.random.choice(all_vals) if all_vals else 0.5
    
    # Test two very different demographic groups
    alpha_cells = list(pmf_tables['alpha'].keys())
    
    # Find cells with most different means
    cell_means = {}
    for demo_key, pmf_data in pmf_tables['alpha'].items():
        vals, probs = pmf_data['values'], pmf_data['probabilities']
        cell_means[demo_key] = sum(v * p for v, p in zip(vals, probs))
    
    sorted_cells = sorted(cell_means.items(), key=lambda x: x[1])
    low_group = sorted_cells[0][0]
    high_group = sorted_cells[-1][0]
    
    print(f"Testing two extreme groups:")
    print(f"  Low group: {low_group} (mean={sorted_cells[0][1]:.3f})")
    print(f"  High group: {high_group} (mean={sorted_cells[-1][1]:.3f})")
    
    # Sample many times from each group
    np.random.seed(42)
    low_samples = [sample_from_pmf(low_group, 'alpha') for _ in range(1000)]
    high_samples = [sample_from_pmf(high_group, 'alpha') for _ in range(1000)]
    
    print(f"\nSampling results (1000 samples each):")
    print(f"  Low group samples: mean={np.mean(low_samples):.3f}, std={np.std(low_samples):.3f}")
    print(f"  High group samples: mean={np.mean(high_samples):.3f}, std={np.std(high_samples):.3f}")
    print(f"  Difference in means: {np.mean(high_samples) - np.mean(low_samples):.3f}")
    
    # T-test
    t_stat, p_val = stats.ttest_ind(low_samples, high_samples)
    print(f"  T-test: t={t_stat:.2f}, p={p_val:.6f}")
    
    if p_val < 0.001:
        print(f"  ✓ Groups are significantly different - demographic sampling working")
    else:
        print(f"  ✗ Groups not significantly different - may be uniform sampling")

def test_validation_methodology():
    """Test if validation is comparing the right things"""
    
    print("\n=== 4. VALIDATION METHODOLOGY TEST ===")
    
    # Load survey data
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
    alpha_clean = alpha_clean.dropna()
    
    with open("demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)
    
    def sample_from_pmf(demo_key, param):
        if demo_key in pmf_tables[param]:
            pmf = pmf_tables[param][demo_key]
            vals, probs = pmf['values'], pmf['probabilities']
            return np.random.choice(vals, p=np.array(probs)/sum(probs))
        return 0.5
    
    demo_vars = ['gender', 'age_group', 'incquart', 'educlevel']
    
    # Test validation on subset
    test_sample = alpha_clean.sample(500, random_state=42)
    
    actual_vals = []
    sampled_vals = []
    fallback_count = 0
    
    for _, row in test_sample.iterrows():
        demo_key = tuple(row[demo_vars])
        actual_val = row['alpha']
        
        # Check if using fallback
        if demo_key not in pmf_tables['alpha']:
            fallback_count += 1
        
        sampled_val = sample_from_pmf(demo_key, 'alpha')
        
        actual_vals.append(actual_val)
        sampled_vals.append(sampled_val)
    
    actual_vals = np.array(actual_vals)
    sampled_vals = np.array(sampled_vals)
    
    print(f"Validation test on {len(test_sample)} individuals:")
    print(f"  Fallback usage: {fallback_count}/{len(test_sample)} ({100*fallback_count/len(test_sample):.1f}%)")
    print(f"  Individual R²: {np.corrcoef(actual_vals, sampled_vals)[0,1]**2:.3f}")
    
    # Check for uniform pattern
    unique_sampled = len(set(sampled_vals))
    print(f"  Unique sampled values: {unique_sampled}")
    
    if unique_sampled < 10:
        print(f"  WARNING: Very few unique values - may indicate uniform sampling")
        print(f"  Sampled values: {sorted(set(sampled_vals))}")

def plot_diagnostic():
    """Create diagnostic plots"""
    
    print("\n=== 5. DIAGNOSTIC PLOTS ===")
    
    # Load survey data
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
    alpha_clean = alpha_clean.dropna()
    
    with open("demographic_pmfs.pkl", 'rb') as f:
        pmf_tables = pickle.load(f)
    
    def sample_from_pmf(demo_key, param):
        if demo_key in pmf_tables[param]:
            pmf = pmf_tables[param][demo_key]
            vals, probs = pmf['values'], pmf['probabilities']
            return np.random.choice(vals, p=np.array(probs)/sum(probs))
        return 0.5
    
    demo_vars = ['gender', 'age_group', 'incquart', 'educlevel']
    test_sample = alpha_clean.sample(500, random_state=42)
    
    actual_vals = []
    sampled_vals = []
    
    for _, row in test_sample.iterrows():
        demo_key = tuple(row[demo_vars])
        actual_vals.append(row['alpha'])
        sampled_vals.append(sample_from_pmf(demo_key, 'alpha'))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Individual validation (problematic)
    axes[0,0].scatter(actual_vals, sampled_vals, alpha=0.6, s=20)
    axes[0,0].plot([0, 1], [0, 1], 'r--')
    axes[0,0].set_xlabel('Actual Alpha')
    axes[0,0].set_ylabel('PMF Sampled Alpha')
    axes[0,0].set_title('Individual Validation (Current Issue)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Distribution comparison
    axes[0,1].hist(actual_vals, alpha=0.7, label='Actual', bins=20)
    axes[0,1].hist(sampled_vals, alpha=0.7, label='PMF Sampled', bins=20)
    axes[0,1].set_xlabel('Alpha Value')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Distribution Comparison')
    axes[0,1].legend()
    
    # Group means validation
    group_actual = test_sample.groupby(demo_vars)['alpha'].mean()
    group_sampled = []
    for demo_key in group_actual.index:
        samples = [sample_from_pmf(demo_key, 'alpha') for _ in range(100)]
        group_sampled.append(np.mean(samples))
    
    axes[1,0].scatter(group_actual.values, group_sampled, alpha=0.7)
    axes[1,0].plot([group_actual.min(), group_actual.max()], 
                   [group_actual.min(), group_actual.max()], 'r--')
    axes[1,0].set_xlabel('Actual Group Mean')
    axes[1,0].set_ylabel('PMF Group Mean')
    axes[1,0].set_title('Group-Level Validation')
    axes[1,0].grid(True, alpha=0.3)
    
    # Sampled values pattern
    axes[1,1].hist(sampled_vals, bins=50, alpha=0.7)
    axes[1,1].set_xlabel('PMF Sampled Values')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('PMF Sample Distribution Pattern')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_demographic_predictive_power()
    test_pmf_cell_differences()
    test_sampling_behavior()
    test_validation_methodology()
    plot_diagnostic()