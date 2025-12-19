"""
Stratified sampling utility to ensure demographic representativeness
when selecting N < 5602 agents from survey data.
"""
import pandas as pd
import numpy as np

def stratified_sample_agents(df, n_target, strata_cols=['gender', 'age_group', 'incquart', 'educlevel'],
                              random_state=42):
    """
    Sample n_target agents using stratified sampling to preserve demographic distributions.

    Args:
        df: DataFrame with survey participants
        n_target: Target sample size
        strata_cols: Columns defining demographic strata
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with n_target sampled agents maintaining demographic proportions
    """
    if n_target >= len(df):
        print(f"INFO: n_target ({n_target}) >= data size ({len(df)}), returning all data")
        return df.copy()

    # Create stratification groups
    df['_strata'] = df[strata_cols].astype(str).agg('_'.join, axis=1)
    strata_counts = df['_strata'].value_counts()

    print(f"\nSTRATIFIED SAMPLING: N={n_target} from {len(df)} participants")
    print(f"Using strata: {', '.join(strata_cols)}")
    print(f"Number of unique strata: {len(strata_counts)}")

    # Calculate proportional allocation for each stratum
    strata_props = strata_counts / len(df)
    strata_targets = (strata_props * n_target).round().astype(int)

    # Handle rounding errors to ensure exact n_target
    diff = n_target - strata_targets.sum()
    if diff != 0:
        # Add/subtract from largest strata
        largest_strata = strata_targets.nlargest(abs(diff)).index
        for stratum in largest_strata:
            strata_targets[stratum] += np.sign(diff)

    # Sample from each stratum
    sampled_dfs = []
    np.random.seed(random_state)

    for stratum, target_n in strata_targets.items():
        if target_n == 0:
            continue

        stratum_data = df[df['_strata'] == stratum]

        # If stratum smaller than target, sample with replacement (rare)
        replace = len(stratum_data) < target_n
        if replace:
            print(f"WARNING: Stratum '{stratum}' has only {len(stratum_data)} members but needs {target_n}")
            print(f"         Sampling WITH replacement for this stratum")

        sampled = stratum_data.sample(n=target_n, replace=replace, random_state=random_state)
        sampled_dfs.append(sampled)

    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.drop(columns=['_strata'])

    # Validation
    print(f"\nValidation:")
    print(f"  Target N: {n_target}")
    print(f"  Actual N: {len(result)}")

    # Compare distributions
    print(f"\n{'Demographic':<15} {'Original %':<12} {'Sampled %':<12} {'Difference':<12}")
    print("-" * 55)

    for col in strata_cols:
        orig_dist = df[col].value_counts(normalize=True).sort_index()
        samp_dist = result[col].value_counts(normalize=True).sort_index()

        for category in orig_dist.index:
            orig_pct = orig_dist.get(category, 0) * 100
            samp_pct = samp_dist.get(category, 0) * 100
            diff = samp_pct - orig_pct
            print(f"{col}:{category:<10} {orig_pct:>10.2f}%  {samp_pct:>10.2f}%  {diff:>+10.2f}%")

    return result


def compare_sampling_methods(df, n_target, n_trials=10, random_state=42):
    """
    Compare simple random sampling vs stratified sampling.
    """
    print("=" * 80)
    print(f"COMPARING SAMPLING METHODS: N={n_target} from {len(df)} participants")
    print("=" * 80)

    strata_cols = ['gender', 'age_group', 'incquart', 'educlevel']

    # Get original distributions
    orig_dists = {col: df[col].value_counts(normalize=True) for col in strata_cols}

    # Simple random sampling (multiple trials)
    print("\nSIMPLE RANDOM SAMPLING (10 trials):")
    random_diffs = {col: [] for col in strata_cols}

    for trial in range(n_trials):
        sample = df.sample(n=n_target, random_state=random_state + trial)
        for col in strata_cols:
            sample_dist = sample[col].value_counts(normalize=True)
            max_diff = max(abs(sample_dist.get(cat, 0) - orig_dists[col].get(cat, 0))
                          for cat in orig_dists[col].index)
            random_diffs[col].append(max_diff * 100)

    for col in strata_cols:
        mean_diff = np.mean(random_diffs[col])
        max_diff = np.max(random_diffs[col])
        print(f"  {col:<15}: mean max deviation = {mean_diff:.2f}%, worst = {max_diff:.2f}%")

    # Stratified sampling
    print("\nSTRATIFIED SAMPLING:")
    stratified_diffs = {col: [] for col in strata_cols}

    for trial in range(n_trials):
        sample = stratified_sample_agents(df, n_target, strata_cols, random_state + trial)
        for col in strata_cols:
            sample_dist = sample[col].value_counts(normalize=True)
            max_diff = max(abs(sample_dist.get(cat, 0) - orig_dists[col].get(cat, 0))
                          for cat in orig_dists[col].index)
            stratified_diffs[col].append(max_diff * 100)

    print("\nComparison (maximum category deviation from original distribution):")
    print(f"{'Demographic':<15} {'Random (mean)':<15} {'Stratified (mean)':<18} {'Improvement'}")
    print("-" * 70)

    for col in strata_cols:
        random_mean = np.mean(random_diffs[col])
        strat_mean = np.mean(stratified_diffs[col])
        improvement = ((random_mean - strat_mean) / random_mean) * 100 if random_mean > 0 else 0
        print(f"{col:<15} {random_mean:>13.2f}%  {strat_mean:>15.2f}%  {improvement:>12.1f}%")

    print("\n" + "=" * 80)
    print("RECOMMENDATION: Use stratified sampling to preserve demographic distributions")
    print("=" * 80)


if __name__ == "__main__":
    # Load survey data
    df = pd.read_csv('../data/hierarchical_agents.csv')

    # Test stratified sampling at N=2000
    print("\n" + "=" * 80)
    print("TESTING STRATIFIED SAMPLING AT N=2000")
    print("=" * 80)

    sampled_df = stratified_sample_agents(df, n_target=2000)

    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)

    compare_sampling_methods(df, n_target=2000, n_trials=5)

    # Save stratified sample
    output_path = '../data/stratified_sample_n2000.csv'
    sampled_df.to_csv(output_path, index=False)
    print(f"\nSaved stratified sample to: {output_path}")
