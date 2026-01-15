"""
Sampling utilities for agent initialization.
Provides stratified sampling to ensure demographic representativeness.
"""
import pandas as pd
import numpy as np

def stratified_sample_agents(df, n_target, strata_cols=['gender', 'age_group', 'incquart', 'educlevel'],
                              random_state=None, verbose=True):
    """
    Sample n_target agents using stratified sampling to preserve demographic distributions.

    Args:
        df: DataFrame with survey participants
        n_target: Target sample size
        strata_cols: Columns defining demographic strata
        random_state: Random seed for reproducibility
        verbose: Print sampling details

    Returns:
        DataFrame with n_target sampled agents maintaining demographic proportions
    """
    if n_target >= len(df):
        if verbose:
            print(f"INFO: n_target ({n_target}) >= data size ({len(df)}), returning all data")
        return df.copy()

    # Create stratification groups
    df = df.copy()
    df['_strata'] = df[strata_cols].astype(str).agg('_'.join, axis=1)
    strata_counts = df['_strata'].value_counts()

    if verbose:
        print(f"INFO: Stratified sampling N={n_target} from {len(df)} participants")
        print(f"INFO: Using strata: {', '.join(strata_cols)}")
        print(f"INFO: Number of unique strata: {len(strata_counts)}")

    # Calculate proportional allocation for each stratum
    strata_props = strata_counts / len(df)
    strata_targets = (strata_props * n_target).round().astype(int)

    # Handle rounding errors to ensure exact n_target
    diff = n_target - strata_targets.sum()
    if diff != 0:
        largest_strata = strata_targets.nlargest(abs(diff)).index
        for stratum in largest_strata:
            strata_targets[stratum] += np.sign(diff)

    # Sample from each stratum
    sampled_dfs = []
    if random_state is not None:
        np.random.seed(random_state)

    for stratum, target_n in strata_targets.items():
        if target_n == 0:
            continue

        stratum_data = df[df['_strata'] == stratum]
        replace = len(stratum_data) < target_n

        if replace and verbose:
            print(f"WARNING: Stratum '{stratum}' has only {len(stratum_data)} members but needs {target_n}")
            print(f"WARNING: Sampling WITH replacement for this stratum")

        sampled = stratum_data.sample(n=target_n, replace=replace,
                                     random_state=random_state if random_state else None)
        sampled_dfs.append(sampled)

    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.drop(columns=['_strata'])

    if verbose:
        max_diff = 0
        for col in strata_cols:
            orig_dist = df[col].value_counts(normalize=True)
            samp_dist = result[col].value_counts(normalize=True)
            col_max_diff = max(abs(samp_dist.get(cat, 0) - orig_dist.get(cat, 0))
                              for cat in orig_dist.index)
            max_diff = max(max_diff, col_max_diff)

        print(f"INFO: Maximum demographic deviation: {max_diff*100:.2f}%")

    return result

def load_sample_max_agents(filepath):
    """Load 385 demographically representative complete-case agents (has_alpha=True & has_rho=True)"""
    df = pd.read_csv(filepath)
    complete = df[df['has_alpha'] & df['has_rho']].copy()

    # Sort by nomem_encr to ensure stable row ordering across runs
    complete = complete.sort_values('nomem_encr').reset_index(drop=True)

    # Target stratification for n=385 (perfect age representation)
    age_targets = {
        '18-29': 56,  # All available (bottleneck)
        '30-39': 54,
        '40-49': 56,
        '50-59': 68,
        '60-69': 80,
        '70+': 71
    }

    sampled = []
    for age_group, n_target in age_targets.items():
        group = complete[complete['age_group'] == age_group].reset_index(drop=True)
        if len(group) < n_target:
            print(f"WARNING: Only {len(group)} agents in {age_group}, need {n_target}")
            sampled.append(group)
        else:
            sampled.append(group.sample(n=n_target, replace=False, random_state=42))

    result = pd.concat(sampled, ignore_index=True)
    print(f"Sample-max mode: {len(result)} agents with perfect age stratification")
    return result
