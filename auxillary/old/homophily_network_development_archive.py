"""
HOMOPHILY NETWORK DEVELOPMENT ARCHIVE
======================================

This archive consolidates experimental homophily network development scripts.
These were used during development of homophily_network_v2.py (production version).

PRODUCTION CODE: Use homophily_network_v2.py instead
ARCHIVED: 2025-02-13

Scripts consolidated:
- homophily_network.py (v1 - deprecated)
- compute_normalized_homophily.py
- debug_weights.py
- diagnose_homophily.py
- diagnose_homophily_resistance.py
- test_attribute_weighting.py
- test_final_calibration.py
- test_homophily_weights_literature.py
- test_similarity_exponent.py
- test_weighting_v2.py
- visualize_homophily_network.py

Key concepts developed:
1. Normalized homophily: H_norm = (observed - expected)/(1 - expected)
2. Attribute weighting for similarity calculation
3. Similarity exponent for strengthening homophily (σ^exponent)
4. Literature calibration (gender: 0.10-0.20, age/edu: 0.15-0.25)

Reference implementations below - DO NOT USE IN PRODUCTION
"""

# ============================================================================
# NORMALIZED HOMOPHILY CALCULATION
# ============================================================================
# From: compute_normalized_homophily.py
# Purpose: Account for null expectation in multi-category attributes
#
# Formula: H_norm = (observed - expected) / (1 - expected)
# Where: expected = sum(p_i^2) for each category proportion p_i
#
# Range: [-1, 1]
#   H_norm =  1: perfect homophily
#   H_norm =  0: random connections
#   H_norm = -1: perfect heterophily

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Optional, Tuple

def compute_normalized_homophily(G: nx.Graph, agents_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute normalized homophily coefficients for categorical attributes.
    Reference: Newman (2003) Mixing patterns in networks, Phys Rev E
    """
    results = {}
    n_edges = G.number_of_edges()

    for attr in ['gender', 'age_group', 'incquart', 'educlevel', 'diet']:
        same_attr = sum(1 for i, j in G.edges() if G.nodes[i][attr] == G.nodes[j][attr])
        frac_observed = same_attr / n_edges if n_edges > 0 else 0

        # Expected fraction if random
        value_counts = agents_df[attr].value_counts(normalize=True)
        frac_expected = (value_counts ** 2).sum()

        # Normalized homophily
        H_norm = (frac_observed - frac_expected) / (1 - frac_expected) if frac_expected < 1.0 else 0.0
        enrichment = frac_observed / frac_expected if frac_expected > 0 else 1.0

        results[attr] = {
            'frac_observed': frac_observed,
            'frac_expected': frac_expected,
            'H_norm': H_norm,
            'enrichment': enrichment
        }

    return results


# ============================================================================
# SIMILARITY EXPONENT METHOD
# ============================================================================
# From: test_similarity_exponent.py
# Purpose: Strengthen homophily by raising similarity to power > 1
#
# P(i,j) ∝ σ(i,j)^exponent
#   exponent > 1: winner-take-all (strengthens homophily)
#   exponent = 1: original Lackner algorithm
#   exponent < 1: weakens homophily

def generate_network_with_exponent(
    N, avg_degree, agents_df, attribute_weights, similarity_exponent=1.0, seed=None
):
    """
    Network generator with adjustable similarity exponent.
    Higher exponent → stronger homophily (winner-take-all dynamics).
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute similarity matrix (implementation-specific)
    # sim_matrix = compute_similarity_matrix(agents_df, attribute_weights)
    # sim_matrix_exp = sim_matrix ** similarity_exponent

    # Use exponentiated similarities for link probability
    # P(i,j) = sim_exp[i,j] / sum(sim_exp[i,:])

    pass  # See test_similarity_exponent.py for full implementation


# ============================================================================
# ATTRIBUTE WEIGHTING
# ============================================================================
# From: test_attribute_weighting.py, test_weighting_v2.py
# Purpose: Find optimal weights for similarity calculation
#
# Tested weight configurations:
# - Equal weights: [0.2, 0.2, 0.2, 0.2, 0.2]
# - Demographics-heavy: [0.25, 0.25, 0.25, 0.25, 0.0]
# - Age-heavy: [0.2, 0.4, 0.2, 0.2, 0.0]
# - Best found: [0.20, 0.35, 0.18, 0.32, 0.05]
#
# Target ranges (from literature):
# - Gender: 0.10 - 0.20
# - Age: 0.15 - 0.25
# - Education: 0.15 - 0.25

LITERATURE_TARGETS = {
    'gender': (0.10, 0.20),
    'age_group': (0.15, 0.25),
    'educlevel': (0.15, 0.25)
}

BEST_WEIGHTS = np.array([0.20, 0.35, 0.18, 0.32, 0.05])  # [gender, age, income, edu, theta]


# ============================================================================
# HOMOPHILY NETWORK V1 (DEPRECATED)
# ============================================================================
# From: homophily_network.py
# Status: Superseded by homophily_network_v2.py
#
# Key functions (for reference only):
# - normalize_attributes(agents_df): Normalize to [0,1]
# - compute_similarity(attr_i, attr_j, weights): L1 similarity
# - compute_similarity_matrix(attr_matrix, weights): Pairwise similarities
# - generate_homophily_network(N, avg_degree, agents_df, weights): Lackner algorithm
# - compute_homophily_coefficients(G, agents_df): Traditional H = (same-diff)/(same+diff)
# - get_network_stats(G): Basic network statistics

def normalize_attributes_v1(agents_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    V1 attribute normalization (deprecated - see v2).
    Gender: {Male, Female} -> {0, 1}
    Age: categorical -> [0, 1]
    Income: {1,2,3,4} -> {0, 0.33, 0.67, 1}
    Education: categorical -> [0, 1]
    Alpha: [0, 1]
    Theta: [-1, 1] -> [0, 1]
    Rho: [0, 1]
    """
    pass  # See homophily_network.py for full implementation


# ============================================================================
# DIAGNOSTIC SCRIPTS
# ============================================================================
# From: debug_weights.py, diagnose_homophily.py, diagnose_homophily_resistance.py
# Purpose: Debug why homophily wasn't reaching literature values
#
# Key findings:
# 1. Need higher weights on age/education (0.35, 0.32)
# 2. Similarity exponent helps (2.0-2.5)
# 3. Theta has minimal impact on demographic homophily (weight 0.05)
# 4. Combined approach works best: optimized weights + moderate exponent

# diagnose_homophily_resistance.py tested:
# - Different weight configurations
# - Similarity exponents
# - Final recommendation: weights=[0.20,0.35,0.18,0.32,0.05], exponent=2.0


# ============================================================================
# CALIBRATION TESTING
# ============================================================================
# From: test_final_calibration.py, test_homophily_weights_literature.py
# Purpose: Validate homophily values against literature
#
# Literature sources:
# - McPherson et al. (2001): Strong demographic homophily in social networks
# - Newman (2003): Mixing patterns methodology
# - Lackner et al. (2024): Dietary behavior model with homophily
#
# Final calibrated parameters (integrated into v2):
# - Weights: [0.20, 0.35, 0.18, 0.32, 0.05]
# - Exponent: 2.0
# - Achieves H_norm in literature range for gender, age, education


# ============================================================================
# VISUALIZATION (OLD)
# ============================================================================
# From: visualize_homophily_network.py
# Status: Superseded by visualize_homophilic_emp_network.py
#
# Visualizations included:
# - Network colored by attribute
# - Degree distribution
# - Similarity matrix heatmap
# - Homophily coefficient bar plots

def visualize_network_by_attribute_v1(G, attribute, layout='spring', save_path=None):
    """Deprecated - see visualize_homophilic_emp_network.py for current version"""
    pass


# ============================================================================
# USAGE NOTES
# ============================================================================
#
# DO NOT use these archived implementations in production code.
#
# For current homophily network generation:
#   from auxillary.homophily_network_v2 import generate_homophily_network_v2
#
# For normalized homophily calculation:
#   from auxillary.compute_normalized_homophily import compute_normalized_homophily
#   (If this becomes needed in production, extract from this archive)
#
# For visualization:
#   Use auxillary/visualize_homophilic_emp_network.py
#
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("HOMOPHILY NETWORK DEVELOPMENT ARCHIVE")
    print("="*80)
    print("\nThis is a reference archive - not for production use.")
    print("\nFor current homophily network generation:")
    print("  from auxillary.homophily_network_v2 import generate_homophily_network_v2")
    print("\nKey development outcomes:")
    print("  1. Normalized homophily metric: H_norm = (obs - exp)/(1 - exp)")
    print("  2. Optimized attribute weights: [0.20, 0.35, 0.18, 0.32, 0.05]")
    print("  3. Similarity exponent: 2.0")
    print("  4. Literature-calibrated homophily values")
    print("="*80)
