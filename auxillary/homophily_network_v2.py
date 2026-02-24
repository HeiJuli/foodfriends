"""
Homophily-based network generation for agent-based dietary behavior models.
VERSION 2: Demographics + theta only (excludes diet, alpha, rho from similarity)

Based on Lackner et al. (2024) Supplementary Information Section B.3.
Generates social networks where link formation depends on demographic and
intrinsic preference (theta) similarity between agents.

Design decisions:
- Diet excluded: should be outcome, not network formation mechanism
- Alpha/rho excluded: too many missing values (77% incomplete)
- Theta included: available for all agents, represents stable preference

Author: Generated for foodfriends project
Date: 2025-01-20
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt

def normalize_attributes_v2(agents_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Normalize agent attributes to [0, 1] range for similarity calculation.

    Attributes included (5 dimensions):
    - Gender: {male, female} -> {0, 1}
    - Age group: categorical -> normalized to [0, 1]
    - Income quartile: {1, 2, 3, 4} -> {0, 0.33, 0.67, 1}
    - Education level: categorical -> normalized to [0, 1]
    - Theta: [-1, 1] -> [0, 1]

    Attributes EXCLUDED:
    - Diet: should be outcome, not mechanism
    - Alpha/rho: too many missing values

    Parameters
    ----------
    agents_df : pd.DataFrame
        Agent data with columns: gender, age_group, incquart, educlevel, theta

    Returns
    -------
    attr_matrix : np.ndarray
        Normalized attribute matrix (N x 5)
    agents_df : pd.DataFrame
        Copy of input with normalized columns added
    """
    df = agents_df.copy()
    N = len(df)
    attr_matrix = np.zeros((N, 5))

    # Gender: Male=0, Female=1
    attr_matrix[:, 0] = (df['gender'] == 'Female').astype(float)

    # Age group: convert to numeric then normalize
    age_map = {'18-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70+': 5}
    age_numeric = df['age_group'].map(age_map)
    age_min, age_max = age_numeric.min(), age_numeric.max()
    attr_matrix[:, 1] = (age_numeric - age_min) / (age_max - age_min) if age_max > age_min else 0.5

    # Income quartile: {1,2,3,4} -> {0, 0.33, 0.67, 1}
    attr_matrix[:, 2] = (df['incquart'] - 1) / 3.0

    # Education level: normalize to [0, 1]
    edu_min, edu_max = df['educlevel'].min(), df['educlevel'].max()
    attr_matrix[:, 3] = (df['educlevel'] - edu_min) / (edu_max - edu_min) if edu_max > edu_min else 0.5

    # Theta: [-1, 1] -> [0, 1]
    attr_matrix[:, 4] = (df['theta'].values + 1) / 2.0

    # Add normalized columns to dataframe
    df['gender_norm'] = attr_matrix[:, 0]
    df['age_norm'] = attr_matrix[:, 1]
    df['income_norm'] = attr_matrix[:, 2]
    df['edu_norm'] = attr_matrix[:, 3]
    df['theta_norm'] = attr_matrix[:, 4]

    return attr_matrix, df


def compute_similarity_v2(attr_i: np.ndarray, attr_j: np.ndarray,
                         weights: Optional[np.ndarray] = None) -> float:
    """
    Compute similarity between two agents using weighted Manhattan distance.

    From Lackner et al. Equation B.3:
    σ(vi, vj) = (W - Σ w_m|a^i_m - a^j_m|) / W, where W = Σ w_m

    Parameters
    ----------
    attr_i, attr_j : np.ndarray
        Normalized attribute vectors (length M=5)
    weights : np.ndarray, optional
        Attribute weights (default: equal weights)

    Returns
    -------
    similarity : float
        Similarity in (0, 1), where 1 = identical
    """
    if weights is None:
        weights = np.ones(len(attr_i))

    weighted_dist = np.sum(weights * np.abs(attr_i - attr_j))
    weight_sum = np.sum(weights)
    return (weight_sum - weighted_dist) / weight_sum


def compute_similarity_matrix_v2(attr_matrix: np.ndarray,
                                 weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute pairwise similarity matrix for all agents.

    Parameters
    ----------
    attr_matrix : np.ndarray
        Normalized attribute matrix (N x M=5)
    weights : np.ndarray, optional
        Attribute weights (default: equal weights)

    Returns
    -------
    sim_matrix : np.ndarray
        Symmetric similarity matrix (N x N)
    """
    N = attr_matrix.shape[0]
    sim_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1, N):
            sim = compute_similarity_v2(attr_matrix[i], attr_matrix[j], weights)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    # Diagonal: self-similarity = 1
    np.fill_diagonal(sim_matrix, 1.0)

    return sim_matrix


def _homophily_target(source, target_set, sim_matrix, G):
    """Select target weighted by similarity * (degree + epsilon), ala PATCH."""
    targets = np.array(list(target_set))
    if len(targets) == 0:
        return None
    EPSILON = 1e-5
    sims = sim_matrix[source, targets]
    degrees = np.array([G.degree(t) + EPSILON for t in targets])
    weights = sims * degrees
    s = weights.sum()
    if s <= 0:
        return int(np.random.choice(targets))
    return int(np.random.choice(targets, p=weights / s))


def _tc_target(source, special_targets):
    """Select target from FOF accumulator weighted by encounter count only (ala PATCH)."""
    if not special_targets:
        return None
    targets = np.array(list(special_targets.keys()))
    counts = np.array([special_targets[t] for t in targets], dtype=float)
    s = counts.sum()
    if s <= 0:
        return int(np.random.choice(targets))
    return int(np.random.choice(targets, p=counts / s))


def generate_homophily_network_v2(
    N: int,
    avg_degree: int,
    agents_df: pd.DataFrame,
    attribute_weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    tc: float = 0.95,
    target_clustering: Optional[float] = None
) -> nx.Graph:
    """
    Generate homophily-based social network with triadic closure.
    Follows Holme-Kim (2002) incremental growth model with homophily
    replacing preferential attachment:

    1. Nodes arrive one at a time, each adding m = avg_degree // 2 edges
    2. First edge per node: similarity-weighted selection from existing nodes
    3. Subsequent edges: with probability tc, select from FOF accumulator
       (weighted by occurrence count * similarity); otherwise homophily
    4. FOF accumulator tracks friends-of-friends across ALL current neighbors

    Parameters
    ----------
    N : int
        Number of agents
    avg_degree : int
        Target average degree (m = avg_degree // 2 edges per arriving node)
    agents_df : pd.DataFrame
        Agent data with demographics and theta
    attribute_weights : np.ndarray, optional
        Weights for 5 attributes: [gender, age, income, education, theta]
    seed : int, optional
        Random seed for reproducibility
    tc : float
        Triadic closure probability (0-1). Higher = more clustering. Default 0.95.
    target_clustering : float, optional
        If provided, print warning if achieved clustering differs by >0.05.

    Returns
    -------
    G : nx.Graph
        NetworkX graph with N nodes, homophilic edges, and triadic closure
    sim_matrix : np.ndarray
        Pairwise similarity matrix (N x N), for use in rewiring
    """
    if seed is not None:
        np.random.seed(seed)

    if len(agents_df) != N:
        raise ValueError(f"agents_df has {len(agents_df)} rows but N={N}")

    m = max(1, avg_degree // 2)  # edges per arriving node

    attr_matrix, df_norm = normalize_attributes_v2(agents_df)
    sim_matrix = compute_similarity_matrix_v2(attr_matrix, attribute_weights)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Add agent attributes to nodes
    for i in range(N):
        for col in ('gender', 'age_group', 'incquart', 'educlevel', 'theta', 'diet'):
            G.nodes[i][col] = agents_df.iloc[i][col]

    # Seed network: first m nodes form a clique (Holme-Kim initialisation)
    m0 = min(m + 1, N)  # need at least m+1 nodes for the seed
    for i in range(m0):
        for j in range(i + 1, m0):
            G.add_edge(i, j)

    # Incremental node arrival (Holme-Kim growth model)
    for source in range(m0, N):
        # Available targets: all nodes with id < source, excluding self
        target_set = {t for t in range(source) if not G.has_edge(source, t)}
        special_targets = defaultdict(int)  # FOF accumulator

        for idx in range(min(m, len(target_set))):
            target = None

            # First edge: always homophily. Subsequent: try TC with prob tc.
            if idx > 0 and special_targets and np.random.random() < tc:
                # Filter special_targets to only non-connected nodes
                valid_st = {t: c for t, c in special_targets.items()
                            if t in target_set}
                if valid_st:
                    target = _tc_target(source, valid_st)

            # Fallback to homophily+PA if TC wasn't attempted or failed
            if target is None:
                target = _homophily_target(source, target_set, sim_matrix, G)

            if target is None:
                break

            # Add edge
            G.add_edge(source, target)

            # Update FOF accumulator (netin-style): add all neighbors of
            # the newly connected target as candidates, increment counts
            target_set.discard(target)
            if target in special_targets:
                del special_targets[target]
            # Only accumulate if more edges to add (per Holme-Kim)
            if idx < m - 1:
                for neighbor in G.neighbors(target):
                    if neighbor != source and not G.has_edge(source, neighbor):
                        special_targets[neighbor] += 1

    if target_clustering is not None:
        achieved = nx.average_clustering(G)
        if abs(achieved - target_clustering) > 0.05:
            print(f"WARNING: clustering {achieved:.3f} differs from target "
                  f"{target_clustering:.3f} by "
                  f"{abs(achieved - target_clustering):.3f}")

    return G, sim_matrix


def get_network_stats_v2(G: nx.Graph) -> Dict[str, float]:
    """Compute basic network statistics."""
    degrees = [d for n, d in G.degree()]

    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'mean_degree': np.mean(degrees),
        'median_degree': np.median(degrees),
        'std_degree': np.std(degrees),
        'min_degree': np.min(degrees),
        'max_degree': np.max(degrees),
        'clustering': nx.average_clustering(G),
        'n_components': nx.number_connected_components(G),
        'largest_component_size': len(max(nx.connected_components(G), key=len)),
        'density': nx.density(G)
    }

    return stats


def compute_homophily_coefficients_v2(G: nx.Graph, agents_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute homophily coefficients for each attribute.

    Homophily coefficient measures tendency for similar nodes to connect:
    H = (E_same - E_diff) / (E_same + E_diff)
    """
    homophily = {}

    # Gender homophily
    same, diff = 0, 0
    for i, j in G.edges():
        if G.nodes[i]['gender'] == G.nodes[j]['gender']:
            same += 1
        else:
            diff += 1
    homophily['gender'] = (same - diff) / (same + diff) if (same + diff) > 0 else 0

    # Age group homophily
    same, diff = 0, 0
    for i, j in G.edges():
        if G.nodes[i]['age_group'] == G.nodes[j]['age_group']:
            same += 1
        else:
            diff += 1
    homophily['age_group'] = (same - diff) / (same + diff) if (same + diff) > 0 else 0

    # Income homophily
    same, diff = 0, 0
    for i, j in G.edges():
        if G.nodes[i]['incquart'] == G.nodes[j]['incquart']:
            same += 1
        else:
            diff += 1
    homophily['income'] = (same - diff) / (same + diff) if (same + diff) > 0 else 0

    # Education homophily
    same, diff = 0, 0
    for i, j in G.edges():
        if G.nodes[i]['educlevel'] == G.nodes[j]['educlevel']:
            same += 1
        else:
            diff += 1
    homophily['education'] = (same - diff) / (same + diff) if (same + diff) > 0 else 0

    # Diet homophily (for monitoring, NOT used in network formation)
    same, diff = 0, 0
    for i, j in G.edges():
        if G.nodes[i]['diet'] == G.nodes[j]['diet']:
            same += 1
        else:
            diff += 1
    homophily['diet'] = (same - diff) / (same + diff) if (same + diff) > 0 else 0

    return homophily


if __name__ == "__main__":
    print("Testing homophily network module V2 (with triadic closure)...")
    print("(demographics + theta only, excludes diet/alpha/rho)")

    agents_df = pd.read_csv('../data/hierarchical_agents.csv')
    N_test = 500
    agents_sample = agents_df.sample(n=N_test, random_state=42).reset_index(drop=True)

    # Test multiple tc values
    for tc_val in [0.0, 0.5, 0.8, 0.95]:
        print(f"\n--- tc={tc_val} ---")
        print(f"Generating network with N={N_test}, avg_degree=8, tc={tc_val}...")
        G, _ = generate_homophily_network_v2(
            N=N_test, avg_degree=8, agents_df=agents_sample,
            seed=42, tc=tc_val
        )
        stats = get_network_stats_v2(G)
        print(f"  Clustering: {stats['clustering']:.4f}")
        print(f"  Mean degree: {stats['mean_degree']:.2f}")
        print(f"  Components: {stats['n_components']}")

        if tc_val == 0.95:
            homophily = compute_homophily_coefficients_v2(G, agents_sample)
            print("  Homophily coefficients:")
            for key, val in homophily.items():
                print(f"    {key}: {val:.4f}")

    print("\nModule test complete!")
