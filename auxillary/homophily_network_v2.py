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


def generate_homophily_network_v2(
    N: int,
    avg_degree: int,
    agents_df: pd.DataFrame,
    attribute_weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate homophily-based social network using Lackner et al. algorithm.
    VERSION 2: Uses demographics + theta only.

    From Lackner et al. Equation B.2:
    P(vi, vj) = σ(vi, vj) / Σ_n σ(vi, vn)

    Each agent forms connections based on similarity.

    Parameters
    ----------
    N : int
        Number of agents
    avg_degree : int
        Target average degree
    agents_df : pd.DataFrame
        Agent data with demographics and theta
    attribute_weights : np.ndarray, optional
        Weights for 5 attributes: [gender, age, income, education, theta]
        Default: equal weights
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    G : nx.Graph
        NetworkX graph with N nodes and homophilic edges
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure we have N agents
    if len(agents_df) != N:
        raise ValueError(f"agents_df has {len(agents_df)} rows but N={N}")

    # Normalize attributes (demographics + theta only)
    attr_matrix, df_norm = normalize_attributes_v2(agents_df)

    # Compute similarity matrix (with optional weights)
    sim_matrix = compute_similarity_matrix_v2(attr_matrix, attribute_weights)

    # Initialize graph
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Add agent attributes to nodes
    for i in range(N):
        G.nodes[i]['gender'] = agents_df.iloc[i]['gender']
        G.nodes[i]['age_group'] = agents_df.iloc[i]['age_group']
        G.nodes[i]['incquart'] = agents_df.iloc[i]['incquart']
        G.nodes[i]['educlevel'] = agents_df.iloc[i]['educlevel']
        G.nodes[i]['theta'] = agents_df.iloc[i]['theta']
        G.nodes[i]['diet'] = agents_df.iloc[i]['diet']

    # Network generation: each agent forms connections based on similarity
    for i in range(N):
        # Get similarity scores to all other agents
        similarities = sim_matrix[i].copy()

        # Exclude self
        similarities[i] = 0

        # Exclude already connected neighbors
        for neighbor in G.neighbors(i):
            similarities[neighbor] = 0

        # Normalize to get probabilities
        sim_sum = similarities.sum()
        if sim_sum > 0:
            probs = similarities / sim_sum
        else:
            # If no similarity, uniform random
            probs = np.ones(N) / N
            probs[i] = 0
            probs = probs / probs.sum()

        # Number of connections to make (accounting for existing degree)
        current_degree = G.degree(i)
        n_to_add = max(0, avg_degree - current_degree)

        # Sample connections
        if n_to_add > 0:
            available_nodes = [j for j in range(N) if j != i and not G.has_edge(i, j)]
            n_to_add = min(n_to_add, len(available_nodes))

            if n_to_add > 0:
                # Sample without replacement
                chosen = np.random.choice(
                    available_nodes,
                    size=n_to_add,
                    replace=False,
                    p=probs[available_nodes] / probs[available_nodes].sum()
                )

                # Add edges
                for j in chosen:
                    G.add_edge(i, j)

    return G


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
    # Test module
    print("Testing homophily network module V2...")
    print("(demographics + theta only, excludes diet/alpha/rho)")

    # Load data
    agents_df = pd.read_csv('../data/hierarchical_agents.csv')

    # Test with full sample (no need for complete cases - theta is always present)
    N_test = 500
    agents_sample = agents_df.sample(n=N_test, random_state=42).reset_index(drop=True)

    print(f"\nGenerating network with N={N_test}, avg_degree=8...")
    G = generate_homophily_network_v2(
        N=N_test,
        avg_degree=8,
        agents_df=agents_sample,
        seed=42
    )

    # Compute statistics
    stats = get_network_stats_v2(G)
    print("\nNetwork Statistics:")
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    # Compute homophily
    homophily = compute_homophily_coefficients_v2(G, agents_sample)
    print("\nHomophily Coefficients:")
    for key, val in homophily.items():
        print(f"  {key}: {val:.4f}")

    print("\nModule test complete!")
