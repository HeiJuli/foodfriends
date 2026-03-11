#!/usr/bin/env python3
"""
Demonstration script for empirical homophily network topology.
Visualizes network structure colored by dietary choice (veg/meat).
Always draws agents from the sample-max pool; generates network directly on the N sampled agents.
Adjusts vegetarian fraction to 6% by flipping the most diet-inclined meat-eaters (same as main model).
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auxillary.homophily_network_v2 import generate_homophily_network_v2

SURVEY_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'hierarchical_agents.csv')
ATTR_WEIGHTS = np.array([0.20, 0.35, 0.18, 0.32, 0.05])
COLOR_VEG  = (0.2, 0.7, 0.2, 0.8)
COLOR_MEAT = (0.8, 0.2, 0.2, 0.7)


def load_sample_max_agents(filepath=SURVEY_FILE):
    df = pd.read_csv(filepath)
    complete = df[df['has_alpha'] & df['has_rho']].copy().sort_values('nomem_encr').reset_index(drop=True)
    age_targets = {'18-29': 56, '30-39': 54, '40-49': 56, '50-59': 68, '60-69': 80, '70+': 71}
    sampled = [
        group.sample(n=min(len(group), n_target), replace=False, random_state=42)
        for age_group, n_target in age_targets.items()
        for group in [complete[complete['age_group'] == age_group].reset_index(drop=True)]
    ]
    result = pd.concat(sampled, ignore_index=True)
    print(f"Sample-max: {len(result)} agents loaded")
    return result


def adjust_veg_fraction(agents_df, target=0.06, seed=42):
    """Flip highest-(rho,alpha) meat-eaters to veg until target fraction reached."""
    df = agents_df.copy()
    current_veg = (df['diet'] == 'veg').sum()
    needed = max(0, int(target * len(df)) - current_veg)
    if needed == 0:
        return df
    candidates = df[df['diet'] == 'meat'].sort_values(['rho', 'alpha'], ascending=False)
    flip_idx = candidates.index[:needed]
    df.loc[flip_idx, 'diet'] = 'veg'
    actual = (df['diet'] == 'veg').sum() / len(df)
    print(f"INFO: Flipped {needed} meat-eaters -> veg (target: {target:.3f}, actual: {actual:.3f})")
    return df


def create_and_visualize_homophilic(N=25, output_file=None, seed=42):
    # Always draw agents from sample-max pool
    all_agents = load_sample_max_agents()
    N_full = len(all_agents)

    # Sample N agents from pool (random, seeded); generate network directly on them
    if N < N_full:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(N_full, size=N, replace=False))
        agents_df = all_agents.iloc[idx].reset_index(drop=True)
        print(f"Sampled {N} agents from {N_full} sample-max pool")
    else:
        agents_df = all_agents

    # Adjust veg fraction to 6% (same mechanism as main model)
    agents_df = adjust_veg_fraction(agents_df, target=0.06, seed=seed)

    print(f"Generating homophilic_emp network on {len(agents_df)} agents")
    G, _ = generate_homophily_network_v2(
        N=len(agents_df), avg_degree=8,
        agents_df=agents_df,
        attribute_weights=ATTR_WEIGHTS,
        seed=seed, tc=0.7
    )
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_colors = [COLOR_VEG if agents_df.iloc[i]['diet'] == 'veg' else COLOR_MEAT
                   for i in range(len(G.nodes()))]

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors, node_size=80,
        edge_color=[(0, 0, 0, 0.2)], width=0.6,
        with_labels=False
    )
    ax.set_title(f"Homophilic empirical network  (N={G.number_of_nodes()})", fontsize=13)
    ax.axis('off')

    out = output_file or 'network_output.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate homophilic empirical network topology")
    parser.add_argument("-N", type=int, default=50, help="Number of nodes to display (default: 50)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output filename (e.g., homophilic_network.png)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    create_and_visualize_homophilic(N=args.N, output_file=args.output, seed=args.seed)
