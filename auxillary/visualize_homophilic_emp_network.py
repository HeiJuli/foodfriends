"""
Visualize the homophilic_emp network with meat eaters and vegetarians shown.
Shows network layout colored by diet and demographic attributes.

Usage:
    python visualize_homophilic_emp_network.py [--N 500] [--layout spring]
"""

import sys
sys.path.append('../model_src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import argparse
from pathlib import Path
from model_main_single import Model
from compute_normalized_homophily import compute_normalized_homophily

def visualize_diet_network(G, pos, title="Homophilic Network by Diet", ax=None):
    """Visualize network colored by diet"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # Get diet colors
    node_colors = []
    for node in G.nodes():
        diet = G.nodes[node]['diet']
        if diet == 'veg':
            node_colors.append('#2a9d8f')  # Teal for vegetarian
        else:
            node_colors.append('#e76f51')  # Coral for meat

    # Draw network
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=30, alpha=0.8, edgecolors='black', linewidths=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2a9d8f', edgecolor='black', label='Vegetarian'),
        Patch(facecolor='#e76f51', edgecolor='black', label='Meat eater')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    return ax

def visualize_attribute_network(G, pos, attribute='gender', title=None, ax=None):
    """Visualize network colored by demographic attribute"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique values and assign colors
    unique_vals = list(set(G.nodes[node][attribute] for node in G.nodes()))
    cmap = plt.cm.Set3
    color_map = {val: cmap(i/len(unique_vals)) for i, val in enumerate(unique_vals)}

    node_colors = [color_map[G.nodes[node][attribute]] for node in G.nodes()]

    # Draw network
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=30, alpha=0.8, edgecolors='black', linewidths=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[val], edgecolor='black', label=str(val))
                      for val in sorted(unique_vals)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)

    if title is None:
        title = f"Network by {attribute.replace('_', ' ').title()}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    return ax

def main():
    parser = argparse.ArgumentParser(description='Visualize homophilic_emp network')
    parser.add_argument('--N', type=int, default=500,
                       help='Number of agents (default: 500)')
    parser.add_argument('--layout', type=str, default='spring',
                       choices=['spring', 'spectral', 'kamada_kawai'],
                       help='Network layout algorithm (default: spring)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()

    print("="*80)
    print("VISUALIZING HOMOPHILIC_EMP NETWORK")
    print("="*80)

    # Load PMF tables
    print("\nLoading PMF tables...")
    with open("../data/demographic_pmfs.pkl", "rb") as f:
        pmf_tables = pickle.load(f)

    # Model parameters
    params = {
        "N": args.N,
        "steps": 100,
        "k": 8,
        "erdos_p": 0.001,
        "p_rewire": 0.1,
        "rewire_h": 0.1,
        "tc": 0.3,
        'topology': "homophilic_emp",
        "alpha": 0.36,
        "rho": 0.45,
        "theta": 0.44,
        "agent_ini": "twin",
        "survey_file": "../data/hierarchical_agents.csv",
        "adjust_veg_fraction": False,
        "target_veg_fraction": 0.06,
        "immune_n": 0.0,
        "M": 7,
        "veg_f": 0.1,
        "meat_f": 0.9,
        "seed": args.seed,
        "veg_CO2": 1.5,
        "meat_CO2": 7.2
    }

    print(f"\nGenerating network with N={args.N}...")
    model = Model(params, pmf_tables=pmf_tables)
    model.agent_ini()

    G = model.G1

    print(f"\nNetwork statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Mean degree: {np.mean([d for n,d in G.degree()]):.2f}")
    print(f"  Connected components: {nx.number_connected_components(G)}")

    # Diet composition
    diets = [G.nodes[node]['diet'] for node in G.nodes()]
    veg_count = sum(d == 'veg' for d in diets)
    print(f"\nDiet composition:")
    print(f"  Vegetarian: {veg_count} ({veg_count/len(diets)*100:.1f}%)")
    print(f"  Meat eater: {len(diets)-veg_count} ({(len(diets)-veg_count)/len(diets)*100:.1f}%)")

    # Compute homophily
    print(f"\nComputing homophily coefficients...")
    homophily = compute_normalized_homophily(G, model.survey_data)
    print(f"\nHomophily (H_norm):")
    for attr in ['gender', 'age_group', 'educlevel', 'incquart']:
        h = homophily[attr]['H_norm']
        print(f"  {attr:15s}: {h:.3f}")

    # Compute layout
    print(f"\nComputing {args.layout} layout...")
    if args.layout == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=args.seed)
    elif args.layout == 'spectral':
        pos = nx.spectral_layout(G, seed=args.seed)
    else:  # kamada_kawai
        pos = nx.kamada_kawai_layout(G)

    # Create output directory
    output_dir = Path("../visualisations_output/homophily")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print(f"\nGenerating visualizations...")

    # 1. Main plot: 2x2 grid of networks by different attributes
    print("  - Main 4-panel plot")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    visualize_diet_network(G, pos, "Network by Diet", ax=axes[0, 0])
    visualize_attribute_network(G, pos, 'gender', "Network by Gender", ax=axes[0, 1])
    visualize_attribute_network(G, pos, 'age_group', "Network by Age Group", ax=axes[1, 0])
    visualize_attribute_network(G, pos, 'educlevel', "Network by Education Level", ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(output_dir / f'homophilic_emp_network_N{args.N}.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_dir / f'homophilic_emp_network_N{args.N}.png'}")
    plt.close()

    # 2. Large single plot focused on diet
    print("  - Large diet network plot")
    fig, ax = plt.subplots(figsize=(14, 12))
    visualize_diet_network(G, pos,
                          f"Homophilic Empirical Network (N={args.N}, Option A)",
                          ax=ax)

    # Add network stats text
    stats_text = (f"Network Statistics:\n"
                 f"Edges: {G.number_of_edges()}\n"
                 f"Mean degree: {np.mean([d for n,d in G.degree()]):.2f}\n"
                 f"Vegetarian: {veg_count/len(diets)*100:.1f}%\n"
                 f"\n"
                 f"Homophily (H_norm):\n"
                 f"Gender: {homophily['gender']['H_norm']:.3f}\n"
                 f"Age: {homophily['age_group']['H_norm']:.3f}\n"
                 f"Education: {homophily['educlevel']['H_norm']:.3f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / f'homophilic_emp_diet_N{args.N}.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_dir / f'homophilic_emp_diet_N{args.N}.png'}")
    plt.close()

    # 3. Degree distribution by diet
    print("  - Degree distribution by diet")
    fig, ax = plt.subplots(figsize=(10, 6))

    veg_degrees = [G.degree(node) for node in G.nodes() if G.nodes[node]['diet'] == 'veg']
    meat_degrees = [G.degree(node) for node in G.nodes() if G.nodes[node]['diet'] == 'meat']

    ax.hist(veg_degrees, bins=range(min(veg_degrees), max(veg_degrees)+2),
           alpha=0.6, label='Vegetarian', color='#2a9d8f', edgecolor='black')
    ax.hist(meat_degrees, bins=range(min(meat_degrees), max(meat_degrees)+2),
           alpha=0.6, label='Meat eater', color='#e76f51', edgecolor='black')

    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Degree Distribution by Diet', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'degree_distribution_diet_N{args.N}.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_dir / f'degree_distribution_diet_N{args.N}.png'}")
    plt.close()

    print(f"\n{'='*80}")
    print(f"DONE! All plots saved to {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
