#!/usr/bin/env python3
"""
Demonstration script for PATCH network topology from NETIN package.
Visualizes network structure with minority/majority node types.
"""

import netin
import argparse


def create_and_visualize_patch(N=50, minority_fraction=0.1, output_file=None):
    """
    Create and visualize a PATCH network with minority/majority groups.

    Parameters:
    -----------
    N : int
        Number of nodes (default: 50)
    minority_fraction : float
        Fraction of minority nodes (default: 0.1)
    output_file : str or None
        Output filename (default: None, displays instead)
    """
    print(f"Generating PATCH network with N={N}, minority_fraction={minority_fraction}")

    # Calculate minority and majority sizes
    n_minority = int(N * minority_fraction)
    n_majority = N - n_minority

    print(f"  Minority nodes: {n_minority}")
    print(f"  Majority nodes: {n_majority}")

    # Create PATCH network with two types
    graph = netin.PATCH(
        n=[n_minority, n_majority],
        f=[[1, 0], [0, 1]],  # Strong homophily: types prefer their own
        tc=0.5,              # Triadic closure probability
        h_dd=0.1,            # Degree-degree correlation
        plo_M=2.5            # Power-law exponent for mixing
    )

    print(f"Network generated: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Visualize with enhanced parameters for presentation
    netin.viz.handlers.plot_graph(
        graph,
        fn=output_file,
        ignore_singletons=False,
        cell_size=12,            # Large figure size
        node_size=300,           # Large nodes for visibility
        node_shape='o',
        edge_width=1.5,          # Thicker edges
        edge_style='solid',
        edge_arrows=False
    )

    if output_file:
        print(f"Visualization saved to: {output_file}")
    else:
        print("Displaying visualization...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate PATCH network topology")
    parser.add_argument("-N", type=int, default=50, help="Number of nodes (default: 50)")
    parser.add_argument("-m", "--minority", type=float, default=0.1,
                        help="Minority fraction (default: 0.1)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output filename (e.g., patch_network.png)")

    args = parser.parse_args()

    create_and_visualize_patch(
        N=args.N,
        minority_fraction=args.minority,
        output_file=args.output
    )
