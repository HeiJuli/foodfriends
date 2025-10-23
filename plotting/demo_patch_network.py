#!/usr/bin/env python3
"""
Demonstration script for PATCH network topology from NETIN package.
Visualizes network structure with minority/majority node types.
"""

import netin
import netin.viz.handlers
import matplotlib.pyplot as plt
import matplotlib
import argparse
import tempfile
import shutil


def create_and_visualize_patch(N=25, minority_fraction=0.1, output_file=None):
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
        n=N,                 # Total number of nodes
        k=3,                 # Minimum degree per node
        f_m=minority_fraction,  # Fraction of minority nodes
        h_mm=0.9,            # Homophily between minority nodes
        h_MM=0.9,            # Homophily between majority nodes
        tc=0.5               # Triadic closure probability
    )

    # Generate the network
    graph.generate()

    print(f"Network generated: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Monkey-patch netin to fix the parameter bug and customize colors
    # Save originals
    original_save_plot = netin.viz.handlers._save_plot
    original_get_edge_color = netin.viz.handlers._get_edge_color
    original_color_majority = netin.viz.handlers.COLOR_MAJORITY
    original_color_minority = netin.viz.handlers.COLOR_MINORITY
    original_color_mixed = netin.viz.handlers.COLOR_MIXED

    def fixed_save_plot(fig, fn=None, **kwargs):
        # Pop the visualization params that netin incorrectly leaks to savefig
        kwargs.pop('node_size', None)
        kwargs.pop('node_shape', None)
        kwargs.pop('edge_width', None)
        kwargs.pop('edge_style', None)
        kwargs.pop('edge_arrows', None)
        kwargs.pop('arrow_style', None)
        kwargs.pop('arrow_size', None)
        # Call original with cleaned kwargs
        return original_save_plot(fig, fn, **kwargs)

    def custom_edge_color(s, t, g, maj_val=None, min_val=None):
        # Make edges black unless they connect to minority nodes
        from netin.utils import constants as const
        maj_val = const.MAJORITY_VALUE if maj_val is None else maj_val
        min_val = const.MINORITY_VALUE if min_val is None else min_val

        s_val = g.get_class_value(s)
        t_val = g.get_class_value(t)

        # If either node is minority, use green for the edge
        if s_val == min_val or t_val == min_val:
            return (0.2, 0.7, 0.2, 0.6)  # Green with alpha
        # Otherwise use black
        return (0, 0, 0, 0.3)  # Black with lower alpha

    # Apply monkey patches
    netin.viz.handlers._save_plot = fixed_save_plot
    netin.viz.handlers._get_edge_color = custom_edge_color
    netin.viz.handlers.COLOR_MAJORITY = (0.8, 0.2, 0.2, 0.7)  # Softer red with alpha
    netin.viz.handlers.COLOR_MINORITY = (0.2, 0.7, 0.2, 0.8)  # Softer green with alpha
    netin.viz.handlers.COLOR_MIXED = (0, 0, 0, 0.3)  # Black for mixed edges

    # Now plot with custom node/edge sizes
    netin.viz.handlers.plot_graph(
        graph,
        fn=output_file if output_file else 'network_output.png',
        ignore_singletons=False,
        cell_size=10,
        node_size=80,
        edge_width=0.8,
        dpi=300,
        bbox_inches='tight'
    )

    # Restore originals
    netin.viz.handlers._save_plot = original_save_plot
    netin.viz.handlers._get_edge_color = original_get_edge_color
    netin.viz.handlers.COLOR_MAJORITY = original_color_majority
    netin.viz.handlers.COLOR_MINORITY = original_color_minority
    netin.viz.handlers.COLOR_MIXED = original_color_mixed

    print(f"Visualization saved to: {output_file if output_file else 'network_output.png'}")
    

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
