# -*- coding: utf-8 -*-
"""
Network analysis and visualization functions for dietary contagion model
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

def plot_network_influence_map(cluster_result, save_path=None):
    """
    Visualize the network with node sizes proportional to influence (reduction_out)
    
    Args:
        cluster_result (dict): Result from cluster analysis (containing G and attributes)
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib axis: The plot axis
    """
    # Extract required elements from cluster_result
    G = cluster_result['G']
    
    # Get reduction values and diet information
    if 'reductions' in cluster_result:
        reductions = cluster_result['reductions']
        diets = cluster_result['diets']
    else:
        # Assume these are the attributes directly in the result
        reductions = cluster_result.get('reduction_out', [0])
        diets = cluster_result.get('diets', ['veg'])
    
    # Normalize reduction values for visualization
    max_reduction = max(reductions) if max(reductions) > 0 else 1
    node_sizes = [100 + (r / max_reduction) * 500 for r in reductions]
    
    # Set node colors based on diet
    node_colors = ['green' if d == 'veg' else 'red' for d in diets]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Use force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with size based on influence
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors, 
                          alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Create legend
    veg_patch = Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Vegetarian')
    meat_patch = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                       markersize=10, label='Meat Eater')
    influence_patch = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                            markersize=15, label='Higher Influence')
    
    plt.legend(handles=[veg_patch, meat_patch, influence_patch], 
              loc='upper right', fontsize=12)
    
    final_veg_fraction = cluster_result.get('final_veg_fraction', 'N/A')
    plt.title(f'Network Influence Map (Final Veg. Fraction: {final_veg_fraction})', 
             fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gca()

def plot_vegetarian_clusters(cluster_result, save_path=None):
    """
    Visualize vegetarian clusters in the network
    
    Args:
        cluster_result (dict): Result from cluster analysis
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib axis: The plot axis
    """
    # Extract required elements from cluster_result
    G = cluster_result['G']
    clusters = cluster_result['clusters']
    diets = cluster_result.get('diets', [])
    
    # Create visualization of clusters
    plt.figure(figsize=(12, 10))
    
    # Use spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw all nodes in gray first
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=50, alpha=0.5)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    
    # Then draw each vegetarian cluster with a different color
    cluster_colors = plt.cm.tab10.colors
    for i, cluster in enumerate(clusters):
        color = cluster_colors[i % len(cluster_colors)]
        nx.draw_networkx_nodes(G, pos, nodelist=list(cluster), 
                              node_color=color, node_size=100, alpha=0.8)
    
    # Get cluster statistics
    num_clusters = cluster_result.get('num_clusters', len(clusters))
    avg_cluster_size = cluster_result.get('avg_cluster_size', np.mean([len(c) for c in clusters]) if clusters else 0)
    max_cluster_size = cluster_result.get('max_cluster_size', max([len(c) for c in clusters]) if clusters else 0)
    
    plt.title(f'Vegetarian Clusters\n'
             f'Number of Clusters: {num_clusters}, Avg Size: {avg_cluster_size:.1f}, '
             f'Max Size: {max_cluster_size}', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gca()

def plot_cluster_statistics(stats_df, save_path=None):
    """
    Plot cluster statistics vs initial vegetarian fraction
    
    Args:
        stats_df (DataFrame): DataFrame with cluster statistics
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib figure: The plot figure
    """
    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of clusters vs initial fraction
    axs[0, 0].plot(stats_df['initial_veg_fraction'], stats_df['num_clusters'], 'o-', color='#1f77b4')
    axs[0, 0].set_xlabel('Initial Vegetarian Fraction')
    axs[0, 0].set_ylabel('Number of Clusters')
    axs[0, 0].set_title('Number of Vegetarian Clusters')
    axs[0, 0].grid(alpha=0.3)
    
    # Plot 2: Average cluster size vs initial fraction
    axs[0, 1].plot(stats_df['initial_veg_fraction'], stats_df['avg_cluster_size'], 'o-', color='#ff7f0e')
    axs[0, 1].set_xlabel('Initial Vegetarian Fraction')
    axs[0, 1].set_ylabel('Average Cluster Size')
    axs[0, 1].set_title('Average Vegetarian Cluster Size')
    axs[0, 1].grid(alpha=0.3)
    
    # Plot 3: Maximum cluster size vs initial fraction
    axs[1, 0].plot(stats_df['initial_veg_fraction'], stats_df['max_cluster_size'], 'o-', color='#2ca02c')
    axs[1, 0].set_xlabel('Initial Vegetarian Fraction')
    axs[1, 0].set_ylabel('Maximum Cluster Size')
    axs[1, 0].set_title('Maximum Vegetarian Cluster Size')
    axs[1, 0].grid(alpha=0.3)
    
    # Plot 4: Final vs initial vegetarian fraction
    axs[1, 1].plot(stats_df['initial_veg_fraction'], stats_df['final_veg_fraction'], 'o-', color='#d62728')
    axs[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)  # y=x reference line
    axs[1, 1].set_xlabel('Initial Vegetarian Fraction')
    axs[1, 1].set_ylabel('Final Vegetarian Fraction')
    axs[1, 1].set_title('Growth in Vegetarian Population')
    axs[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_topology_comparison(results_df, save_path=None):
    """
    Create visualizations comparing different network topologies
    
    Args:
        results_df (DataFrame): Results from topology comparison
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib figure: The plot figure
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Growth in vegetarian fraction by topology
    sns.lineplot(
        data=results_df, 
        x='initial_veg_f', 
        y='growth',
        hue='topology',
        marker='o',
        ci=95,
        ax=ax1
    )
    
    ax1.set_xlabel('Initial Vegetarian Fraction', fontsize=12)
    ax1.set_ylabel('Growth in Vegetarian Fraction', fontsize=12)
    ax1.set_title('Dietary Contagion Efficiency by Network Structure', fontsize=14)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Final CO2 by topology
    sns.lineplot(
        data=results_df, 
        x='initial_veg_f', 
        y='final_CO2',
        hue='topology',
        marker='o',
        ci=95,
        ax=ax2
    )
    
    ax2.set_xlabel('Initial Vegetarian Fraction', fontsize=12)
    ax2.set_ylabel('Final Average CO2 Emissions [kg/year]', fontsize=12)
    ax2.set_title('Emissions Reduction by Network Structure', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_targeted_interventions(results_df, save_path=None, timeseries_path=None):
    """
    Plot the effect of targeted vs random interventions
    
    Args:
        results_df (DataFrame): Results from intervention analysis
        save_path (str, optional): Path to save figure for barplot
        timeseries_path (str, optional): Path to save figure for timeseries
        
    Returns:
        tuple: (barplot figure, timeseries figure)
    """
    # Create barplot of final vegetarian fractions
    barplot_fig = plt.figure(figsize=(10, 6))
    
    sns.barplot(data=results_df, x='intervention', y='final_veg_f', ci=95)
    
    plt.xlabel('Intervention Strategy', fontsize=12)
    plt.ylabel('Final Vegetarian Fraction', fontsize=12)
    plt.title('Effectiveness of Different Intervention Strategies', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Create figure for time series
    timeseries_fig = plt.figure(figsize=(12, 6))
    
    # Plot vegetarian fraction over time for different interventions
    intervention_types = results_df['intervention'].unique()
    
    for intervention in intervention_types:
        intervention_data = results_df[results_df['intervention'] == intervention]
        
        # Get the time series data and average across iterations
        trajectories = np.vstack(intervention_data['veg_trajectory'].values)
        mean_trajectory = np.mean(trajectories, axis=0)
        
        # Plot the mean trajectory
        plt.plot(mean_trajectory, label=f'{intervention}')
    
    plt.xlabel('Simulation Steps', fontsize=12)
    plt.ylabel('Vegetarian Fraction', fontsize=12)
    initial_veg_f = results_df.loc[results_df['intervention'] != 'none', 'initial_veg_f'].iloc[0]
    plt.title(f'Impact of Targeted Interventions ({initial_veg_f*100:.0f}% Initial Vegetarians)', 
             fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if timeseries_path:
        plt.savefig(timeseries_path, dpi=300, bbox_inches='tight')
    
    return barplot_fig, timeseries_fig

def plot_centrality_vs_influence(centrality_data, save_path=None):
    """
    Plot the relationship between agent centrality and influence
    
    Args:
        centrality_data (DataFrame): DataFrame with centrality and influence data
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib figure: The plot figure
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Degree centrality vs reduction
    sns.scatterplot(
        data=centrality_data, 
        x='degree', 
        y='reduction',
        hue='diet',
        palette={'veg': 'green', 'meat': 'red'},
        alpha=0.7,
        s=80,
        ax=ax1
    )
    
    ax1.set_xlabel('Degree Centrality', fontsize=12)
    ax1.set_ylabel('Emissions Reduction Attributed [kg/CO2]', fontsize=12)
    ax1.set_title('Degree Centrality vs Agent Influence', fontsize=14)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Betweenness centrality vs reduction
    sns.scatterplot(
        data=centrality_data, 
        x='betweenness', 
        y='reduction',
        hue='diet',
        palette={'veg': 'green', 'meat': 'red'},
        alpha=0.7,
        s=80,
        ax=ax2
    )
    
    ax2.set_xlabel('Betweenness Centrality', fontsize=12)
    ax2.set_ylabel('Emissions Reduction Attributed [kg/CO2]', fontsize=12)
    ax2.set_title('Betweenness Centrality vs Agent Influence', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig