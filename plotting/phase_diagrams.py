# -*- coding: utf-8 -*-
"""
Phase diagram and parameter analysis plots for dietary contagion model
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

def plot_tipping_point_phase_diagram(results_df, fixed_veg_f=0.2, save_path=None):
    """
    Create phase diagram showing where tipping points occur based on alpha and beta values
    
    Args:
        results_df (DataFrame): Results from parameter sensitivity analysis
        fixed_veg_f (float): Initial vegetarian fraction for labeling
        save_path (str, optional): Path to save figure
    
    Returns:
        Matplotlib axis: The plot axis
    """
    # Get alpha and beta values from the dataframe
    alpha_values = sorted(results_df['alpha'].unique())
    beta_values = sorted(results_df['beta'].unique())
    
    # Create pivot tables for the heatmap
    pivot_table = results_df.pivot_table(index='beta', columns='alpha', values='change')
    
    # Create a custom diverging colormap
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    ax = sns.heatmap(pivot_table, cmap=cmap, center=0)
    
    # Add contour lines to highlight tipping point boundaries
    pivot_tipped = results_df.pivot_table(index='beta', columns='alpha', values='tipped').astype(int)
    contour_levels = [0.5]  # Boundary between tipping and not tipping
    CS = plt.contour(np.arange(len(alpha_values)) + 0.5, 
                     np.arange(len(beta_values)) + 0.5,
                     pivot_tipped.values, 
                     levels=contour_levels, 
                     colors='black', 
                     linewidths=2)
    
    # Set labels and title
    plt.xlabel('Individual preference weight (α)', fontsize=12)
    plt.ylabel('Social influence weight (β)', fontsize=12)
    plt.title(f'Parameter Combinations Leading to Tipping Points (Initial Veg={fixed_veg_f})', 
             fontsize=14)
    
    # Fix axis labels
    plt.xticks(np.arange(len(alpha_values)) + 0.5, np.round(alpha_values, 2), rotation=0)
    plt.yticks(np.arange(len(beta_values)) + 0.5, np.round(beta_values, 2), rotation=0)
    
    # Add a colorbar with proper label
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(ax.collections[0], cax=cax)
    cbar.set_label('Change in vegetarian fraction', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax

def plot_threshold_variations(results_df, thresholds=[0.05, 0.1, 0.2, 0.3], save_path=None):
    """
    Create plots showing how tipping behavior varies with different thresholds
    
    Args:
        results_df (DataFrame): Results from parameter sensitivity analysis
        thresholds (list): List of thresholds to test
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib figure: The plot figure
    """
    fig, axes = plt.subplots(1, len(thresholds), figsize=(16, 4), sharey=True)
    
    # Get alpha and beta ranges
    alpha_values = sorted(results_df['alpha'].unique())
    beta_values = sorted(results_df['beta'].unique())
    
    for i, threshold in enumerate(thresholds):
        # Create tipped variable based on different thresholds
        results_df[f'tipped_{threshold}'] = results_df['change'] > threshold
        
        # Create pivot table for this threshold
        pivot = results_df.pivot_table(
            index='beta', columns='alpha', values=f'tipped_{threshold}'
        ).astype(int)
        
        # Plot filled contour
        ax = axes[i]
        CS = ax.contourf(
            alpha_values, beta_values, pivot.values,
            levels=[-0.5, 0.5, 1.5],  # Boundaries between 0, 1, and 2
            colors=['#ffcccc', '#ccffcc'],
            alpha=0.7
        )
        
        # Add contour lines
        CS2 = ax.contour(
            alpha_values, beta_values, pivot.values,
            levels=[0.5],  # Just the boundary between tipping and not tipping
            colors=['black'],
            linewidths=2
        )
        
        ax.set_title(f'Threshold = {threshold}')
        ax.set_xlabel('Individual preference (α)')
        
        if i == 0:
            ax.set_ylabel('Social influence (β)')
    
    plt.suptitle('Tipping Regions with Different Thresholds', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_multi_parameter_phase_diagrams(combined_df, save_path=None):
    """
    Plot multiple phase diagrams for different initial vegetarian fractions
    
    Args:
        combined_df (DataFrame): Combined results with 'initial_veg_f' column
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib figure: The plot figure
    """
    # Get unique vegetarian fractions
    veg_fractions = sorted(combined_df['initial_veg_f'].unique())
    
    # Create plot grid
    fig, axes = plt.subplots(1, len(veg_fractions), figsize=(6*len(veg_fractions), 5))
    
    # Custom colormap
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    for i, veg_f in enumerate(veg_fractions):
        # Filter data for this vegetarian fraction
        vf_data = combined_df[combined_df['initial_veg_f'] == veg_f]
        
        # Get alpha and beta values
        alpha_values = sorted(vf_data['alpha'].unique())
        beta_values = sorted(vf_data['beta'].unique())
        
        # Create pivot table for heatmap
        pivot = vf_data.pivot_table(index='beta', columns='alpha', values='change')
        
        # Create heatmap
        ax = axes[i] if len(veg_fractions) > 1 else axes
        sns.heatmap(pivot, cmap=cmap, ax=ax, center=0, 
                   cbar_kws={'label': 'Change in Veg. Fraction'})
        
        # Overlay contour for tipping point boundary
        pivot_tipped = vf_data.pivot_table(index='beta', columns='alpha', values='tipped').astype(int)
        CS = ax.contour(np.arange(len(alpha_values)) + 0.5, 
                       np.arange(len(beta_values)) + 0.5,
                       pivot_tipped.values, 
                       levels=[0.5], 
                       colors='black', 
                       linewidths=2)
        
        ax.set_title(f'Initial Veg. Fraction: {veg_f}', fontsize=14)
        ax.set_xlabel('Individual Preference (α)', fontsize=12)
        
        # Fix axis labels
        ax.set_xticks(np.arange(len(alpha_values)) + 0.5)
        ax.set_xticklabels(np.round(alpha_values, 2), rotation=0)
        ax.set_yticks(np.arange(len(beta_values)) + 0.5)
        ax.set_yticklabels(np.round(beta_values, 2), rotation=0)
        
        if i == 0:
            ax.set_ylabel('Social Influence (β)', fontsize=12)
        else:
            ax.set_ylabel('')
    
    plt.suptitle('Parameter Conditions for Tipping Points at Different Initial Vegetarian Fractions', 
                fontsize=16, y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_critical_dynamics(results_df, near_tipping=0.28, save_path=None):
    """
    Plot variance and autocorrelation near the tipping point to demonstrate critical slowing down
    
    Args:
        results_df (DataFrame): Results with variance and autocorrelation metrics
        near_tipping (float): Estimated tipping point value
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib figure: The plot figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot variance
    ax1.plot(results_df['veg_fraction'], results_df['variance'], 'o-', 
            color='#1f77b4', linewidth=2)
    ax1.set_xlabel('Vegetarian Fraction', fontsize=12)
    ax1.set_ylabel('Variance', fontsize=12)
    ax1.set_title('Variance Near Tipping Point', fontsize=14)
    ax1.axvline(x=near_tipping, color='r', linestyle='--', label='Est. Tipping Point')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Plot autocorrelation
    ax2.plot(results_df['veg_fraction'], results_df['autocorrelation'], 'o-', 
            color='#ff7f0e', linewidth=2)
    ax2.set_xlabel('Vegetarian Fraction', fontsize=12)
    ax2.set_ylabel('Lag-1 Autocorrelation', fontsize=12)
    ax2.set_title('Autocorrelation Near Tipping Point', fontsize=14)
    ax2.axvline(x=near_tipping, color='r', linestyle='--', label='Est. Tipping Point')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_bifurcation_diagram(results_df, parameter_name='veg_f', save_path=None):
    """
    Create a bifurcation diagram showing final vegetarian fraction vs a parameter
    
    Args:
        results_df (DataFrame): Results with param_value, param_name, and final_veg_f columns
        parameter_name (str): Parameter to plot (for label)
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib axis: The plot axis
    """
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot final vegetarian fraction vs parameter value
    sns.scatterplot(
        data=results_df, 
        x='param_value', 
        y='final_veg_f',
        hue='run' if 'run' in results_df.columns else None,
        alpha=0.7,
        s=80
    )
    
    # If parameter is veg_f, add y=x reference line
    if parameter_name == 'veg_f':
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No Change')
        plt.legend()
    
    plt.xlabel(f'{parameter_name} value', fontsize=12)
    plt.ylabel('Final Vegetarian Fraction', fontsize=12)
    plt.title(f'Bifurcation Analysis for {parameter_name}', fontsize=14)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gca()

def plot_combined_tipping_boundaries(combined_df, save_path=None):
    """
    Create a single plot showing tipping point boundaries for multiple vegetarian fractions
    
    Args:
        combined_df (DataFrame): Combined results with 'initial_veg_f' column
        save_path (str, optional): Path to save figure
        
    Returns:
        Matplotlib axis: The plot axis
    """
    # Get unique vegetarian fractions
    veg_fractions = sorted(combined_df['initial_veg_f'].unique())
    
    # Get alpha and beta ranges
    alpha_values = sorted(combined_df['alpha'].unique())
    beta_values = sorted(combined_df['beta'].unique())
    
    # Create mesh grid for plotting
    alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot overlapping contours for different veg fractions
    colors = plt.cm.viridis(np.linspace(0, 1, len(veg_fractions)))
    
    for i, veg_f in enumerate(veg_fractions):
        # Filter data for this vegetarian fraction
        vf_data = combined_df[combined_df['initial_veg_f'] == veg_f]
        
        # Create pivot table
        pivot_tipped = vf_data.pivot_table(index='beta', columns='alpha', values='tipped').astype(int)
        
        # Plot contour
        cs = plt.contour(alpha_grid, beta_grid, pivot_tipped.values, 
                        levels=[0.5], colors=[colors[i]], linewidths=2)
        cs.collections[0].set_label(f'Veg Fraction = {veg_f}')
    
    plt.xlabel('Individual Preference Weight (α)', fontsize=12)
    plt.ylabel('Social Influence Weight (β)', fontsize=12)
    plt.title('Tipping Point Boundaries for Different Initial Vegetarian Fractions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gca()