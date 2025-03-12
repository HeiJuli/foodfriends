#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 21:32:29 2025

@author: jpoveralls
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focused publication plotting script for dietary contagion model
Creates only the four specific plots requested for publication
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom plotting style
from plotting.plot_styles import (
    set_publication_style, 
    apply_axis_style, 
    COLORS, 
    ECO_CMAP, 
    ECO_DIV_CMAP
)

def ensure_output_dir():
    """Ensure visualizations output directory exists"""
    output_dir = '../visualisations_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_data_file(file_path):
    """Load a DataFrame from a pickle file"""
    try:
        data = pd.read_pickle(file_path)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def plot_heatmap_alpha_beta(data=None, file_path=None, save=True):
    """
    Create heatmap showing how alpha and beta affect vegetarian growth
    
    Args:
        data (DataFrame): DataFrame with alpha, beta, and change columns
        file_path (str): Path to data file if data not provided
        save (bool): Whether to save the plot
    """
    # Set publication style
    set_publication_style()
    
    # Load data if not provided
    if data is None:
        if file_path is None:
            print("Either data or file_path must be provided")
            return None
        data = load_data_file(file_path)
        if data is None:
            return None
    
    # Get alpha and beta values
    alpha_values = sorted(data['alpha'].unique())
    beta_values = sorted(data['beta'].unique())
    
    # Create pivot table for heatmap
    pivot_table = data.pivot_table(index='beta', columns='alpha', values='change')
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap
    ax = sns.heatmap(
        pivot_table, 
        cmap=ECO_DIV_CMAP, 
        center=0,
        cbar_kws={'label': 'Change in Vegetarian Fraction'}
    )
    
    # Add contour lines if 'tipped' column exists
    if 'tipped' in data.columns:
        pivot_tipped = data.pivot_table(index='beta', columns='alpha', values='tipped').astype(int)
        CS = plt.contour(
            np.arange(len(alpha_values)) + 0.5, 
            np.arange(len(beta_values)) + 0.5,
            pivot_tipped.values, 
            levels=[0.5], 
            colors=COLORS['neutral'], 
            linewidths=2
        )
    
    # Set labels and title
    plt.xlabel('Individual preference weight (α)', fontsize=12)
    plt.ylabel('Social influence weight (β)', fontsize=12)
    plt.title('Parameter Effect on Growth in Vegetarian Population', fontsize=14)
    
    # Fix axis labels
    plt.xticks(np.arange(len(alpha_values)) + 0.5, [f"{x:.1f}" for x in alpha_values], rotation=0)
    plt.yticks(np.arange(len(beta_values)) + 0.5, [f"{y:.1f}" for y in beta_values], rotation=0)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'heatmap_alpha_beta.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return ax

def plot_emissions_vs_veg_fraction(data=None, file_path=None, save=True):
    """
    Create scatter plot of final CO2 consumption vs vegetarian fraction
    
    Args:
        data (DataFrame): DataFrame with veg_fraction and final_CO2 columns
        file_path (str): Path to data file if data not provided
        save (bool): Whether to save the plot
    """
    # Set publication style
    set_publication_style()
    
    # Load data if not provided
    if data is None:
        if file_path is None:
            print("Either data or file_path must be provided")
            return None
        data = load_data_file(file_path)
        if data is None:
            return None
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create scatter plot
    scatter = plt.scatter(
        data['veg_fraction'], 
        data['final_CO2'],
        s=70, 
        alpha=0.8,
        color=COLORS['primary'],
        edgecolor='white',
        linewidth=1.0
    )
    
    # Fit and plot trend line
    if len(data) > 2:  # Need at least 3 points for good regression
        x = data['veg_fraction']
        y = data['final_CO2']
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(
            np.sort(x), 
            poly1d_fn(np.sort(x)), 
            '--', 
            color=COLORS['neutral'],
            alpha=0.7,
            label=f'Trend: y = {coef[0]:.1f}x + {coef[1]:.1f}'
        )
    
    # Format plot
    plt.xlabel('Vegetarian Fraction', fontsize=12)
    plt.ylabel('Final Average CO₂ Emissions [kg/year]', fontsize=12)
    plt.title('Impact of Vegetarian Population on CO₂ Emissions', fontsize=14)
    
    if 'Trend' in plt.gca().get_legend_handles_labels()[1]:
        plt.legend()
    
    # Apply style to axis
    apply_axis_style(plt.gca())
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'emissions_vs_veg_fraction.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return plt.gca()

def plot_growth_in_veg_population(data=None, file_path=None, save=True, max_veg_fraction=0.6):
    """
    Create standalone plot showing growth in vegetarian population
    
    Args:
        data (DataFrame): DataFrame with initial_veg_fraction and final_veg_fraction columns
        file_path (str): Path to data file if data not provided
        save (bool): Whether to save the plot
        max_veg_fraction (float): Maximum vegetarian fraction to show on plot
    """
    # Set publication style
    set_publication_style()
    
    # Load data if not provided
    if data is None:
        if file_path is None:
            print("Either data or file_path must be provided")
            return None
        data = load_data_file(file_path)
        if data is None:
            return None
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Extract data - handle different column naming conventions
    x_values = data.get('initial_veg_fraction', data.get('initial_veg_f'))
    y_values = data.get('final_veg_fraction', data.get('final_veg_f'))
    
    if x_values is None or y_values is None:
        print("Data must contain initial_veg_fraction/initial_veg_f and final_veg_fraction/final_veg_f columns")
        return None
    
    # Filter data based on max_veg_fraction
    mask = x_values <= max_veg_fraction
    x_filtered = x_values[mask]
    y_filtered = y_values[mask]
    
    # Create scatter plot
    scatter = plt.scatter(
        x_filtered, 
        y_filtered,
        s=70, 
        alpha=0.8,
        color=COLORS['vegetation'],
        edgecolor='white',
        linewidth=1.0
    )
    
    # Plot y=x reference line
    plt.plot([0, max_veg_fraction], [0, max_veg_fraction], 'k--', alpha=0.5, label='No Change')
    
    # Calculate and plot growth trend
    if len(x_filtered) > 2:  # Need at least 3 points for good regression
        coef = np.polyfit(x_filtered, y_filtered, 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(
            np.sort(x_filtered), 
            poly1d_fn(np.sort(x_filtered)), 
            '-', 
            color=COLORS['neutral'],
            alpha=0.7,
            label=f'Trend: y = {coef[0]:.2f}x + {coef[1]:.2f}'
        )
    
    # Format plot
    plt.xlabel('Initial Vegetarian Fraction', fontsize=12)
    plt.ylabel('Final Vegetarian Fraction', fontsize=12)
    plt.title('Growth in Vegetarian Population', fontsize=14)
    plt.xlim(0, max_veg_fraction)
    plt.ylim(0, 1)
    plt.legend()
    
    # Apply style to axis
    apply_axis_style(plt.gca())
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'growth_in_veg_population.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return plt.gca()

def plot_individual_reductions_distribution(data=None, file_path=None, save=True):
    """
    Create distribution plot of individual emission reductions
    
    Args:
        data (DataFrame): DataFrame with individual_reductions column
        file_path (str): Path to data file if data not provided
        save (bool): Whether to save the plot
    """
    # Set publication style
    set_publication_style()
    
    # Load data if not provided
    if data is None:
        if file_path is None:
            print("Either data or file_path must be provided")
            return None
        data = load_data_file(file_path)
        if data is None:
            return None
    
    # Extract individual reductions
    if 'individual_reductions' not in data.columns:
        print("Data must contain individual_reductions column")
        return None
    
    # Flatten the list of individual reductions
    reductions = []
    for _, row in data.iterrows():
        if isinstance(row['individual_reductions'], list):
            reductions.extend(row['individual_reductions'])
        else:
            reductions.append(row['individual_reductions'])
    
    reductions = np.array(reductions)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    sns.histplot(
        reductions,
        bins=30,
        color=COLORS['secondary'],
        ax=ax1,
        kde=True
    )
    ax1.set_xlabel('Emissions Reduction Attributed [kg CO₂]', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Individual Emission Reductions', fontsize=14)
    apply_axis_style(ax1)
    
    # ECDF
    sns.ecdfplot(
        reductions,
        color=COLORS['primary'],
        linewidth=2,
        ax=ax2
    )
    ax2.set_xlabel('Emissions Reduction Attributed [kg CO₂]', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Empirical CDF of Emission Reductions', fontsize=14)
    apply_axis_style(ax2)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'individual_reductions_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig, (ax1, ax2)

def main():
    """Main function to run all plots"""
    print("Running publication plots...")
    
    # Ask user which plots to create
    print("\nAvailable plots:")
    print("[1] Heatmap of Alpha and Beta")
    print("[2] Emissions vs Vegetarian Fraction")
    print("[3] Growth in Vegetarian Population")
    print("[4] Individual Reductions Distribution")
    print("[5] All plots")
    
    choice = input("\nSelect plot to create (1-5): ")
    
    if choice not in ['1', '2', '3', '4', '5']:
        print("Invalid choice")
        return
    
    # Ask for file paths
    tipping_file = None
    emissions_file = None
    cluster_file = None
    reduction_file = None
    
    if choice in ['1', '5']:
        tipping_file = input("\nEnter path to tipping point analysis file: ")
        if not os.path.exists(tipping_file):
            print(f"File not found: {tipping_file}")
            tipping_file = None
    
    if choice in ['2', '5']:
        emissions_file = input("\nEnter path to emissions analysis file: ")
        if not os.path.exists(emissions_file):
            print(f"File not found: {emissions_file}")
            emissions_file = None
    
    if choice in ['3', '5']:
        cluster_file = input("\nEnter path to cluster analysis file: ")
        if not os.path.exists(cluster_file):
            print(f"File not found: {cluster_file}")
            cluster_file = None
    
    if choice in ['4', '5']:
        reduction_file = input("\nEnter path to parameter sweep file with individual reductions: ")
        if not os.path.exists(reduction_file):
            print(f"File not found: {reduction_file}")
            reduction_file = None
    
    # Create plots
    if choice in ['1', '5'] and tipping_file:
        plot_heatmap_alpha_beta(file_path=tipping_file)
    
    if choice in ['2', '5'] and emissions_file:
        plot_emissions_vs_veg_fraction(file_path=emissions_file)
    
    if choice in ['3', '5'] and cluster_file:
        plot_growth_in_veg_population(file_path=cluster_file)
    
    if choice in ['4', '5'] and reduction_file:
        plot_individual_reductions_distribution(file_path=reduction_file)

if __name__ == "__main__":
    main()