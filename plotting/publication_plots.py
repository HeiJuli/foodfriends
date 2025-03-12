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
from plot_styles import (
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
    Create heatmap showing how alpha and beta affect system metrics using parameter sweep data
    
    Args:
        data (DataFrame): DataFrame with alpha, beta columns
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
    
    # Ensure alpha, beta columns exist
    if 'alpha' not in data.columns or 'beta' not in data.columns:
        print("Data must contain alpha and beta columns")
        return None
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create pivot table for heatmap - average final_veg_fraction by alpha and beta
    pivot_data = data.groupby(['alpha', 'beta'])['final_veg_fraction'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='beta', columns='alpha', values='final_veg_fraction')
    
    # Plot heatmap
    ax = sns.heatmap(
        pivot_table, 
        cmap=ECO_CMAP, 
        cbar_kws={'label': 'Final Vegetarian Fraction'}
    )
    
    # Set labels and title
    plt.xlabel('Individual preference weight (α)', fontsize=12)
    plt.ylabel('Social influence weight (β)', fontsize=12)
    plt.title('Parameter Effect on Growth in Vegetarian Population', fontsize=14)
    
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
    
    # Format plot
    plt.xlabel('Vegetarian Fraction', fontsize=12)
    plt.ylabel('Final Average CO₂ Emissions [kg/year]', fontsize=12)
    plt.title('Impact of Vegetarian Population on CO₂ Emissions', fontsize=14)
    
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
        
        



def select_file(pattern):
    """Display a list of files matching the pattern and let user select one"""
    import os
    import glob
    from datetime import datetime
    
    # Look for files in model_output directory
    model_output_dir = '../model_output'
    search_pattern = os.path.join(model_output_dir, f'{pattern}_*.pkl')
    
    # Get all matching files
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    # Sort files by modification time (newest first)
    matching_files.sort(key=os.path.getmtime, reverse=True)
    
    # Display files to user
    print(f"\nAvailable {pattern} files:")
    for idx, file in enumerate(matching_files):
        base_name = os.path.basename(file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file))
        print(f"[{idx+1}] {base_name} (Modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    # Let user select a file
    while True:
        try:
            selection = input(f"\nSelect {pattern} file (1-{len(matching_files)}, or press Enter for latest): ")
            
            # Default to the latest file if user just presses Enter
            if selection == "":
                return matching_files[0]
                
            selection_idx = int(selection) - 1
            if 0 <= selection_idx < len(matching_files):
                return matching_files[selection_idx]
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(matching_files)}")
        except ValueError:
            print("Please enter a valid number or press Enter for the latest file")

def main():
    """Main function to run publication plots with file selection"""
    print("Running publication plots...")
    
    # Ask user which plots to create
    print("\nAvailable plots:")
    print("[1] Heatmap of Alpha and Beta (from parameter sweep)")
    print("[2] Emissions vs Vegetarian Fraction")
    print("[3] Growth in Vegetarian Population")
    print("[4] Individual Reductions Distribution (from parameter sweep)")
    print("[5] All plots")
    
    choice = input("\nSelect plot to create (1-5): ")
    
    if choice not in ['1', '2', '3', '4', '5']:
        print("Invalid choice")
        return
    
    # Let user select files for each plot type as needed
    parameter_sweep_file = None
    emissions_file = None
    veg_growth_file = None
    
    if choice in ['1', '4', '5']:
        parameter_sweep_file = select_file('parameter_sweep')
        
    if choice in ['2', '5']:
        emissions_file = select_file('emissions')
        
    if choice in ['3', '5']:
        veg_growth_file = select_file('veg_growth')
    
    # Create plots based on user selection and available files
    if choice in ['1', '5'] and parameter_sweep_file:
        plot_heatmap_alpha_beta(file_path=parameter_sweep_file)
    elif choice == '1' and not parameter_sweep_file:
        print("Cannot create heatmap: No parameter sweep file found")
    
    if choice in ['2', '5'] and emissions_file:
        plot_emissions_vs_veg_fraction(file_path=emissions_file)
    elif choice == '2' and not emissions_file:
        print("Cannot create emissions plot: No emissions analysis file found")
    
    if choice in ['3', '5'] and veg_growth_file:
        plot_growth_in_veg_population(file_path=veg_growth_file)
    elif choice == '3' and not veg_growth_file:
        print("Cannot create growth plot: No vegetarian growth analysis file found")
    
    if choice in ['4', '5'] and parameter_sweep_file:
        plot_individual_reductions_distribution(file_path=parameter_sweep_file)
    elif choice == '4' and not parameter_sweep_file:
        print("Cannot create distribution plot: No parameter sweep file found")

if __name__ == "__main__":
    main()