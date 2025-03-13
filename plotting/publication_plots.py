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
from matplotlib.colors import LinearSegmentedColormap

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
    
    # Round tick labels to one decimal place
    alpha_values = sorted(data['alpha'].unique())
    beta_values = sorted(data['beta'].unique())
    
    plt.xticks(
        np.arange(len(alpha_values)) + 0.5, 
        [f"{v:.1f}" for v in alpha_values], 
        rotation=0
    )
    plt.yticks(
        np.arange(len(beta_values)) + 0.5, 
        [f"{v:.1f}" for v in beta_values], 
        rotation=0
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

def plot_tipping_point_heatmap(data=None, file_path=None, save=True):
    """
    Create heatmap showing parameter combinations leading to tipping points
    (significant changes in vegetarian fraction)
    
    Args:
        data (DataFrame): DataFrame with alpha, beta and change columns
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
    
    # Create change column if not exist - difference between final and initial vegetarian fraction
    if 'change' not in data.columns:
        if 'final_veg_fraction' in data.columns and 'initial_veg_fraction' in data.columns:
            data['change'] = data['final_veg_fraction'] - data['initial_veg_fraction']
        elif 'final_veg_f' in data.columns and 'initial_veg_f' in data.columns:
            data['change'] = data['final_veg_f'] - data['initial_veg_f']
        else:
            # If we have final but not initial, we can calculate change if we know the fixed initial value
            if 'final_veg_fraction' in data.columns:
                # Try to infer a fixed initial vegetarian fraction
                if 'fixed_veg_f' in data.columns:
                    data['change'] = data['final_veg_fraction'] - data['fixed_veg_f']
                else:
                    # Estimate initial veg fraction based on typical values
                    data['change'] = data['final_veg_fraction'] - 0.2  # Assuming 20% initial
            else:
                print("Data must contain columns to calculate change in vegetarian fraction")
                return None
    
    # Create tipped column if not exist - binary indicator of whether a tipping point occurred
    if 'tipped' not in data.columns:
        # Define tipping as change greater than 20%
        data['tipped'] = data['change'] > 0.2
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create pivot tables for the heatmap
    pivot_table = data.pivot_table(index='beta', columns='alpha', values='change')
    
    # Use the diverging colormap from plot_styles for our heatmap
    # This is ideal since our data has a meaningful zero point (no change)
    
    # Plot heatmap
    ax = sns.heatmap(
        pivot_table, 
        cmap=ECO_DIV_CMAP,  # Use our custom diverging colormap
        center=0, 
        cbar_kws={'label': 'Change in Vegetarian Fraction'}
    )
    
    # Add contour lines to highlight tipping point boundaries
    pivot_tipped = data.pivot_table(index='beta', columns='alpha', values='tipped').astype(int)
    CS = plt.contour(
        np.arange(len(pivot_table.columns)) + 0.5, 
        np.arange(len(pivot_table.index)) + 0.5,
        pivot_tipped.values, 
        levels=[0.5], 
        colors=COLORS['neutral'],  # Use our neutral color for contours
        linewidths=2
    )
    
    # Set labels and title
    plt.xlabel('Individual preference weight (α)', fontsize=12)
    plt.ylabel('Social influence weight (β)', fontsize=12)
    plt.title('Parameter Combinations Leading to Tipping Points', fontsize=14)
    
    # Round tick labels to one decimal place
    alpha_values = sorted(data['alpha'].unique())
    beta_values = sorted(data['beta'].unique())
    
    plt.xticks(
        np.arange(len(alpha_values)) + 0.5, 
        [f"{v:.1f}" for v in alpha_values], 
        rotation=0
    )
    plt.yticks(
        np.arange(len(beta_values)) + 0.5, 
        [f"{v:.1f}" for v in beta_values], 
        rotation=0
    )
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'tipping_point_heatmap.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return ax

def plot_3d_parameter_surface(data=None, file_path=None, save=True):
    """
    Create 3D surface plot showing how alpha, beta and initial vegetarian fraction 
    affect final vegetarian fraction
    
    Args:
        data (DataFrame): DataFrame with alpha, beta, initial_veg_fraction/initial_veg_f and final_veg_fraction/final_veg_f
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
            
    # Check if we have the required columns
    if not all(col in data.columns for col in ['alpha', 'beta']):
        print("Data must contain 'alpha' and 'beta' columns")
        return None
        
    # Get the vegetarian fraction columns
    init_veg_col = 'initial_veg_fraction' if 'initial_veg_fraction' in data.columns else 'initial_veg_f'
    final_veg_col = 'final_veg_fraction' if 'final_veg_fraction' in data.columns else 'final_veg_f'
    
    if init_veg_col not in data.columns or final_veg_col not in data.columns:
        print("Data must contain initial and final vegetarian fraction columns")
        return None
    
    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique values for each axis
    alpha_values = sorted(data['alpha'].unique())
    beta_values = sorted(data['beta'].unique())
    veg_values = sorted(data[init_veg_col].unique())
    
    # Create meshgrid for surface
    alpha_grid, beta_grid, veg_grid = np.meshgrid(alpha_values, beta_values, veg_values)
    
    # Prepare data for surface
    final_veg = np.zeros_like(alpha_grid)
    
    # Populate the grid with final vegetarian fractions
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            for k, veg in enumerate(veg_values):
                subset = data[(data['alpha'] == alpha) & 
                              (data['beta'] == beta) & 
                              (data[init_veg_col] == veg)]
                if len(subset) > 0:
                    final_veg[j, i, k] = subset[final_veg_col].mean()
    
    # Create the 3D surface plot
    surf = ax.plot_surface(
        alpha_grid[..., 0], 
        beta_grid[..., 0], 
        final_veg[..., 0],
        cmap=ECO_CMAP,
        edgecolor='none',
        alpha=0.8
    )
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Final Vegetarian Fraction')
    
    # Set labels
    ax.set_xlabel('Individual preference (α)')
    ax.set_ylabel('Social influence (β)')
    ax.set_zlabel('Final Vegetarian Fraction')
    ax.set_title('3D Parameter Surface')
    
    # Set ticks to one decimal place
    ax.set_xticks(alpha_values)
    ax.set_xticklabels([f"{v:.1f}" for v in alpha_values])
    ax.set_yticks(beta_values)
    ax.set_yticklabels([f"{v:.1f}" for v in beta_values])
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, '3d_parameter_surface.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return ax

def plot_trajectory_grid(data=None, file_path=None, save=True):
    """
    Create a grid of trajectory plots for different parameter combinations
    
    Args:
        data (DataFrame): DataFrame with system_C_trajectory and fraction_veg_trajectory
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
    
    # Check if we have trajectory data
    if 'fraction_veg_trajectory' not in data.columns:
        print("Data must contain fraction_veg_trajectory column")
        return None
    
    # Define parameter combinations to plot
    # If alpha/beta are in the data, we'll use those
    if 'alpha' in data.columns and 'beta' in data.columns:
        param_combinations = [
            {'alpha': 0.25, 'beta': 0.75},
            {'alpha': 0.75, 'beta': 0.25},
            {'alpha': 0.5, 'beta': 0.5},
            {'alpha': 0.75, 'beta': 0.75}
        ]
    else:
        # Fall back to simpler approach - just plot the first 4 trajectories
        param_combinations = None
    
    # Create figure with 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    # If we have parameter combinations, select data accordingly
    if param_combinations:
        for i, params in enumerate(param_combinations):
            if i >= 4:  # Only plot in the 2x2 grid
                break
                
            # Filter data for this parameter combination
            # Allow some tolerance to find closest matches
            alpha_tol = 0.05
            beta_tol = 0.05
            subset = data[
                (data['alpha'] >= params['alpha'] - alpha_tol) & 
                (data['alpha'] <= params['alpha'] + alpha_tol) &
                (data['beta'] >= params['beta'] - beta_tol) & 
                (data['beta'] <= params['beta'] + beta_tol)
            ]
            
            if len(subset) == 0:
                print(f"No data found for α={params['alpha']}, β={params['beta']}")
                continue
            
            # Plot all trajectories for this parameter combination
            for _, row in subset.iterrows():
                trajectory = row['fraction_veg_trajectory']
                if isinstance(trajectory, list) and len(trajectory) > 0:
                    time_steps = np.arange(len(trajectory))
                    axs[i].plot(time_steps, trajectory, color='black', alpha=0.5, linewidth=0.8)
            
            # Set title and labels
            axs[i].set_title(f"α={params['alpha']:.2f}, β={params['beta']:.2f}")
            axs[i].set_xlabel("Time Steps")
            axs[i].set_ylabel("Vegetarian Fraction")
            axs[i].set_ylim(0, 1)
            
            # Apply style
            apply_axis_style(axs[i])
    else:
        # Just plot the first 4 trajectories
        for i in range(min(4, len(data))):
            trajectory = data.iloc[i]['fraction_veg_trajectory']
            if isinstance(trajectory, list) and len(trajectory) > 0:
                time_steps = np.arange(len(trajectory))
                axs[i].plot(time_steps, trajectory, color='black', linewidth=1.2)
                
            # Set title and labels
            axs[i].set_title(f"Trajectory {i+1}")
            axs[i].set_xlabel("Time Steps")
            axs[i].set_ylabel("Vegetarian Fraction")
            axs[i].set_ylim(0, 1)
            
            # Apply style
            apply_axis_style(axs[i])
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'trajectory_grid.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig
def plot_growth_in_veg_population(data=None, file_path=None, save=True):
    """
    Create standalone plot showing growth in vegetarian population
    
    Args:
        data (DataFrame): DataFrame with initial_veg_fraction and final_veg_fraction columns
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
    
    # Extract data - handle different column naming conventions
    x_values = data.get('initial_veg_fraction', data.get('initial_veg_f'))
    y_values = data.get('final_veg_fraction', data.get('final_veg_f'))
    
    if x_values is None or y_values is None:
        print("Data must contain initial_veg_fraction/initial_veg_f and final_veg_fraction/final_veg_f columns")
        return None
    
    # Create scatter plot
    scatter = plt.scatter(
        x_values, 
        y_values,
        s=70, 
        alpha=0.8,
        color=COLORS['vegetation'],
        edgecolor='white',
        linewidth=1.0
    )
    
    # Plot y=x reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No Change')
    
    # Format plot
    plt.xlabel('Initial Vegetarian Fraction', fontsize=12)
    plt.ylabel('Final Vegetarian Fraction', fontsize=12)
    plt.title('Growth in Vegetarian Population', fontsize=14)
    plt.xlim(0, 1)
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
    print("[1] Heatmap of Alpha and Beta (final vegetarian fraction)")
    print("[2] Emissions vs Vegetarian Fraction")
    print("[3] Growth in Vegetarian Population")
    print("[4] Individual Reductions Distribution")
    print("[5] Tipping Point Heatmap (change in vegetarian fraction)")
    print("[6] 3D Parameter Surface")
    print("[7] Trajectory Grid")
    print("[8] All plots")
    
    choice = input("\nSelect plot to create (1-8): ")
    
    if choice not in ['1', '2', '3', '4', '5', '6', '7', '8']:
        print("Invalid choice")
        return
    
    # Let user select files for each plot type as needed
    parameter_sweep_file = None
    emissions_file = None
    veg_growth_file = None
    tipping_point_file = None
    trajectory_file = None
    
    if choice in ['1', '4', '5', '6', '8']:
        parameter_sweep_file = select_file('parameter_sweep')
        
    if choice in ['2', '8']:
        emissions_file = select_file('emissions')
        
    if choice in ['3', '8']:
        veg_growth_file = select_file('veg_growth')
    
    if choice in ['7', '8']:
        trajectory_file = parameter_sweep_file if parameter_sweep_file else select_file('parameter_sweep')
    
    # Create plots based on user selection and available files
    if choice in ['1', '8'] and parameter_sweep_file:
        plot_heatmap_alpha_beta(file_path=parameter_sweep_file)
    elif choice == '1' and not parameter_sweep_file:
        print("Cannot create heatmap: No parameter sweep file found")
    
    if choice in ['2', '8'] and emissions_file:
        plot_emissions_vs_veg_fraction(file_path=emissions_file)
    elif choice == '2' and not emissions_file:
        print("Cannot create emissions plot: No emissions analysis file found")
    
    if choice in ['3', '8'] and veg_growth_file:
        plot_growth_in_veg_population(file_path=veg_growth_file)
    elif choice == '3' and not veg_growth_file:
        print("Cannot create growth plot: No vegetarian growth analysis file found")
    
    if choice in ['4', '8'] and parameter_sweep_file:
        plot_individual_reductions_distribution(file_path=parameter_sweep_file)
    elif choice == '4' and not parameter_sweep_file:
        print("Cannot create distribution plot: No parameter sweep file found")
        
    if choice in ['5', '8'] and parameter_sweep_file:
        plot_tipping_point_heatmap(file_path=parameter_sweep_file)
    elif choice == '5' and not parameter_sweep_file:
        print("Cannot create tipping point heatmap: No parameter sweep file found")
        
    if choice in ['6', '8'] and parameter_sweep_file:
        plot_3d_parameter_surface(file_path=parameter_sweep_file)
    elif choice == '6' and not parameter_sweep_file:
        print("Cannot create 3D parameter surface: No parameter sweep file found")
        
    if choice in ['7', '8'] and trajectory_file:
        plot_trajectory_grid(file_path=trajectory_file)
    elif choice == '7' and not trajectory_file:
        print("Cannot create trajectory grid: No suitable file found")

if __name__ == "__main__":
    main()