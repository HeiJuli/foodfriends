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


#%% Helper Functions


 # Function to create color variations (lighter/darker)
def create_color_variations(base_color, num_variations):
    import matplotlib.colors as mcolors
    base_rgb = mcolors.to_rgb(base_color)
    colors = []
    
    for i in range(num_variations):
        # Create variations by adjusting brightness
        # Start from 70% brightness for good visibility
        factor = 0.7 + (0.3 * i / max(1, num_variations-1))
        # Ensure we don't exceed RGB limits
        new_color = tuple(min(1.0, c * factor) for c in base_rgb)
        colors.append(new_color)
        
    return colors
    

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


def _prep_data(data, file_path, theta_values):
    """Prepare data for heatmap plotting"""
    if data is None:
        if file_path is None: return None
        data = load_data_file(file_path)
        if data is None: return None
    
    # Check required columns
    if 'alpha' not in data.columns or 'beta' not in data.columns:
        print("Data must contain alpha and beta columns")
        return None
    
    # Handle theta column
    if 'theta' not in data.columns:
        print("Warning: No theta column found, assuming theta=0 for all data")
        data['theta'] = 0
        theta_values = [0]
    
    # Filter available theta values
    available_thetas = sorted(data['theta'].unique())
    theta_values = [t for t in theta_values if t in available_thetas]
    if not theta_values:
        print("No specified theta values found in data")
        return None
    
    return data, theta_values

def _get_veg_columns(data):
    """Find vegetarian fraction column names in dataframe and create derived columns"""
    # Find column names for initial and final vegetarian fractions
    init_col = next((c for c in ['initial_veg_f', 'initial_veg_fraction', 'veg_f'] 
                    if c in data.columns), None)
    final_col = next((c for c in ['final_veg_f', 'final_veg_fraction'] 
                     if c in data.columns), None)
    
    # Handle case where both columns exist
    if init_col and final_col:
        # Create change column
        if 'change' not in data.columns:
            data['change'] = data[final_col] - data[init_col]
            print(f"Created 'change' column from {final_col} - {init_col}")
        
        # Create tipped column
        if 'tipped' not in data.columns:
            data['tipped'] = data['change'] > 0.2
            print(f"Created 'tipped' column (change > 0.2)")
    
    # Handle case where only final column exists
    elif final_col and not init_col:
        print(f"Warning: Initial vegetarian fraction column not found, using default value of 0.2")
        data['change'] = data[final_col] - 0.2
        data['tipped'] = data[final_col] > 0.4  # 0.2 + 0.2
    
    return init_col, final_col, data
def _create_heatmap(ax, pivot_table, alpha_vals, beta_vals, cmap, vmin, vmax, 
                   center=None, show_cbar=False, cbar_label=None):
    """Create a single heatmap on the given axis"""
    sns.heatmap(
        pivot_table, 
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar=show_cbar,
        cbar_kws={'label': cbar_label} if show_cbar else None
    )
    
    # Format tick labels
    ax.set_xticks(np.arange(len(alpha_vals)) + 0.5)
    ax.set_xticklabels([f"{round(v, 1):.1f}" for v in alpha_vals], rotation=0)
    
    ax.set_yticks(np.arange(len(beta_vals)) + 0.5)
    ax.set_yticklabels([f"{round(v, 1):.1f}" for v in beta_vals], rotation=0)


#%% Main plotting functions



def plot_heatmap_alpha_beta(data=None, file_path=None, save=True, theta_values=[-0.5, 0, 0.5], 
                          aggregate_veg_fractions=True):
    """Create heatmap of alpha and beta effects, optionally aggregated across veg fractions"""
    # Set publication style and prepare data
    set_publication_style()
    result = _prep_data(data, file_path, theta_values)
    if result is None: return None
    data, theta_values = result
    
    init_col, final_col, data = _get_veg_columns(data)
    if not final_col:
        print("Error: No final vegetarian fraction column found in data")
        return None
        
    can_aggregate = aggregate_veg_fractions and init_col and final_col
    value_col = final_col  # Default column to plot
    
    # Create figure with subplots
    fig = plt.figure(figsize=(6*len(theta_values)+1, 6))
    gs = fig.add_gridspec(1, len(theta_values), width_ratios=[1]*len(theta_values))
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(theta_values))]
    if len(theta_values) == 1: axes = [axes]
    
    # Calculate vmin/vmax for consistent colorbar
    vmin, vmax = float('inf'), float('-inf')
    for theta in theta_values:
        theta_data = data[data['theta'] == theta]
        if not theta_data.empty:
            if can_aggregate:
                agg = theta_data.groupby(['alpha', 'beta'])[value_col].mean()
            else:
                agg = theta_data.groupby(['alpha', 'beta'])[value_col].mean()
            
            if not agg.empty:
                vmin = min(vmin, agg.min())
                vmax = max(vmax, agg.max())
    
    # Create heatmaps for each theta value
    for i, theta in enumerate(theta_values):
        theta_data = data[data['theta'] == theta]
        
        # Aggregate data if requested
        if can_aggregate:
            agg = theta_data.groupby(['alpha', 'beta']).agg({value_col: ['mean', 'std']})
            agg.columns = ['final_veg_mean', 'final_veg_std']
            agg = agg.reset_index()
            pivot_data = agg
            plot_col = 'final_veg_mean'
        else:
            pivot_data = theta_data.groupby(['alpha', 'beta'])[value_col].mean().reset_index()
            plot_col = value_col
        
        # Get parameter values and create pivot table
        beta_vals = sorted(pivot_data['beta'].unique(), reverse=True)
        alpha_vals = sorted(pivot_data['alpha'].unique())
        
        pivot_table = pd.pivot_table(
            pivot_data,
            values=plot_col,
            index=pd.CategoricalIndex(pivot_data['beta'], categories=beta_vals),
            columns=pd.CategoricalIndex(pivot_data['alpha'], categories=alpha_vals)
        )
        
        # Create heatmap
        cbar_label = 'Final Vegetarian Fraction (Mean)' if can_aggregate else 'Final Vegetarian Fraction'
        _create_heatmap(
            axes[i], pivot_table, alpha_vals, beta_vals, ECO_CMAP, vmin, vmax,
            show_cbar=(i == len(theta_values)-1), cbar_label=cbar_label
        )
        
        # Set labels
        title = f'θ = {theta:.1f}'
        if can_aggregate: title += ' (Aggregated)'
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel('Individual preference (α)', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('Social influence (β)', fontsize=12)
        else:
            axes[i].set_ylabel('')
    
    plt.tight_layout(pad=1.8)
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        suffix = 'aggregated' if can_aggregate else 'theta'
        plt.savefig(f'{output_dir}/heatmap_alpha_beta_{suffix}.pdf', dpi=300, bbox_inches='tight')
    
    return fig



def plot_tipping_point_heatmap(data=None, file_path=None, save=True, theta_values=[-0.5, 0, 0.5], 
                              aggregate_veg_fractions=True):
    """Create tipping point heatmap, optionally aggregated across veg fractions"""
    # Set publication style and prepare data
    set_publication_style()
    result = _prep_data(data, file_path, theta_values)
    if result is None: return None
    data, theta_values = result
    
    # Get vegetarian fraction columns and create change column if possible
    init_col, final_col, data = _get_veg_columns(data)
    
    # Check if we can create a tipping point plot
    if 'change' not in data.columns:
        if final_col:
            print(f"Warning: 'change' column couldn't be created. Using '{final_col}' instead.")
            data['change'] = data[final_col]  # Use final value as fallback
        else:
            print("Error: Neither 'change' nor final vegetarian fraction columns found in data")
            return None
    
    can_aggregate = aggregate_veg_fractions and init_col
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(theta_values), figsize=(6*len(theta_values)+1, 6), sharey=True)
    if len(theta_values) == 1: axes = [axes]
    
    # Calculate vmin/vmax for consistent colorbar
    vmin, vmax = float('inf'), float('-inf')
    for theta in theta_values:
        theta_data = data[data['theta'] == theta]
        if not theta_data.empty:
            if can_aggregate:
                agg = theta_data.groupby(['alpha', 'beta'])['change'].mean()
            else:
                agg = theta_data.groupby(['alpha', 'beta'])['change'].mean()
            
            if not agg.empty:
                vmin = min(vmin, agg.min())
                vmax = max(vmax, agg.max())
    
    # Center colormap at 0
    vabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vabs, vabs
    
    # Create a heatmap for each theta value
    for i, theta in enumerate(theta_values):
        theta_data = data[data['theta'] == theta]
        
        # Process data based on aggregation setting
        if can_aggregate and 'tipped' in data.columns:
            agg = theta_data.groupby(['alpha', 'beta']).agg({
                'change': ['mean', 'std'],
                'tipped': ['mean']
            })
            agg.columns = ['change_mean', 'change_std', 'tipping_prob']
            agg = agg.reset_index()
            pivot_data = agg
            change_col = 'change_mean'
        else:
            pivot_data = theta_data.groupby(['alpha', 'beta'])['change'].mean().reset_index()
            change_col = 'change'
        
        # Get parameter values and create pivot table
        alpha_vals = sorted(pivot_data['alpha'].unique())
        beta_vals = sorted(pivot_data['beta'].unique(), reverse=True)
        
        pivot_table = pd.pivot_table(
            pivot_data,
            values=change_col,
            index=pd.CategoricalIndex(pivot_data['beta'], categories=beta_vals),
            columns=pd.CategoricalIndex(pivot_data['alpha'], categories=alpha_vals)
        )
        
        # Create heatmap
        cbar_label = 'Change in Vegetarian Fraction (Mean)' if can_aggregate else 'Change in Vegetarian Fraction'
        _create_heatmap(
            axes[i], pivot_table, alpha_vals, beta_vals, ECO_DIV_CMAP, vmin, vmax,
            center=0, show_cbar=(i == len(theta_values)-1), cbar_label=cbar_label
        )
        
        # Set labels
        title = f'θ = {theta:.1f}'
        if can_aggregate: title += ' (Aggregated)'
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel('Individual preference (α)', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('Social influence (β)', fontsize=12)
        else:
            axes[i].set_ylabel('')
        
        # Add black border to the plot
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
            
        # Make sure all axes share the y-axis
        if i > 0: axes[i].sharey(axes[0])
    
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        suffix = 'aggregated' if can_aggregate else 'theta'
        plt.savefig(f'{output_dir}/tipping_point_heatmap_{suffix}.pdf', dpi=300, bbox_inches='tight')
    
    return fig

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

def plot_3d_parameter_surface(data=None, file_path=None, save=True, plot_type='change', initial_veg_value=None):
    """
    Create 3D surface showing parameter effects on vegetarian fraction
    
    Args:
        data (DataFrame): DataFrame with alpha, beta, initial_veg_f and final_veg_f
        file_path (str): Path to data file if data not provided
        save (bool): Whether to save the plot
        plot_type (str): Type of plot to create:
                        'change' - z-axis shows change in veg fraction
                        'final' - z-axis shows final veg fraction
        initial_veg_value (float): Specific initial vegetarian fraction to use for surface.
                                  If None, creates surfaces for all available values.
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
    
    # Calculate change in vegetarian fraction
    data['change'] = data[final_veg_col] - data[init_veg_col]
    
    # Group by alpha, beta, and initial veg fraction to get average values
    grouped_data = data.groupby(['alpha', 'beta', init_veg_col])[
        [final_veg_col, 'change']].mean().reset_index()
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine what to plot on z-axis based on plot_type
    if plot_type == 'change':
        z_col = 'change'
        color_col = init_veg_col
        z_label = 'Change in Vegetarian Fraction'
        color_label = 'Initial Vegetarian Fraction'
        title = 'Effect of Parameters on Change in Vegetarian Population'
        # Use a diverging colormap for change (centered at 0)
        plot_cmap = 'coolwarm'
    else:  # 'final'
        z_col = final_veg_col
        color_col = init_veg_col
        z_label = 'Final Vegetarian Fraction'
        color_label = 'Initial Vegetarian Fraction'
        title = 'Final Vegetarian Fraction by Parameter Combination'
        # Use our eco-friendly colormap for final values
        plot_cmap = ECO_CMAP
    
    # Choose what to visualize based on initial_veg_value parameter
    if initial_veg_value is not None:
        # Find closest initial vegetarian fraction in the data
        veg_values = sorted(grouped_data[init_veg_col].unique())
        closest_veg = min(veg_values, key=lambda x: abs(x - initial_veg_value))
        
        # Filter to this specific initial vegetarian fraction
        surface_data = grouped_data[grouped_data[init_veg_col] == closest_veg]
        
        # Create pivot table for the surface
        pivot = surface_data.pivot_table(index='beta', columns='alpha', values=z_col)
        
        # Create meshgrid for surface
        alpha_vals = sorted(surface_data['alpha'].unique())
        beta_vals = sorted(surface_data['beta'].unique())
        alpha_grid, beta_grid = np.meshgrid(alpha_vals, beta_vals)
        
        # Create surface
        surf = ax.plot_surface(
            alpha_grid, beta_grid, pivot.values,
            cmap=plot_cmap,
            edgecolor='none',
            alpha=0.8,
            antialiased=True
        )
        
        # Add colorbar for the surface
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(z_label)
        
        # Add title with initial vegetarian fraction
        title = f"{title} (Initial Veg. Fraction = {closest_veg:.2f})"
        
    else:
        # Create a surface for each initial vegetarian fraction
        veg_values = sorted(grouped_data[init_veg_col].unique())
        
        # Create empty array for colorbar mappable
        norm = plt.Normalize(grouped_data[color_col].min(), grouped_data[color_col].max())
        sm = plt.cm.ScalarMappable(cmap=plot_cmap, norm=norm)
        sm.set_array([])
        
        # Plot each surface with transparency
        for veg_f in veg_values:
            surface_data = grouped_data[grouped_data[init_veg_col] == veg_f]
            
            # Create pivot table for the surface
            pivot = surface_data.pivot_table(index='beta', columns='alpha', values=z_col)
            
            # Create meshgrid for surface
            alpha_vals = sorted(surface_data['alpha'].unique())
            beta_vals = sorted(surface_data['beta'].unique())
            alpha_grid, beta_grid = np.meshgrid(alpha_vals, beta_vals)
            
            # Create surface with color based on initial veg fraction
            surf = ax.plot_surface(
                alpha_grid, beta_grid, pivot.values,
                color=plt.cm.viridis(norm(veg_f)),
                edgecolor='none',
                alpha=0.7,
                antialiased=True,
                label=f'Initial Veg = {veg_f:.2f}'
            )
        
        # Add colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(color_label)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(color_label)
    
    # Set labels
    ax.set_xlabel('Individual preference (α)', labelpad=10)
    ax.set_ylabel('Social influence (β)', labelpad=10)
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(title)
    
    # Improve appearance
    ax.view_init(elev=30, azim=45)
    ax.grid(True, alpha=0.3)
    
    # Make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    plt.tight_layout()
    
    # Add a custom legend for multiple surfaces if needed
    if initial_veg_value is None and len(veg_values) > 1:
        # Create custom proxy artists for legend
        legend_elements = []
        for i, veg_f in enumerate(veg_values):
            color = plt.cm.viridis(norm(veg_f))
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, fc=color, ec="none", alpha=0.7,
                             label=f'Initial Veg = {veg_f:.2f}')
            )
        
        # Add legend outside the plot
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.05, 1), title="Initial Veg. Fraction")
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, f'3d_parameter_{plot_type}_surface.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return ax



def plot_trajectory_param_twin(data=None, file_path=None, save=True):
    """Create side-by-side plot of parameterized and twin modes"""
    # Set publication style
    set_publication_style()
    linewidth = 0.8
    
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
    
    # Define colorblind-friendly base colors
    parameterized_base = "#ff7f00"  # Orange
    twin_base = "#984ea3"  # Purple
    
    # Create figure with 2 subplots (1 for parameterized, 1 for twin)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    # 1. Plot parameterized mode
    ax = axs[0]
    param_data = data[data['agent_ini'] == 'parameterized']
    
    if len(param_data) > 0:
        # Group by run index for consistent coloring
        run_groups = param_data.groupby('run')
        num_runs = len(run_groups)
        
        # Create color variations
        colors = create_color_variations(parameterized_base, num_runs)
        
        for run_idx, (_, group) in enumerate(run_groups):
            color = colors[run_idx % len(colors)]
            
            for _, row in group.iterrows():
                trajectory = row['fraction_veg_trajectory']
                if isinstance(trajectory, list) and len(trajectory) > 0:
                    time_steps = np.arange(len(trajectory))
                    ax.plot(time_steps, trajectory, color=color, linewidth=linewidth)
                
        # Get parameter values for title
        if len(param_data) > 0:
            alpha = param_data.iloc[0]['alpha']
            beta = param_data.iloc[0]['beta']
            theta = param_data.iloc[0]['theta'] if 'theta' in param_data.columns else 0
            ax.set_title(f"Parameterized: α={alpha:.2f}, β={beta:.2f}, θ={theta:.2f}")
        else:
            ax.set_title("Parameterized")
    else:
        ax.set_title("Parameterized (no data)")
        
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Vegetarian Fraction")
    ax.set_ylim(0, 1)
    apply_axis_style(ax)
    
    # 2. Plot twin mode
    ax = axs[1]
    twin_data = data[data['agent_ini'] == 'twin']
    
    if len(twin_data) > 0:
        # Group by run index for consistent coloring
        run_groups = twin_data.groupby('run')
        num_runs = len(run_groups)
        
        # Create color variations
        colors = create_color_variations(twin_base, num_runs)
        
        for run_idx, (_, group) in enumerate(run_groups):
            color = colors[run_idx % len(colors)]
            
            for _, row in group.iterrows():
                trajectory = row['fraction_veg_trajectory']
                if isinstance(trajectory, list) and len(trajectory) > 0:
                    time_steps = np.arange(len(trajectory))
                    ax.plot(time_steps, trajectory, color=color, linewidth=linewidth)
                
        ax.set_title("Twin: Survey Individual Parameters")
    else:
        ax.set_title("Twin (no data)")
        
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Vegetarian Fraction")
    ax.set_ylim(0, 1)
    apply_axis_style(ax)
    
    # Add a legend showing the different modes
    param_patch = plt.Line2D([0], [0], color=parameterized_base, linewidth=linewidth+1, label='Parameterized')
    twin_patch = plt.Line2D([0], [0], color=twin_base, linewidth=linewidth+1, label='Twin')
    
    fig.legend(handles=[param_patch, twin_patch], 
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)  # More space at bottom for legend
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'param_twin_trajectories.pdf')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig

def plot_trajectory_synthetic(data=None, file_path=None, save=True):
    """Create grid of trajectory plots for synthetic mode with theta parameter"""
    # Set publication style
    set_publication_style()
    linewidth = 0.8
    
    # Load data if not provided
    if data is None:
        if file_path is None:
            print("Either data or file_path must be provided")
            return None
        data = load_data_file(file_path)
        if data is None:
            return None
    
    # Filter for synthetic mode only
    synthetic_data = data[data['agent_ini'] == 'synthetic']
    
    if len(synthetic_data) == 0:
        print("No synthetic data found")
        return None
    
    # Get unique parameter values
    theta_values = sorted(synthetic_data['theta'].unique())
    alpha_values = sorted(synthetic_data['alpha'].unique())
    beta_values = sorted(synthetic_data['beta'].unique())
    
    # Determine grid size based on theta values
    n_theta = len(theta_values)
    n_plots_per_theta = min(4, len(alpha_values) * len(beta_values))  # Limit to 4 plots per theta
    
    # Select which alpha-beta combinations to show
    # For simplicity, we'll just take the first n_plots_per_theta combinations
    ab_combinations = []
    for a in alpha_values:
        for b in beta_values:
            ab_combinations.append((a, b))
            if len(ab_combinations) >= n_plots_per_theta:
                break
        if len(ab_combinations) >= n_plots_per_theta:
            break
    
    # Create figure with rows for each theta and columns for each alpha-beta combination
    fig, axs = plt.subplots(n_theta, n_plots_per_theta, figsize=(3*n_plots_per_theta, 3*n_theta))
    
    # Make axs a 2D array even if it's 1D
    if n_theta == 1 and n_plots_per_theta == 1:
        axs = np.array([[axs]])
    elif n_theta == 1:
        axs = np.array([axs])
    elif n_plots_per_theta == 1:
        axs = np.array([[ax] for ax in axs])
    
    # Define color for synthetic mode
    synthetic_base = "#377eb8"  # Blue
    
    # Plot each combination
    for t_idx, theta in enumerate(theta_values):
        for ab_idx, (alpha, beta) in enumerate(ab_combinations):
            if ab_idx >= n_plots_per_theta:
                continue
                
            ax = axs[t_idx, ab_idx]
            
            # Filter data for this parameter combination
            alpha_tol = 0.05
            beta_tol = 0.05
            theta_tol = 0.05
            subset = synthetic_data[
                (synthetic_data['alpha'] >= alpha - alpha_tol) & 
                (synthetic_data['alpha'] <= alpha + alpha_tol) &
                (synthetic_data['beta'] >= beta - beta_tol) & 
                (synthetic_data['beta'] <= beta + beta_tol) &
                (synthetic_data['theta'] >= theta - theta_tol) &
                (synthetic_data['theta'] <= theta + theta_tol)
            ]
            
            # Plot trajectories with different shades based on run index
            if len(subset) > 0:
                # Group by run index to assign consistent colors
                run_groups = subset.groupby('run')
                num_runs = len(run_groups)
                
                # Create color variations
                colors = create_color_variations(synthetic_base, num_runs)
                
                for run_idx, (_, group) in enumerate(run_groups):
                    color = colors[run_idx % len(colors)]
                    
                    for _, row in group.iterrows():
                        trajectory = row['fraction_veg_trajectory']
                        if isinstance(trajectory, list) and len(trajectory) > 0:
                            time_steps = np.arange(len(trajectory))
                            ax.plot(time_steps, trajectory, color=color, linewidth=linewidth)
            
            # Set title and labels
            ax.set_title(f"α={alpha:.2f}, β={beta:.2f}, θ={theta:.2f}")
            
            # Only add x-label for bottom row
            if t_idx == len(theta_values) - 1:
                ax.set_xlabel("Time Steps")
            
            # Only add y-label for leftmost column
            if ab_idx == 0:
                ax.set_ylabel("Vegetarian Fraction")
                
            ax.set_ylim(0, 1)
            apply_axis_style(ax)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'synthetic_trajectories.pdf')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig


def plot_trajectory_grid(data=None, file_path=None, save=True):
    """Create separate plots for different agent initialization modes"""
    # Load data if not provided
    if data is None:
        if file_path is None:
            print("Either data or file_path must be provided")
            return None
        data = load_data_file(file_path)
        if data is None:
            return None
    
    # Create separate plots
    fig_param_twin = plot_trajectory_param_twin(data, save=save)
    fig_synthetic = plot_trajectory_synthetic(data, save=save)
    
    return (fig_param_twin, fig_synthetic)

def plot_individual_reductions_distribution(data=None, file_path=None, save=True):
    """Create distribution plot of individual emission reductions"""
    set_publication_style()
    
    # Load data if not provided
    if data is None:
        if file_path is None:
            print("Either data or file_path must be provided")
            return None
        data = load_data_file(file_path)
        if data is None:
            return None
    
    # Extract and flatten individual reductions
    reductions = []
    for _, row in data.iterrows():
        if isinstance(row['individual_reductions'], list):
            reductions.extend(row['individual_reductions'])
        else:
            reductions.append(row['individual_reductions'])
    
    reductions = np.array(reductions)
    
    # Calculate statistics before truncation
    max_value = np.max(reductions)
    percentile_99 = np.percentile(reductions, 99)
    
    # Truncate data to 99th percentile
    truncated_data = reductions[reductions <= percentile_99]
    
    # Create figure
    fig, ax = plt.figure(figsize=(8, 5)), plt.gca()
    
    # Create explicit bin edges starting at exactly 0
    bin_edges = np.linspace(0, percentile_99, 31)  # 31 edges = 30 bins
    
    # Plot histogram with explicit bins
    ax.hist(truncated_data, bins=bin_edges, color=COLORS['secondary'], alpha=0.7, 
            edgecolor='white', linewidth=0.5)
    
    # Add KDE with lower scaling
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(truncated_data, bw_method=0.1)
    x_kde = np.linspace(0, percentile_99, 1000)
    
    # Scale KDE to 80% of our y limit to ensure it stays within bounds
    max_hist_height = min(ax.get_ylim()[1], 20000)
    y_kde = kde(x_kde) * (0.8 * max_hist_height / kde(kde.dataset.min()))
    
    ax.plot(x_kde, y_kde, color='#e29578', linewidth=2.5)
    
    # Set y-axis limit AFTER plotting both histogram and KDE
    ax.set_ylim(0, 20000)
    
    # Annotation about truncation
    ax.annotate(f"Data truncated at 99th percentile ({percentile_99:.0f} kg CO₂)\nMax value: {max_value:.0f} kg CO₂", 
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    ax.set_xlabel('Emissions Reduction Attributed [kg CO₂]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    apply_axis_style(ax)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'individual_reductions_distribution.pdf')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig


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
    three_d_param_file = None
    trajectory_file = None
    
    # Load 3D parameter file for plots that need aggregation across vegetarian fractions
    if choice in ['1', '5', '6', '8']:
        print("\nLoading 3D parameter file for plots that need multiple vegetarian fractions:")
        three_d_param_file = select_file('3d_parameter_analysis')
        if three_d_param_file is None:
            print("Warning: 3D parameter file not found. Falling back to parameter_sweep file.")
    
    # Load parameter sweep file for plots that don't need aggregation
    if (choice in ['1', '4', '5', '8'] and three_d_param_file is None) or choice in ['4', '8']:
        parameter_sweep_file = select_file('parameter_sweep')
        
    if choice in ['2', '8']:
        emissions_file = select_file('emissions')
        
    if choice in ['3', '8']:
        veg_growth_file = select_file('veg_growth')
    
    if choice in ['7', '8']:
        trajectory_file = select_file('trajectory_analysis')
    
    # Create plots based on user selection and available files
    if choice in ['1', '8']:
        if three_d_param_file:
            print("\nCreating heatmap with aggregation across vegetarian fractions...")
            plot_heatmap_alpha_beta(file_path=three_d_param_file, aggregate_veg_fractions=True)
        elif parameter_sweep_file:
            print("\nCreating heatmap without aggregation (single vegetarian fraction)...")
            plot_heatmap_alpha_beta(file_path=parameter_sweep_file, aggregate_veg_fractions=False)
        else:
            print("Cannot create heatmap: No suitable data file found")
    
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
        
    if choice in ['5', '8']:
        if three_d_param_file:
            print("\nCreating tipping point heatmap with aggregation across vegetarian fractions...")
            plot_tipping_point_heatmap(file_path=three_d_param_file, aggregate_veg_fractions=True)
        elif parameter_sweep_file:
            print("\nCreating tipping point heatmap without aggregation (single vegetarian fraction)...")
            plot_tipping_point_heatmap(file_path=parameter_sweep_file, aggregate_veg_fractions=False)
        else:
            print("Cannot create tipping point heatmap: No suitable data file found")
        
    if choice in ['6', '8'] and three_d_param_file:
        plot_3d_parameter_surface(file_path=three_d_param_file)
    elif choice == '6' and not three_d_param_file:
        print("Cannot create 3D parameter surface: No 3D parameter analysis file found")
        
    if choice in ['7', '8'] and trajectory_file:
        plot_trajectory_grid(file_path=trajectory_file)
    elif choice == '7' and not trajectory_file:
        print("Cannot create trajectory grid: No trajectory analysis file found")
        
if  __name__ ==  '__main__': 
    
    main()