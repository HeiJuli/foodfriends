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

def plot_heatmap_alpha_beta(data=None, file_path=None, save=True, theta_values=[-0.5, 0, 0.5]):
    """
    Create heatmap showing how alpha and beta affect system metrics for different theta values
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
    
    # Ensure alpha, beta, theta columns exist
    if 'alpha' not in data.columns or 'beta' not in data.columns:
        print("Data must contain alpha and beta columns")
        return None
    
    # Check if theta column exists, if not, assume theta=0 for all
    if 'theta' not in data.columns:
        print("Warning: No theta column found, assuming theta=0 for all data")
        data['theta'] = 0
        theta_values = [0]
    
    # Filter theta values that exist in the data
    available_thetas = sorted(data['theta'].unique())
    theta_values = [t for t in theta_values if t in available_thetas]
    
    if not theta_values:
        print("No specified theta values found in data")
        return None
    
    # Create figure with subplots using GridSpec for more control
    fig = plt.figure(figsize=(6*len(theta_values)+1, 6))
    gs = fig.add_gridspec(1, len(theta_values), width_ratios=[1]*len(theta_values))
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(theta_values))]
    
    # Ensure axes is always an array, even with one subplot
    if len(theta_values) == 1:
        axes = [axes]
    
    # Store vmin/vmax across all heatmaps for consistent colorbar
    vmin, vmax = float('inf'), float('-inf')
    for theta in theta_values:
        theta_data = data[data['theta'] == theta]
        pivot_data = theta_data.groupby(['alpha', 'beta'])['final_veg_fraction'].mean().reset_index()
        if not pivot_data.empty:
            curr_min = pivot_data['final_veg_fraction'].min()
            curr_max = pivot_data['final_veg_fraction'].max()
            vmin = min(vmin, curr_min)
            vmax = max(vmax, curr_max)
    
    # Create a heatmap for each theta value
    for i, theta in enumerate(theta_values):
        # Filter data for this theta value
        theta_data = data[data['theta'] == theta]
        
        # Create pivot table for heatmap with sorted beta values (descending order for y-axis reversal)
        pivot_data = theta_data.groupby(['alpha', 'beta'])['final_veg_fraction'].mean().reset_index()
        beta_values = sorted(theta_data['beta'].unique(), reverse=True)  # Sort in descending order
        alpha_values = sorted(theta_data['alpha'].unique())
        
        # Create pivot table with explicitly ordered indices
        pivot_table = pd.pivot_table(
            pivot_data,
            values='final_veg_fraction',
            index=pd.CategoricalIndex(pivot_data['beta'], categories=beta_values),
            columns=pd.CategoricalIndex(pivot_data['alpha'], categories=alpha_values)
        )
        
        # Plot heatmap on the corresponding subplot
        ax = axes[i]
        sns.heatmap(
            pivot_table, 
            cmap=ECO_CMAP,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar=(i == len(theta_values)-1),  # Only add colorbar to last plot
            cbar_kws={'label': 'Final Vegetarian Fraction'} if i == len(theta_values)-1 else None
        )
        
        # Set subplot title with theta value
        ax.set_title(f'θ = {theta:.1f}', fontsize=12)
        
        # Format tick labels to one decimal place
        ax.set_xticks(np.arange(len(alpha_values)) + 0.5)
        # Explicitly round to 1 decimal place
        ax.set_xticklabels([f"{round(v, 1):.1f}" for v in alpha_values], rotation=0)
        
        # For y ticks, we need to handle the reversed order
        # Set y ticks for all subplots explicitly for consistent formatting
        ax.set_yticks(np.arange(len(beta_values)) + 0.5)
        # Explicitly round to 1 decimal place before formatting
        rounded_labels = [f"{round(v, 1):.1f}" for v in beta_values]
        ax.set_yticklabels(rounded_labels, rotation=0)
        
        if i == 0:  # Only set y-axis label on first subplot
            ax.set_ylabel('Social influence (β)', fontsize=12)
        else:
            ax.set_ylabel('')  # Remove y label for other plots
        
        # Set x label for all plots
        ax.set_xlabel('Individual preference (α)', fontsize=12)
    
    # Adjust layout with custom parameters for better spacing
    plt.tight_layout(pad=1.8)
    
    # Set equal heights for all subplots
    for ax in axes:
        ax.set_box_aspect(1)
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'heatmap_alpha_beta_theta.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
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

def plot_tipping_point_heatmap(data=None, file_path=None, save=True, theta_values=[-0.5, 0, 0.5]):
    """
    Create heatmap showing parameter combinations leading to tipping points for different theta values
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
    
    # Check if theta column exists, if not, assume theta=0 for all
    if 'theta' not in data.columns:
        print("Warning: No theta column found, assuming theta=0 for all data")
        data['theta'] = 0
        theta_values = [0]
    
    # Filter theta values that exist in the data
    available_thetas = sorted(data['theta'].unique())
    theta_values = [t for t in theta_values if t in available_thetas]
    
    if not theta_values:
        print("No specified theta values found in data")
        return None
    
    # Create change column if not exist
    if 'change' not in data.columns:
        if 'final_veg_fraction' in data.columns and 'initial_veg_fraction' in data.columns:
            data['change'] = data['final_veg_fraction'] - data['initial_veg_fraction']
        elif 'final_veg_f' in data.columns and 'initial_veg_f' in data.columns:
            data['change'] = data['final_veg_f'] - data['initial_veg_f']
        else:
            # Try to infer change if possible
            if 'final_veg_fraction' in data.columns:
                if 'fixed_veg_f' in data.columns:
                    data['change'] = data['final_veg_fraction'] - data['fixed_veg_f']
                else:
                    # Estimate initial veg fraction
                    data['change'] = data['final_veg_fraction'] - 0.2
            else:
                print("Data must contain columns to calculate change in vegetarian fraction")
                return None
    
    # Create tipped column if not exist
    if 'tipped' not in data.columns:
        # Define tipping as change greater than 20%
        data['tipped'] = data['change'] > 0.2
    
    # Create figure with subplots - make the figure a bit wider to accommodate colorbar 
    fig, axes = plt.subplots(1, len(theta_values), figsize=(6*len(theta_values)+1, 6), sharey=True)
    
    # Ensure axes is always an array, even with one subplot
    if len(theta_values) == 1:
        axes = [axes]
    
    # Store vmin/vmax across all heatmaps for consistent colorbar
    vmin, vmax = float('inf'), float('-inf')
    for theta in theta_values:
        theta_data = data[data['theta'] == theta]
        if not theta_data.empty:
            curr_min = theta_data['change'].min()
            curr_max = theta_data['change'].max()
            vmin = min(vmin, curr_min)
            vmax = max(vmax, curr_max)
    
    # Center colormap at 0
    vabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vabs, vabs
    
    # Create a heatmap for each theta value
    for i, theta in enumerate(theta_values):
        # Filter data for this theta value
        theta_data = data[data['theta'] == theta]
        
        # Get sorted parameter values
        alpha_values = sorted(theta_data['alpha'].unique())
        beta_values = sorted(theta_data['beta'].unique(), reverse=True)  # Sort in descending order
        
        # Create pivot tables with explicitly ordered indices
        pivot_data = theta_data.groupby(['alpha', 'beta'])['change'].mean().reset_index()
        pivot_table = pd.pivot_table(
            pivot_data,
            values='change',
            index=pd.CategoricalIndex(pivot_data['beta'], categories=beta_values),
            columns=pd.CategoricalIndex(pivot_data['alpha'], categories=alpha_values)
        )
        
        # Plot heatmap on the corresponding subplot
        ax = axes[i]
        sns.heatmap(
            pivot_table, 
            cmap=ECO_DIV_CMAP,
            ax=ax,
            center=0,
            vmin=vmin,
            vmax=vmax,
            cbar=(i == len(theta_values)-1),  # Only add colorbar to last plot
            cbar_kws={'label': 'Change in Vegetarian Fraction'} if i == len(theta_values)-1 else None
        )
        
        # Set subplot title with theta value
        ax.set_title(f'θ = {theta:.1f}', fontsize=12)
        
        # Set labels
        ax.set_xlabel('Individual preference (α)', fontsize=12)
        if i == 0:
            ax.set_ylabel('Social influence (β)', fontsize=12)
        else:
            ax.set_ylabel('')  # Remove y label for other plots
        
        # Format tick labels to one decimal place
        ax.set_xticks(np.arange(len(alpha_values)) + 0.5)
        ax.set_xticklabels([f"{v:.1f}" for v in alpha_values], rotation=0)
        
        # Set y ticks for all subplots for consistent formatting
        ax.set_yticks(np.arange(len(beta_values)) + 0.5)
        # Explicitly round to 1 decimal place
        rounded_labels = [f"{round(v, 1):.1f}" for v in beta_values]
        ax.set_yticklabels(rounded_labels, rotation=0)
        
        # Add black border to the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
            
        # Make sure all axes share the y-axis
        if i > 0:
            ax.sharey(axes[0])
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        output_dir = ensure_output_dir()
        output_file = os.path.join(output_dir, 'tipping_point_heatmap_theta.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig

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
        output_file = os.path.join(output_dir, 'individual_reductions_distribution.pdf')
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
    three_d_param_file = None
    trajectory_file = None
    
    if choice in ['1', '4', '5', '8']:
        parameter_sweep_file = select_file('parameter_sweep')
        
    if choice in ['2', '8']:
        emissions_file = select_file('emissions')
        
    if choice in ['3', '8']:
        veg_growth_file = select_file('veg_growth')
    
    if choice in ['6', '8']:
        three_d_param_file = select_file('3d_parameter_analysis')
    
    if choice in ['7', '8']:
        trajectory_file = select_file('trajectory_analysis')
    
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
        
    if choice in ['6', '8'] and three_d_param_file:
        plot_3d_parameter_surface(file_path=three_d_param_file)
    elif choice == '6' and not three_d_param_file:
        print("Cannot create 3D parameter surface: No 3D parameter analysis file found")
        
    if choice in ['7', '8'] and trajectory_file:
        plot_trajectory_grid(file_path=trajectory_file)
    elif choice == '7' and not trajectory_file:
        print("Cannot create trajectory grid: No trajectory analysis file found")

if __name__ == "__main__":
    main()