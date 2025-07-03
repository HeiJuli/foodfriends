#!/usr/bin/env python3
"""Streamlined publication plotting script"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from plot_styles import set_publication_style, apply_axis_style, COLORS, ECO_CMAP, ECO_DIV_CMAP

#%%


cm = 1/2.54
def ensure_output_dir():
    output_dir = '../visualisations_output'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data(file_path):
    try:
        return pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_color_variations(base_color, n):
    import matplotlib.colors as mcolors
    base_rgb = mcolors.to_rgb(base_color)
    return [tuple(min(1.0, c * (0.7 + 0.3 * i / max(1, n-1))) for c in base_rgb) for i in range(n)]


#%%
def plot_heatmap(data=None, file_path=None, value_type='final', theta_values=[-0.5, 0, 0.5], save=True):
    """Unified heatmap plotting for final values or change values"""
    set_publication_style()
    
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    
    # Handle missing theta column
    if 'theta' not in data.columns:
        data['theta'] = 0
        theta_values = [0]
    
    # Get available theta values
    available_thetas = sorted(data['theta'].unique())
    theta_values = [t for t in theta_values if t in available_thetas]
    
    # Create change column if needed
    if 'change' not in data.columns and 'final_veg_f' in data.columns:
        init_col = next((c for c in ['initial_veg_f', 'veg_f'] if c in data.columns), None)
        if init_col:
            data['change'] = data['final_veg_f'] - data[init_col]
        else:
            data['change'] = data['final_veg_f'] - 0.2  # Default assumption
    
    # Set value column and colormap
    if value_type == 'change':
        value_col = 'change'
        cmap = ECO_DIV_CMAP
        center = 0
        label = 'Change in Vegetarian Fraction'
    else:
        value_col = 'final_veg_f'
        cmap = ECO_CMAP
        center = None
        label = 'Final Vegetarian Fraction'
    
    # Create figure
    fig, axes = plt.subplots(1, len(theta_values), figsize=(6*len(theta_values)+1, 6), sharey=True)
    if len(theta_values) == 1: axes = [axes]
    
    # Calculate global vmin/vmax
    vmin, vmax = float('inf'), float('-inf')
    for theta in theta_values:
        theta_data = data[data['theta'] == theta]
        if not theta_data.empty:
            agg = theta_data.groupby(['alpha', 'beta'])[value_col].mean()
            if not agg.empty:
                vmin = min(vmin, agg.min())
                vmax = max(vmax, agg.max())
    
    if center == 0:
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs
    
    # Create heatmaps
    for i, theta in enumerate(theta_values):
        theta_data = data[data['theta'] == theta]
        pivot_data = theta_data.groupby(['alpha', 'beta'])[value_col].mean().reset_index()
        
        alpha_vals = sorted(pivot_data['alpha'].unique())
        beta_vals = sorted(pivot_data['beta'].unique(), reverse=True)
        
        pivot_table = pd.pivot_table(
            pivot_data, values=value_col,
            index=pd.CategoricalIndex(pivot_data['beta'], categories=beta_vals),
            columns=pd.CategoricalIndex(pivot_data['alpha'], categories=alpha_vals)
        )
        
        sns.heatmap(pivot_table, cmap=cmap, ax=axes[i], vmin=vmin, vmax=vmax, center=center,
                   cbar=(i == len(theta_values)-1), cbar_kws={'label': label} if i == len(theta_values)-1 else None)
        
        axes[i].set_title(f'θ = {theta:.1f}')
        axes[i].set_xlabel('Individual preference (α)')
        if i == 0:
            axes[i].set_ylabel('Social influence (β)')
        
        # Format tick labels
        axes[i].set_xticks(np.arange(len(alpha_vals)) + 0.5)
        axes[i].set_xticklabels([f"{v:.1f}" for v in alpha_vals], rotation=0)
        axes[i].set_yticks(np.arange(len(beta_vals)) + 0.5)
        axes[i].set_yticklabels([f"{v:.1f}" for v in beta_vals], rotation=0)
    
    plt.tight_layout()
    
    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/heatmap_{value_type}.pdf', dpi=300, bbox_inches='tight')
        print(f"Saved heatmap_{value_type}.pdf")
    
    return fig

#%%
def plot_network_agency_evolution(data=None, file_path=None, save=True, log_scale=None):
    """Simplified 8-panel plot: 4 network snapshots + 4 reduction distributions
    log_scale: None, 'y', or 'loglog' for scaling options"""
    set_publication_style()
    
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    
    # Get median twin trajectory
    median_row = data[data['is_median_twin']].iloc[0]
    snapshots = median_row['snapshots']
    
    # Create figure
    fig = plt.figure(figsize=(17.8*cm, 10*cm)) 
    gs = fig.add_gridspec(2, 4, height_ratios=[2.5, 1], hspace=0.05, wspace=0.05)
    
    # Network layout
    G = snapshots['final']['graph']

    try:
        pos = nx.spectral_layout(G, seed=42)
    except:
        pos = nx.spring_layout(G, k=4.5, iterations=100, seed=42)
    
    # Get position bounds for consistent aspect ratio
    pos_array = np.array(list(pos.values()))
    x_min, x_max = pos_array[:, 0].min(), pos_array[:, 0].max()
    y_min, y_max = pos_array[:, 1].min(), pos_array[:, 1].max()
    
    # Get 4 time points
    all_times = sorted([t for t in snapshots.keys() if isinstance(t, int)])
    if len(all_times) >= 2:
        time_points = [0, all_times[len(all_times)//3], all_times[2*len(all_times)//3], 'final']
    else:
        time_points = [0] + all_times + ['final']
    time_points = time_points[:4]
    
    # Plot networks and histograms
    for i, t in enumerate(time_points):
        # Network subplot
        net_ax = fig.add_subplot(gs[0, i])
        snap = snapshots[t]
        
        # Draw network
        node_colors = ['#2a9d8f' if d == 'veg' else '#e76f51' for d in snap['diets']]
        nx.draw_networkx_edges(G, pos, ax=net_ax, alpha=0.3, width=0.2)
        nx.draw_networkx_nodes(G, pos, ax=net_ax, node_color=node_colors, node_size=1, alpha=0.9)
        
        # Highlight top reducers and add labels
        reductions = np.array(snap['reductions'])
        if np.max(reductions) > 0:
            top3_idx = np.argsort(reductions)[-3:]
            top3_nodes = [list(G.nodes())[j] for j in top3_idx if reductions[j] > 0]
            if top3_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=top3_nodes, ax=net_ax, 
                                     node_color='#f4a261', node_size=10, alpha=1.0)
                
                # Add reduction labels
                for j, node in zip(top3_idx, top3_nodes):
                    if reductions[j] > 0:
                        x, y = pos[node]
                        net_ax.annotate(f'{reductions[j]:.0f}', (x, y), 
                                      xytext=(8, 8), textcoords='offset points',
                                      fontsize=8, fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        
        title = 'Initial' if t == 0 else 'Final' if t == 'final' else f'{t//1000}k steps'
        net_ax.set_title(title, fontsize=12)
        
        # Set consistent bounds and aspect ratio
        padding = 0.02
        net_ax.set_xlim(x_min - padding, x_max + padding)
        net_ax.set_ylim(y_min - padding, y_max + padding)
        net_ax.set_aspect('equal', adjustable='box')
        net_ax.axis('off')
        
        # Histogram subplot
        hist_ax = fig.add_subplot(gs[1, i])
        reductions_tonnes = reductions / 1000  # Convert to tonnes
        
        if np.max(reductions) == 0:
            hist_ax.text(0.5, 0.5, 'No reductions yet', ha='center', va='center', 
                        transform=hist_ax.transAxes, fontsize=8, color='gray')
            # Apply consistent scaling even for empty plots
            if log_scale == 'loglog':
                hist_ax.set_yscale('log')
                hist_ax.set_xscale('log')
                hist_ax.set_xlim(1e-1, 1e2)
                hist_ax.set_ylim(1e-3, 1e1)
                hist_ax.yaxis.set_major_formatter(plt.LogFormatterMathtext())
                hist_ax.xaxis.set_major_formatter(plt.LogFormatterMathtext())
            elif log_scale == 'y':
                hist_ax.set_yscale('log')
                hist_ax.set_ylim(1e-3, 1e1)
                hist_ax.yaxis.set_major_formatter(plt.LogFormatterMathtext())
        else:
            # Plot normalized histogram
            hist_ax.hist(reductions_tonnes, bins=15, density=True, color=COLORS['secondary'], 
                        alpha=0.7, edgecolor='white', linewidth=0.5)
            
            if log_scale == 'y':
                hist_ax.set_yscale('log')
                hist_ax.yaxis.set_major_formatter(plt.LogFormatterMathtext())
            elif log_scale == 'loglog':
                hist_ax.set_yscale('log')
                hist_ax.set_xscale('log')
                hist_ax.yaxis.set_major_formatter(plt.LogFormatterMathtext())
                hist_ax.xaxis.set_major_formatter(plt.LogFormatterMathtext())
        
        # Clean axis styling
        hist_ax.spines['top'].set_visible(False)
        hist_ax.spines['right'].set_visible(False)
        
        if i == 0:
            hist_ax.set_ylabel('Normalised count', fontsize=8)
        else:
            hist_ax.set_ylabel('')
            hist_ax.tick_params(labelleft=False)
        
        if i == 1 or i == 2:  # Only middle plots get x-label
            hist_ax.set_xlabel('Reduction [tonnes CO₂]', fontsize=8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2a9d8f', label='Vegetarian'),
        Patch(facecolor='#e76f51', label='Meat Eater'),
        Patch(facecolor='#f4a261', label='Top Reducers')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/network_agency_evolution.pdf', dpi=300, bbox_inches='tight')
        print("Saved network_agency_evolution.pdf")
    
    return fig

#%%
def plot_emissions_vs_veg_fraction(data=None, file_path=None, save=True):
    set_publication_style()
    
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data['veg_fraction'], data['final_CO2'], s=70, alpha=0.8,
               color=COLORS['primary'], edgecolor='white', linewidth=1.0)
    
    plt.xlabel('Vegetarian Fraction')
    plt.ylabel('Final Average CO₂ Emissions [kg/year]')
    plt.title('Impact of Vegetarian Population on CO₂ Emissions')
    apply_axis_style(plt.gca())
    plt.tight_layout()
    
    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/emissions_vs_veg_fraction.pdf', dpi=300, bbox_inches='tight')
        print("Saved emissions_vs_veg_fraction.pdf")
    
    return plt.gca()
#%%
def plot_veg_growth(data=None, file_path=None, save=True):
    set_publication_style()
    
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    
    plt.figure(figsize=(8, 6))
    plt.plot(data['initial_veg_fraction'], data['final_veg_fraction'], 'o-', 
             color=COLORS['vegetation'], linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No Change')
    
    plt.xlabel('Initial Vegetarian Fraction')
    plt.ylabel('Final Vegetarian Fraction')
    plt.title('Growth in Vegetarian Population')
    plt.legend()
    apply_axis_style(plt.gca())
    plt.tight_layout()
    
    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/veg_growth.pdf', dpi=300, bbox_inches='tight')
        print("Saved veg_growth.pdf")
    
    return plt.gca()
#%%
def plot_individual_reductions_distribution(data=None, file_path=None, save=True):
    set_publication_style()
    
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    
    # Extract reductions
    reductions = []
    for _, row in data.iterrows():
        if isinstance(row['individual_reductions'], list):
            reductions.extend(row['individual_reductions'])
    
    reductions = np.array(reductions)
    percentile_99 = np.percentile(reductions, 99)
    truncated_data = reductions[reductions <= percentile_99]
    
    plt.figure(figsize=(8, 5))
    plt.hist(truncated_data, bins=30, color=COLORS['secondary'], alpha=0.7, 
             edgecolor='white', linewidth=0.5)
    
    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(truncated_data, bw_method=0.1)
    x_kde = np.linspace(0, percentile_99, 1000)
    y_kde = kde(x_kde) * (15000 / kde(kde.dataset.min()))
    plt.plot(x_kde, y_kde, color='#e29578', linewidth=2.5)
    
    plt.ylim(0, 20000)
    plt.xlabel('Emissions Reduction Attributed [kg CO₂]')
    plt.ylabel('Frequency')
    
    # Annotation
    plt.annotate(f"Truncated at 99th percentile ({percentile_99:.0f} kg CO₂)", 
                xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    apply_axis_style(plt.gca())
    plt.tight_layout()
    
    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/individual_reductions_distribution.pdf', dpi=300, bbox_inches='tight')
        print("Saved individual_reductions_distribution.pdf")
    
    return plt.gca()
#%%
def plot_trajectory_param_twin(data=None, file_path=None, save=True):
    set_publication_style()
    
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    
    fig, axs = plt.subplots(1, 2, figsize=(17.8*cm, 8*cm))

    
    lw = 0.8
    # Parameterized mode
    param_data = data[data['agent_ini'] == 'parameterized']
    if len(param_data) > 0:
        colors = create_color_variations("#ff7f00", len(param_data))
        for i, (_, row) in enumerate(param_data.iterrows()):
            trajectory = row['fraction_veg_trajectory']
            if isinstance(trajectory, list):
                t_thousands = np.arange(len(trajectory)) / 1000
                axs[0].plot(t_thousands, trajectory, color=colors[i % len(colors)], alpha=0.7, linewidth=lw)
        
        alpha, beta, theta = param_data.iloc[0][['alpha', 'beta', 'theta']]
        axs[0].set_title(f"Parameterized: α={alpha:.2f}, β={beta:.2f}, θ={theta:.2f}")
    
    # Twin mode  
    twin_data = data[data['agent_ini'] == 'twin']
    if len(twin_data) > 0:
        colors = create_color_variations("#984ea3", len(twin_data))
        for i, (_, row) in enumerate(twin_data.iterrows()):
            trajectory = row['fraction_veg_trajectory']
            if isinstance(trajectory, list):
                t_thousands = np.arange(len(trajectory)) / 1000
                axs[1].plot(t_thousands, trajectory, color=colors[i % len(colors)], alpha=0.7, linewidth=lw)
        
        axs[1].set_title("Twin: Survey Individual Parameters")
    
    for ax in axs:
        ax.set_xlabel("t (thousands)")
        ax.set_ylabel("Vegetarian Fraction")
        ax.set_ylim(0, 1)
        apply_axis_style(ax)
    
    plt.tight_layout()
    
    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/param_twin_trajectories.pdf', dpi=300, bbox_inches='tight')
        print("Saved param_twin_trajectories.pdf")
    
    return fig
#%%
def plot_parameter_sweep_trajectories(data=None, file_path=None, save=True):
    """Plot trajectories grouped by parameter combinations for supplement"""
    set_publication_style()
    
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    
    # Get unique parameter combinations
    param_sets = data['parameter_set'].unique()
    
    # Create subplot grid
    n_sets = len(param_sets)
    cols = min(3, n_sets)
    rows = (n_sets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharey=True)
    if n_sets == 1: axes = [axes]
    axes = axes.flatten() if n_sets > 1 else axes
    
    for i, param_set in enumerate(param_sets):
        if i >= len(axes): break
        
        ax = axes[i]
        subset = data[data['parameter_set'] == param_set]
        
        # Plot all trajectories for this parameter set
        colors = create_color_variations(COLORS['primary'], len(subset))
        for j, (_, row) in enumerate(subset.iterrows()):
            trajectory = row['fraction_veg_trajectory']
            if isinstance(trajectory, list):
                ax.plot(np.arange(len(trajectory)), trajectory, 
                       color=colors[j % len(colors)], alpha=0.7, linewidth=1)
        
        ax.set_title(param_set, fontsize=10)
        ax.set_xlabel('Time Steps')
        if i % cols == 0:
            ax.set_ylabel('Vegetarian Fraction')
        ax.set_ylim(0, 1)
        apply_axis_style(ax)
    
    # Hide unused subplots
    for i in range(n_sets, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/parameter_sweep_trajectories.pdf', dpi=300, bbox_inches='tight')
        print("Saved parameter_sweep_trajectories.pdf")
    
    return fig
#%%
def select_file(pattern):
    import glob
    from datetime import datetime
    
    files = glob.glob(f'../model_output/{pattern}_*.pkl')
    if not files:
        print(f"No {pattern} files found")
        return None
    
    files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"\nAvailable {pattern} files:")
    for i, f in enumerate(files):
        name = os.path.basename(f)
        time = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
        print(f"[{i+1}] {name} ({time})")
    
    choice = input(f"Select file (1-{len(files)}, Enter for latest): ")
    try:
        return files[int(choice)-1] if choice else files[0]
    except (ValueError, IndexError):
        return files[0]
def main():
    print("=== Publication Plots ===")
    
    while True:
        print("\n[1] Final Vegetarian Fraction Heatmap")
        print("[2] Tipping Point (Change) Heatmap") 
        print("[3] Emissions vs Vegetarian Fraction")
        print("[4] Vegetarian Growth Analysis")
        print("[5] Individual Reductions Distribution")
        print("[6] Parameter vs Twin Trajectories")
        print("[7] Network Agency Evolution")
        print("[8] Parameter Sweep Trajectories (supplement)")
        print("[0] Exit")
        
        choice = input("Select: ")
        
        if choice == '1':
            file_path = select_file('parameter_analysis')
            if file_path: plot_heatmap(file_path=file_path, value_type='final')
        elif choice == '2':
            file_path = select_file('parameter_analysis')
            if file_path: plot_heatmap(file_path=file_path, value_type='change')
        elif choice == '3':
            file_path = select_file('emissions')
            if file_path: plot_emissions_vs_veg_fraction(file_path=file_path)
        elif choice == '4':
            file_path = select_file('veg_growth')
            if file_path: plot_veg_growth(file_path=file_path)
        elif choice == '5':
            file_path = select_file('parameter_analysis')
            if file_path: plot_individual_reductions_distribution(file_path=file_path)
        elif choice == '6':
            file_path = select_file('trajectory_analysis')
            if file_path: plot_trajectory_param_twin(file_path=file_path)
        elif choice == '7':
            file_path = select_file('trajectory_analysis')
            if file_path: 
                data = load_data(file_path)
                if data is not None and 'is_median_twin' in data.columns:
                    plot_network_agency_evolution(file_path=file_path, log_scale="loglog")
                else:
                    print("No twin mode snapshots found. Run trajectory analysis with twin mode first.")
        elif choice == '8':
            file_path = select_file('parameter_trajectories')
            if file_path: 
                data = load_data(file_path)
                if data is not None:
                    plot_parameter_sweep_trajectories(data=data)
                else:
                    print("No parameter sweep trajectory data found.")
        elif choice == '0':
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()