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
def plot_network_agency_evolution(data=None, file_path=None, save=True):
    """8-panel plot: 4 network snapshots + 4 reduction distributions"""
    set_publication_style()
    
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    
    # Get median twin trajectory
    median_row = data[data['is_median_twin']].iloc[0]
    snapshots = median_row['snapshots']
    
    # Create figure with custom height ratios - networks larger than histograms
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], hspace=0.05, wspace=0.1)
    
    # Create axes
    net_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    hist_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    
    # Better layout for clustered networks - try spectral layout for community structure
    G = snapshots['final']['graph']
    try:
        # Use spectral layout which often reveals cluster structure better
        pos = nx.spectral_layout(G, seed=42)
    except:
        # Fallback to spring layout with very loose clustering
        pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42)
    nodes = list(G.nodes())
    
    # Get 4 time points: t=0, 33%, 66%, final
    all_times = sorted([t for t in snapshots.keys() if isinstance(t, int)])
    if len(all_times) >= 2:
        time_points = [0, all_times[len(all_times)//3], all_times[2*len(all_times)//3], 'final']
    else:
        time_points = [0] + all_times + ['final']
    time_points = time_points[:4]  # Ensure exactly 4
    
    # Top row: Network snapshots
    for i, t in enumerate(time_points):
        ax = net_axes[i]
        snap = snapshots[t]
        
        # Node colors
        node_colors = ['#2a9d8f' if d == 'veg' else '#e76f51' for d in snap['diets']]
        
        # Draw network with thinner edges to show structure better
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.3)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=6, alpha=0.9)
        
        # Highlight top reducers
        reductions = np.array(snap['reductions'])
        if np.max(reductions) > 0:
            top3_idx = np.argsort(reductions)[-3:]
            top3_nodes = [nodes[j] for j in top3_idx if reductions[j] > 0]
            if top3_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=top3_nodes, ax=ax, 
                                     node_color='#f4a261', node_size=20, alpha=1.0)
                
                # Add reduction labels
                for j, node in zip(top3_idx, top3_nodes):
                    if reductions[j] > 0:
                        x, y = pos[node]
                        ax.annotate(f'{reductions[j]:.0f}', (x, y), 
                                  xytext=(8, 8), textcoords='offset points',
                                  fontsize=7, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        
        title = 'Initial' if t == 0 else f'Final' if t == 'final' else f'{t/1000:.0f}k steps'
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Bottom row: Histograms
    for i, t in enumerate(time_points):
        ax = hist_axes[i]
        reductions = np.array(snapshots[t]['reductions'])
        
        if np.max(reductions) > 0:
            ax.hist(reductions, bins=12, color=COLORS['secondary'], alpha=0.7, 
                   edgecolor='white', linewidth=0.5)
            # Mark top reducer
            top_val = np.max(reductions)
            ax.axvline(top_val, color='#f4a261', linewidth=2, alpha=0.9)
        else:
            ax.text(0.5, 0.5, 'No reductions', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=8)
        
        ax.set_xlabel('Reduction [kg CO₂]', fontsize=8)
        if i == 0:
            ax.set_ylabel('Count', fontsize=8)
        apply_axis_style(ax)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2a9d8f', label='Vegetarian'),
        Patch(facecolor='#e76f51', label='Meat Eater'),
        Patch(facecolor='#f4a261', label='Top Reducers')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/network_agency_evolution.pdf', dpi=300, bbox_inches='tight')
        print("Saved network_agency_evolution.pdf")
    
    return fig

def plot_network_agency_evolution_old(data=None, file_path=None, save=True):
    """4-panel plot: network snapshots + reduction distribution"""
    set_publication_style()
    
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    
    # Get median twin trajectory
    median_row = data[data['is_median_twin']].iloc[0]
    snapshots = median_row['snapshots']
    
    # Get final reductions and identify top 3
    final_reductions = np.array(snapshots['final']['reductions'])
    top3_idx = np.argsort(final_reductions)[-3:]
    top3_values = final_reductions[top3_idx]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Network layout (use same layout for all panels)
    G = snapshots['final']['graph']
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    nodes = list(G.nodes())  # Map array indices to node IDs
    
    # Time points for snapshots
    time_points = sorted([t for t in snapshots.keys() if isinstance(t, int)])
    
    # Panels 1-3: Network snapshots
    for i, t in enumerate(time_points):
        ax = axes[i]
        snap = snapshots[t]
        
        # Node colors: veg=green, meat=red
        node_colors = ['#2a9d8f' if d == 'veg' else '#e76f51' for d in snap['diets']]
        
        # Draw network
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=0.3)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=15, alpha=0.9)
        
        # For panels 2&3, highlight top 3 and add labels
        if i > 0:  # Skip first panel
            reductions = np.array(snap['reductions'])
            current_top3_idx = np.argsort(reductions)[-3:]
            current_top3_nodes = [nodes[j] for j in current_top3_idx]  # Map to node IDs
            
            # Highlight top 3 with larger gold nodes
            if len(current_top3_nodes) > 0:
                nx.draw_networkx_nodes(G, pos, nodelist=current_top3_nodes, ax=ax, 
                                     node_color='#f4a261', node_size=120, alpha=0.9, 
                                     edgecolors='black', linewidths=1)
                
                # Add reduction labels
                for j, node in zip(current_top3_idx, current_top3_nodes):
                    if reductions[j] > 0:
                        x, y = pos[node]
                        ax.annotate(f'{reductions[j]:.0f}', (x, y), 
                                  xytext=(8, 8), textcoords='offset points',
                                  fontsize=8, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        
        ax.set_title(f'Time {t/1000:.0f}k steps' if isinstance(t, int) else 'Initial')
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Panel 4: Distribution
    ax = axes[3]
    
    # Use all reductions, including zeros for full picture
    if len(final_reductions) > 0:
        # Histogram
        n, bins, patches = ax.hist(final_reductions, bins=20, color=COLORS['secondary'], 
                                  alpha=0.7, edgecolor='white', linewidth=0.5)
        
        # Mark top 3 positions with vertical lines only
        for val in top3_values:
            if val > 0:
                ax.axvline(val, color='#f4a261', linewidth=2, alpha=0.9)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel('Emissions Reduction Attributed [kg CO₂]')
    ax.set_ylabel('Number of Agents')
    ax.set_title('Final Reduction Distribution')
    apply_axis_style(ax)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2a9d8f', label='Vegetarian'),
        Patch(facecolor='#e76f51', label='Meat Eater'),
        Patch(facecolor='#f4a261', label='Top 3 Reducers')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
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
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    # Parameterized mode
    param_data = data[data['agent_ini'] == 'parameterized']
    if len(param_data) > 0:
        colors = create_color_variations("#ff7f00", len(param_data))
        for i, (_, row) in enumerate(param_data.iterrows()):
            trajectory = row['fraction_veg_trajectory']
            if isinstance(trajectory, list):
                axs[0].plot(np.arange(len(trajectory)), trajectory, color=colors[i % len(colors)], linewidth=0.8)
        
        alpha, beta, theta = param_data.iloc[0][['alpha', 'beta', 'theta']]
        axs[0].set_title(f"Parameterized: α={alpha:.2f}, β={beta:.2f}, θ={theta:.2f}")
    
    # Twin mode  
    twin_data = data[data['agent_ini'] == 'twin']
    if len(twin_data) > 0:
        colors = create_color_variations("#984ea3", len(twin_data))
        for i, (_, row) in enumerate(twin_data.iterrows()):
            trajectory = row['fraction_veg_trajectory']
            if isinstance(trajectory, list):
                axs[1].plot(np.arange(len(trajectory)), trajectory, color=colors[i % len(colors)], linewidth=0.8)
        
        axs[1].set_title("Twin: Survey Individual Parameters")
    
    for ax in axs:
        ax.set_xlabel("Time Steps")
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
                    plot_network_agency_evolution(file_path=file_path)
                else:
                    print("No twin mode snapshots found. Run trajectory analysis with twin mode first.")
        elif choice == '0':
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()