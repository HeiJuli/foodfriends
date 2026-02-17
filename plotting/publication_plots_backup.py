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
    """12-panel plot: 4 network snapshots + 4 CCDF distributions + 4 trajectories
    log_scale: None, 'y', or 'loglog' for scaling options (CCDF always uses loglog)"""
    from matplotlib.ticker import LogLocator, NullFormatter
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    set_publication_style()

    # Muted publication colors for reducer highlights
    COL_TOP10 = '#6a994e'   # sage green - top 10% reducers
    COL_TOP1  = '#d4a029'   # muted gold - top reducer

    if data is None:
        data = load_data(file_path)
        if data is None: return None

    median_row = data[data['is_median_twin']].iloc[0]
    snapshots = median_row['snapshots']

    trajectory = median_row['fraction_veg_trajectory']
    if isinstance(trajectory, list) and len(trajectory) > 0:
        print(f"INFO: Initial veg fraction = {trajectory[0]:.3f}, Final = {trajectory[-1]:.3f}")
        print(f"INFO: Trajectory length = {len(trajectory)} timesteps")
        traj_y_max = max(trajectory) * 1.1
    else:
        traj_y_max = 0.5

    # Figure layout: nested gridspecs
    fig = plt.figure(figsize=(17.8*cm, 13.5*cm))
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.7],
                                hspace=0.3, top=0.94, bottom=0.06, left=0.10, right=0.97)
    gs_top = outer_gs[0].subgridspec(2, 4, height_ratios=[2.5, 1], hspace=0.18, wspace=0.12)
    gs_bot = outer_gs[1].subgridspec(1, 4, wspace=0.12)

    # Network layout
    G = snapshots['final']['graph']
    try:
        pos = nx.spectral_layout(G, seed=42)
    except Exception:
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)

    pos_array = np.array(list(pos.values()))
    x_min, x_max = pos_array[:, 0].min(), pos_array[:, 0].max()
    y_min, y_max_net = pos_array[:, 1].min(), pos_array[:, 1].max()

    # Time points
    all_times = sorted([t for t in snapshots.keys() if isinstance(t, int) and t > 0])
    time_points = ([0] + (all_times[:2] if len(all_times) >= 2 else all_times) + ['final'])[:4]
    print(f"INFO: Plotting snapshots at times: {time_points}")

    # Pre-compute global CCDF x-range for consistent axes
    all_red_t = []
    for tp in time_points:
        r = np.array(snapshots[tp]['reductions'])
        if np.max(r) > 0:
            all_red_t.extend(r[r > 0] / 1000)
    if all_red_t:
        ccdf_xmin = min(all_red_t) * 0.5
        ccdf_xmax = max(all_red_t) * 1.5
    else:
        ccdf_xmin, ccdf_xmax = 1e-1, 1e2

    # --- Top legend (diet types only) ---
    net_legend = [
        Patch(facecolor='#2a9d8f', edgecolor='#333', linewidth=0.4, label='Vegetarian'),
        Patch(facecolor='#e76f51', edgecolor='#333', linewidth=0.4, label='Meat eater'),
        Patch(facecolor=COL_TOP10, edgecolor='#333', linewidth=0.4, label='Top 10% reducers'),
        Patch(facecolor=COL_TOP1, edgecolor='#333', linewidth=0.4, label='Top reducer'),
    ]
    fig.legend(handles=net_legend, loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=4, fontsize=6.5, frameon=False, handletextpad=0.4, columnspacing=1.0)

    for i, t in enumerate(time_points):
        snap = snapshots[t]
        reductions = np.array(snap['reductions'])

        # === Row 0: Network ===
        net_ax = fig.add_subplot(gs_top[0, i])
        node_colors = ['#2a9d8f' if d == 'veg' else '#e76f51' for d in snap['diets']]
        nx.draw_networkx_edges(G, pos, ax=net_ax, alpha=0.3, width=0.05)
        nx.draw_networkx_nodes(G, pos, ax=net_ax, node_color=node_colors, node_size=2, alpha=0.9,
                              edgecolors='#333', linewidths=0.15)

        top_reducer_value = 0
        if np.max(reductions) > 0:
            n_top = max(1, int(0.1 * len(reductions)))
            top_idx = np.argsort(reductions)[-n_top:]
            nodes_list = list(G.nodes())

            top_10_nodes = [nodes_list[j] for j in top_idx[:-1] if reductions[j] > 0]
            if top_10_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=top_10_nodes, ax=net_ax,
                                     node_color=COL_TOP10, node_size=5, alpha=0.9,
                                     edgecolors='#333', linewidths=0.2)

            top_reducer_idx = top_idx[-1]
            if reductions[top_reducer_idx] > 0:
                nx.draw_networkx_nodes(G, pos, nodelist=[nodes_list[top_reducer_idx]], ax=net_ax,
                                     node_color=COL_TOP1, node_size=6, alpha=1.0,
                                     edgecolors='#333', linewidths=0.3)
                top_reducer_value = reductions[top_reducer_idx]

        title = '$t_0$' if t == 0 else '$t_{end}$' if t == 'final' else f't = {t//1000}k'
        net_ax.set_title(title, fontsize=10, pad=2)

        pad = 0.02
        net_ax.set_xlim(x_min - pad, x_max + pad)
        net_ax.set_ylim(y_min - pad, y_max_net + pad)
        net_ax.set_aspect('equal', adjustable='box')
        net_ax.axis('off')

        if top_reducer_value > 0:
            net_ax.text(0.5, -0.08, f'{top_reducer_value/1000:.1f} t CO$_2$e',
                       transform=net_ax.transAxes, ha='center', va='top',
                       fontsize=5.5, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                edgecolor=COL_TOP1, linewidth=0.8, alpha=0.9))

        # === Row 1: CCDF (complementary cumulative distribution) ===
        ccdf_ax = fig.add_subplot(gs_top[1, i])
        reductions_tonnes = reductions / 1000

        if np.max(reductions) == 0:
            ccdf_ax.text(0.5, 0.5, 'No reductions yet', ha='center', va='center',
                        transform=ccdf_ax.transAxes, fontsize=7, color='gray', style='italic')
        else:
            # Empirical CCDF: P(X > x)
            pos_red = np.sort(reductions_tonnes[reductions_tonnes > 0])
            ccdf_y = 1.0 - np.arange(1, len(pos_red) + 1) / len(pos_red)
            ccdf_ax.step(pos_red, ccdf_y, where='post', color=COLORS['secondary'],
                        linewidth=1.0, alpha=0.9)

            # Percentile markers
            n_top = max(1, int(0.1 * len(reductions)))
            top_10_thr = np.sort(reductions_tonnes)[-n_top] if n_top < len(reductions_tonnes) else 0
            top_1_val = np.max(reductions_tonnes)

            if top_10_thr > 0:
                ccdf_ax.axvline(top_10_thr, color=COL_TOP10, linestyle='--', linewidth=1.0, alpha=0.9, zorder=10)
            if top_1_val > 0 and top_1_val != top_10_thr:
                ccdf_ax.axvline(top_1_val, color=COL_TOP1, linestyle='--', linewidth=1.0, alpha=0.9, zorder=10)

        # Always use log-log for CCDF (standard for power-law visualization)
        ccdf_ax.set_xscale('log')
        ccdf_ax.set_yscale('log')
        ccdf_ax.set_xlim(ccdf_xmin, ccdf_xmax)
        ccdf_ax.set_ylim(1e-3, 1.5)
        ccdf_ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ccdf_ax.xaxis.set_minor_formatter(NullFormatter())

        ccdf_ax.spines['top'].set_visible(False)
        ccdf_ax.spines['right'].set_visible(False)
        ccdf_ax.tick_params(axis='both', labelsize=5.5)

        if i == 0:
            ccdf_ax.set_ylabel('$P(X > x)$', fontsize=7)
        else:
            ccdf_ax.set_ylabel('')
            ccdf_ax.set_yticklabels([])

        ccdf_ax.set_xlabel('Reduction [t CO$_2$e]', fontsize=6)

    # --- Trajectory row ---
    for i, t in enumerate(time_points):
        traj_ax = fig.add_subplot(gs_bot[0, i])
        trajectory = median_row['fraction_veg_trajectory']

        if isinstance(trajectory, list):
            end_idx = 1 if t == 0 else (len(trajectory) if t == 'final' else t)
            t_range = np.arange(end_idx) / 1000
            traj_ax.plot(t_range, trajectory[:end_idx], color=COLORS['vegetation'],
                        linewidth=1.0, alpha=0.9)
            traj_ax.scatter(t_range[-1], trajectory[end_idx-1], color=COLORS['vegetation'],
                          s=12, zorder=5, edgecolors='#333', linewidths=0.3)

            traj_ax.set_ylim(0, traj_y_max)
            traj_ax.spines['top'].set_visible(False)
            traj_ax.spines['right'].set_visible(False)

            if i == 0:
                traj_ax.set_ylabel('$F_{veg}$', fontsize=7)
            else:
                traj_ax.set_ylabel('')
                traj_ax.set_yticklabels([])

            traj_ax.set_xlabel('$t$ [thousands]', fontsize=6)
            traj_ax.tick_params(axis='both', labelsize=5.5)
            traj_ax.set_xlim(0, len(trajectory) / 1000)

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
def plot_trajectory_param_twin(data=None, file_path=None, save=True, xlim_max=None):
    """Plot twin mode trajectories only

    Args:
        xlim_max: Maximum x-axis limit in thousands (e.g., 20 for 20k timesteps).
                  If None, uses full data range. Arrows always show steady state.
    """
    set_publication_style()

    if data is None:
        data = load_data(file_path)
        if data is None: return None

    fig, ax = plt.subplots(1, 1, figsize=(9*cm, 8*cm))

    lw = 0.8
    # Twin mode only
    twin_data = data[data['agent_ini'] == 'twin']
    if len(twin_data) > 0:
        colors = create_color_variations("#984ea3", len(twin_data))
        trajectories_data = []
        for i, (_, row) in enumerate(twin_data.iterrows()):
            trajectory = row['fraction_veg_trajectory']
            if isinstance(trajectory, list):
                t_thousands = np.arange(len(trajectory)) / 1000
                line, = ax.plot(t_thousands, trajectory, color=colors[i % len(colors)], alpha=0.7, linewidth=lw)
                trajectories_data.append((line, trajectory[-1]))

        # Set x-axis limits if specified
        if xlim_max is not None:
            ax.set_xlim(0, xlim_max)

        # Add arrows at right edge showing steady state values
        xlim = ax.get_xlim()
        x_arrow = xlim[1]
        for line, final_val in trajectories_data:
            ax.annotate('', xy=(x_arrow, final_val),
                       xytext=(x_arrow*0.98, final_val),
                       arrowprops=dict(arrowstyle='->', color=line.get_color(), lw=1.5))

        ax.set_title("Twin: Survey Individual Parameters")
    else:
        print("WARNING: No twin mode data found")

    ax.set_xlabel("t (thousands)")
    ax.set_ylabel("Vegetarian Fraction")
    ax.set_ylim(0, 0.5)
    apply_axis_style(ax)

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/twin_trajectories.pdf', dpi=300, bbox_inches='tight')
        print("Saved twin_trajectories.pdf")

    return fig
#%%
def plot_tipping_comparison_ccdf(data=None, file_path=None, save=True, tipping_threshold=0.6):
    """Compare CCDF of emissions reductions for tipping vs non-tipping runs.

    Args:
        tipping_threshold: Final veg fraction above which a run is considered 'tipped' (default 0.6)
    """
    set_publication_style()

    if data is None:
        data = load_data(file_path)
        if data is None: return None

    # Classify runs as tipping or non-tipping based on final veg fraction
    data['tipped'] = data['fraction_veg_trajectory'].apply(
        lambda traj: traj[-1] if isinstance(traj, list) and len(traj) > 0 else 0
    ) > tipping_threshold

    tipping_runs = data[data['tipped']]
    non_tipping_runs = data[~data['tipped']]

    print(f"INFO: {len(tipping_runs)} tipping runs, {len(non_tipping_runs)} non-tipping runs")

    if len(tipping_runs) == 0 or len(non_tipping_runs) == 0:
        print("WARNING: Need both tipping and non-tipping runs for comparison")
        return None

    # Extract reductions from final snapshots
    def get_reductions(run_data):
        all_reductions = []
        for _, row in run_data.iterrows():
            if 'snapshots' in row and 'final' in row['snapshots']:
                reductions = np.array(row['snapshots']['final']['reductions'])
                all_reductions.extend(reductions[reductions > 0] / 1000)  # Convert to tonnes
        return np.array(all_reductions)

    tip_reductions = get_reductions(tipping_runs)
    no_tip_reductions = get_reductions(non_tipping_runs)

    if len(tip_reductions) == 0 or len(no_tip_reductions) == 0:
        print("WARNING: No reduction data available")
        return None

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14*cm, 7*cm), sharey=True)

    # Global x-axis limits
    all_red = np.concatenate([tip_reductions, no_tip_reductions])
    xmin, xmax = all_red.min() * 0.5, all_red.max() * 1.5

    for i, (ax, reductions, label, color) in enumerate([
        (axes[0], no_tip_reductions, 'Non-Tipping Runs', COLORS['secondary']),
        (axes[1], tip_reductions, 'Tipping Runs', COLORS['vegetation'])
    ]):
        # Empirical CCDF
        sorted_red = np.sort(reductions)
        ccdf_y = 1.0 - np.arange(1, len(sorted_red) + 1) / len(sorted_red)
        ax.step(sorted_red, ccdf_y, where='post', color=color, linewidth=1.5, alpha=0.9, label='Actual')

        # Equal distribution baseline
        mean_red = np.mean(reductions)
        ax.axvline(mean_red, color='#888', linestyle='--', linewidth=1.2, alpha=0.8, label='Equal distribution')

        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(1e-3, 1.5)
        ax.set_xlabel('Reduction [t CO$_2$e]', fontsize=8)
        ax.set_title(label, fontsize=10, pad=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=7)

        if i == 0:
            ax.set_ylabel('$P(X > x)$', fontsize=8)
            ax.legend(fontsize=7, frameon=True, loc='upper right')

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/tipping_comparison_ccdf.pdf', dpi=300, bbox_inches='tight')
        print("Saved tipping_comparison_ccdf.pdf")

    return fig

def plot_parameter_sweep_trajectories(data=None, file_path=None, save=True, xlim_max=None):
    """Plot trajectories grouped by parameter combinations for supplement. Supports parameterized mode.

    Args:
        xlim_max: Maximum x-axis limit in timesteps (e.g., 20000 for 20k timesteps).
                  If None, uses full data range. Arrows always show steady state.
    """
    set_publication_style()

    if data is None:
        data = load_data(file_path)
        if data is None: return None

    # Check if agent_ini column exists
    if 'agent_ini' in data.columns:
        agent_modes = data['agent_ini'].unique()
        print(f"INFO: Found agent modes: {agent_modes}")
    else:
        agent_modes = ['other']

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

        # Color by agent_ini if available
        if 'agent_ini' in data.columns and 'parameterized' in subset['agent_ini'].values:
            color_base = "#ff7f00"  # Orange for parameterized
            label_suffix = " (Parameterized)"
        else:
            color_base = COLORS['primary']
            label_suffix = ""

        # Plot all trajectories for this parameter set
        colors = create_color_variations(color_base, len(subset))
        trajectories_data = []
        for j, (_, row) in enumerate(subset.iterrows()):
            trajectory = row['fraction_veg_trajectory']
            if isinstance(trajectory, list):
                line, = ax.plot(np.arange(len(trajectory)), trajectory,
                       color=colors[j % len(colors)], alpha=0.7, linewidth=1)
                trajectories_data.append((line, trajectory[-1]))

        # Set x-axis limits if specified
        if xlim_max is not None:
            ax.set_xlim(0, xlim_max)

        # Add arrows at right edge showing steady state values
        xlim = ax.get_xlim()
        x_arrow = xlim[1]
        for line, final_val in trajectories_data:
            ax.annotate('', xy=(x_arrow, final_val),
                       xytext=(x_arrow*0.98, final_val),
                       arrowprops=dict(arrowstyle='->', color=line.get_color(), lw=1.5))

        ax.set_title(param_set + label_suffix, fontsize=10)
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
        print("[6] Twin Trajectories")
        print("[7] Network Agency Evolution")
        print("[8] Parameter Sweep Trajectories (supplement)")
        print("[9] Tipping vs Non-Tipping CCDF Comparison")
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
            if file_path:
                xlim_input = input("X-axis max limit in thousands (e.g., 20 for 20k) [Enter for auto]: ")
                xlim_max = float(xlim_input) if xlim_input else None
                plot_trajectory_param_twin(file_path=file_path, xlim_max=xlim_max)
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
                xlim_input = input("X-axis max limit in timesteps (e.g., 20000 for 20k) [Enter for auto]: ")
                xlim_max = float(xlim_input) if xlim_input else None
                data = load_data(file_path)
                if data is not None:
                    plot_parameter_sweep_trajectories(data=data, xlim_max=xlim_max)
                else:
                    print("No parameter sweep trajectory data found.")
        elif choice == '9':
            file_path = select_file('trajectory_analysis')
            if file_path:
                threshold_input = input("Tipping threshold (final veg fraction, e.g., 0.6) [Enter for 0.6]: ")
                threshold = float(threshold_input) if threshold_input else 0.6
                plot_tipping_comparison_ccdf(file_path=file_path, tipping_threshold=threshold)
        elif choice == '0':
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()