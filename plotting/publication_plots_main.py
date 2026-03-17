#!/usr/bin/env python3
"""Main publication plots"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from plot_styles import set_publication_style, apply_axis_style, COLORS, ECO_CMAP, ECO_DIV_CMAP

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

def plot_network_agency_evolution(data=None, file_path=None, save=True, log_scale=None,
                                  truncate_steps=None, analysis_t_end=None, mid_t=None):
    """7-panel plot: 4 network snapshots (top) + trajectory + CCDF + Lorenz (bottom)

    Args:
        analysis_t_end: If set, panels B (CCDF) and C (Lorenz) use the snapshot closest
                        to this timestep instead of the true final. Panel A still shows
                        the full trajectory for visual context.
    """
    from matplotlib.ticker import LogLocator, NullFormatter
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    set_publication_style()

    COL_TOP10 = '#6a994e'
    COL_TOP1  = '#d4a029'

    if data is None:
        data = load_data(file_path)
        if data is None: return None

    median_row = data[data['is_median_twin']].iloc[0]
    snapshots = median_row['snapshots']
    trajectory = median_row['fraction_veg_trajectory']

    if truncate_steps is not None and isinstance(trajectory, list):
        trajectory = trajectory[:truncate_steps]
        valid_int_times = sorted([t for t in snapshots if isinstance(t, int) and t <= truncate_steps])
        last_t = valid_int_times[-1] if valid_int_times else 0
        # Preserve 'steady' if it occurred before truncation point
        steady_t = median_row.get('steady_state_t', None)
        keep_steady = 'steady' in snapshots and (steady_t is not None and steady_t <= truncate_steps)
        snapshots = {t: snapshots[t] for t in ([0] + valid_int_times) if t in snapshots}
        if keep_steady:
            snapshots['steady'] = median_row['snapshots']['steady']
        snapshots['final'] = median_row['snapshots'][last_t]
        print(f"INFO: Truncated to {truncate_steps} steps; 'final' snapshot -> t={last_t}"
              f"{'; steady snapshot preserved' if keep_steady else ''}")

    # Build analysis snapshots: remap 'final' for B/C panels
    # Priority: explicit analysis_t_end > auto-detected 'steady' snapshot > true 'final'
    analysis_snapshots = snapshots
    if analysis_t_end is not None:
        int_times = sorted(t for t in snapshots if isinstance(t, int) and t > 0)
        best_t = min(int_times, key=lambda t: abs(t - analysis_t_end)) if int_times else 'final'
        analysis_snapshots = dict(snapshots)
        analysis_snapshots['final'] = snapshots[best_t] if best_t != 'final' else snapshots['final']
        print(f"INFO: analysis_t_end={analysis_t_end} -> B/C use snapshot t={best_t}")
    elif 'steady' in snapshots:
        analysis_snapshots = dict(snapshots)
        analysis_snapshots['final'] = snapshots['steady']
        ss_t_print = median_row.get('steady_state_t')
        print(f"INFO: Auto-detected steady-state snapshot used for B/C panels"
              + (f" (t={int(ss_t_print)})" if ss_t_print is not None else ""))

    # Resolve the integer t that analysis 'final' corresponds to (for trajectory marker)
    if analysis_t_end is not None:
        analysis_final_t = best_t if isinstance(best_t, int) else len(trajectory) - 1
    elif 'steady' in snapshots:
        ss_t = median_row.get('steady_state_t')
        analysis_final_t = int(ss_t) if ss_t is not None else len(trajectory) - 1
    else:
        analysis_final_t = len(trajectory) - 1

    if isinstance(trajectory, list) and len(trajectory) > 0:
        print(f"INFO: Initial veg fraction = {trajectory[0]:.3f}, Final = {trajectory[-1]:.3f}")
        traj_y_max = max(trajectory) * 1.1
    else:
        traj_y_max = 0.5

    # Figure layout: 4 networks top, trajectory + CCDF + Lorenz bottom
    fig = plt.figure(figsize=(17.8*cm, 12.5*cm))
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[2.8, 1.4],
                                hspace=0.18, top=0.93, bottom=0.08, left=0.08, right=0.97)
    gs_top = outer_gs[0].subgridspec(1, 3, wspace=0.08)
    gs_bot = outer_gs[1].subgridspec(1, 3, wspace=0.35, width_ratios=[1.2, 1, 1])

    # Network layout: giant component only (fixed across all snapshots)
    G_full = analysis_snapshots['final']['graph']
    giant_nodes = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(giant_nodes).copy()
    N_gc = G.number_of_nodes()
    giant_list = list(G.nodes())
    print(f"INFO: Network full N={G_full.number_of_nodes()}, giant component={N_gc} nodes, "
          f"{G.number_of_edges()} edges, "
          f"{nx.number_connected_components(G_full)} components")

    pos = nx.spring_layout(G, k=4/N_gc**0.5, iterations=80, seed=42)

    pos_array = np.array(list(pos.values()))
    x_min, x_max = pos_array[:, 0].min(), pos_array[:, 0].max()
    y_min, y_max_net = pos_array[:, 1].min(), pos_array[:, 1].max()

    # Time points
    all_times = sorted([t for t in snapshots.keys() if isinstance(t, int) and t > 0])
    if all_times:
        _mid_target = mid_t if mid_t is not None else (
            (len(trajectory) // 2) if isinstance(trajectory, list) else (all_times[-1] // 2))
        mid = min(all_times, key=lambda t: abs(t - _mid_target))
        if mid_t is not None:
            print(f"INFO: mid_t={mid_t} -> middle snapshot t={mid}")
    else:
        mid = None
    time_points = [t for t in [0, mid, 'final'] if t is not None]
    print(f"INFO: Plotting snapshots at times: {time_points}")

    # --- Legend ---
    net_legend = [
        Patch(facecolor='#2a9d8f', edgecolor='#333', linewidth=0.4, label='Vegetarian'),
        Patch(facecolor='#e76f51', edgecolor='#333', linewidth=0.4, label='Meat eater'),
        Patch(facecolor=COL_TOP10, edgecolor='#333', linewidth=0.4, label='Top 10% reducers'),
        Patch(facecolor=COL_TOP1, edgecolor='#333', linewidth=0.4, label='Top reducer'),
    ]
    fig.legend(handles=net_legend, loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=4, fontsize=6.5, frameon=False, handletextpad=0.4, columnspacing=1.0)

    # === Row 0: Network snapshots ===
    for i, t in enumerate(time_points):
        snap = analysis_snapshots[t] if t in analysis_snapshots else snapshots[t]
        all_diets = snap['diets']
        all_red   = np.array(snap['reductions'])
        net_ax = fig.add_subplot(gs_top[0, i])

        # Index by node id into the full diet/reduction arrays
        node_colors = ['#2a9d8f' if all_diets[n] == 'veg' else '#e76f51' for n in giant_list]
        reductions  = np.array([all_red[n] for n in giant_list])

        nx.draw_networkx_edges(G, pos, ax=net_ax, alpha=0.25, width=0.03)
        nx.draw_networkx_nodes(G, pos, ax=net_ax, node_color=node_colors, node_size=4,
                              alpha=0.9, edgecolors='#333', linewidths=0.15)

        top_reducer_value = 0
        if np.max(reductions) > 0:
            n_top = max(1, int(0.1 * len(reductions)))
            top_idx = np.argsort(reductions)[-n_top:]

            top_10_nodes = [giant_list[j] for j in top_idx[:-1] if reductions[j] > 0]
            if top_10_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=top_10_nodes, ax=net_ax,
                                     node_color=COL_TOP10, node_size=10, alpha=0.9,
                                     edgecolors='#333', linewidths=0.2)

            top_reducer_idx = top_idx[-1]
            if reductions[top_reducer_idx] > 0:
                nx.draw_networkx_nodes(G, pos, nodelist=[giant_list[top_reducer_idx]], ax=net_ax,
                                     node_color=COL_TOP1, node_size=12, alpha=1.0,
                                     edgecolors='#333', linewidths=0.3)
                top_reducer_value = reductions[top_reducer_idx]

        title = '$t_0$' if t == 0 else '$t_{end}$' if t == 'final' else f't = {t//1000}k'
        net_ax.set_title(title, fontsize=10, pad=2)

        pad_n = 0.02
        net_ax.set_xlim(x_min - pad_n, x_max + pad_n)
        net_ax.set_ylim(y_min - pad_n, y_max_net + pad_n)
        net_ax.set_aspect('equal', adjustable='box')
        net_ax.axis('off')

        if top_reducer_value > 0:
            net_ax.text(0.5, -0.06, f'{top_reducer_value/1000:.1f} t CO$_2$e',
                       transform=net_ax.transAxes, ha='center', va='top', fontsize=5.5,
                       fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white',
                       edgecolor=COL_TOP1, linewidth=0.8, alpha=0.9))

    # === Bottom-left: Single trajectory with snapshot markers ===
    traj_ax = fig.add_subplot(gs_bot[0, 0])
    if isinstance(trajectory, list):
        from scipy.signal import savgol_filter
        t_thousands = np.arange(len(trajectory)) / 1000
        traj_ax.plot(t_thousands, trajectory, color=COLORS['vegetation'], linewidth=1.0, alpha=0.9)

        # Vertical markers at snapshot times
        time_labels = []
        for t in time_points:
            if t == 0:
                t_val = 0
            elif t == 'final':
                t_val = min(analysis_final_t, len(trajectory) - 1)
            else:
                t_val = min(t, len(trajectory) - 1)
            t_k = t_val / 1000
            traj_ax.axvline(t_k, color='#888', linestyle=':', linewidth=0.7, alpha=0.6)
            traj_ax.scatter(t_k, trajectory[t_val], color=COLORS['vegetation'],
                          s=14, zorder=5, edgecolors='#333', linewidths=0.4)
            time_labels.append((t_k, t))

        # Tipping point: max of 2nd derivative (max adoption acceleration)
        COL_TIP = '#9b59b6'  # purple -- no clash with teal/orange/grey/green/gold
        traj_arr = np.array(trajectory)
        _burnin = 5000
        if len(traj_arr) > _burnin + 2001 * 2:
            _d2 = savgol_filter(traj_arr, window_length=2001, polyorder=3, deriv=2)
            _sm = savgol_filter(traj_arr, window_length=2001, polyorder=3)
            _d2[:_burnin] = 0
            _d2[_sm > 0.5] = 0
            _t_tip = int(np.argmax(_d2))
            _t_tip_k = _t_tip / 1000
            traj_ax.axvline(_t_tip_k, color=COL_TIP, linestyle=':', linewidth=0.9, alpha=0.8)
            print(f"INFO: Tipping point (max d2F/dt2) at t={_t_tip} ({_t_tip_k:.1f}k), "
                  f"F_veg={traj_arr[_t_tip]:.3f}")

    traj_ax.set_ylim(0, 1.0)
    traj_ax.set_xlim(-0.5, len(trajectory) / 1000)
    traj_ax.set_ylabel('$F_{veg}$', fontsize=7)
    traj_ax.set_xlabel('$t$ [thousands]', fontsize=7)
    traj_ax.spines['top'].set_visible(False)
    traj_ax.spines['right'].set_visible(False)
    traj_ax.tick_params(axis='both', labelsize=6)
    traj_ax.text(0.02, 0.95, 'A', transform=traj_ax.transAxes, fontsize=10,
                fontweight='bold', va='top')

    # === Bottom-right: Combined CCDF with time-colored curves ===
    ccdf_ax = fig.add_subplot(gs_bot[0, 1])

    # Time colormap: grey shades (light to dark) to avoid conflict with diet colors
    n_curves = len([t for t in time_points if t != 0])
    time_cmap = [plt.cm.Greys(v) for v in np.linspace(0.35, 0.85, n_curves)]

    curve_idx = 0
    for t in time_points:
        snap = analysis_snapshots[t] if t in analysis_snapshots else snapshots[t]
        reductions = np.array(snap['reductions'])
        if np.max(reductions) == 0:
            continue
        reductions_tonnes = reductions / 1000
        pos_red = np.sort(reductions_tonnes[reductions_tonnes > 1e-6])
        ccdf_y = 1.0 - np.arange(1, len(pos_red) + 1) / len(pos_red)

        label = '$t_0$' if t == 0 else '$t_{end}$' if t == 'final' else f'{t//1000}k'
        ccdf_ax.step(pos_red, ccdf_y, where='post', color=time_cmap[curve_idx],
                    linewidth=1.2, alpha=0.9, label=label)
        curve_idx += 1

    ccdf_ax.set_xscale('log')
    ccdf_ax.set_yscale('log')
    ccdf_ax.set_ylim(1e-3, 1.5)
    ccdf_ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ccdf_ax.xaxis.set_minor_formatter(NullFormatter())
    ccdf_ax.set_ylabel('$P(X > x)$', fontsize=7)
    ccdf_ax.set_xlabel('Reduction [t CO$_2$e]', fontsize=7)
    ccdf_ax.spines['top'].set_visible(False)
    ccdf_ax.spines['right'].set_visible(False)
    ccdf_ax.tick_params(axis='both', labelsize=6)
    ccdf_ax.legend(fontsize=5, frameon=False, loc='center', title='Snapshot',
                  title_fontsize=5)
    ccdf_ax.text(0.02, 0.95, 'B', transform=ccdf_ax.transAxes, fontsize=10,
                fontweight='bold', va='top')

    # === Bottom-right: Lorenz curve (Newman-style, sorted from largest reducer) ===
    lorenz_ax = fig.add_subplot(gs_bot[0, 2])
    for ci, t in enumerate(t for t in time_points if t != 0):
        snap = analysis_snapshots[t] if t in analysis_snapshots else snapshots[t]
        reductions = np.array(snap['reductions'])
        if np.max(reductions) == 0:
            continue
        sorted_red = np.sort(reductions)[::-1]
        cum_frac = np.cumsum(sorted_red) / sorted_red.sum()
        pop_frac = np.arange(1, len(sorted_red) + 1) / len(sorted_red)
        label = '$t_0$' if t == 0 else '$t_{end}$' if t == 'final' else f'{t//1000}k'
        lorenz_ax.plot(pop_frac, cum_frac, color=time_cmap[ci], lw=1.2, alpha=0.9, label=label)
    lorenz_ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    lorenz_ax.set_xlabel('$F_{pop}$', fontsize=7)
    lorenz_ax.set_ylabel('$F_{red}$', fontsize=7)
    lorenz_ax.set_xlim(0, 1)
    lorenz_ax.set_ylim(0, 1)
    lorenz_ax.set_aspect('equal', adjustable='datalim')
    lorenz_ax.spines['top'].set_visible(False)
    lorenz_ax.spines['right'].set_visible(False)
    lorenz_ax.tick_params(axis='both', labelsize=6)
    # Legend only in CCDF panel (B) to avoid duplication
    lorenz_ax.text(0.02, 0.95, 'C', transform=lorenz_ax.transAxes, fontsize=10,
                   fontweight='bold', va='top')

    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/network_agency_evolution.pdf', dpi=300, bbox_inches='tight')
        print("Saved network_agency_evolution.pdf")

    return fig

def _resolve_snapshot(data, analysis_t_end=None):
    """Resolve the analysis snapshot: explicit t_end > steady > final."""
    if 'is_median_twin' in data.columns and data['is_median_twin'].any():
        median_row = data[data['is_median_twin']].iloc[0]
    else:
        twin_data = data[data['agent_ini'].isin(['twin', 'sample-max'])]
        median_row = (twin_data if len(twin_data) else data).iloc[len(data) // 2]
    snapshots = median_row['snapshots']
    if analysis_t_end is not None:
        int_times = sorted(t for t in snapshots if isinstance(t, int) and t > 0)
        best_t = min(int_times, key=lambda t: abs(t - analysis_t_end)) if int_times else 'final'
        snap = snapshots[best_t] if best_t != 'final' else snapshots['final']
        print(f"INFO: analysis_t_end={analysis_t_end} -> using snapshot t={best_t}")
    elif 'steady' in snapshots:
        snap = snapshots['steady']
        ss_t_print = median_row.get('steady_state_t')
        print(f"INFO: Using auto-detected steady-state snapshot"
              + (f" (t={int(ss_t_print)})" if ss_t_print is not None else ""))
    else:
        snap = snapshots['final']
    return median_row, snap

def plot_amplification(data=None, file_path=None, save=True, analysis_t_end=None):
    """Standalone figure: rank-ordered amplification factor distribution."""
    set_publication_style()

    if data is None:
        data = load_data(file_path)
        if data is None: return None

    fig, ax = plt.subplots(figsize=(8.9*cm, 7*cm))

    DIRECT_REDUCTION_KG = 664  # 2054 - 1390 kg CO2/year

    median_row, snap = _resolve_snapshot(data, analysis_t_end)
    reductions_kg = np.array(snap['reductions'])
    pos_mask = reductions_kg > 1e-3
    multipliers = reductions_kg[pos_mask] / DIRECT_REDUCTION_KG

    # Early-adopter cutoff: converted before F_veg crosses 0.5
    change_times = snap.get('change_times')
    traj = median_row.get('fraction_veg_trajectory', median_row.get('fraction_veg_trajectory'))
    early_mult = None
    if change_times is not None and traj is not None:
        traj = np.array(traj)
        crossed = np.where(traj >= 0.5)[0]
        t_half = int(crossed[0]) if len(crossed) > 0 else len(traj)
        ct = np.array(change_times, dtype=float)
        early_mask = pos_mask & (ct > 0) & (ct <= t_half)  # converted before F_veg=0.5
        if early_mask.sum() > 0:
            early_mult = np.mean(reductions_kg[early_mask] / DIRECT_REDUCTION_KG)
            print(f"INFO: Early-adopter cutoff t={t_half} ({early_mask.sum()} agents, "
                  f"mean amplification {early_mult:.1f}x)")

    # Sort for rank-ordered plot
    multipliers_sorted = np.sort(multipliers)[::-1]
    ranks = np.arange(1, len(multipliers_sorted) + 1)
    rank_pct = ranks / len(multipliers_sorted) * 100

    # Rank-ordered curve with subtle fill
    ax.fill_between(rank_pct, multipliers_sorted, alpha=0.15, color=COLORS['secondary'])
    ax.plot(rank_pct, multipliers_sorted, color=COLORS['secondary'], linewidth=1.2)

    # Mean line
    mean_mult = np.mean(multipliers)
    ax.axhline(mean_mult, color='#555', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.text(97, mean_mult * 0.88, f'Mean: {mean_mult:.0f}x', fontsize=6, color='#555',
            va='top', ha='right')

    # Early-adopter mean line
    if early_mult is not None:
        ax.axhline(early_mult, color='#2ca02c', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.text(3, early_mult * 1.12, f'Early adopters: {early_mult:.1f}x', fontsize=6,
                color='#2ca02c', va='bottom', ha='left')

    # 1x baseline (personal only)
    ax.axhline(1.0, color='#aaa', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(50, 1.25, 'Personal only (1x)', fontsize=5.5, color='#aaa',
            va='bottom', ha='center')

    ax.set_xlabel('Agent rank [percentile]', fontsize=8)
    ax.set_ylabel('Amplification factor', fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=7)

    # Summary stats
    p90 = np.percentile(multipliers, 90)
    top_mult = np.max(multipliers)
    early_str = f'  |  Early: {early_mult:.1f}x' if early_mult else ''
    stats_text = f'Top: {top_mult:.0f}x  |  p90: {p90:.0f}x{early_str}  |  N = {len(multipliers)}'
    ax.text(0.50, 0.03, stats_text, transform=ax.transAxes, fontsize=5.5,
            va='bottom', ha='center', color='#444',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.8))

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/amplification_factor.pdf', dpi=300, bbox_inches='tight')
        print("Saved amplification_factor.pdf")

    return fig

def plot_agency_predictors(data=None, file_path=None, save=True, analysis_t_end=None):
    """Standalone single-column figure: what predicts high individual agency?
    Scatter of amplification factor vs agent properties (degree, theta).

    NOTE: This is exploratory. Further analysis needed to properly identify
    predictors of high agency (e.g., regression, SHAP values, network centrality
    measures beyond degree). See claude_stuff/agency_predictor_analysis_TODO.md
    """
    from scipy.stats import spearmanr
    set_publication_style()

    if data is None:
        data = load_data(file_path)
        if data is None: return None

    median_row, snap = _resolve_snapshot(data, analysis_t_end)
    reductions_kg = np.array(snap['reductions'])
    G = snap['graph']
    nodes = list(G.nodes())
    N = len(nodes)

    DIRECT_REDUCTION_KG = 664
    multipliers = reductions_kg / DIRECT_REDUCTION_KG

    # Extract agent properties
    degrees = np.array([G.degree(n) for n in nodes])
    thetas = np.array([G.nodes[n].get('theta', 0) for n in nodes])
    initial_diets = np.array([G.nodes[n].get('diet', 'meat') for n in nodes])

    # Only agents with meaningful positive reductions (filter FP noise)
    pos_mask = reductions_kg > 1e-3
    mult_pos = multipliers[pos_mask]
    deg_pos = degrees[pos_mask]
    theta_pos = thetas[pos_mask]

    fig, (ax_deg, ax_theta) = plt.subplots(2, 1, figsize=(8.9*cm, 14*cm), sharex=False)

    # --- Panel A: Multiplier vs Degree ---
    sc1 = ax_deg.scatter(deg_pos, mult_pos, s=8, alpha=0.5, c=COLORS['primary'],
                         edgecolors='none', rasterized=True)

    # Binned means for trend
    degree_bins = np.percentile(deg_pos, np.linspace(0, 100, 8))
    bin_centers, bin_means = [], []
    for lo, hi in zip(degree_bins[:-1], degree_bins[1:]):
        mask = (deg_pos >= lo) & (deg_pos < hi)
        if mask.sum() > 2:
            bin_centers.append(np.mean(deg_pos[mask]))
            bin_means.append(np.mean(mult_pos[mask]))
    if bin_centers:
        ax_deg.plot(bin_centers, bin_means, 'o-', color=COLORS['secondary'], linewidth=1.5,
                   markersize=4, zorder=5, label='Binned mean')

    rho_deg, p_deg = spearmanr(deg_pos, mult_pos)
    ax_deg.text(0.97, 0.97, f'$\\rho_s$ = {rho_deg:.2f} (p = {p_deg:.1e})',
               transform=ax_deg.transAxes, fontsize=6, va='top', ha='right',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.8))

    ax_deg.set_ylabel('Amplification factor', fontsize=8)
    ax_deg.set_xlabel('Node degree', fontsize=8)
    ax_deg.set_yscale('log')
    ax_deg.spines['top'].set_visible(False)
    ax_deg.spines['right'].set_visible(False)
    ax_deg.tick_params(axis='both', labelsize=7)
    ax_deg.legend(fontsize=6, frameon=False, loc='lower right')
    ax_deg.text(0.02, 0.97, 'A', transform=ax_deg.transAxes, fontsize=11,
               fontweight='bold', va='top')

    # --- Panel B: Multiplier vs Theta ---
    sc2 = ax_theta.scatter(theta_pos, mult_pos, s=8, alpha=0.5, c=COLORS['primary'],
                           edgecolors='none', rasterized=True)

    # Binned means
    theta_bins = np.linspace(theta_pos.min(), theta_pos.max(), 8)
    bin_centers_t, bin_means_t = [], []
    for lo, hi in zip(theta_bins[:-1], theta_bins[1:]):
        mask = (theta_pos >= lo) & (theta_pos < hi)
        if mask.sum() > 2:
            bin_centers_t.append(np.mean(theta_pos[mask]))
            bin_means_t.append(np.mean(mult_pos[mask]))
    if bin_centers_t:
        ax_theta.plot(bin_centers_t, bin_means_t, 'o-', color=COLORS['secondary'], linewidth=1.5,
                     markersize=4, zorder=5, label='Binned mean')

    rho_theta, p_theta = spearmanr(theta_pos, mult_pos)
    ax_theta.text(0.97, 0.97, f'$\\rho_s$ = {rho_theta:.2f} (p = {p_theta:.1e})',
                 transform=ax_theta.transAxes, fontsize=6, va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.8))

    ax_theta.set_ylabel('Amplification factor', fontsize=8)
    ax_theta.set_xlabel('Intrinsic preference ($\\theta$)', fontsize=8)
    ax_theta.set_yscale('log')
    ax_theta.spines['top'].set_visible(False)
    ax_theta.spines['right'].set_visible(False)
    ax_theta.tick_params(axis='both', labelsize=7)
    ax_theta.legend(fontsize=6, frameon=False, loc='lower right')
    ax_theta.text(0.02, 0.97, 'B', transform=ax_theta.transAxes, fontsize=11,
                 fontweight='bold', va='top')

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/agency_predictors.pdf', dpi=300, bbox_inches='tight')
        print("Saved agency_predictors.pdf")

    return fig

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
    print("=== Main Publication Plots ===")

    while True:
        print("\n[1] Network Agency Evolution (6-panel)")
        print("[2] Amplification Factor (standalone)")
        print("[3] Agency Predictors (degree/theta scatter)")
        print("[0] Exit")

        choice = input("Select: ")

        if choice == '1':
            file_path = select_file('trajectory_analysis')
            if file_path:
                data = load_data(file_path)
                if data is not None and 'is_median_twin' in data.columns:
                    trunc_input = input("Truncate to N timesteps (Enter for full run): ")
                    truncate_steps = int(trunc_input) if trunc_input else None
                    mid_input = input("Middle snapshot target t (Enter for trajectory//2): ")
                    mid_t = int(mid_input) if mid_input else None
                    at_input = input("Analysis t_end for B/C panels (Enter for same as trajectory): ")
                    analysis_t_end = int(at_input) if at_input else None
                    plot_network_agency_evolution(file_path=file_path, log_scale="loglog",
                                                 truncate_steps=truncate_steps, mid_t=mid_t,
                                                 analysis_t_end=analysis_t_end)
                else:
                    print("No twin mode snapshots found. Run trajectory analysis with twin mode first.")
        elif choice == '2':
            file_path = select_file('trajectory_analysis')
            if file_path:
                at_input = input("Analysis t_end (Enter for auto-steady): ")
                analysis_t_end = int(at_input) if at_input else None
                plot_amplification(file_path=file_path, analysis_t_end=analysis_t_end)
        elif choice == '3':
            file_path = select_file('trajectory_analysis')
            if file_path:
                at_input = input("Analysis t_end (Enter for auto-steady): ")
                analysis_t_end = int(at_input) if at_input else None
                plot_agency_predictors(file_path=file_path, analysis_t_end=analysis_t_end)
        elif choice == '0':
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()
