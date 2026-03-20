#!/usr/bin/env python3
"""Main publication plots"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import yaml
from matplotlib.colors import LinearSegmentedColormap
from plot_styles import set_publication_style, apply_axis_style, COLORS, ECO_CMAP, ECO_DIV_CMAP

cm = 1/2.54
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'plot_config.yaml')
MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'model_output')

def load_config():
    """Load plot config from YAML. Returns dict or None."""
    if not os.path.exists(CONFIG_PATH):
        print(f"WARNING: No config at {CONFIG_PATH}")
        return None
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    print(f"INFO: Loaded config from {CONFIG_PATH}")
    return cfg

def _cfg_path(filename):
    """Resolve a pkl filename to full path under model_output/."""
    if filename is None:
        return None
    if os.path.isabs(filename):
        return filename
    return os.path.join(MODEL_OUTPUT, filename)

def _find_crossing(traj, ref_fveg):
    """Find first timestep where trajectory crosses ref_fveg. Returns int or None."""
    arr = np.array(traj)
    idx = np.where(arr >= ref_fveg)[0]
    return int(idx[0]) if len(idx) > 0 else None

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

def _get_median(df):
    if 'is_median_twin' in df.columns and df['is_median_twin'].any():
        return df[df['is_median_twin']].iloc[0]
    twin = df[df['agent_ini'].isin(['twin', 'sample-max'])] if 'agent_ini' in df.columns else df
    return (twin if len(twin) else df).iloc[len(df) // 2]

def _resolve_analysis(snapshots, row, t_end_override=None):
    """Return (snapshot_dict, final_t_int) for analysis panels."""
    if t_end_override is not None:
        int_ts = sorted(t for t in snapshots if isinstance(t, int) and t > 0)
        best = min(int_ts, key=lambda t: abs(t - t_end_override)) if int_ts else None
        if best is not None:
            return snapshots[best], best
        return snapshots.get('final', snapshots[max(t for t in snapshots if isinstance(t, int))]), None
    if 'steady' in snapshots:
        ss_t = row.get('steady_state_t')
        return snapshots['steady'], int(ss_t) if ss_t is not None else None
    # fallback to last integer key
    last = max((t for t in snapshots if isinstance(t, int)), default=None)
    return snapshots.get('final', snapshots.get(last)), last

def _truncate_snaps(snapshots, row, max_t):
    """Truncate snapshot dict to times <= max_t, remapping 'final'."""
    valid = sorted([t for t in snapshots if isinstance(t, int) and t <= max_t])
    last_t = valid[-1] if valid else 0
    ss_t = row.get('steady_state_t')
    keep_steady = 'steady' in snapshots and ss_t is not None and ss_t <= max_t
    out = {t: snapshots[t] for t in ([0] + valid) if t in snapshots}
    if keep_steady:
        out['steady'] = snapshots['steady']
    out['final'] = snapshots[last_t]
    return out


def plot_network_agency_evolution(data=None, file_path=None,
                                  small_data=None, small_file_path=None,
                                  save=True, log_scale=None,
                                  truncate_steps=None, analysis_t_end=None,
                                  small_truncate_steps=None, small_mid_t=None,
                                  small_analysis_t_end=None,
                                  rescale_ref=0.5, savgol_window=10001):
    """7-panel: 3 network snapshots (top) + trajectory + CCDF + Lorenz (bottom).

    Dual-mode: big N provides primary trajectory, CCDF, Lorenz.
               small N provides network visualisations and grey trajectory overlay.
    If small_file_path not provided, falls back to single-source mode.

    Args:
        data/file_path:              Big N dataset (primary)
        small_data/small_file_path:  Small N dataset (illustrative networks)
        truncate_steps:              Big N display window (x-axis max)
        analysis_t_end:              Big N: snapshot for CCDF/Lorenz (Enter=auto-steady)
        small_truncate_steps:        Small N display window
        small_mid_t:                 Small N: middle network snapshot target
        small_analysis_t_end:        Small N: which snapshot is t_end for networks
        rescale_ref:                 F_veg crossing to align small N onto big N time axis
                                     (None to disable rescaling)
        savgol_window:               Savitzky-Golay window for tipping point detection
    """
    from matplotlib.ticker import LogLocator, NullFormatter
    from matplotlib.patches import Patch
    from scipy.signal import savgol_filter
    set_publication_style()

    COL_TOP10, COL_TOP1 = '#6a994e', '#d4a029'

    # === Load big N (primary) ===
    if data is None:
        data = load_data(file_path)
        if data is None: return None
    big_row = _get_median(data)
    big_traj = list(big_row['fraction_veg_trajectory'])
    big_snaps = big_row['snapshots']
    if truncate_steps:
        big_traj = big_traj[:truncate_steps]
    big_snap, big_final_t = _resolve_analysis(big_snaps, big_row, analysis_t_end)
    print(f"INFO: Big N: F_veg {big_traj[0]:.3f} -> {big_traj[-1]:.3f} ({len(big_traj)} steps)")

    # === Load small N (optional) ===
    dual = small_file_path is not None or small_data is not None
    if dual:
        if small_data is None:
            small_data = load_data(small_file_path)
        if small_data is None:
            dual = False
    if dual:
        sm_row = _get_median(small_data)
        sm_traj = list(sm_row['fraction_veg_trajectory'])
        sm_snaps = dict(sm_row['snapshots'])
        if small_truncate_steps:
            sm_traj = sm_traj[:small_truncate_steps]
            sm_snaps = _truncate_snaps(sm_snaps, sm_row, small_truncate_steps)
        net_snap_final, sm_final_t = _resolve_analysis(sm_snaps, sm_row, small_analysis_t_end)
        sm_analysis = dict(sm_snaps)
        sm_analysis['final'] = net_snap_final
        if sm_final_t is None:
            sm_final_t = len(sm_traj) - 1
        print(f"INFO: Small N: F_veg {sm_traj[0]:.3f} -> {sm_traj[-1]:.3f} ({len(sm_traj)} steps)")
    else:
        # Fallback: single-source mode (big N for everything)
        sm_snaps = dict(big_snaps)
        sm_traj = big_traj
        if truncate_steps:
            sm_snaps = _truncate_snaps(sm_snaps, big_row, truncate_steps)
        net_snap_final, sm_final_t = _resolve_analysis(sm_snaps, big_row, analysis_t_end)
        sm_analysis = dict(sm_snaps)
        sm_analysis['final'] = net_snap_final
        if sm_final_t is None:
            sm_final_t = len(sm_traj) - 1

    # === Compute time rescaling for small N overlay ===
    scale_factor = 1.0
    if dual and rescale_ref is not None:
        t_cross_big = _find_crossing(big_traj, rescale_ref)
        t_cross_sm = _find_crossing(sm_traj, rescale_ref)
        if t_cross_big is not None and t_cross_sm is not None and t_cross_sm > 0:
            scale_factor = t_cross_big / t_cross_sm
            print(f"INFO: Rescaling small N: F_veg={rescale_ref} at t_big={t_cross_big}, "
                  f"t_small={t_cross_sm}, scale={scale_factor:.2f}")
        else:
            print(f"INFO: Rescaling skipped (crossing F_veg={rescale_ref} not found in both trajectories)")

    # === Figure layout ===
    fig = plt.figure(figsize=(17.8*cm, 11.5*cm))
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[2.8, 1.4],
                                hspace=0.08, top=0.93, bottom=0.08, left=0.08, right=0.97)
    gs_top = outer_gs[0].subgridspec(1, 3, wspace=0.08)
    gs_bot = outer_gs[1].subgridspec(1, 3, wspace=0.35, width_ratios=[1.2, 1, 1])

    # === Network layout (from small N) ===
    G_full = sm_analysis['final']['graph']
    giant_nodes = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(giant_nodes).copy()
    giant_list = list(G.nodes())
    N_gc = G.number_of_nodes()
    pos = nx.spring_layout(G, k=4/N_gc**0.5, iterations=80, seed=42)
    pos_arr = np.array(list(pos.values()))
    x_min, x_max = pos_arr[:, 0].min(), pos_arr[:, 0].max()
    y_min, y_max_net = pos_arr[:, 1].min(), pos_arr[:, 1].max()
    print(f"INFO: Network N={G_full.number_of_nodes()}, giant={N_gc}, edges={G.number_of_edges()}")

    # Time points for network snapshots (from small N)
    all_times = sorted([t for t in sm_snaps if isinstance(t, int) and t > 0])
    if all_times:
        _mid_target = small_mid_t if small_mid_t is not None else (
            len(sm_traj) // 2 if isinstance(sm_traj, list) else all_times[-1] // 2)
        mid = min(all_times, key=lambda t: abs(t - _mid_target))
    else:
        mid = None
    time_points = [t for t in [0, mid, 'final'] if t is not None]
    print(f"INFO: Network snapshots at: {time_points}")

    # === Legend ===
    net_legend = [
        Patch(facecolor='#2a9d8f', edgecolor='#333', linewidth=0.4, label='Vegetarian'),
        Patch(facecolor='#e76f51', edgecolor='#333', linewidth=0.4, label='Meat eater'),
        Patch(facecolor=COL_TOP10, edgecolor='#333', linewidth=0.4, label='Top 10% reducers'),
        Patch(facecolor=COL_TOP1, edgecolor='#333', linewidth=0.4, label='Top reducer'),
    ]
    fig.legend(handles=net_legend, loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=4, fontsize=6.5, frameon=False, handletextpad=0.4, columnspacing=1.0)

    # === Row 0: Network snapshots (from small N) ===
    for i, t in enumerate(time_points):
        snap = sm_analysis.get(t, sm_snaps.get(t))
        if snap is None:
            continue
        net_ax = fig.add_subplot(gs_top[0, i])
        all_diets = snap['diets']
        all_red = np.array(snap['reductions'])
        node_colors = ['#2a9d8f' if all_diets[n] == 'veg' else '#e76f51' for n in giant_list]
        reductions = np.array([all_red[n] for n in giant_list])

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

    # === Panel A: Trajectory (big N primary + small N grey overlay) ===
    traj_ax = fig.add_subplot(gs_bot[0, 0])

    # Small N grey line (rescaled, drawn first so big N is on top)
    if dual:
        t_k_sm = np.arange(len(sm_traj)) * scale_factor / 1000
        traj_ax.plot(t_k_sm, sm_traj, color='#999', linewidth=0.6, alpha=0.6)

    # Big N primary line
    t_k_big = np.arange(len(big_traj)) / 1000
    traj_ax.plot(t_k_big, big_traj, color=COLORS['vegetation'], linewidth=1.0, alpha=0.9)

    # Vertical markers at small N snapshot times (rescaled, intersect the grey line)
    ref_traj = sm_traj if dual else big_traj
    marker_color = '#999' if dual else COLORS['vegetation']
    for t in time_points:
        t_val = 0 if t == 0 else (min(sm_final_t, len(ref_traj) - 1) if t == 'final'
                                   else min(t, len(ref_traj) - 1))
        t_k = t_val * scale_factor / 1000  # rescaled to big N time axis
        traj_ax.axvline(t_k, color='#888', linestyle=':', linewidth=0.7, alpha=0.6)
        traj_ax.scatter(t_k, ref_traj[t_val], color=marker_color,
                       s=14, zorder=5, edgecolors='#333', linewidths=0.4)

    # Tipping point from big N
    _sw = savgol_window
    traj_arr = np.array(big_traj)
    _burnin = 5000
    if len(traj_arr) > _burnin + _sw * 2:
        COL_TIP = '#9b59b6'
        _d2 = savgol_filter(traj_arr, window_length=_sw, polyorder=3, deriv=2)
        _sm = savgol_filter(traj_arr, window_length=_sw, polyorder=3)
        _d2[:_burnin] = 0
        _d2[_sm > 0.5] = 0
        _t_tip = int(np.argmax(_d2))
        _t_tip_k = _t_tip / 1000
        traj_ax.axvline(_t_tip_k, color=COL_TIP, linestyle=':', linewidth=0.9, alpha=0.8)
        print(f"INFO: Tipping point at t={_t_tip} ({_t_tip_k:.1f}k), F_veg={traj_arr[_t_tip]:.3f}")

    traj_ax.set_ylim(0, 1.0)
    traj_ax.set_xlim(-0.5, len(big_traj) / 1000)
    traj_ax.set_ylabel('$F_{veg}$', fontsize=7)
    traj_ax.set_xlabel('$t$ [thousands]', fontsize=7)
    traj_ax.spines['top'].set_visible(False)
    traj_ax.spines['right'].set_visible(False)
    traj_ax.tick_params(axis='both', labelsize=6)
    traj_ax.text(0.02, 0.95, 'A', transform=traj_ax.transAxes, fontsize=10,
                fontweight='bold', va='top')

    # === Panel B: CCDF (big N steady state, single curve) ===
    ccdf_ax = fig.add_subplot(gs_bot[0, 1])
    reductions_all = np.array(big_snap['reductions'])
    if np.max(reductions_all) > 0:
        red_t = np.sort(reductions_all[reductions_all > 1e-6] / 1000)
        ccdf_y = 1.0 - np.arange(1, len(red_t) + 1) / len(red_t)
        ccdf_ax.step(red_t, ccdf_y, where='post', color='#555', linewidth=1.2, alpha=0.9)

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
    ccdf_ax.text(0.02, 0.95, 'B', transform=ccdf_ax.transAxes, fontsize=10,
                fontweight='bold', va='top')

    # === Panel C: Lorenz (big N steady state, single curve) ===
    lorenz_ax = fig.add_subplot(gs_bot[0, 2])
    if np.max(reductions_all) > 0:
        sorted_red = np.sort(reductions_all)[::-1]
        cum_frac = np.cumsum(sorted_red) / sorted_red.sum()
        pop_frac = np.arange(1, len(sorted_red) + 1) / len(sorted_red)
        lorenz_ax.plot(pop_frac, cum_frac, color='#555', lw=1.2, alpha=0.9)
    lorenz_ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    lorenz_ax.set_xlabel('$F_{pop}$', fontsize=7)
    lorenz_ax.set_ylabel('$F_{red}$', fontsize=7)
    lorenz_ax.set_xlim(0, 1)
    lorenz_ax.set_ylim(0, 1)
    lorenz_ax.set_aspect('equal', adjustable='datalim')
    lorenz_ax.spines['top'].set_visible(False)
    lorenz_ax.spines['right'].set_visible(False)
    lorenz_ax.tick_params(axis='both', labelsize=6)
    lorenz_ax.text(0.02, 0.95, 'C', transform=lorenz_ax.transAxes, fontsize=10,
                   fontweight='bold', va='top')

    if save:
        output_dir = ensure_output_dir()
        plt.savefig(f'{output_dir}/network_agency_evolution.pdf', dpi=300, bbox_inches='tight')
        print("Saved network_agency_evolution.pdf")

    return fig

def _resolve_snapshot(data, analysis_t_end=None):
    """Resolve the analysis snapshot: explicit t_end > steady > final."""
    median_row = _get_median(data)
    snapshots = median_row['snapshots']
    snap, final_t = _resolve_analysis(snapshots, median_row, analysis_t_end)
    if final_t is not None:
        print(f"INFO: Using snapshot t={final_t}")
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

def _run_from_config(cfg, plot_key):
    """Run a plot using config dict entries."""
    if plot_key == 'network_agency_evolution':
        c = cfg[plot_key]
        big = c.get('big_n', {})
        sm = c.get('small_n', {})
        rsc = c.get('rescale', {})
        plot_network_agency_evolution(
            file_path=_cfg_path(big.get('file')),
            small_file_path=_cfg_path(sm.get('file')),
            truncate_steps=big.get('truncate_steps'),
            analysis_t_end=big.get('analysis_t_end'),
            small_truncate_steps=sm.get('truncate_steps'),
            small_mid_t=sm.get('mid_t'),
            small_analysis_t_end=sm.get('analysis_t_end'),
            rescale_ref=rsc.get('reference_fveg', 0.5),
            savgol_window=c.get('savgol_window', 10001))
    elif plot_key == 'amplification':
        c = cfg[plot_key]
        plot_amplification(file_path=_cfg_path(c.get('file')),
                          analysis_t_end=c.get('analysis_t_end'))
    elif plot_key == 'agency_predictors':
        c = cfg[plot_key]
        plot_agency_predictors(file_path=_cfg_path(c.get('file')),
                              analysis_t_end=c.get('analysis_t_end'))

def main():
    print("=== Main Publication Plots ===")

    while True:
        print("\n[1] Network Agency Evolution (6-panel)")
        print("[2] Amplification Factor (standalone)")
        print("[3] Agency Predictors (degree/theta scatter)")
        print("[c] Load from config (plot_config.yaml)")
        print("[0] Exit")

        choice = input("Select: ")

        if choice == 'c':
            cfg = load_config()
            if cfg is None:
                continue
            print("\nAvailable in config:")
            keys = list(cfg.keys())
            for i, k in enumerate(keys):
                print(f"  [{i+1}] {k}")
            print(f"  [a] All")
            sel = input("Select: ").strip()
            if sel == 'a':
                for k in keys:
                    print(f"\n--- {k} ---")
                    _run_from_config(cfg, k)
            else:
                try:
                    _run_from_config(cfg, keys[int(sel) - 1])
                except (ValueError, IndexError):
                    print("Invalid selection")

        elif choice == '1':
            print("\n--- Big N (primary: trajectory, CCDF, Lorenz) ---")
            big_file = select_file('trajectory_analysis')
            if not big_file:
                continue

            use_small = input("Add small-N illustrative network? (y/N): ").strip().lower()
            small_file = None
            if use_small == 'y':
                print("\n--- Small N (illustrative: network panels, grey trajectory) ---")
                small_file = select_file('trajectory_analysis')

            trunc = input("Big N: truncate steps (Enter=full): ").strip()
            truncate_steps = int(trunc) if trunc else None
            at = input("Big N: analysis t_end for CCDF/Lorenz (Enter=auto-steady): ").strip()
            analysis_t_end = int(at) if at else None

            small_truncate_steps = small_mid_t = small_analysis_t_end = None
            if small_file:
                st = input("Small N: truncate steps (Enter=full): ").strip()
                small_truncate_steps = int(st) if st else None
                sm = input("Small N: mid snapshot target t (Enter=auto): ").strip()
                small_mid_t = int(sm) if sm else None
                sa = input("Small N: analysis t_end for networks (Enter=auto-steady): ").strip()
                small_analysis_t_end = int(sa) if sa else None

            rr = input("Rescale ref F_veg (Enter=0.5, 'none'=disable): ").strip()
            rescale_ref = None if rr.lower() == 'none' else (float(rr) if rr else 0.5)

            plot_network_agency_evolution(
                file_path=big_file, small_file_path=small_file,
                truncate_steps=truncate_steps, analysis_t_end=analysis_t_end,
                small_truncate_steps=small_truncate_steps,
                small_mid_t=small_mid_t, small_analysis_t_end=small_analysis_t_end,
                rescale_ref=rescale_ref)

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
