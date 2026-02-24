#!/usr/bin/env python3
"""Isolated test for the network agency evolution plot.

Loads the most recent trajectory_analysis pkl and renders ONLY the
4-panel network snapshot row so layout issues can be debugged quickly
without waiting for the full 6-panel figure.
"""
import sys, glob, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(__file__))
from plot_styles import set_publication_style, COLORS

cm = 1/2.54

def load_latest(pattern='trajectory_analysis'):
    files = glob.glob(f'../model_output/{pattern}_*.pkl')
    if not files:
        sys.exit(f"ERROR: no {pattern} files in ../model_output/")
    f = sorted(files, key=os.path.getmtime)[-1]
    print(f"Using: {os.path.basename(f)}")
    import pandas as pd
    return pd.read_pickle(f)

def test_network_layout(clip_pct=90, node_size=2, edge_alpha=0.3, edge_width=0.05,
                        layout='spectral', spring_k=1.5):
    """Render 4-panel network snapshot row in isolation.

    Args:
        clip_pct   : percentile for position clipping (0=no clipping, 90=default)
        node_size  : node size passed to draw_networkx_nodes
        edge_alpha : edge transparency
        edge_width : edge line width
        layout     : 'spectral' or 'spring'
    """
    set_publication_style()
    data = load_latest()

    if 'is_median_twin' not in data.columns or not data['is_median_twin'].any():
        sys.exit("ERROR: no median twin row found")

    median_row = data[data['is_median_twin']].iloc[0]
    snapshots  = median_row['snapshots']
    G_full     = snapshots['final']['graph']
    N_full     = G_full.number_of_nodes()

    # Use only the giant component for drawing
    giant_nodes = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(giant_nodes).copy()
    N = G.number_of_nodes()
    print(f"INFO: full N={N_full}, components={nx.number_connected_components(G_full)}, "
          f"min_degree={min(d for _,d in G_full.degree())}")
    print(f"INFO: giant component N={N}, edges={G.number_of_edges()}")

    # --- Layout ---
    if layout == 'spring':
        pos = nx.spring_layout(G, k=spring_k/N**0.5, iterations=80, seed=42)
    else:
        pos = nx.spectral_layout(G)

    if clip_pct < 100:
        pos_arr = np.array([pos[n] for n in G.nodes()])
        center  = pos_arr.mean(axis=0)
        dists   = np.linalg.norm(pos_arr - center, axis=1)
        clip_r  = np.percentile(dists, clip_pct)
        for n in G.nodes():
            d = np.linalg.norm(pos[n] - center)
            if d > clip_r:
                pos[n] = center + (pos[n] - center) * (clip_r / d)

    pos_array = np.array(list(pos.values()))
    x_min, x_max = pos_array[:, 0].min(), pos_array[:, 0].max()
    y_min, y_max = pos_array[:, 1].min(), pos_array[:, 1].max()

    all_times  = sorted([t for t in snapshots if isinstance(t, int) and t > 0])
    time_points = ([0] + (all_times[:2] if len(all_times) >= 2 else all_times) + ['final'])[:4]
    print(f"INFO: snapshot times = {time_points}")

    COL_TOP10, COL_TOP1 = '#6a994e', '#d4a029'

    fig, axes = plt.subplots(1, 4, figsize=(17.8*cm, 5*cm))
    fig.suptitle(f"layout={layout}, clip_pct={clip_pct}, node_size={node_size}, "
                 f"edge_alpha={edge_alpha}, edge_width={edge_width}", fontsize=6)

    # Node index mapping: full graph node id -> position in reductions array
    full_nodes = list(G_full.nodes())
    giant_list = list(G.nodes())

    for ax, t in zip(axes, time_points):
        snap         = snapshots[t]
        all_diets    = snap['diets']
        all_red      = np.array(snap['reductions'])

        # Subset to giant component
        colors     = ['#2a9d8f' if all_diets[n] == 'veg' else '#e76f51' for n in giant_list]
        reductions = np.array([all_red[n] for n in giant_list])

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha, width=edge_width)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                               node_size=node_size, alpha=0.9,
                               edgecolors='#333', linewidths=0.15)

        if np.max(reductions) > 0:
            n_top   = max(1, int(0.1 * len(reductions)))
            top_idx = np.argsort(reductions)[-n_top:]
            top10 = [giant_list[j] for j in top_idx[:-1] if reductions[j] > 0]
            if top10:
                nx.draw_networkx_nodes(G, pos, nodelist=top10, ax=ax,
                                       node_color=COL_TOP10, node_size=node_size*2.5,
                                       alpha=0.9, edgecolors='#333', linewidths=0.2)
            top1 = giant_list[top_idx[-1]]
            if reductions[top_idx[-1]] > 0:
                nx.draw_networkx_nodes(G, pos, nodelist=[top1], ax=ax,
                                       node_color=COL_TOP1, node_size=node_size*3,
                                       alpha=1.0, edgecolors='#333', linewidths=0.3)

        title = '$t_0$' if t == 0 else '$t_{end}$' if t == 'final' else f't={t//1000}k'
        ax.set_title(title, fontsize=8, pad=2)
        pad = 0.02
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

    out = f'../visualisations_output/test_network_layout_{layout}_clip{clip_pct}.pdf'
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--clip',   type=float, default=90,      help='percentile clip (0=off)')
    p.add_argument('--nsize',  type=float, default=2,       help='node size')
    p.add_argument('--ealpha', type=float, default=0.3,     help='edge alpha')
    p.add_argument('--ewidth', type=float, default=0.05,    help='edge width')
    p.add_argument('--layout',   default='spectral', help='spectral or spring')
    p.add_argument('--k',        type=float, default=1.5, help='spring k multiplier')
    args = p.parse_args()

    test_network_layout(clip_pct=args.clip, node_size=args.nsize,
                        edge_alpha=args.ealpha, edge_width=args.ewidth,
                        layout=args.layout, spring_k=args.k)
