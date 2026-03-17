#!/usr/bin/env python3
"""Compare network layout algorithms for the agency evolution plot.

Tests multiple layout strategies on the actual model network (giant component)
to find the best visual representation of community structure and homophily.

Current production layout: spring_layout (Fruchterman-Reingold), k=4/N^0.5, iter=80

Literature context:
  - Fruchterman & Reingold (1991): classic force-directed, good general purpose
  - Kamada & Kawai (1989): stress minimization on graph distances, better global structure
  - Noack (2009) LinLog: r-PolyLog energy model, stronger community separation than FR
  - Jacomy et al. (2014) ForceAtlas2: Gephi default, gravity+repulsion tuned for communities
    (not in networkx; we approximate via spring_layout weight manipulation)
  - Spectral layout (Koren 2005): eigenvector-based, reveals algebraic community structure
  - UMAP/t-SNE on shortest-path distances (McInnes et al. 2018): nonlinear embedding,
    excellent community separation for medium networks
  - Community-weighted layouts: detect communities first, place centroids, then refine
    with FR within each cluster (Nocaj et al. 2015; Hu 2005 multilevel approach)

For homophilic networks with binary attributes (~300-2000 nodes):
  - Community-aware layouts generally outperform vanilla FR
  - Spectral layout reveals algebraic structure but can look "blobby"
  - UMAP on shortest-path distances gives strong visual clustering
  - Kamada-Kawai preserves global distances better than FR
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(__file__))
from plot_styles import set_publication_style, COLORS

cm = 1/2.54

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_network(pkl_path=None):
    """Load network from trajectory analysis pickle, return giant component + metadata."""
    if pkl_path is None:
        import glob
        # Prefer sample-max (smaller, faster) for layout testing
        files = sorted(glob.glob('../model_output/trajectory_analysis_sample-max_*.pkl'),
                       key=os.path.getmtime, reverse=True)
        if not files:
            files = sorted(glob.glob('../model_output/trajectory_analysis_*.pkl'),
                           key=os.path.getmtime, reverse=True)
        if not files:
            print("ERROR: No trajectory_analysis files found"); return None, None, None
        pkl_path = files[0]
        print(f"Using: {os.path.basename(pkl_path)}")

    data = pd.read_pickle(pkl_path)
    if 'is_median_twin' in data.columns and data['is_median_twin'].any():
        row = data[data['is_median_twin']].iloc[0]
    else:
        row = data.iloc[0]

    snap = row['snapshots'].get('steady', row['snapshots'].get('final',
           row['snapshots'][max(t for t in row['snapshots'] if isinstance(t, int))]))

    G_full = snap['graph']
    giant_nodes = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(giant_nodes).copy()
    diets = snap['diets']
    reductions = np.array(snap['reductions'])
    print(f"Network: N={G.number_of_nodes()}, E={G.number_of_edges()}, "
          f"avg_deg={2*G.number_of_edges()/G.number_of_nodes():.1f}")
    return G, diets, reductions


def draw_network(ax, G, pos, diets, reductions, title, giant_list=None):
    """Draw network on axis with diet coloring and top reducer highlights."""
    if giant_list is None:
        giant_list = list(G.nodes())

    COL_TOP10 = '#6a994e'
    COL_TOP1  = '#d4a029'

    node_colors = ['#2a9d8f' if diets[n] == 'veg' else '#e76f51' for n in giant_list]
    reds = np.array([reductions[n] for n in giant_list])

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=3,
                          alpha=0.9, edgecolors='#333', linewidths=0.1)

    if np.max(reds) > 0:
        n_top = max(1, int(0.1 * len(reds)))
        top_idx = np.argsort(reds)[-n_top:]
        top10_nodes = [giant_list[j] for j in top_idx[:-1] if reds[j] > 0]
        if top10_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=top10_nodes, ax=ax,
                                  node_color=COL_TOP10, node_size=7, alpha=0.9,
                                  edgecolors='#333', linewidths=0.15)
        top1 = top_idx[-1]
        if reds[top1] > 0:
            nx.draw_networkx_nodes(G, pos, nodelist=[giant_list[top1]], ax=ax,
                                  node_color=COL_TOP1, node_size=10, alpha=1.0,
                                  edgecolors='#333', linewidths=0.2)

    ax.set_title(title, fontsize=8, pad=3)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')


# ── Layout algorithms ────────────────────────────────────────────────────────

def layout_spring_current(G):
    """Current production layout: FR with k=4/sqrt(N)."""
    N = G.number_of_nodes()
    return nx.spring_layout(G, k=4/N**0.5, iterations=80, seed=42)

def layout_spring_tight(G):
    """FR with smaller k for tighter clustering."""
    N = G.number_of_nodes()
    return nx.spring_layout(G, k=2/N**0.5, iterations=120, seed=42)

def layout_kamada_kawai(G):
    """Kamada-Kawai: stress minimization on graph-theoretic distances.
    Better global structure preservation than FR (Kamada & Kawai 1989)."""
    return nx.kamada_kawai_layout(G)

def layout_spectral(G):
    """Spectral layout: eigenvectors of Laplacian (Koren 2005).
    Algebraic community structure; often reveals modular organization."""
    return nx.spectral_layout(G)

def layout_spectral_refined(G):
    """Spectral initial positions refined with FR (multilevel approach, Hu 2005).
    Combines spectral's global structure with FR's local aesthetics."""
    pos0 = nx.spectral_layout(G)
    N = G.number_of_nodes()
    return nx.spring_layout(G, pos=pos0, k=3/N**0.5, iterations=60, seed=42)

def layout_community_weighted(G):
    """Community-aware FR: increase edge weight within communities.
    Approximates Noack (2009) LinLog community separation.
    Detects communities with greedy modularity, then boosts intra-community attraction."""
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    node_comm = {}
    for ci, comm in enumerate(communities):
        for n in comm:
            node_comm[n] = ci
    print(f"  Community-weighted: {len(communities)} communities detected "
          f"(sizes: {sorted([len(c) for c in communities], reverse=True)[:5]}...)")

    G_w = G.copy()
    for u, v in G_w.edges():
        G_w[u][v]['weight'] = 3.0 if node_comm[u] == node_comm[v] else 0.5

    N = G.number_of_nodes()
    return nx.spring_layout(G_w, k=4/N**0.5, iterations=100, seed=42, weight='weight')

def layout_community_init(G):
    """Community-seeded: place community centroids on a circle, then FR refine.
    Inspired by Nocaj et al. (2015) multi-level graph drawing."""
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    node_comm = {}
    for ci, comm in enumerate(communities):
        for n in comm:
            node_comm[n] = ci

    n_comm = len(communities)
    # Place community centroids on a circle
    angles = np.linspace(0, 2*np.pi, n_comm, endpoint=False)
    centroids = {ci: (np.cos(a), np.sin(a)) for ci, a in enumerate(angles)}

    # Initial pos: jitter around community centroid
    rng = np.random.RandomState(42)
    pos0 = {}
    for n in G.nodes():
        cx, cy = centroids[node_comm[n]]
        pos0[n] = (cx + rng.normal(0, 0.15), cy + rng.normal(0, 0.15))

    N = G.number_of_nodes()
    return nx.spring_layout(G, pos=pos0, k=3/N**0.5, iterations=80, seed=42)

def layout_forceatlas2(G):
    """ForceAtlas2 (Jacomy et al. 2014, PLOS ONE): degree-dependent repulsion.
    The gold standard for social network community visualization (Gephi default).
    Vectorized NumPy implementation (fa2/fa2l packages broken on networkx 3.x).
    Key: repulsion ~ deg(i)*deg(j)/dist, so hubs push communities apart."""
    nodes = list(G.nodes())
    N = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    deg = np.array([G.degree(n) + 1 for n in nodes], dtype=float)

    rng = np.random.RandomState(42)
    pos = rng.uniform(-1, 1, (N, 2))

    edges_u = np.array([node_idx[u] for u, v in G.edges()])
    edges_v = np.array([node_idx[v] for u, v in G.edges()])
    gravity, kr, ka = 1.0, 1.0, 0.1
    prev_force = np.zeros((N, 2))

    for it in range(200):
        force = np.zeros((N, 2))

        # Repulsion: F_rep = kr * deg(i) * deg(j) / dist (vectorized)
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, 1.0)
        rep = kr * (deg[:, None] * deg[None, :]) / dist
        force[:, 0] = np.sum(dx / dist * rep, axis=1)
        force[:, 1] = np.sum(dy / dist * rep, axis=1)

        # Attraction: F_attr = ka * dist (linear, along edges)
        diff = pos[edges_v] - pos[edges_u]
        d_e = np.linalg.norm(diff, axis=1, keepdims=True)
        f_attr = ka * diff  # linear attraction
        np.add.at(force, edges_u, f_attr)
        np.add.at(force, edges_v, -f_attr)

        # Gravity toward center
        d_center = np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 0.01)
        force -= gravity * deg[:, None] * pos / d_center

        # Adaptive speed (Jacomy et al. swing/traction heuristic)
        swing = np.linalg.norm(force - prev_force, axis=1)
        traction = np.linalg.norm(force + prev_force, axis=1) / 2
        global_swing = np.sum((deg + 1) * swing)
        global_traction = np.sum((deg + 1) * traction)
        jt = 1.0
        global_speed = jt * global_traction / max(global_swing, 1e-6)
        node_speed = global_speed / (1 + global_speed * swing)

        pos += force * node_speed[:, None]
        prev_force = force.copy()

    return {nodes[i]: pos[i] for i in range(N)}

def layout_diet_supergraph(G):
    """Diet-aware supergraph layout (NX cluster layout pattern).
    Places veg and meat groups at separate centroids, then runs FR within each.
    Guarantees spatial separation of dietary groups -- ideal for homophily visualization.
    See: networkx.org/documentation/stable/auto_examples/drawing/plot_clusters.html"""
    # Partition by diet attribute
    veg_nodes = [n for n in G.nodes() if G.nodes[n].get('diet', 'meat') == 'veg']
    meat_nodes = [n for n in G.nodes() if n not in veg_nodes]

    if not veg_nodes or not meat_nodes:
        # Fallback: if diet attr missing, use greedy modularity
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        veg_nodes, meat_nodes = list(comms[0]), [n for c in comms[1:] for n in c]

    print(f"  Diet supergraph: {len(veg_nodes)} veg, {len(meat_nodes)} meat")

    # Position group centers
    centers = {0: np.array([-1.0, 0.0]), 1: np.array([1.0, 0.0])}

    pos = {}
    for group_nodes, center in [(veg_nodes, centers[0]), (meat_nodes, centers[1])]:
        sub = G.subgraph(group_nodes)
        N_sub = len(group_nodes)
        if N_sub > 1:
            sub_pos = nx.spring_layout(sub, k=2/N_sub**0.5, iterations=80, seed=42,
                                       center=center, scale=0.7)
        else:
            sub_pos = {group_nodes[0]: center}
        pos.update(sub_pos)

    return pos

def layout_shell_degree(G):
    """Shell layout: concentric rings by degree quantile.
    High-degree hubs at center, periphery nodes outside."""
    degrees = dict(G.degree())
    nodes = list(G.nodes())
    deg_arr = np.array([degrees[n] for n in nodes])
    q33, q66 = np.percentile(deg_arr, [33, 66])
    shells = [
        [n for n in nodes if degrees[n] > q66],   # inner: high degree
        [n for n in nodes if q33 < degrees[n] <= q66],  # middle
        [n for n in nodes if degrees[n] <= q33],   # outer: low degree
    ]
    shells = [s for s in shells if s]  # remove empty
    return nx.shell_layout(G, nlist=shells)

def layout_umap_distance(G):
    """UMAP on shortest-path distance matrix (McInnes et al. 2018).
    Nonlinear dimensionality reduction; excellent community separation.
    Falls back to t-SNE if UMAP unavailable, or MDS if neither."""
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Compute shortest-path distance matrix (dense, avoids scipy sparse dtype issues)
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(np.float64)
    dist = shortest_path(A, directed=False, unweighted=True)
    dist[dist == np.inf] = dist[dist != np.inf].max() + 1

    # Try UMAP -> t-SNE -> MDS
    try:
        from umap import UMAP
        emb = UMAP(n_components=2, metric='precomputed', random_state=42,
                    n_neighbors=15, min_dist=0.1).fit_transform(dist)
        method = 'UMAP'
    except ImportError:
        try:
            from sklearn.manifold import TSNE
            emb = TSNE(n_components=2, metric='precomputed', random_state=42,
                       perplexity=min(30, len(nodes)//4)).fit_transform(dist)
            method = 't-SNE'
        except Exception:
            from sklearn.manifold import MDS
            emb = MDS(n_components=2, dissimilarity='precomputed',
                      random_state=42, n_init=4).fit_transform(dist)
            method = 'MDS'

    print(f"  Distance embedding: using {method}")
    return {nodes[i]: emb[i] for i in range(len(nodes))}

def layout_mds_distance(G):
    """Classical MDS on shortest-path distances.
    Linear embedding; preserves global distance structure (Kruskal 1964)."""
    from sklearn.manifold import MDS
    nodes = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(np.float64)
    dist = shortest_path(A, directed=False, unweighted=True)
    dist[dist == np.inf] = dist[dist != np.inf].max() + 1
    emb = MDS(n_components=2, dissimilarity='precomputed',
              random_state=42, n_init=4, normalized_stress='auto').fit_transform(dist)
    return {nodes[i]: emb[i] for i in range(len(nodes))}


# ── Main comparison ──────────────────────────────────────────────────────────

LAYOUTS = {
    'Spring (current)':       layout_spring_current,
    'Kamada-Kawai':           layout_kamada_kawai,
    'Spectral + FR refine':   layout_spectral_refined,
    'Community-weighted FR':  layout_community_weighted,
    'Community-seeded FR':    layout_community_init,
    'ForceAtlas2':            layout_forceatlas2,
    'Shell (degree rings)':   layout_shell_degree,
    'UMAP/t-SNE/MDS on dist': layout_umap_distance,
    'MDS on dist':            layout_mds_distance,
}


def main():
    set_publication_style()
    G, diets, reductions = load_network()
    if G is None:
        return
    giant_list = list(G.nodes())

    n_layouts = len(LAYOUTS)
    ncols = 4
    nrows = (n_layouts + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(24*cm, nrows*6*cm))
    axes = axes.flatten()

    for i, (name, func) in enumerate(LAYOUTS.items()):
        print(f"Computing: {name}...")
        try:
            pos = func(G)
            draw_network(axes[i], G, pos, diets, reductions, name, giant_list)
        except Exception as e:
            axes[i].text(0.5, 0.5, f'FAILED\n{e}', transform=axes[i].transAxes,
                        ha='center', va='center', fontsize=7, color='red')
            axes[i].set_title(name, fontsize=8, pad=3)
            axes[i].axis('off')
            print(f"  FAILED: {e}")

    # Hide unused axes
    for j in range(n_layouts, len(axes)):
        axes[j].axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#2a9d8f', edgecolor='#333', lw=0.4, label='Vegetarian'),
        Patch(facecolor='#e76f51', edgecolor='#333', lw=0.4, label='Meat eater'),
        Patch(facecolor='#6a994e', edgecolor='#333', lw=0.4, label='Top 10% reducers'),
        Patch(facecolor='#d4a029', edgecolor='#333', lw=0.4, label='Top reducer'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=4, fontsize=7, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = '../visualisations_output'
    os.makedirs(output_dir, exist_ok=True)
    out_path = f'{output_dir}/layout_comparison.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
