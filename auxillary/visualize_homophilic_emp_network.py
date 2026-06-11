"""
Network comparison: θ-homophilic vs. configuration model (N=385, sample-max).

Two output modes:
  default        three-panel: two networks + adoption trajectory
  --no-traj      two-panel:   networks only (standalone)

Nodes sized by degree, coloured by mean neighbour similarity.

Usage:
    python visualize_homophilic_emp_network.py [--seed 42] [--traj-pkl PATH] [--no-traj]
"""

import sys
sys.path.append('../model_src')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import networkx as nx
import pickle
import argparse
from pathlib import Path
from scipy import stats
from model_main import Model

TRAJ_PKL_DEFAULT = "../model_output/topology_comparison_smax_20260529.pkl"
COLORS = {"homophilic_theta": "#9b2226", "CM": "#adb5bd"}


def mean_neighbor_similarity(G, sim_matrix):
    scores = np.zeros(G.number_of_nodes())
    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if nbrs:
            scores[node] = np.mean(sim_matrix[node, nbrs])
    return scores


def theta_assortativity(G, agents):
    theta = {a.i: a.theta for a in agents}
    nx.set_node_attributes(G, theta, 'theta')
    return nx.numeric_assortativity_coefficient(G, 'theta')


def run_stats(sim_hom, sim_er, G_hom, G_er, agents):
    u_stat, p_val = stats.mannwhitneyu(sim_hom, sim_er, alternative='greater')
    r_hom = theta_assortativity(G_hom, agents)
    r_er  = theta_assortativity(G_er,  agents)
    print("\n--- Homophily comparison ---")
    print(f"Mean sim  hom={sim_hom.mean():.4f}  ER={sim_er.mean():.4f}")
    print(f"Std  sim  hom={sim_hom.std():.4f}  ER={sim_er.std():.4f}")
    print(f"Mann-Whitney U={u_stat:.0f}, p={p_val:.2e} (one-sided: hom > CM)")
    print(f"Theta assortativity  hom={r_hom:.4f}  CM={r_er:.4f}")
    print("----------------------------\n")
    return {"u": u_stat, "p": p_val, "r_hom": r_hom, "r_er": r_er,
            "mean_hom": sim_hom.mean(), "mean_er": sim_er.mean()}


def _degree_sizes(G, mean_deg, base=18, exponent=0.7):
    """Node sizes scaled by degree relative to mean; hubs pop without dominating."""
    return [max(6, base * (d / mean_deg) ** exponent) for _, d in G.degree()]


def draw_network_panel(G, pos, sim_scores, ax, title, vmin, vmax,
                       node_sizes, stats_line=None):
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.35, edge_color='#999999')
    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=[sim_scores[n] for n in G.nodes()],
        node_size=node_sizes,
        cmap=cm.plasma,
        vmin=vmin, vmax=vmax,
        alpha=0.93,
        linewidths=0.45,
        edgecolors='white',
    )
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    if stats_line:
        ax.text(0.5, -0.03, stats_line, transform=ax.transAxes,
                ha='center', va='top', fontsize=8.5, color='#333333')
    ax.axis('off')
    return sc


def draw_trajectory_panel(ax, traj_pkl):
    """Plot theta-homophily vs ER adoption trajectories from topology comparison pkl."""
    try:
        with open(traj_pkl, "rb") as f:
            results = pickle.load(f)
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"pkl not found:\n{traj_pkl}",
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        ax.axis('off')
        return

    labels = {"homophilic_theta": r"$\theta$-homophilic", "CM": "Configuration model"}
    for topo in ("homophilic_theta", "CM"):
        rows = [r for r in results if r['topology'] == topo]
        if not rows:
            continue
        min_len = min(len(r['fraction_veg_trajectory']) for r in rows)
        mat = np.array([r['fraction_veg_trajectory'][:min_len] for r in rows])
        t = np.arange(mat.shape[1])
        mu = mat.mean(0)
        lo, hi = np.percentile(mat, 5, 0), np.percentile(mat, 95, 0)
        ax.plot(t, mu, color=COLORS[topo], lw=2.0, label=labels[topo])
        ax.fill_between(t, lo, hi, color=COLORS[topo], alpha=0.18)

    ax.set_xlabel("Timestep", fontsize=9)
    ax.set_ylabel("Vegetarian fraction", fontsize=9)
    ax.set_ylim(-0.03, 1.06)
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, ls='--', alpha=0.25, zorder=-10)
    ax.set_title("(c) Adoption trajectory\n(mean ± 90% CI, 30 runs)", fontsize=11,
                 fontweight='bold', pad=8)
    ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--traj-pkl', default=TRAJ_PKL_DEFAULT)
    parser.add_argument('--no-traj', action='store_true',
                        help='Produce two-panel networks-only figure')
    args = parser.parse_args()

    with open("../data/demographic_pmfs.pkl", "rb") as f:
        pmf_tables = pickle.load(f)

    params = {
        "N": 385,
        "steps": 100,
        "k": 8,
        "erdos_p": 0.001,
        "p_rewire": 0.1,
        "rewire_h": 0.1,
        "tc": 0.3,
        "topology": "homophilic_theta",
        "theta_w": 1.0,
        "sim_power": 4.0,
        "pa_power": 1.0,
        "tc_sim": True,
        "alpha": 0.36,
        "rho": 0.45,
        "theta": 0.44,
        "agent_ini": "sample-max",
        "survey_file": "../data/hierarchical_agents.csv",
        "adjust_veg_fraction": False,
        "target_veg_fraction": 0.06,
        "beta": 20,
        "immune_n": 0.0,
        "M": 7,
        "veg_f": 0.1,
        "meat_f": 0.9,
        "seed": args.seed,
        "veg_CO2": 1.5,
        "meat_CO2": 7.2,
    }

    print("Generating homophilic network (sample-max mode)...")
    model = Model(params, pmf_tables=pmf_tables)
    model.agent_ini()
    G_hom = model.G1
    sim_matrix = model.sim_matrix
    agents = model.agents
    N = len(agents)

    mean_deg = np.mean([d for _, d in G_hom.degree()])
    degree_seq = [d for _, d in G_hom.degree()]
    print(f"Generating configuration model (N={N}, same degree sequence)...")
    G_cm = nx.configuration_model(degree_seq, seed=args.seed)
    G_cm = nx.Graph(G_cm)
    G_cm.remove_edges_from(nx.selfloop_edges(G_cm))

    sim_hom = mean_neighbor_similarity(G_hom, sim_matrix)
    sim_er  = mean_neighbor_similarity(G_cm,  sim_matrix)

    st = run_stats(sim_hom, sim_er, G_hom, G_cm, agents)

    vmin = min(sim_hom.min(), sim_er.min())
    vmax = max(sim_hom.max(), sim_er.max())

    print("Computing layouts...")
    pos_hom = nx.kamada_kawai_layout(G_hom)
    pos_cm  = nx.kamada_kawai_layout(G_cm)

    cm_deg = np.mean([d for _, d in G_cm.degree()])
    sizes_hom = _degree_sizes(G_hom, mean_deg)
    sizes_er  = _degree_sizes(G_cm, cm_deg)

    hom_stats = (f"$\\bar{{\\sigma}}$ = {st['mean_hom']:.3f},  "
                 f"$r_\\theta$ = {st['r_hom']:.3f}")
    er_stats  = (f"$\\bar{{\\sigma}}$ = {st['mean_er']:.3f},  "
                 f"$r_\\theta$ = {st['r_er']:.3f}")

    if args.no_traj:
        fig = plt.figure(figsize=(13, 6), facecolor='white')
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 0.55], wspace=0.04)
        ax_hom = fig.add_subplot(gs[0])
        ax_er  = fig.add_subplot(gs[1])

        sc = draw_network_panel(G_hom, pos_hom, sim_hom, ax_hom,
                                f"(a) $\\theta$-homophilic network\n$N={N}$,  $\\langle k \\rangle={mean_deg:.1f}$",
                                vmin, vmax, sizes_hom, stats_line=hom_stats)
        draw_network_panel(G_cm, pos_cm, sim_er, ax_er,
                           f"(b) Configuration model (matched degree sequence)\n$N={N}$,  $\\langle k \\rangle={cm_deg:.1f}$",
                           vmin, vmax, sizes_er, stats_line=er_stats)

        cbar = fig.colorbar(sc, ax=[ax_hom, ax_er], fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=8.5)
        cbar.set_label('Mean similarity to neighbours  $\\bar{\\sigma}_i$', fontsize=10)

        output_dir = Path("../visualisations_output/homophily")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"network_comparison_N{N}_networks.pdf"
    else:
        fig = plt.figure(figsize=(19, 6), facecolor='white')
        gs = gridspec.GridSpec(1, 4, width_ratios=[2, 2, 0.55, 1.7], wspace=0.04)
        ax_hom  = fig.add_subplot(gs[0])
        ax_er   = fig.add_subplot(gs[1])
        ax_traj = fig.add_subplot(gs[3])

        sc = draw_network_panel(G_hom, pos_hom, sim_hom, ax_hom,
                                f"(a) $\\theta$-homophilic network\n$N={N}$,  $\\langle k \\rangle={mean_deg:.1f}$",
                                vmin, vmax, sizes_hom, stats_line=hom_stats)
        draw_network_panel(G_cm, pos_cm, sim_er, ax_er,
                           f"(b) Configuration model (matched degree sequence)\n$N={N}$,  $\\langle k \\rangle={cm_deg:.1f}$",
                           vmin, vmax, sizes_er, stats_line=er_stats)
        draw_trajectory_panel(ax_traj, args.traj_pkl)

        cbar = fig.colorbar(sc, ax=[ax_hom, ax_er], fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=8.5)
        cbar.set_label('Mean similarity to neighbours  $\\bar{\\sigma}_i$', fontsize=10)

        output_dir = Path("../visualisations_output/homophily")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"network_comparison_N{N}.pdf"

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
