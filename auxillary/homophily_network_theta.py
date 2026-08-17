"""
Theta-homophily network generation (EXPERIMENTAL, post-submission).

Parameterized variant of `homophily_network_v2.generate_homophily_network_v2`
that can produce a network ASSORTATIVE ON THETA (intrinsic dietary preference),
the trait that actually drives the dynamics. Built as a separate module so the
in-review production path (`homophily_network_v2` / topology="homophilic_emp")
is left byte-for-byte untouched.

Knobs added over v2:
  - sim_power : sharpen the similarity kernel (sim ** sim_power). Near-identical
                agents become strongly preferred. Primary lever for theta-assort.
  - pa_power  : exponent on the preferential-attachment (degree) term.
                pa_power=1.0 reproduces v2 exactly (homophily x PA); 0.0 removes
                PA. Kept ON by default (=1.0) because degree heterogeneity is
                load-bearing for the amplification DV, and v2 added PA
                deliberately to fix core-periphery collapse (network_homophily.md
                section 3). We tune sim_power / theta_w to hit the assortativity
                target rather than removing PA.
  - tc_sim    : if True, triadic-closure edges ALSO respect similarity (counts x
                sim**sim_power) instead of encounter-count only.

Calibration target: Newman numeric theta-assortativity ~0.2-0.25 (empirical Moran's
I 0.17-0.39, Groningen 2025 / Nezlek 2020), subject to keeping degree std near
the production value and clustering ~0.30 with a single connected component.
NB: the FROZEN production config (theta_w=1.0, sim_power=4.0, pa_power=1.0, tc_sim=True
in model_main) actually measures ~0.27 (5-seed avg 0.283; visual-script seed 42 = 0.266),
i.e. slightly above the 0.2-0.25 target but inside the empirical range. The self-test
below uses theta_w=3.0 (~0.36) as a STRONGER illustration, not the production setting.

See claude_stuff/Infrastructure/homophily_network_audit_2026-05-29.md sections 6-7.
"""
import os
import sys
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from auxillary.homophily_network_v2 import (
    normalize_attributes_v2, compute_similarity_matrix_v2)


def _homophily_target(source, target_set, sim_matrix, G, sim_power, pa_power):
    """Select target weighted by sim**sim_power * (degree+eps)**pa_power."""
    targets = np.array(list(target_set))
    if len(targets) == 0:
        return None
    EPSILON = 1e-5
    weights = sim_matrix[source, targets] ** sim_power
    if pa_power > 0:
        degrees = np.array([G.degree(t) + EPSILON for t in targets])
        weights = weights * degrees ** pa_power
    s = weights.sum()
    if s <= 0:
        return int(np.random.choice(targets))
    return int(np.random.choice(targets, p=weights / s))


def _tc_target(source, special_targets, sim_matrix=None, sim_power=1.0):
    """Select FOF target weighted by encounter count, optionally x sim**sim_power."""
    if not special_targets:
        return None
    targets = np.array(list(special_targets.keys()))
    counts = np.array([special_targets[t] for t in targets], dtype=float)
    if sim_matrix is not None:
        counts = counts * sim_matrix[source, targets] ** sim_power
    s = counts.sum()
    if s <= 0:
        return int(np.random.choice(targets))
    return int(np.random.choice(targets, p=counts / s))


def generate_homophily_network_theta(
    N: int,
    avg_degree: int,
    agents_df,
    attribute_weights: Optional[np.ndarray] = None,
    sim_power: float = 1.0,
    pa_power: float = 1.0,
    tc_sim: bool = False,
    seed: Optional[int] = None,
    tc: float = 0.7,
    target_clustering: Optional[float] = None,
) -> Tuple[nx.Graph, np.ndarray]:
    """Holme-Kim growth + triadic closure with tunable homophily/PA.

    Identical growth structure to v2; only the per-edge selection weights are
    parameterized. With sim_power=1.0, pa_power=1.0, tc_sim=False this reproduces
    `generate_homophily_network_v2` exactly.

    Parameters
    ----------
    attribute_weights : 5-vector [gender, age, income, edu, theta]. Raise the
        theta entry (e.g. [0.20,0.35,0.18,0.32, 3.0]) to dial up theta-homophily.
    sim_power : similarity kernel exponent (>=1 sharpens preference).
    pa_power  : degree (PA) exponent. 1.0 = v2 behaviour, 0.0 = no PA.
    tc_sim    : let triadic-closure edges respect similarity.

    Returns (G, sim_matrix). sim_matrix is the RAW (un-powered) similarity, so it
    stays drop-in compatible with model_main.rewire().
    """
    if seed is not None:
        np.random.seed(seed)
    if len(agents_df) != N:
        raise ValueError(f"agents_df has {len(agents_df)} rows but N={N}")

    m = max(1, avg_degree // 2)
    attr_matrix, _ = normalize_attributes_v2(agents_df)
    sim_matrix = compute_similarity_matrix_v2(attr_matrix, attribute_weights)
    tc_sim_arg = sim_matrix if tc_sim else None

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for col in ('gender', 'age_group', 'incquart', 'educlevel', 'theta', 'diet'):
            G.nodes[i][col] = agents_df.iloc[i][col]

    m0 = min(m + 1, N)
    for i in range(m0):
        for j in range(i + 1, m0):
            G.add_edge(i, j)

    for source in range(m0, N):
        target_set = {t for t in range(source) if not G.has_edge(source, t)}
        special_targets = defaultdict(int)

        for idx in range(min(m, len(target_set))):
            target = None
            if idx > 0 and special_targets and np.random.random() < tc:
                valid_st = {t: c for t, c in special_targets.items() if t in target_set}
                if valid_st:
                    target = _tc_target(source, valid_st, tc_sim_arg, sim_power)
            if target is None:
                target = _homophily_target(source, target_set, sim_matrix, G,
                                           sim_power, pa_power)
            if target is None:
                break

            G.add_edge(source, target)
            target_set.discard(target)
            if target in special_targets:
                del special_targets[target]
            if idx < m - 1:
                for neighbor in G.neighbors(target):
                    if neighbor != source and not G.has_edge(source, neighbor):
                        special_targets[neighbor] += 1

    if target_clustering is not None:
        achieved = nx.average_clustering(G)
        if abs(achieved - target_clustering) > 0.05:
            print(f"WARNING: clustering {achieved:.3f} differs from target "
                  f"{target_clustering:.3f} by {abs(achieved - target_clustering):.3f}")

    return G, sim_matrix


if __name__ == "__main__":
    # Quick self-test against v2 equivalence + a theta-assortative config.
    import pandas as pd
    from auxillary.homophily_network_v2 import generate_homophily_network_v2

    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  '../data/hierarchical_agents.csv'))
    df = df.sample(n=385, random_state=42).reset_index(drop=True)
    prod_w = np.array([0.20, 0.35, 0.18, 0.32, 0.05])

    def _assort(G):
        nx.set_node_attributes(G, {i: float(df['theta'].iloc[i]) for i in G.nodes()}, 'theta')
        return nx.numeric_assortativity_coefficient(G, 'theta')

    G_v2, _ = generate_homophily_network_v2(385, 8, df, prod_w, seed=1, tc=0.7)
    G_eq, _ = generate_homophily_network_theta(385, 8, df, prod_w, sim_power=1.0,
                                               pa_power=1.0, tc_sim=False, seed=1, tc=0.7)
    print(f"v2  edges={G_v2.number_of_edges()} CC={nx.average_clustering(G_v2):.3f} "
          f"theta_assort={_assort(G_v2):+.3f}")
    print(f"new(=v2) edges={G_eq.number_of_edges()} CC={nx.average_clustering(G_eq):.3f} "
          f"theta_assort={_assort(G_eq):+.3f}  (should match v2)")

    # NB: theta_w=3.0 here is a STRONGER demo (~0.36); production uses theta_w=1.0 (~0.27).
    theta_w = np.array([0.20, 0.35, 0.18, 0.32, 3.0])
    G_t, _ = generate_homophily_network_theta(385, 8, df, theta_w, sim_power=4.0,
                                              pa_power=1.0, tc_sim=True, seed=1, tc=0.7)
    degs = [d for _, d in G_t.degree()]
    print(f"theta-homophily: CC={nx.average_clustering(G_t):.3f} "
          f"theta_assort={_assort(G_t):+.3f} deg_std={np.std(degs):.1f} "
          f"max_deg={max(degs)} comps={nx.number_connected_components(G_t)}")
