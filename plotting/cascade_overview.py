"""Illustrative overview: one run's cascade structure, one arm of one tree highlighted.

The event graph is the one the ledger walks (analysis/attribution_ledger.py): nodes are
conversion events plus the initial vegetarians at t = 0; a child is linked to each
exposure-proportional source's stint-start event if that stint had begun when the child
sampled it. Reducing it to the main-cause spanning forest (each event keeps its
largest-share source) gives trees. The root is the initial vegetarian with the largest
tree; the arm is one of its direct children and that child's subtree.

--layout tree (default): Didelot et al. 2017 (MBE 34:997) Fig. 5 style. x = conversion
  time, y = depth-first order of the tree, so no link jumps; elbow connectors run along
  the parent's row (--links rounded softens the corner, --links straight is Didelot's
  original diagonal). Grey is the rest of the root's tree (--scope all: the whole forest).
  Arm dots are filled if the stint lasts to t_end, open if the agent later reverts.
--layout agents: agent x sweep lattice with every event-graph link in grey; rows are the
  arm's agents as a depth-first block, the rest in spectral order.
Row position carries no meaning beyond adjacency in either layout.

Usage: python cascade_overview.py <run.pkl> [--layout L] [--root I] [--arm R] [--out PATH]
  run.pkl: a trajectory ensemble DataFrame (row --run) or a dict with events/initial_diets/params
"""
import argparse
import os
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from attribution_ledger import _exposure_parents

GREY_LINK, GREY_NODE, INK = "#bdbdbd", "#8c8c8c", "#5b2a86"


def load(path, run):
    d = pd.read_pickle(path)
    if isinstance(d, pd.DataFrame):
        d = d.iloc[run]
    return d["events"], list(d["initial_diets"]), d["params"]


def event_graph(events, initial_diets, params, t_end):
    """nodes: node -> (agent, t); links: (source node, child node, share)."""
    g = params.get("gamma", 0.3)
    nodes = {("init", i): (i, 0) for i, d in enumerate(initial_diets) if d == "veg"}
    stint = {n[1]: n for n in nodes}                # agent -> node that opened its stint
    links = []
    for k, ev in enumerate(events):
        t = ev[1]
        if t > t_end:
            break
        if ev[0] == "rev":
            stint.pop(ev[2], None)
            continue
        _, _, j, _, _, buf = ev
        for src, share, ts in _exposure_parents(buf, j, g, t):
            n = stint.get(src)
            if n is not None and nodes[n][1] <= ts:
                links.append((n, k, share))
        stint[j], nodes[k] = k, (j, t)
    return nodes, links


def main_cause_tree(links):
    best = {}
    for s, c, sh in links:
        if sh > best.get(c, (None, 0))[1]:
            best[c] = (s, sh)
    kids = defaultdict(list)
    for c, (s, _) in best.items():
        kids[s].append(c)
    return kids


def subtree(kids, n):
    out, st = [], [n]
    while st:
        m = st.pop(); out.append(m); st += kids.get(m, [])
    return out


def tree_rows(kids, roots, nodes):
    """Didelot et al. 2017 Fig. 5 layout: depth-first preorder, children in time order,
    one row per event. Consecutive rows along a chain, so no link jumps."""
    y, i = {}, 0
    for r in roots:
        st = [r]
        while st:
            n = st.pop(); y[n] = i; i += 1
            st += sorted(kids.get(n, []), key=lambda c: nodes[c][1], reverse=True)
    return y


def reverted(events, initial_diets, t_end):
    """Event nodes whose stint ended before t_end."""
    open_ = {i: ("init", i) for i, d in enumerate(initial_diets) if d == "veg"}
    out = set()
    for k, ev in enumerate(events):
        if ev[1] > t_end:
            break
        if ev[0] == "conv":
            open_[ev[2]] = k
        elif ev[2] in open_:
            out.add(open_.pop(ev[2]))
    return out


def curves(p, q, bow=0.12, n=14):
    """Quadratic Beziers p->q (arrays (L, 2), display units), bowed perpendicular."""
    d = q - p
    ctrl = (p + q) / 2 + bow * np.sign(d[:, 1:2] + 1e-9) * np.c_[-d[:, 1], d[:, 0]]
    s = np.linspace(0, 1, n)[None, :, None]
    return (1 - s) ** 2 * p[:, None] + 2 * s * (1 - s) * ctrl[:, None] + s ** 2 * q[:, None]


def rounded_elbow(p, q, rad, n=9):
    """Elbows p->q with the corner at (q_x, p_y) rounded by a quadratic Bezier of radius
    rad (display units), clamped to half the shorter leg so short links stay sharp combs
    and only long drops visibly soften."""
    c = np.c_[q[:, 0], p[:, 1]]
    legs = [p - c, q - c]
    lens = [np.hypot(e[:, 0], e[:, 1]) for e in legs]
    d = np.minimum(rad, 0.5 * np.minimum(*lens))
    b = [c + np.divide(d, l, out=np.zeros_like(d), where=l > 0)[:, None] * e
         for e, l in zip(legs, lens)]
    s = np.linspace(0, 1, n)[None, :, None]
    bez = (1 - s) ** 2 * b[0][:, None] + 2 * s * (1 - s) * c[:, None] + s ** 2 * b[1][:, None]
    return np.concatenate([p[:, None], bez, q[:, None]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--root", type=int, default=None, help="default: largest main-cause tree")
    ap.add_argument("--arm", type=int, default=0, help="rank of the root's child subtrees by size")
    ap.add_argument("--layout", choices=["tree", "agents"], default="tree",
                    help="tree: main-cause forest, Didelot Fig. 5 style; agents: agent x sweep lattice")
    ap.add_argument("--scope", choices=["root", "all"], default="root",
                    help="tree layout: the root's tree only, or the whole forest")
    ap.add_argument("--links", choices=["elbow", "rounded", "straight"], default="elbow",
                    help="tree layout: elbow connectors, rounded elbows, or straight lines "
                         "(Didelot Fig. 5); straight fades grey links by density so hub "
                         "fans read as gradients")
    ap.add_argument("--t-end", type=int, default=None)
    ap.add_argument("--size", type=float, nargs=2, default=(7.2, 2.4), help="inches")
    ap.add_argument("--out", default="../visualisations_output/cascade_overview.pdf")
    a = ap.parse_args()

    events, diets, params = load(a.pkl, a.run)
    N, sweep = len(diets), 2 * len(diets)
    t_end = a.t_end if a.t_end is not None else params["steps"]
    nodes, links = event_graph(events, diets, params, t_end)
    kids = main_cause_tree(links)
    inits = [n for n in nodes if isinstance(n, tuple)]      # conversion nodes are ints
    if not inits:
        sys.exit("no initial vegetarians in this run; nothing to draw")
    if a.root is not None:
        root = ("init", a.root)
        if root not in nodes:
            sys.exit(f"--root {a.root}: agent {a.root} is not an initial vegetarian "
                     f"(valid: {sorted(n[1] for n in inits)})")
    else:
        root = max(inits, key=lambda n: len(subtree(kids, n)))
    arms = sorted(kids.get(root, []), key=lambda c: len(subtree(kids, c)), reverse=True)
    if not arms:
        sys.exit(f"root agent {root[1]}'s tree has no arms (no conversions); pick another --root")
    if not 0 <= a.arm < len(arms):
        sys.exit(f"--arm {a.arm} out of range: root agent {root[1]} has {len(arms)} arms "
                 f"(0..{len(arms) - 1})")
    arm = [root] + subtree(kids, arms[a.arm])
    armset = set(arm)
    print(f"{len(nodes)} event nodes, {len(links)} links; root agent {root[1]}: "
          f"{len(subtree(kids, root)) - 1} events in its tree, {len(arms)} arms; "
          f"arm {a.arm}: {len(arm) - 1} events, {len({nodes[n][0] for n in arm})} agents")

    W, H = a.size
    kid_of = {c: s for s, cs in kids.items() for c in cs}
    if a.layout == "agents":
        # rows: the arm's agents as one block in depth-first tree order (parent and child
        # rows adjacent), the rest by spectral ordering of the share-weighted influence
        # graph, split either side of the block. At N=2000 this takes arm-link spans from
        # a median 270 rows (RCM) to 2, and all-link spans 415 -> 225.
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for s, c, w in links:
            u, v = nodes[s][0], nodes[c][0]
            if u != v:
                G.add_edge(u, v, weight=G.get_edge_data(u, v, {"weight": 0})["weight"] + w)
        block = list(dict.fromkeys(nodes[n][0] for n in arm))
        rest = [i for i in nx.spectral_ordering(G, weight="weight", seed=0) if i not in set(block)]
        h = len(rest) // 2
        row = np.empty(N)
        row[rest[:h] + block + rest[h:]] = np.arange(N)
        ncol = t_end // sweep + 1
        xy = {n: np.array([(t // sweep) / ncol * W, row[i] / N * H]) for n, (i, t) in nodes.items()}
        grey = [(s, c) for s, c, _ in links]
        draw = curves
        glw, galpha = 0.08, min(0.55, 12000 / max(1, len(grey)))
    else:
        # the event graph reduced to its main-cause spanning forest: every event keeps one
        # parent, so the layout is a set of trees and no link jumps
        if a.scope == "root":
            roots = [root]
        else:
            others = [n for n in nodes if n not in kid_of and n != root]
            roots = [root] + sorted(others, key=lambda n: (nodes[n][1], -len(subtree(kids, n))))
        y = tree_rows(kids, roots, nodes)
        xy = {n: np.array([nodes[n][1] / t_end * W, y[n] / len(y) * H]) for n in y}
        grey = [(kid_of[c], c) for c in y if c in kid_of]
        # elbow: along the parent's row to the child's time, then up to the child. A hub's
        # many children become a comb on one line instead of a fan (the root has ~300).
        if a.links == "elbow":
            draw = lambda p, q: np.stack([p, np.c_[q[:, 0], p[:, 1]], q], axis=1)
        elif a.links == "rounded":
            draw = lambda p, q: rounded_elbow(p, q, rad=0.015 * H)
        else:
            # straight: hub fans are coherent (children are contiguous rows in time
            # order), but dense; thin + fade the grey so fans read as gradients
            draw = lambda p, q: np.stack([p, q], axis=1)
        glw, galpha = (0.15, 1.0) if a.links != "straight" \
            else (0.1, min(0.8, 2500 / max(1, len(grey))))

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    if grey:
        P = np.array([[xy[s], xy[c]] for s, c in grey])
        ax.add_collection(LineCollection(draw(P[:, 0], P[:, 1]), colors=GREY_LINK,
                                         lw=glw, alpha=galpha, rasterized=True, zorder=1))
    G_xy = np.array(list(xy.values()))
    ax.scatter(*G_xy.T, s=0.25, c=GREY_NODE, lw=0, rasterized=True, zorder=2)

    if len(arm) > 1:
        Q = np.array([[xy[kid_of[c]], xy[c]] for c in arm[1:]])
        ax.add_collection(LineCollection(draw(Q[:, 0], Q[:, 1]), colors=INK, lw=0.45, zorder=3))
    gone = reverted(events, diets, t_end)
    for sel, face in ((lambda n: n not in gone, INK), (lambda n: n in gone, "white")):
        pts = np.array([xy[n] for n in arm[1:] if sel(n)]).reshape(-1, 2)
        ax.scatter(*pts.T, s=3, c=face, ec=INK, lw=0.35, zorder=4)
    ax.scatter(*xy[root], s=22, c=INK, ec="white", lw=0.8, zorder=5)
    ax.set_xlim(-0.02 * W, 1.01 * W)
    ax.set_ylim(-0.02 * H, 1.02 * H)
    ax.axis("off")
    fig.savefig(a.out, dpi=600)
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=300)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
