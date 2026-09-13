"""Is the diet-group subgraph clustering anything but a group-size effect?

Observed induced-subgraph clustering for each diet group at t0 and t_end, against a
null of same-size random node subsets of the same graph. 10 runs, 20 draws.
"""
import glob, pickle, sys
import numpy as np, networkx as nx

DIR = 'model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced'
T_END, NDRAW = 310000, 20
rng = np.random.default_rng(0)

def graph(s):
    G = nx.Graph(); G.add_nodes_from(range(s['n_nodes'])); G.add_edges_from(map(tuple, s['edges'])); return G

def sub_cl(G, idx):
    return nx.average_clustering(G.subgraph(list(idx)))

rows = []
for p in sorted(glob.glob(f'{DIR}/*.pkl'))[:10]:
    d = pickle.load(open(p, 'rb'))
    S = d['snapshots']; s0, s1 = S[0], S[T_END]
    G0, G1 = graph(s0), graph(s1)
    veg0 = np.array(d['initial_diets']) == 'veg'
    veg1 = s1['diets'].astype(bool)
    for tag, G, veg in (('t0', G0, veg0), ('tend', G1, veg1)):
        n = G.number_of_nodes()
        for grp, m in (('veg', veg), ('omni', ~veg)):
            idx = np.flatnonzero(m)
            obs = sub_cl(G, idx)
            null = [sub_cl(G, rng.choice(n, len(idx), replace=False)) for _ in range(NDRAW)]
            rows.append((tag, grp, len(idx) / n, obs, float(np.mean(null)), float(np.std(null))))

import pandas as pd
df = pd.DataFrame(rows, columns=['t', 'group', 'share', 'obs', 'null', 'null_sd'])
print(df.groupby(['t', 'group']).mean().round(3))
