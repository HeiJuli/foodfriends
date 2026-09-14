"""Same-diet neighbour share per diet group through the run, against the degree-decile null.

Diet assortativity r (topology_local_effects.py, measure 1) is a single coefficient for the
whole partition, so it cannot say which group clusters. Here each group's mean same-diet
neighbour share is measured against the same degree-decile permutation null (diet labels
permuted within degree deciles, which holds F and any degree-diet link fixed).

Diets are replayed from the event log onto the same time grid as measure 1, on the network
snapshot nearest below each time (the reducer kept edges every 10k from 280k and at 0, 100k
and 200k before that; rewiring moves about 15% of edges by t_end).

Usage (repo root): python analysis/topology_group_assortativity.py [n_runs] [workers]
Writes topology_group_assortativity.csv beside the run pkls.
"""
import sys
import glob
import bisect
import numpy as np
import pandas as pd
from multiprocessing import Pool

DIR = 'model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced'
N_PERM = 50
GRID = [0, 1000, 5000] + list(range(10000, 400001, 10000))


def degree(e, n):
    return np.bincount(e[:, 0], None, n) + np.bincount(e[:, 1], None, n)


def shares(x, e, n, k, ok):
    """Mean same-diet neighbour share, (vegetarians, omnivores)."""
    v = x.astype(float)
    nv = np.bincount(e[:, 0], v[e[:, 1]], n) + np.bincount(e[:, 1], v[e[:, 0]], n)
    s = nv[ok] / k[ok]
    xo = x[ok]
    return s[xo].mean(), (1 - s[~xo]).mean()


def net(S, key, n, cache):
    """Edges, degree, non-isolated mask and degree-decile index sets of one network snapshot."""
    if key not in cache:
        e = np.asarray(S[key]['edges'])
        k = degree(e, n)
        bins = np.searchsorted(np.quantile(k, np.linspace(0, 1, 11)[1:-1]), k, side='right')
        cache[key] = (e, k, k > 0, [np.flatnonzero(bins == b) for b in np.unique(bins)])
    return cache[key]


def one_run(path):
    d = pd.read_pickle(path)
    S, run = d['snapshots'], d['run']
    n = S[0]['n_nodes']
    rng = np.random.default_rng(1)
    gts = sorted(t for t in S if isinstance(t, int) and 'edges' in S[t])
    cache, states, gi = {}, {}, 0
    veg = np.array(d['initial_diets']) == 'veg'
    for ev in d['events']:
        while gi < len(GRID) and GRID[gi] <= ev[1]:
            states[GRID[gi]] = veg.copy(); gi += 1
        veg[ev[2]] = ev[0] == 'conv'
    for t in GRID[gi:]:
        states[t] = veg.copy()

    rows = []
    for t in GRID:
        key = gts[bisect.bisect_right(gts, t) - 1]
        e, k, ok, ii = net(S, key, n, cache)
        x = states[t]
        perm = []
        for _ in range(N_PERM):
            xp = x.copy()
            for i in ii:
                xp[i] = x[rng.permutation(i)]
            perm.append(shares(xp, e, n, k, ok))
        p = np.array(perm)
        obs = shares(x, e, n, k, ok)
        rows.append(dict(run=run, t=t, gkey=key, F=x.mean(), s_veg=obs[0], s_omni=obs[1],
                         s_veg_null=p[:, 0].mean(), s_omni_null=p[:, 1].mean(),
                         s_veg_sd=p[:, 0].std(), s_omni_sd=p[:, 1].std()))
    return rows


if __name__ == '__main__':
    paths = sorted(glob.glob(f'{DIR}/*.pkl'))[:int(sys.argv[1]) if len(sys.argv) > 1 else None]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 7) as pool:
        rows = [r for rs in pool.map(one_run, paths) for r in rs]
    df = pd.DataFrame(rows)
    out = f'{DIR}/topology_group_assortativity.csv'
    df.to_csv(out, index=False)

    df['ex_veg'] = df.s_veg - df.s_veg_null
    df['ex_omni'] = df.s_omni - df.s_omni_null
    g = df.groupby('t')
    print(f'{"t":>8}{"F":>8}{"veg obs":>10}{"veg null":>10}{"excess":>10}'
          f'{"omni obs":>10}{"omni null":>10}{"excess":>10}')
    for t in (0, 10000, 50000, 100000, 200000, 310000, 400000):
        if t in g.groups:
            q = g.get_group(t).median(numeric_only=True)
            print(f'{t:>8}{q.F:>8.3f}{q.s_veg:>10.3f}{q.s_veg_null:>10.3f}{q.ex_veg:>10.3f}'
                  f'{q.s_omni:>10.3f}{q.s_omni_null:>10.3f}{q.ex_omni:>10.3f}')
    pk = df.loc[df.groupby('run').ex_veg.idxmax()]
    print(f'INFO: peak vegetarian excess (median over runs) {pk.ex_veg.median():.3f} '
          f'at F {pk.F.median():.2f}; omnivore excess at the same rows {pk.ex_omni.median():.3f}')
    print(f'INFO: wrote {out} ({len(df)} rows, {df.run.nunique()} runs)')
