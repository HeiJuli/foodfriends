#!/usr/bin/env python3
"""What does psychology add, and what does the empirical network add?

A ladder of arms. Each rung turns on one ingredient, so the change in the
amplification distribution between consecutive rungs is attributable to it.

  Lc  complete    + naive       degree-free reference: no degree variance at all
  L0  ER          + naive       pure cascade accounting, random graph
  L1  empirical   + naive       + empirical topology (degree tail, clustering)
  L2  ER          + synthetic   + behaviour (Boltzmann, theta gate, memory), random graph
  L3  empirical   + synthetic   + behaviour on empirical topology
  L4  empirical   + twin        + empirical calibration (survey theta/rho/alpha)  = the paper

Deltas:
  L1 - L0   what empirical topology alone buys
  L2 - L0   what behaviour alone buys
  L3 - L1   behaviour, given empirical topology
  L4 - L3   what EMPIRICAL CALIBRATION buys over generic parametric psychology

Caveat on L3. The homophilic_emp builder needs survey agents (it wires on their
attributes), so L3 feeds the empirical graph in as `prebuilt` and puts synthetic
agents on it. That keeps the topology and breaks the attribute-topology
correlation. So "empirical network construction" splits in two here: L3 - L1 is
the topology, L4 - L3 is empirical psychology *plus* its homophilic alignment
with the network. They are not separable without a node-permutation arm.

Usage:
    python amplification_decomposition.py [--runs 3] [--steps 150000] [--jobs 6]
    python amplification_decomposition.py --arms L0,L1,L4      # subset
"""
import os, sys, argparse, pickle
from multiprocessing import Pool
import numpy as np
import pandas as pd
import networkx as nx

sys.path.append('../model_src')
from naive_counterfactuals import naive_cascade, gini, slope, BANDS, DIRECT_REDUCTION_KG

TWIN_PKL = '../model_output/trajectory_analysis_twin_20260820.pkl'
T_END = 144000
OUT = '../visualisations_output'

ARMS = {
    'Lc complete + naive':  dict(net='complete', dyn='naive'),
    'L0 ER + naive':        dict(net='ER',       dyn='naive'),
    'L1 emp + naive':       dict(net='emp',      dyn='naive'),
    'L2 ER + synthetic':    dict(net='ER',       dyn='model', agent_ini='synthetic'),
    'L3 emp + synthetic':   dict(net='emp',      dyn='model', agent_ini='synthetic'),
    'L4 emp + twin':        dict(net='emp',      dyn='model', agent_ini='twin'),
}


# ------------------------------------------------------------------- metrics
def ccdf_alpha(a):
    """Hill estimator above the median of the positive part."""
    p = np.sort(a[a > 0])
    xmin = np.median(p); t = p[p >= xmin]
    return 1 + len(t) / np.sum(np.log(t / xmin)) if len(t) > 1 else np.nan


def band_slope(a, deg):
    """Degree scaling from per-band means, zeros included (the unbiased one)."""
    kk, mu = [], []
    for lo, hi in BANDS:
        m = (deg >= lo) & (deg <= hi)
        if m.sum() >= 5:
            kk.append(deg[m].mean()); mu.append(a[m].mean())
    kk, mu = np.array(kk), np.array(mu)
    g = mu > 0
    return np.polyfit(np.log10(kk[g]), np.log10(mu[g]), 1)[0] if g.sum() > 2 else np.nan


def metrics(a, deg, adopters, f_veg):
    a = np.asarray(a, float)
    pos = a[a > 0]; srt = np.sort(a)[::-1]; tot = a.sum()
    n_top20 = max(1, int(0.2 * (a > 0).sum()))
    return dict(
        f_veg=f_veg, mean_ad=a[adopters].mean() if adopters.sum() else np.nan,
        mean_pos=pos.mean(), gini=gini(a[adopters]) if adopters.sum() else np.nan,
        top20=srt[:n_top20].sum() / tot, top1=srt[:max(1, len(a) // 100)].sum() / tot,
        p90=np.percentile(pos, 90), mx=a.max(), pct_credit=100 * (a > 0).mean(),
        alpha=ccdf_alpha(a), b_band=band_slope(a, deg), b_pos=slope(a, deg))


# ------------------------------------------------------------------ the arms
def empirical_graph():
    d = pd.read_pickle(TWIN_PKL)
    r = d[d['is_median_twin']].iloc[0]
    return r['snapshots'][T_END]['graph']


def run_naive(net, G_emp, N, f_veg, seed):
    G = {'complete': lambda: nx.complete_graph(N),
         'ER': lambda: nx.gnm_random_graph(N, int(round(G_emp.number_of_edges())), seed=seed),
         'emp': lambda: G_emp}[net]()
    a, rank, deg = naive_cascade(G, f_veg, seed=seed)
    return metrics(a, deg.astype(float), rank >= 0, f_veg)


def run_model(net, agent_ini, G_emp, N, steps, seed):
    import random, model_main, model_runn
    p = dict(model_runn.DEFAULT_PARAMS)
    p.update(N=N, steps=steps, agent_ini=agent_ini, seed=seed,
             survey_file='../data/hierarchical_agents.csv',
             topology={'ER': 'ER', 'emp': 'prebuilt'}[net])
    p['tau_persistence'] = p['M'] * 2 * N          # never let N default in
    np.random.seed(seed); random.seed(seed)
    m = model_runn.get_model(p) if agent_ini in ('twin', 'sample-max') else model_main.Model(p)
    if net == 'emp':
        m.G1 = nx.convert_node_labels_to_integers(G_emp)
    m.run()
    snaps = m.snapshots
    key = min((t for t in snaps if isinstance(t, int)), key=lambda t: abs(t - steps))
    s = snaps[key]
    a = np.array(s['reductions']) / DIRECT_REDUCTION_KG
    deg = np.array([m.G1.degree(v) for v in m.G1.nodes()], float)
    adopters = np.array([c is not None for c in s['change_times']])
    return metrics(a, deg, adopters, s['veg_fraction'])


def one(job):
    name, cfg, G_emp, N, steps, seed = job
    try:
        if cfg['dyn'] == 'naive':
            r = run_naive(cfg['net'], G_emp, N, cfg['f_veg'], seed)
        else:
            r = run_model(cfg['net'], cfg['agent_ini'], G_emp, N, steps, seed)
        return name, seed, r
    except Exception as e:                     # one arm failing must not kill the sweep
        print(f"ERROR: {name} seed={seed}: {type(e).__name__}: {e}")
        return name, seed, None


# ------------------------------------------------------------------- reporting
COLS = [('f_veg', 'F_veg', 3), ('mean_ad', 'mean/ad', 2), ('mean_pos', 'mean+', 2),
        ('gini', 'Gini', 3), ('top20', 'top20%', 3), ('top1', 'top1%', 3),
        ('p90', 'p90', 2), ('mx', 'max', 1), ('pct_credit', '%cred', 1),
        ('alpha', 'CCDF a', 2), ('b_band', 'b_band', 2), ('b_pos', 'b_pos', 2)]


def report(res):
    hdr = f"{'arm':<22}" + "".join(f"{lab:>9}" for _, lab, _ in COLS)
    print("\n" + hdr); print("-" * len(hdr))
    med = {}
    for name in ARMS:
        rows = [r for r in res.get(name, []) if r]
        if not rows:
            continue
        med[name] = {k: np.nanmedian([r[k] for r in rows]) for k, _, _ in COLS}
        print(f"{name:<22}" + "".join(f"{med[name][k]:>9.{d}f}" for k, _, d in COLS)
              + f"   (n={len(rows)})")

    print("\ndeltas (median to median, x = ratio)")
    for a, b, what in [('L0 ER + naive', 'L1 emp + naive', 'empirical topology, naive dyn'),
                       ('L0 ER + naive', 'L2 ER + synthetic', 'behaviour, random net'),
                       ('L1 emp + naive', 'L3 emp + synthetic', 'behaviour, empirical net'),
                       ('L3 emp + synthetic', 'L4 emp + twin', 'EMPIRICAL CALIBRATION')]:
        if a in med and b in med:
            print(f"  {what:<32} Gini {med[a]['gini']:.3f} -> {med[b]['gini']:.3f}"
                  f"   top1% {med[a]['top1']:.3f} -> {med[b]['top1']:.3f}"
                  f"   %cred {med[a]['pct_credit']:.0f} -> {med[b]['pct_credit']:.0f}"
                  f"   b_band {med[a]['b_band']:.2f} -> {med[b]['b_band']:.2f}")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--steps', type=int, default=150000)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--out', default='amplification_decomposition.pkl')
    args = ap.parse_args()

    G_emp = empirical_graph()
    N = G_emp.number_of_nodes()
    d = pd.read_pickle(TWIN_PKL)
    f_veg = d[d['is_median_twin']].iloc[0]['snapshots'][T_END]['veg_fraction']
    print(f"empirical net: N={N}, z={2 * G_emp.number_of_edges() / N:.2f}, "
          f"F_veg={f_veg:.3f}; {args.runs} runs/arm, {args.steps} steps")

    want = [a for a in ARMS if any(a.startswith(x.strip()) for x in args.arms.split(','))]
    jobs = [(n, {**ARMS[n], 'f_veg': f_veg}, G_emp, N, args.steps, 42 + i)
            for n in want for i in range(args.runs if ARMS[n]['dyn'] == 'model'
                                         else max(args.runs, 5))]
    print(f"arms: {', '.join(want)}  ({len(jobs)} jobs)")

    res = {}
    with Pool(min(args.jobs, len(jobs))) as pool:
        for name, seed, r in pool.imap_unordered(one, jobs):
            res.setdefault(name, []).append(r)
            print(f"  done {name} seed={seed}" + ("" if r else "  FAILED"))

    med = report(res)
    with open(args.out, 'wb') as f:
        pickle.dump({'raw': res, 'median': med}, f)
    print(f"\nSaved {args.out}")


if __name__ == '__main__':
    main()
