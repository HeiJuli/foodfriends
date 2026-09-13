"""Does network position predict WHO adopts, once churn and the equilibrium snapshot go?

The two-DV null ("structure does not predict adoption") is scored on the agent's diet at
t = 310,000, which is equilibrium with heavy churn, so a snapshot diet is a noisy label.
This re-tests it with churn-robust outcomes (ever-converted, own vegetarian-time share,
conversion count, first-conversion step) and at pre-equilibrium windows t = 100k (F_veg
~0.21) and 200k (~0.51) as well as 310k. It also re-runs the manuscript :127 claim that
high-betweenness nodes "maintain their vegetarian state for longer", and the diet-group
subgraph clustering null (group_clustering_null.py) at the two earlier windows.

Structural features come from the 200,000-step snapshot graph for every window: the
network is quasi-static (p_rewire 0.01), so one graph serves all three.

Correlations are Spearman throughout; on a binary outcome that is the rank-biserial
coefficient, monotone-equivalent to the point-biserial on ranks. Partials residualise the
rank of the feature and of the outcome on the rank of degree.

Writes topology_window_robustness.csv beside the run pickles; prints median and the
fraction of runs with p < 0.05 per statistic.
"""
import pickle, sys
from itertools import product
from multiprocessing import Pool

import numpy as np, networkx as nx, pandas as pd
from scipy.stats import spearmanr, rankdata, t as tdist

sys.path.insert(0, "analysis")
sys.path.insert(0, "plotting")
from attribution_ledger import veg_time, conv_counts
from agency_predictor_analysis import compute_complex_centrality

DIR = "model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced"
G_SNAP = 200000                      # structural features come from here
WINDOWS = (100000, 200000, 310000)
FEATURES = ("degree", "betweenness", "clustering", "cc")
NDRAW, NPROC = 20, 3


def graph(s):
    G = nx.Graph()
    G.add_nodes_from(range(s["n_nodes"]))
    G.add_edges_from(map(tuple, s["edges"]))
    return G


def partial(y, x, z):
    """Spearman partial: correlate rank residuals of y and x on rank z. Returns (r, p)."""
    ry, rx, rz = (rankdata(a) for a in (y, x, z))
    Z = np.column_stack([np.ones_like(rz), rz])
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    r = np.corrcoef(res(ry), res(rx))[0, 1]
    n = len(ry)
    p = 2 * tdist.sf(abs(r) * np.sqrt((n - 3) / max(1e-12, 1 - r ** 2)), n - 3)
    return r, p


def diet_at(events, initial_diets, t):
    """Vegetarian flag per agent at step t, replayed from the event log."""
    veg = np.array(initial_diets) == "veg"
    for ev in events:
        if ev[1] > t:
            break
        veg[ev[2]] = ev[0] == "conv"
    return veg


def ever_conv(events, n, t):
    e = np.zeros(n, bool)
    for ev in events:
        if ev[1] > t:
            break
        if ev[0] == "conv":
            e[ev[2]] = True
    return e


def first_conv(events, n, t):
    f = np.full(n, np.nan)
    for ev in events:
        if ev[1] > t:
            break
        if ev[0] == "conv" and np.isnan(f[ev[2]]):
            f[ev[2]] = ev[1]
    return f


def corr_rows(run, block, win, name, y, feats, deg):
    """Spearman of y with each feature, plus betweenness|degree and cc|degree partials."""
    out = [dict(run=run, block=block, window=win, outcome=name, feature=f,
                stat=spearmanr(feats[f], y).statistic, p=spearmanr(feats[f], y).pvalue,
                n=len(y)) for f in FEATURES]
    for f in ("betweenness", "cc"):
        r, p = partial(y, feats[f], deg)
        out.append(dict(run=run, block=block, window=win, outcome=name,
                        feature=f + "|deg", stat=r, p=p, n=len(y)))
    return out


def group_null(G, veg, rng):
    """Induced-subgraph average clustering of each diet group vs same-size node subsets."""
    n = G.number_of_nodes()
    for grp, m in (("veg", veg), ("omni", ~veg)):
        idx = np.flatnonzero(m)
        obs = nx.average_clustering(G.subgraph(list(idx)))
        null = [nx.average_clustering(G.subgraph(list(rng.choice(n, len(idx), replace=False))))
                for _ in range(NDRAW)]
        mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
        yield grp, obs, mu, sd, (obs - mu) / sd if sd > 0 else np.nan


def one_run(run):
    d = pickle.load(open(f"{DIR}/run_{run:02d}.pkl", "rb"))
    ev, init = d["events"], d["initial_diets"]
    N = len(init)
    G = graph(d["snapshots"][G_SNAP])
    nodes = list(range(N))

    deg = np.array([G.degree(v) for v in nodes], float)
    bet = np.array([v for _, v in sorted(nx.betweenness_centrality(G).items())])
    clu = np.array([v for _, v in sorted(nx.clustering(G).items())])
    cc = np.array([v for _, v in sorted(compute_complex_centrality(G).items())])
    feats = dict(degree=deg, betweenness=bet, clustering=clu, cc=cc)

    immune = np.asarray(d["snapshots"][G_SNAP]["immune"], bool)
    omni0 = np.array(init) != "veg"
    m = omni0 & ~immune                                        # the adoption-eligible set
    print(f"INFO: run {run:02d} N={N} eligible={m.sum()} <k>={deg.mean():.2f}", flush=True)

    rows = []
    for t in WINDOWS:
        vt = veg_time(ev, init, t)
        dt = diet_at(ev, init, t)
        snap = np.asarray(d["snapshots"][t]["diets"], bool)
        if (dt != snap).sum():
            print(f"WARNING: run {run:02d} t={t} replayed diet differs from snapshot on "
                  f"{(dt != snap).sum()} agents", flush=True)
        nc = conv_counts(ev, init, t)
        sub = {f: a[m] for f, a in feats.items()}
        for name, y in (("diet_at_t", dt[m].astype(float)), ("ever_conv", ever_conv(ev, N, t)[m].astype(float)),
                        ("vegtime_share", vt[m] / t), ("n_conv", nc[m].astype(float))):
            rows += corr_rows(run, "C", t, name, y, sub, deg[m])
        # D: betweenness vs own vegetarian time, all agents
        rows += [dict(run=run, block="D", window=t, outcome="vegtime_all", feature=f,
                      stat=spearmanr(feats[f], vt).statistic, p=spearmanr(feats[f], vt).pvalue, n=N)
                 for f in ("betweenness", "degree")]
        r, p = partial(vt, bet, deg)
        rows.append(dict(run=run, block="D", window=t, outcome="vegtime_all",
                         feature="betweenness|deg", stat=r, p=p, n=N))

    # C(iv): first conversion step among eligible agents converting by 310k
    f310 = first_conv(ev, N, WINDOWS[-1])
    ok = m & ~np.isnan(f310)
    rows += corr_rows(run, "C", WINDOWS[-1], "first_conv_step", f310[ok],
                      {f: a[ok] for f, a in feats.items()}, deg[ok])

    # E: diet-group clustering null at the 100k and 200k snapshots
    rng = np.random.default_rng(run)
    for t in (100000, 200000):
        s = d["snapshots"][t]
        Gt = graph(s)
        for grp, obs, mu, sd, z in group_null(Gt, np.asarray(s["diets"], bool), rng):
            rows.append(dict(run=run, block="E", window=t, outcome=grp, feature="avg_clustering",
                             stat=obs, p=np.nan, n=Gt.number_of_nodes(),
                             obs=obs, null_mean=mu, null_sd=sd, z=z))
    print(f"INFO: run {run:02d} done, {len(rows)} rows", flush=True)
    return rows


def main(nruns):
    with Pool(NPROC) as pool:
        rows = [r for part in pool.imap_unordered(one_run, range(nruns)) for r in part]
    df = pd.DataFrame(rows).sort_values(["block", "window", "outcome", "feature", "run"])
    out = f"{DIR}/topology_window_robustness.csv"
    df.to_csv(out, index=False)
    print(f"INFO: wrote {out} ({len(df)} rows, {nruns} runs)")

    summ = df.groupby(["block", "window", "outcome", "feature"]).agg(
        median_stat=("stat", "median"), min_stat=("stat", "min"), max_stat=("stat", "max"),
        frac_sig=("p", lambda s: np.mean(s < 0.05) if s.notna().any() else np.nan),
        median_null=("null_mean", "median"), median_null_sd=("null_sd", "median"),
        median_z=("z", "median"))
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
