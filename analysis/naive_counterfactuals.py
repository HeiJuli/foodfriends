#!/usr/bin/env python3
"""Naive counterfactuals for the amplification factor (exploratory).

With the dwell weight stripped (w=1) the ledger in model_main._cascade_attribute
reduces to a pure property of the influence tree: each conversion pays lam^(d-1)
to its ancestor at tree-distance d, so

    A_i = sum_{j in desc(i)} lam^{d(i,j)-1}          (lam-discounted subtree size)

which is what the naive scenarios below compute.

LEDGER IDENTITY (topology- and rule-independent)
    A conversion at depth D mints 1 + lam + ... + lam^{D-1} units of credit, so
    over M adopters      mean A = c * (1 - E[lam^D]) / (1 - lam),
    c = fraction of conversions that book an influencer. The MEAN amplification
    factor is bookkeeping: 1/(1-lam) = 3.33x at lam=0.7, whatever the topology or
    the conversion rule. Only the distribution -- tail, inequality, degree
    scaling -- carries information about the model.

CF1  complete graph, conversion on contact, credit to the sampled partner.
    The m-th adopter attaches to a uniform draw from the m existing adopters:
    the influence tree is a uniform RANDOM RECURSIVE TREE. By rank k,
        E[subtree size] = M/k               (exact; from a_{j+1} = a_j (j+1)/j)
        E[A_k] = ((M/k)^lam - 1) / lam      (continuum limit of a_{j+1} = a_j(1+lam/j) + 1/j)
    a pure 1/rank law: amplification is decided by arrival order alone. Note this
    is E[A | rank], a conditional mean, NOT the distribution of A -- roughly half
    the nodes are leaves with A = 0 and the law smooths them away.

CF2  ER(N, z), simple contagion at fixed p, credit to the sampled partner.
    p CANCELS EXACTLY: it thins every (converter, partner) attempt uniformly, so
    the embedded jump chain -- and hence the whole tree -- is p-independent. p
    sets the timescale only. No closed form for the rank law -> simulated (cheap).

CF3  empirical network, naive dynamics. Topology held fixed, behaviour stripped:
    isolates what the wiring alone implies.

Usage:
    python naive_counterfactuals.py [pkl] [t_cutoff]
"""
import sys, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

LAM = 0.7                    # params['decay']
DIRECT_REDUCTION_KG = 664
DEFAULT_PKL = os.path.join(os.path.dirname(__file__), '..', 'model_output',
                           'trajectory_analysis_twin_20260820_unwledger.pkl')
OUT = os.path.join(os.path.dirname(__file__), '..', 'visualisations_output')


# ------------------------------------------------------------------ analytics
def rrt_rank_law(M, lam=LAM):
    """CF1: E[A_k] by adoption rank k over M adopters, from the exact recursion
    a_{j+1} = a_j (1 + lam/j) + 1/j, a_k = 0, solved by suffix products in O(M).
    The closed form ((M/k)^lam - 1)/lam is its continuum limit: within 1% for
    k > 20, ~9% low at k = 1. Verified against direct RRT Monte-Carlo."""
    j = np.arange(1, M, dtype=float)                      # j = 1 .. M-1
    R = np.append(np.cumprod((1 + lam / j)[::-1])[::-1][1:], 1.0)   # prod_{i>j}
    return np.append(np.cumsum((R / j)[::-1])[::-1], 0.0)  # a_k, k = 1 .. M


def rrt_closed(M, lam=LAM):
    """Interpretable continuum form of rrt_rank_law: a pure 1/rank^lam law."""
    return ((M / np.arange(1, M + 1)) ** lam - 1) / lam


# ----------------------------------------------------------------- simulation
def naive_cascade(G, frac=1.0, lam=LAM, seed=0):
    """Conversion on contact; credit to the sampled veg partner and its ancestors.
    Returns per-node (credit, adoption rank, degree); rank -1 = never converted."""
    G = nx.convert_node_labels_to_integers(
        G.subgraph(max(nx.connected_components(G), key=len)))
    n = G.number_of_nodes()
    nbr = [np.fromiter(G.neighbors(v), int) for v in range(n)]
    deg = np.array([len(x) for x in nbr])
    rng = np.random.default_rng(seed)

    veg = np.zeros(n, bool); par = np.full(n, -1); rank = np.full(n, -1)
    credit = np.zeros(n)
    s = rng.integers(n); veg[s] = True; rank[s] = 0
    target, order = max(1, int(round(frac * n))), 1
    while order < target:
        u = rng.integers(n)
        if veg[u]:
            continue
        v = nbr[u][rng.integers(deg[u])]
        if not veg[v]:
            continue
        veg[u] = True; par[u] = v; rank[u] = order; order += 1
        a, w = v, 1.0
        while a != -1:                      # walk the ancestor chain
            credit[a] += w; w *= lam; a = par[a]
    return credit, rank, deg


# -------------------------------------------------------------------- metrics
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def slope(y, x):
    """log-log OLS slope over strictly positive pairs (repo convention)."""
    m = (y > 0) & (x > 0)
    return float(np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)[0])


def slope_binned(y, x, minn=5):
    """Zero-inclusive: mean credit per degree value, then log-log OLS.
    Immune to the leaf-exclusion bias that inflates `slope` when many A == 0."""
    ks = np.array([k for k in np.unique(x) if k > 0 and (x == k).sum() >= minn])
    mu = np.array([y[x == k].mean() for k in ks])
    m = mu > 0
    return float(np.polyfit(np.log10(ks[m]), np.log10(mu[m]), 1)[0])


def summarise(a, label, deg=None, adopter=None):
    """mean_ad = over adopters incl. zero-credit leaves (the ledger-identity
    denominator); mean_pos = over positive credit only (the paper's figure)."""
    a = np.asarray(a, float)
    ad = a if adopter is None else a[adopter]
    pos = a[a > 0]
    r = dict(label=label, n_ad=len(ad), n_pos=len(pos), mean_ad=ad.mean(),
             mean_pos=pos.mean(), med_pos=np.median(pos),
             p90=np.percentile(ad, 90), mx=a.max(), gini=gini(ad),
             b=np.nan, bb=np.nan)
    if deg is not None:
        r['b'], r['bb'] = slope(a, deg), slope_binned(a, deg)
    return r


def table(rows):
    h = (f"{'scenario':<27}{'n_ad':>6}{'n_pos':>6}{'mean_ad':>8}{'mean+':>7}{'med+':>6}"
         f"{'p90':>6}{'max':>7}{'gini':>6}{'b':>6}{'b_bin':>7}")
    print(h); print('-' * len(h))
    f = lambda v, w, p=2: f"{v:>{w}.{p}f}" if np.isfinite(v) else f"{'--':>{w}}"
    for r in rows:
        print(f"{r['label']:<27}{r['n_ad']:>6}{r['n_pos']:>6}{f(r['mean_ad'],8)}"
              f"{f(r['mean_pos'],7)}{f(r['med_pos'],6)}{f(r['p90'],6)}{f(r['mx'],7,1)}"
              f"{f(r['gini'],6,3)}{f(r['b'],6)}{f(r['bb'],7)}")


def ledger_audit(A_u, par, adopt, lam=LAM):
    """Invert the identity total = sum_events (1 - lam^D)/(1 - lam) to get the
    implied number of credited conversion EVENTS, and compare it with the number
    of edges standing in the observed forest. Ratio > 1 => repeat conversions:
    the ledger takes no debits (Zhou et al. 2014), so an agent that reverts and
    re-converts mints a second helping of credit. Depths come from the observed
    forest, so reverted nodes orphan their subtrees and the depths are an
    approximation -- the ratio is indicative, not exact."""
    n = len(par)
    depth = np.zeros(n)
    for i in range(n):                       # memoised walk to root
        c, d = i, 0
        while par[c] >= 0 and d < 200:
            c = par[c]; d += 1
        depth[i] = d + 1                     # root converts at depth 1
    att = par >= 0
    mint = (1 - lam ** depth[att]) / (1 - lam)
    implied = A_u.sum() / mint.mean()
    print(f"\nledger audit (w=1): total credit {A_u.sum():.0f} units; mean mint "
          f"{mint.mean():.2f}/event (E[lam^D]={1 - mint.mean() * (1 - lam):.3f})")
    print(f"  implied credited events {implied:.0f} vs {att.sum()} edges standing "
          f"=> churn factor {implied / att.sum():.2f}")
    print(f"  i.e. ~{implied / adopt.sum():.2f} credited conversions per net adopter")


BANDS = [(3, 4), (5, 6), (7, 8), (9, 11), (12, 15), (16, 20), (21, 30), (31, 50), (51, 400)]


def degree_profile(A, deg, label, bands=BANDS):
    """Amplification vs degree without fitting anything: the A/k ratio per band.
    The repo estimator (log-log OLS over positive-credit agents only) reads low
    here because the zero-credit fraction falls steeply with degree -- the
    excluded agents sit at low k, which flattens the slope. It also drifts with
    the analysis cutoff (0.62 -> 0.79 over t=106k..140k) while A/k does not move:
    the signature of a censoring artefact rather than a scaling law."""
    print(f"\n{label}: amplification by degree band")
    print(f"{'band':>10}{'n':>6}{'mean A':>9}{'A/k':>7}{'%zero':>7}")
    kk, mu = [], []
    for lo, hi in bands:
        m = (deg >= lo) & (deg <= hi)
        if m.sum() < 5:
            continue
        a, k = A[m], deg[m].mean()
        kk.append(k); mu.append(a.mean())
        print(f"{lo:>4}-{hi:<5}{m.sum():>6}{a.mean():>9.2f}{a.mean() / k:>7.3f}"
              f"{100 * (a == 0).mean():>7.1f}")
    kk, mu = np.array(kk), np.array(mu)
    m = mu > 0
    print(f"  conditional-mean slope (zeros included) = "
          f"{np.polyfit(np.log10(kk[m]), np.log10(mu[m]), 1)[0]:.2f}")
    print(f"  repo estimator (positive-only log-log)  = {slope(A, deg):.2f}")


def estimator_bias_test(A, deg, reps=50, a=0.22, seed=0):
    """Settle which estimator to believe: build synthetic credit whose conditional
    mean is EXACTLY linear in degree, carrying the observed P(zero | k) and the
    observed log-spread, then see which estimator recovers the known slope of 1."""
    p0, sd = {}, {}
    for lo, hi in BANDS:
        m = (deg >= lo) & (deg <= hi)
        if m.sum() < 5:
            continue
        x = A[m]; p0[(lo, hi)] = (x == 0).mean(); sd[(lo, hi)] = np.log(x[x > 0]).std()
    bands = list(p0)
    rng = np.random.default_rng(seed)
    kk = np.array([deg[(deg >= l) & (deg <= h)].mean() for l, h in bands])
    rep, cond = [], []
    for _ in range(reps):
        syn = np.zeros(len(deg))
        for i, k in enumerate(deg):
            if k < bands[0][0]:
                continue
            b = next((x for x in bands if x[0] <= k <= x[1]), bands[-1])
            q, sg = p0[b], sd[b]
            if rng.random() < q:
                continue                       # zero with the observed probability
            # lognormal with mean a*k/(1-q)  =>  E[A|k] = a*k exactly
            syn[i] = rng.lognormal(np.log(a * k / (1 - q)) - sg ** 2 / 2, sg)
        m = deg >= bands[0][0]
        mu = np.array([syn[(deg >= l) & (deg <= h)].mean() for l, h in bands])
        rep.append(slope(syn[m], deg[m]))
        cond.append(np.polyfit(np.log10(kk), np.log10(mu), 1)[0])
    rep, cond = np.array(rep), np.array(cond)
    lz = -np.log(1 - np.array([p0[b] for b in bands]))
    bias = np.polyfit(np.log(kk), lz, 1)[0]
    print("\nestimator test on synthetic data, true conditional-mean slope = 1.000")
    print(f"  positive-only log-log OLS recovers {rep.mean():.3f} +- {rep.std():.3f}"
          f"   <- biased low by {1 - rep.mean():.3f}")
    print(f"  conditional-mean recovers          {cond.mean():.3f} +- {cond.std():.3f}")
    print(f"  censoring term d[-log(1-p0)]/d[log k] = {bias:+.3f} accounts for "
          f"{100 * -bias / (1 - rep.mean()):.0f}% of the bias; rest is Jensen")


# ----------------------------------------------------------------------- main
def main(pkl=DEFAULT_PKL, t_cutoff=143600):
    import pandas as pd
    d = pd.read_pickle(pkl)
    row = d[d['is_median_twin']].iloc[0] if d['is_median_twin'].any() else d.iloc[0]
    ts = sorted(t for t in row['snapshots'] if isinstance(t, int))
    key = min(ts, key=lambda t: abs(t - t_cutoff))
    s = row['snapshots'][key]
    G = s['graph']; N = G.number_of_nodes()
    deg = np.array([G.degree(v) for v in G.nodes()])
    A_w = np.array(s['reductions']) / DIRECT_REDUCTION_KG
    A_u = (np.array(s['reductions_unw']) / DIRECT_REDUCTION_KG
           if 'reductions_unw' in s else None)
    par = np.array([-1 if p is None else p for p in s['influence_parents']])
    adopt = np.array([c is not None for c in s['change_times']])   # ever switched
    f_veg = s['veg_fraction']

    print(f"model: {os.path.basename(pkl)}  t={key}  N={N}  z={deg.mean():.2f}  "
          f"F_veg={f_veg:.3f}  lambda={LAM}")
    print(f"adopters={adopt.sum()}  of which credited (influencer booked) "
          f"c={(par[adopt] >= 0).mean():.3f}\n")

    M = int(round(f_veg * N))
    c_kn, r_kn, _ = naive_cascade(nx.complete_graph(N), f_veg)
    c_er, r_er, d_er = naive_cascade(
        nx.gnm_random_graph(N, int(round(deg.mean() * N / 2)), seed=1), f_veg)
    c_em, r_em, d_em = naive_cascade(G, f_veg)
    ana = rrt_rank_law(M)

    rows = [summarise(A_w, 'model (weighted ledger)', deg, adopt)]
    if A_u is not None:
        rows.append(summarise(A_u, 'model (w=1 ledger)', deg, adopt))
    rows += [summarise(c_em, 'CF3 emp. net, naive', d_em, r_em >= 0),
             summarise(c_er, 'CF2 ER(z) naive', d_er, r_er >= 0),
             summarise(c_kn, 'CF1 complete, naive', None, r_kn >= 0),
             summarise(ana, 'CF1 analytic E[A|rank]')]
    table(rows)

    ceil = 1 / (1 - LAM)
    print(f"\nledger identity   1/(1-lambda) = {ceil:.2f}x per credited conversion")
    print(f"  CF1 analytic mean over M={M} adopters   {ana.mean():.2f}x "
          f"(shortfall = shallow nodes, E[lam^D])")
    print(f"  CF1 simulated mean over adopters        {c_kn[r_kn >= 0].mean():.2f}x")
    obs = A_w[adopt].mean()
    print(f"  model, w=1, over adopters               "
          f"{A_u[adopt].mean():.2f}x" if A_u is not None else "")
    print(f"  model, weighted, over adopters          {obs:.2f}x")
    if A_u is not None:
        print(f"  => dwell weight costs a factor {obs / A_u[adopt].mean():.2f}; "
              f"uncredited conversions cost {(par[adopt] >= 0).mean():.2f}")
    if A_u is not None:
        ledger_audit(A_u, par, adopt)
    degree_profile(A_w, deg.astype(float), 'model (weighted ledger)')
    degree_profile(c_em, d_em.astype(float), 'CF3 emp. net, naive')
    estimator_bias_test(A_w, deg.astype(float))
    print(f"\nCF1 top-rank ceiling: exact {ana[0]:.0f}x, closed form "
          f"(M^lam-1)/lam = {rrt_closed(M)[0]:.0f}x;  model max = {A_w.max():.0f}x "
          f"(w=1: {A_u.max():.0f}x)" if A_u is not None else "")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    series = [(A_w, 'model (weighted)', 'C0'), (c_em, 'CF3 emp. net, naive', 'C2'),
              (c_er, 'CF2 ER, naive', 'C3'), (c_kn, 'CF1 complete, naive', 'C4')]
    if A_u is not None:
        series.insert(1, (A_u, 'model (w=1)', 'C1'))
    for v, lab, col in series:
        v = np.sort(np.asarray(v, float))[::-1]
        ax.plot(np.arange(1, len(v) + 1) / len(v) * 100, v, lw=1.2, color=col, label=lab)
    ax.plot(np.arange(1, M + 1) / M * 100, ana, 'k--', lw=1.0, label='CF1 analytic $E[A|k]$')
    ax.axhline(ceil, color='#999', ls=':', lw=0.8)
    ax.text(99, ceil * 1.15, r'$1/(1-\lambda)$', fontsize=6, ha='right', color='#777')
    ax.set(xlabel='Agent rank [%]', ylabel='Amplification factor $A$',
           xlim=(0, 100), ylim=(1e-2, None), yscale='log')
    ax.set_title('a  distribution vs naive counterfactuals', fontsize=8, loc='left')
    ax.legend(fontsize=6, frameon=False)

    for v, dg, lab, col in [(A_w, deg, 'model (weighted)', 'C0'),
                            (c_em, d_em, 'CF3 emp. net, naive', 'C2')]:
        dg = np.asarray(dg, float)
        kk = np.array([dg[(dg >= l) & (dg <= h)].mean() for l, h in BANDS
                       if ((dg >= l) & (dg <= h)).sum() >= 5])
        mu = np.array([v[(dg >= l) & (dg <= h)].mean() for l, h in BANDS
                       if ((dg >= l) & (dg <= h)).sum() >= 5])
        bx.plot(kk, mu, 'o-', ms=3, lw=1.0, color=col, label=f'{lab}, $E[A|k]$')
        b0 = np.polyfit(np.log10(kk[mu > 0]), np.log10(mu[mu > 0]), 1)
        bx.plot(kk, 10 ** np.polyval(b0, np.log10(kk)), '-', lw=0.6, color=col, alpha=0.5)
        if col == 'C0':
            kk_m, mu_m = kk, mu
    k0 = np.array([3, 160], float)
    k1, m1 = kk_m[0], mu_m[0]          # anchor both references on the model's first band
    bx.plot(k0, m1 * (k0 / k1), 'k--', lw=0.9, label='linear, $A \\propto k$')
    bx.plot(k0, m1 * (k0 / k1) ** 0.72, ':', color='#c33', lw=1.1,
            label='reported $A \\propto k^{0.72}$')
    bx.set(xscale='log', yscale='log', xlabel='Degree $k$',
           ylabel='Mean amplification $E[A|k]$')
    bx.set_title('b  degree scaling is linear once zeros are kept', fontsize=8, loc='left')
    bx.legend(fontsize=6, frameon=False)
    for a_ in (ax, bx):
        for sp in ('top', 'right'):
            a_.spines[sp].set_visible(False)
        a_.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/naive_counterfactuals.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{OUT}/naive_counterfactuals.png', dpi=200, bbox_inches='tight')
    print(f"\nSaved {OUT}/naive_counterfactuals.pdf")


if __name__ == '__main__':
    main(*(sys.argv[1:2] or []), *([int(sys.argv[2])] if len(sys.argv) > 2 else []))
