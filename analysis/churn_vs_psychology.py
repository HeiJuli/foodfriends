#!/usr/bin/env python3
"""Is the model's flatter credit distribution psychology, or just churn?

The model spreads credit over ~66% of adopters at Gini ~0.70; a naive contagion
null on the same network reaches only ~36% at Gini ~0.86. Two candidate causes:

  (a) calibrated heterogeneity genuinely flattens leverage
  (b) churn -- the ledger takes no debits, so revert-and-reconvert mints fresh
      credit, and the second conversion usually credits a DIFFERENT parent, which
      mechanically spreads credit over more agents

These are separated by attacking from both sides, and neither needs a model rerun.

TEST A -- take churn out of the model.
    Rebuild each agent's credit from the FINAL influence forest
    (`influence_parents`): the lambda-discounted subtree size, counting every
    adopter exactly once. That is what the ledger would say if each agent could
    be credited only once. Compare its shape to the real ledger.

TEST B -- put churn into the null.
    Run the naive cascade with reversion, so agents detach and re-convert and
    mint fresh credit, sweeping the reversion rate. If the null moves to the
    model's %credited and Gini once its churn matches, churn is sufficient and
    (a) is unsupported.

Reading it:
    A ~ real ledger  and  B(churn) ~ B(0)   -> churn does nothing; psychology it is
    A << real ledger and  B(churn) ~ model  -> churn is sufficient; drop claim (a)
    partial movement in both                -> both contribute; quote the split

Usage:
    python churn_vs_psychology.py [--pkl PATH] [--t 144000]
"""
import sys, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx

from naive_counterfactuals import gini, naive_cascade, DIRECT_REDUCTION_KG, LAM

TWIN_PKL = '../model_output/trajectory_analysis_twin_20260820_unwledger.pkl'


# --------------------------------------------------------------------- shared
def shape(a, adopters=None, label=''):
    a = np.asarray(a, float)
    ad = a if adopters is None else a[adopters]
    srt = np.sort(a)[::-1]; tot = a.sum()
    n20 = max(1, int(0.2 * (a > 0).sum()))
    return dict(label=label, mean_ad=ad.mean(), gini=gini(ad),
                pct_credit=100 * (a > 0).mean(), top1=srt[:max(1, len(a) // 100)].sum() / tot,
                top20=srt[:n20].sum() / tot, mx=a.max())


def show(rows):
    h = f"{'':<34}{'mean/ad':>9}{'Gini':>8}{'%cred':>8}{'top1%':>8}{'top20%':>8}{'max':>8}"
    print(h); print('-' * len(h))
    for r in rows:
        print(f"{r['label']:<34}{r['mean_ad']:>9.2f}{r['gini']:>8.3f}{r['pct_credit']:>8.1f}"
              f"{r['top1']:>8.3f}{r['top20']:>8.3f}{r['mx']:>8.1f}")


# ---------------------------------------------------- TEST A: de-churn the model
def forest_credit(parents, lam=LAM):
    """Lambda-discounted subtree size from the final influence forest: every
    adopter credited exactly once, i.e. the model's own ledger with churn removed.
    Mirrors _cascade_attribute's ancestor walk, run once per standing edge."""
    n = len(parents)
    credit = np.zeros(n)
    for j in range(n):
        a, w, seen = parents[j], 1.0, set()
        while a >= 0 and a not in seen:
            seen.add(a)
            credit[a] += w; w *= lam
            a = parents[a]
    return credit


# ------------------------------------------------- TEST B: churn up the null
def naive_with_reversion(G, f_target, revert_p, lam=LAM, seed=0, max_events=None,
                         f_seed=0.06):
    """Naive cascade plus reversion. Conversion on contact with a veg partner
    mints credit to the partner's chain (no debits, as in the model); reverting
    only detaches. Runs until the credited-event budget is spent, then reports
    the churn actually achieved."""
    G = nx.convert_node_labels_to_integers(
        G.subgraph(max(nx.connected_components(G), key=len)))
    n = G.number_of_nodes()
    nbr = [np.fromiter(G.neighbors(v), int) for v in range(n)]
    deg = np.array([len(x) for x in nbr])
    rng = np.random.default_rng(seed)

    veg = np.zeros(n, bool); par = np.full(n, -1); credit = np.zeros(n)
    first_t = np.full(n, np.inf)          # first adoption time, for rank
    # seed at the model's initial veg fraction, not a single agent: with a high
    # reversion rate one seed goes extinct before it can spread
    veg[rng.choice(n, max(1, int(round(f_seed * n))), replace=False)] = True
    events = 0
    budget = max_events or int(4 * f_target * n)
    ever = np.zeros(n, bool); ever[veg] = True
    # run long enough to equilibrate and spend the budget
    for _ in range(400 * n):
        if events >= budget:
            break
        u = rng.integers(n)
        if deg[u] == 0:
            continue
        if veg[u]:
            if rng.random() < revert_p:                # revert: detach, keep credit
                veg[u] = False; par[u] = -1
            continue
        v = nbr[u][rng.integers(deg[u])]
        if not veg[v]:
            continue
        veg[u] = True; ever[u] = True; par[u] = v; events += 1
        first_t[u] = min(first_t[u], events)
        a, w, seen = v, 1.0, set()
        while a >= 0 and a not in seen:
            seen.add(a); credit[a] += w; w *= lam; a = par[a]
    return credit, ever, veg.mean(), events / max(1, ever.sum()), first_t, deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=TWIN_PKL)
    ap.add_argument('--t', type=int, default=144000)
    a = ap.parse_args()

    d = pd.read_pickle(a.pkl)
    r = d[d['is_median_twin']].iloc[0] if d['is_median_twin'].any() else d.iloc[0]
    s = r['snapshots'][min((t for t in r['snapshots'] if isinstance(t, int)),
                           key=lambda t: abs(t - a.t))]
    G = s['graph']
    par = np.array([-1 if p is None else p for p in s['influence_parents']])
    adopt = np.array([c is not None for c in s['change_times']])
    A_w = np.array(s['reductions']) / DIRECT_REDUCTION_KG
    A_u = np.array(s['reductions_unw']) / DIRECT_REDUCTION_KG
    f_veg = s['veg_fraction']
    print(f"{a.pkl.split('/')[-1]}  t={a.t}  N={G.number_of_nodes()}  F_veg={f_veg:.3f}\n")

    # ---- TEST A
    A_tree = forest_credit(par)
    churn = A_u.sum() / max(A_tree.sum(), 1e-9)
    print("TEST A -- model ledger with churn removed (credit each adopter once)")
    show([shape(A_w, adopt, 'model, real ledger (weighted)'),
          shape(A_u, adopt, 'model, real ledger (w=1)'),
          shape(A_tree, adopt, 'model, churn-free (w=1)')])
    print(f"  total credit ratio real/churn-free = {churn:.2f}\n")

    # ---- TEST B
    print("TEST B -- naive null with reversion, sweeping the reversion rate")
    c0, r0, _ = naive_cascade(G, f_veg)
    rows = [shape(c0, r0 >= 0, 'null, no reversion (churn=1.0)')]
    # model's churn in the same units: credited events per net adopter
    churn_model = (A_u.sum() / 2.31) / adopt.sum()
    for rp in (0.05, 0.10, 0.15, 0.20, 0.30):
        c, ever, fv, ch, _, _ = naive_with_reversion(
            G, f_veg, rp, max_events=int(round(churn_model * G.number_of_nodes())))
        rows.append(shape(c, ever, f'null + reversion p={rp:<4g} '
                                   f'(F_veg={fv:.2f}, churn={ch:.2f})'))
    rows.append(shape(A_w, adopt, 'model, real ledger (weighted)'))
    show(rows)
    print(f"\n  model churn = {churn_model:.2f} credited conversions per net adopter;"
          f" match that column, then compare Gini / %cred / top1%")


if __name__ == '__main__':
    main()
