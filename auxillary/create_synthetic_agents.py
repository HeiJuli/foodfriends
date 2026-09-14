#!/usr/bin/env python3
"""Synthetic stand-in for data/hierarchical_agents.csv (LISS respondent-level data,
which the Centerdata user statement forbids redistributing).

Two stages so that the second runs without LISS access:

  --aggregate   read the real CSV, write data/synthetic_aggregates.json: per
                demographic cell, the count of rows in each (has_rho, has_alpha)
                class; per theta bin, the diet split. Aggregate cell counts only.
  (default)     sample a population of the same shape from that JSON plus the
                conditional PMFs in data/demographic_pmfs.pkl: theta | cell,
                rho | cell, theta bin, alpha | cell; measured-flag pattern per
                cell as in the real file, ids 1..n. Rows flagged unmeasured are
                left blank and imputed by the model at run time exactly as for
                the real file.

The result matches the real file in marginals and in the theta-rho correlation
(rho is theta-stratified); theta-alpha is absent (+0.14 in the real file) because alpha is
conditioned on demographics only, as at run time. A PMF cell missing for a key
falls back to the equal-weight pool over all cells with the same theta bin
(rho) or all cells (alpha). No real respondent appears. Model runs on it give
the paper's qualitative results, not its numbers: the paper's sample is the
real file under random_state=42.
"""
import argparse
import json
import pickle

import numpy as np
import pandas as pd

DEMO = ['gender', 'age_group', 'incquart', 'educlevel']
AGG = "../data/synthetic_aggregates.json"
PMF = "../data/demographic_pmfs.pkl"
OUT = "../data/synthetic_agents.csv"


def aggregate(src="../data/hierarchical_agents.csv"):
    d = pd.read_csv(src)
    meta = pickle.load(open(PMF, 'rb'))['_metadata']
    d['theta_bin'] = pd.cut(d.theta, bins=meta['theta_bins'], labels=meta['theta_labels'],
                            include_lowest=True).astype(str)
    cells = (d.groupby(DEMO + ['has_rho', 'has_alpha']).size().reset_index(name='n'))
    cells['cell'] = cells[DEMO].astype(str).agg('|'.join, axis=1)
    diet = (d.groupby('theta_bin').diet.apply(lambda s: (s == 'veg').mean()))
    agg = {
        'cells': [{'cell': r.cell, 'has_rho': bool(r.has_rho), 'has_alpha': bool(r.has_alpha),
                   'n': int(r.n)} for r in cells.itertuples()],
        'veg_share_by_theta_bin': {k: float(v) for k, v in diet.items()},
        'n_total': int(len(d)),
    }
    json.dump(agg, open(AGG, 'w'), indent=1)
    print(f"Wrote {AGG}: {len(agg['cells'])} cell classes, {agg['n_total']} rows")


def _pool(table, key_filter):
    """Equal-weight pool of all PMFs whose key passes key_filter."""
    acc = {}
    for k, pmf in table.items():
        if key_filter(k):
            for v, p in zip(pmf['values'], pmf['probabilities']):
                acc[v] = acc.get(v, 0.0) + p
    vals = np.array(list(acc)); probs = np.array(list(acc.values()))
    return vals, probs / probs.sum()


def _draw(rng, table, key, n, fallback):
    if key in table:
        pmf = table[key]
        vals, probs = np.array(pmf['values']), np.array(pmf['probabilities'])
        probs = probs / probs.sum()
    else:
        vals, probs = fallback
    return rng.choice(vals, size=n, p=probs)


def _theta_bin(theta, meta):
    bins, labels = meta['theta_bins'], meta['theta_labels']
    idx = np.clip(np.searchsorted(bins, theta, side='right') - 1, 0, len(labels) - 1)
    return np.array(labels)[idx]


def synthesize(seed=0):
    agg = json.load(open(AGG))
    pmf = pickle.load(open(PMF, 'rb'))
    meta = pmf['_metadata']
    rng = np.random.default_rng(seed)
    theta_pool = _pool(pmf['theta'], lambda k: True)
    alpha_pool = _pool(pmf['alpha'], lambda k: True)
    rho_pool = {b: _pool(pmf['rho'], lambda k, b=b: k[-1] == b) for b in meta['theta_labels']}
    veg_share = agg['veg_share_by_theta_bin']

    rows = []
    for c in agg['cells']:
        g, a, inc, edu = c['cell'].split('|')
        key = (g, a, int(inc), int(edu))
        n = c['n']
        theta = _draw(rng, pmf['theta'], key, n, theta_pool)
        tbin = _theta_bin(theta, meta)
        diet = np.where(rng.random(n) < np.array([veg_share[b] for b in tbin]), 'veg', 'meat')
        rho = np.full(n, np.nan); alpha = np.full(n, np.nan)
        if c['has_rho']:
            for b in np.unique(tbin):
                m = tbin == b
                rho[m] = _draw(rng, pmf['rho'], key + (b,), m.sum(), rho_pool[b])
        if c['has_alpha']:
            alpha = _draw(rng, pmf['alpha'], key, n, alpha_pool)
        rows.append(pd.DataFrame({'theta': theta, 'diet': diet, 'has_rho': c['has_rho'],
                                  'has_alpha': c['has_alpha'], 'rho': rho, 'alpha': alpha,
                                  'gender': g, 'age_group': a, 'incquart': int(inc),
                                  'educlevel': int(edu)}))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, 'nomem_encr', np.arange(1, len(df) + 1))
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(df)} synthetic agents, seed {seed}")
    print(f"  veg share {(df.diet == 'veg').mean():.4f}; complete cases "
          f"{(df.has_rho & df.has_alpha).sum()}")
    print(df[['theta', 'rho', 'alpha']].corr().round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--aggregate', action='store_true',
                    help='stage 1: build synthetic_aggregates.json from the real CSV (needs LISS access)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        synthesize(args.seed)
