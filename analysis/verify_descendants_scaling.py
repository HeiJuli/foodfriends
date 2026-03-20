#!/usr/bin/env python3
"""Verify whether direct-conversions vs cascade-credit scaling divergence is real or artefact.

Computes total_descendants (full subtree size) for each agent in the cascade tree,
then compares log-log slopes against degree for:
  - direct_conversions (immediate children only)
  - total_descendants (full subtree)
  - amplification_kg (cascade credit in CO2)

If total_descendants slope ~1.0: divergence is a measurement artefact.
If total_descendants slope is sub-linear: real mechanistic difference.

Usage:
  python verify_descendants_scaling.py <pkl_path>
  python verify_descendants_scaling.py   # defaults to 0317 twin pkl
"""
import sys, os, pickle
import numpy as np
import statsmodels.api as sm
from collections import defaultdict

DIRECT_REDUCTION_KG = 664
DEFAULT_PKL = os.path.join(os.path.dirname(__file__), '..', 'model_output',
                           'trajectory_analysis_twin_20260317.pkl')


def analyze_run(snap, run_label=""):
    """Analyze one run's final snapshot."""
    parents = snap['influence_parents']
    dc = np.array(snap['direct_conversions'])
    reds = np.array(snap['reductions'])
    G = snap['graph']
    nodes = list(G.nodes())
    degrees = np.array([G.degree(n) for n in nodes])
    N = len(reds)

    # Build children map and count total descendants
    children = defaultdict(list)
    for j, p in enumerate(parents):
        if p is not None:
            children[p].append(j)

    total_desc = np.zeros(N, dtype=int)
    for i in range(N):
        stack = list(children[i])
        count = 0
        while stack:
            c = stack.pop()
            count += 1
            stack.extend(children[c])
        total_desc[i] = count

    # Compare slopes
    print(f"\n{'='*70}")
    if run_label:
        print(f"  Run: {run_label}")
    print(f"  N={N}, positive reductions={np.sum(reds > 0)}, "
          f"positive descendants={np.sum(total_desc > 0)}")
    print(f"{'='*70}")
    print(f"  {'metric':<25s} {'slope':>7s}  {'95% CI':>20s}  {'R2':>6s}  {'n':>5s}")
    print(f"  {'-'*70}")

    results = {}
    for label, vals in [('direct_conversions', dc),
                        ('total_descendants', total_desc),
                        ('amplification_kg', reds)]:
        pos = (vals > 0) & (degrees > 0)
        n_pos = pos.sum()
        if n_pos < 10:
            print(f"  {label:<25s}  insufficient data (n={n_pos})")
            continue
        X = sm.add_constant(np.log(degrees[pos].astype(float)))
        y = np.log(vals[pos].astype(float))
        m = sm.OLS(y, X).fit()
        slope = m.params.iloc[1]
        ci = m.conf_int().iloc[1]
        r2 = m.rsquared
        print(f"  {label:<25s} {slope:>7.3f}  [{ci.iloc[0]:>8.3f}, {ci.iloc[1]:>8.3f}]  {r2:>6.3f}  {n_pos:>5d}")
        results[label] = slope

    # Verdict
    if 'total_descendants' in results and 'direct_conversions' in results:
        td = results['total_descendants']
        dc_s = results['direct_conversions']
        print(f"\n  VERDICT:")
        if td > 0.85:
            print(f"  total_descendants slope={td:.3f} (~1.0) -> divergence is likely a MEASUREMENT ARTEFACT")
            print(f"  direct_conversions slope={dc_s:.3f} is sub-linear because it only counts depth-1 children")
        else:
            print(f"  total_descendants slope={td:.3f} (sub-linear) -> REAL mechanistic difference")
            print(f"  CO2 weighting preferentially rewards high-degree early adopters")

    return results


if __name__ == '__main__':
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PKL
    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Handle both list-of-dicts and single-dict formats
    if isinstance(data, list):
        runs = data
    elif isinstance(data, dict) and 'snapshots' in data:
        runs = [data]
    else:
        runs = data if hasattr(data, '__iter__') else [data]

    all_results = []
    for i, run in enumerate(runs):
        snap = run.get('snapshots', run) if isinstance(run, dict) else run
        # Get final snapshot
        if isinstance(snap, dict) and 'final' in snap:
            final = snap['final']
        elif isinstance(snap, dict):
            int_keys = [k for k in snap if isinstance(k, int)]
            final = snap[max(int_keys)] if int_keys else snap
        else:
            final = snap

        # Check required fields exist
        if not all(k in final for k in ['influence_parents', 'direct_conversions', 'reductions', 'graph']):
            print(f"  Run {i}: missing required fields, skipping")
            continue

        res = analyze_run(final, run_label=f"{i}")
        all_results.append(res)

    # Ensemble summary if multiple runs
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  ENSEMBLE SUMMARY (n={len(all_results)} runs)")
        print(f"{'='*70}")
        for label in ['direct_conversions', 'total_descendants', 'amplification_kg']:
            slopes = [r[label] for r in all_results if label in r]
            if slopes:
                print(f"  {label:<25s}  median={np.median(slopes):.3f}  "
                      f"IQR=[{np.percentile(slopes,25):.3f}, {np.percentile(slopes,75):.3f}]")
