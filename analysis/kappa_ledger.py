#!/usr/bin/env python3
"""Attribution ledger replay across the kappa=0.55 headline ensemble.

Runs attribution_ledger.summarise over the reduced per-run artefacts -- same
code path as `attribution_ledger.py <pkl>`, which cannot load this ensemble.
The primary convention (exposure-proportional parents, event-count credit, no
dwell weight, lambda 0.7) is the reported one; the other three ride along for
the SI sensitivity table. The convention is fixed and is not revisited in
response to the output.

Amplification accumulates over the run, so --t-end is not optional in practice:
quote the window and the churn factor with every multiplier.

Usage:
    python kappa_ledger.py <reduced_dir> --t-end 310000 [--validate]
"""
import os, sys, glob, pickle, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_ledger import CONVENTIONS, summarise, validate


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reduced_dir')
    ap.add_argument('--t-end', type=int, default=None)
    ap.add_argument('--validate', action='store_true',
                    help='replay simulation semantics against the in-run ledger')
    a = ap.parse_args()

    rows, worst = [], (0.0, 0.0)
    for path in sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl'))):
        with open(path, 'rb') as f:
            run = pickle.load(f)
        if a.validate:
            sim = np.asarray(run['individual_reductions_unw'], float)
            from attribution_ledger import replay
            rep = replay(run['events'], run['initial_diets'], run['params'],
                         parent='last', weight='none', unit='event', cycle='visited')
            d = np.abs(rep - sim).max()
            worst = (max(worst[0], d), max(worst[1], d / max(sim.max(), 1.0)))
        rows.append(summarise(run, a.t_end))
        print(f"INFO: {os.path.basename(path)} done", flush=True)

    if a.validate:
        print(f"\nvalidate (unweighted, simulation semantics): max abs diff over runs "
              f"{worst[0]:.3e}, relative to the largest credit {worst[1]:.3e}")

    print(f"\n{len(rows)} runs, t_end={a.t_end or 'full run'}")
    ch = [r['churn'] for r in rows]
    print(f"churn: median {np.median(ch):.2f} conversions per converter, "
          f"IQR [{np.percentile(ch, 25):.2f}, {np.percentile(ch, 75):.2f}]")
    print(f"\n{'convention':38s} {'mean':>7s} {'p90':>7s} {'max':>8s} {'credited':>9s}")
    for name in CONVENTIONS:
        m = {k: np.median([r[name][k] for r in rows])
             for k in ('mean', 'p90', 'max', 'n_credited')}
        q = np.percentile([r[name]['mean'] for r in rows], [25, 75])
        print(f"{name:38s} {m['mean']:7.2f} {m['p90']:7.2f} {m['max']:8.1f} "
              f"{m['n_credited']:9.0f}   IQR(mean) [{q[0]:.2f}, {q[1]:.2f}]")


if __name__ == '__main__':
    main()
