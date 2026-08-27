#!/usr/bin/env python3
"""Does the degree-amplification exponent converge, or does it just track run length?

Background: claude_stuff/Infrastructure/gamma_stage_dependence_2026-08-27.md
gamma rises monotonically for the whole of a 150k-step run (ensemble 0.53 at 100k, 0.72 at
t_end, 0.76 at 148k) and has not plateaued when the runs stop, because attributed reduction
is a cumulative ledger and F_veg saturating does not stop credit accruing. This extends the
runs far enough to see whether it settles, and if so where relative to b = 1.

Two things make this a true extension rather than a fresh ensemble: `steps` touches only the
loop bound and the snapshot times (model_main.py:343-346, 714), and seeding is 42 + run_id
exactly as in model_runner_mp.run_single_trajectory_model. So run i here is bit-identical to
run i of trajectory_analysis_twin_20260820.pkl for its first 150,000 steps -- which the
--check flag verifies.

Snapshots are NOT retained: record_snapshot copies the whole graph, and 200 of those per run
is ~800 MB per worker. We override it to fold each snapshot down to one row of statistics.
Output is a CSV of a few hundred KB, no pkl.

Usage (from model_src/testing):
    python gamma_convergence.py --runs 10 --steps 400000 --cores 10
    python gamma_convergence.py --check    # compare against the existing 150k ensemble
"""
import argparse
import os
import sys
import time
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import model_main
from model_runner_mp import DEFAULT_PARAMS, get_model

DIRECT_REDUCTION_KG = 664
OUTDIR = '../model_output'


def summarise(model, t):
    """Everything we want from a snapshot, without keeping the snapshot."""
    G = model.G1
    reds = np.array([a.reduction_out for a in model.agents], float)
    deg = np.array([G.degree(a.i) for a in model.agents], float)
    m = (reds > 0) & (deg > 0)
    row = {'t': t, 'f_veg': model.fraction_veg[-1] if model.fraction_veg else np.nan,
           'n_pos': int(m.sum())}
    if m.sum() < 20:
        return {**row, 'gamma': np.nan, 'r2': np.nan, 'mean_A': np.nan,
                'p90_A': np.nan, 'max_A': np.nan, 'gini': np.nan}

    A = reds[m] / DIRECT_REDUCTION_KG
    lk, lA = np.log10(deg[m]), np.log10(A)
    slope, icept = np.polyfit(lk, lA, 1)
    ss_tot = np.sum((lA - lA.mean()) ** 2)
    r2 = 1 - np.sum((lA - (slope * lk + icept)) ** 2) / ss_tot if ss_tot > 0 else np.nan

    pos = np.sort(reds[m])
    n = len(pos)
    gini = (2 * np.sum(np.arange(1, n + 1) * pos) / (n * pos.sum())) - (n + 1) / n
    return {**row, 'gamma': slope, 'r2': r2, 'mean_A': A.mean(),
            'p90_A': np.percentile(A, 90), 'max_A': A.max(), 'gini': gini}


def run_one(params):
    """One trajectory. Mirrors model_runner_mp.run_single_trajectory_model seeding."""
    import random
    run_id = params['run']
    seed = 42 + run_id
    np.random.seed(seed)
    random.seed(seed)
    params['seed'] = seed

    # get_model, not Model() directly: twin mode needs the demographic PMF tables for
    # alpha and rho, and without them model_main falls back to unconditional draws and the
    # trajectory diverges (F_veg 0.55 against 0.79 at 150k).
    model = get_model(params)
    recs = []

    # Fold each snapshot into one row and throw the snapshot away. Integer times only:
    # 'final' and 'steady' duplicate a t we already have.
    def record(t, _m=model, _r=recs):
        if isinstance(t, (int, np.integer)):
            _r.append(summarise(_m, int(t)))
    model.record_snapshot = record

    t0 = time.time()
    model.run()
    recs.append({**summarise(model, params['steps']), 't': params['steps']})

    df = pd.DataFrame(recs).drop_duplicates('t').sort_values('t')
    df['run'] = run_id
    print(f"  run={run_id:>2d}  F_veg={model.fraction_veg[-1]:.3f}  "
          f"gamma_end={df['gamma'].iloc[-1]:.3f}  points={len(df)}  "
          f"elapsed={time.time() - t0:.0f}s", flush=True)
    return df


def check_against_ensemble(pkl='../model_output/trajectory_analysis_twin_20260820.pkl'):
    """The 150k ensemble is the first 150k of these runs. Confirm the estimator agrees."""
    rows = pd.read_pickle(pkl)
    out = []
    for i in range(min(3, len(rows))):
        snaps = rows.iloc[i]['snapshots']
        for t in sorted(k for k in snaps if isinstance(k, (int, np.integer)))[-3:]:
            sn = snaps[t]
            G = sn['graph']
            reds = np.asarray(sn['reductions'], float)
            deg = np.array([G.degree(n) for n in G.nodes()], float)
            m = (reds > 0) & (deg > 0)
            g = np.polyfit(np.log10(deg[m]), np.log10(reds[m] / DIRECT_REDUCTION_KG), 1)[0]
            out.append({'run': i, 't': t, 'gamma_pkl': g,
                        'f_veg': sn['veg_fraction'], 'n_pos': int(m.sum())})
    print(pd.DataFrame(out).round(4).to_string(index=False))
    print("\nINFO: rerun runs 0-2 here and the same t must give the same gamma to ~1e-9.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--steps', type=int, default=400000)
    ap.add_argument('--cores', type=int, default=10)
    ap.add_argument('--cadence', type=int, default=2000,
                    help='snapshot spacing; 2000 is the hardcoded stride in model_main')
    ap.add_argument('--check', action='store_true',
                    help='print gamma from the existing 150k ensemble and exit')
    args = ap.parse_args()

    if args.check:
        check_against_ensemble()
        return

    params = DEFAULT_PARAMS.copy()
    params.update({'agent_ini': 'twin', 'N': 2000, 'steps': args.steps,
                   'snapshot_dense_start': args.cadence})

    # tau_persistence = M*2*N, independent of steps -- dwell weights saturate against a
    # fixed 36,000 while the run gets longer. Stated so it is on the record.
    print(f"INFO: twin N={params['N']}, steps={args.steps:,}, "
          f"tau_persistence={params['M'] * 2 * params['N']:,}, "
          f"{args.runs} runs on {args.cores} cores")
    print(f"INFO: gamma sampled every {args.cadence} steps "
          f"(~{args.steps // args.cadence} points/run)")

    jobs = [{**params, 'run': i} for i in range(args.runs)]
    t0 = time.time()
    with Pool(args.cores) as pool:
        frames = pool.map(run_one, jobs)
    df = pd.concat(frames, ignore_index=True)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR,
                       f'gamma_convergence_{date.today().strftime("%Y%m%d")}_'
                       f'{args.steps // 1000}k.csv')
    df.to_csv(out, index=False)
    print(f"\nINFO: {len(df)} rows -> {out}  ({time.time() - t0:.0f}s total)")

    g = df.groupby('t').agg(gamma=('gamma', 'mean'), gamma_sd=('gamma', 'std'),
                            f_veg=('f_veg', 'mean'), mean_A=('mean_A', 'mean'),
                            gini=('gini', 'mean'), n_pos=('n_pos', 'mean'))
    marks = [t for t in (100000, 150000, 200000, 250000, 300000, 350000, args.steps)
             if t in g.index]
    print(f"\n{'=' * 72}\n  GAMMA CONVERGENCE\n{'=' * 72}")
    print(g.loc[marks].round(4).to_string())

    # Drift per 10k steps over the last fifth of the run. If this is not ~0, gamma is
    # still tracking run length and b cannot be quoted as a single number.
    tail = g[g.index >= args.steps * 0.8].dropna(subset=['gamma'])
    if len(tail) > 2:
        drift = np.polyfit(tail.index.values.astype(float), tail['gamma'].values, 1)[0]
        print(f"\n  gamma drift over last 20% of run: {drift * 10000:+.4f} per 10k steps")
        print(f"  F_veg drift, same window:          "
              f"{np.polyfit(tail.index.values.astype(float), tail['f_veg'].values, 1)[0] * 10000:+.5f} per 10k")
        print("  -> " + ("CONVERGED (drift within noise); quote b at t_end with this bound"
                         if abs(drift * 10000) < 0.005 else
                         "STILL CLIMBING; b is run-length dependent, see the note"))


if __name__ == '__main__':
    main()
