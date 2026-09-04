#!/usr/bin/env python3
"""Does the two-DV asymmetry depend on who is in each regression?

The shipped specification measures the two DVs on different samples. DV1 is the
initial meat-eaters with a state-at-t outcome; DV2 is every agent holding
positive cascade credit. At N=2000, t=310000 those differ substantially: DV2
admits initially-veg spreaders (115 of 1501 in run 00) and converted-then-
reverted agents (264), and excludes adopters whose credit is zero (249). So a
meat-to-veg-to-meat agent is a failure in DV1 and a success in DV2, and the
headline ratio divides two fits taken over different populations.

This refits both DVs under matched samples to see whether the ratio, the degree
coefficient and the degree-scaling exponent b survive the choice:

  DV1  shipped   initial meat-eaters, adopted = veg at t
       ever      initial meat-eaters, ever converted by t (removes the revert
                 asymmetry from the DV1 side, using the event log)
       noimmune  shipped, less immune agents -- they are selected on the DV1
                 predictors themselves (agency_predictor_analysis.py:180)

  DV2  shipped   any positive cascade credit
       meatonly  drops initially-veg spreaders
       adopters  drops reverted agents too, so DV2's sample is DV1's successes

Feature extraction, predictors and standardisation are the shipped ones, via
kappa_two_dv, so the numbers are comparable with two_dv.csv.

Usage:
    python two_dv_sample_robustness.py <reduced_dir> --t-end 310000 [-o out.csv]
"""
import os, sys, glob, pickle, argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'plotting'))
from kappa_two_dv import rebuild_row, _std
import agency_predictor_analysis as apa
from agency_predictor_analysis import TOPO_PREDS, PSYCH_PREDS, ALL_PREDS

SPECS = [('full', ALL_PREDS), ('topo', TOPO_PREDS), ('psych', PSYCH_PREDS)]


def _b(frame):
    """log-log degree-amplification fit; isolated agents with credit poison log(0)."""
    f = frame[frame['degree'] > 0]
    m = sm.OLS(np.log(f['reduction_kg']), sm.add_constant(np.log(f['degree']))).fit()
    ci = m.conf_int().iloc[1]
    return m.params.iloc[1], ci[1] < 1, ci[0] < 1 < ci[1]


def _one(job):
    path, t_end = job
    with open(path, 'rb') as f:
        run = pickle.load(f)
    t, row = rebuild_row(run, t_end)
    df = apa.extract_features_fast(row)

    # schema is ("conv", t, i, partner, partner_diet, buffer) / ("rev", t, i)
    ever = np.zeros(len(df), dtype=bool)
    for ev in run['events']:
        if ev[0] == 'conv' and ev[1] <= t:
            ever[ev[2]] = True
    df['ever_conv'] = ever.astype(int)

    o = {'run': run['run'], 't': t}
    meat = df[df['init_meat'] == 1]
    pos = df[df['reduction_kg'] > 0].copy()
    pos['log_mult'] = np.log(pos['multiplier'])

    dv1 = {'shipped': (meat, 'adopted'), 'ever': (meat, 'ever_conv'),
           'noimmune': (meat[~meat['immune'].astype(bool)], 'adopted')}
    for k, (fr, y) in dv1.items():
        for lab, preds in SPECS:
            o[f'ado_{k}_{lab}'] = sm.Logit(fr[y], _std(fr, preds)).fit(disp=0).prsquared
        c = sm.Logit(fr[y], _std(fr, ALL_PREDS)).fit(disp=0)
        o[f'ado_{k}_n'], o[f'ado_{k}_pos'] = len(fr), int(fr[y].sum())
        o[f'ado_{k}_alpha'], o[f'ado_{k}_degree'] = c.params['alpha'], c.params['degree']
        o[f'ado_{k}_deg_sig'] = c.pvalues['degree'] < 0.05

    dv2 = {'shipped': pos, 'meatonly': pos[pos['init_meat'] == 1],
           'adopters': pos[pos['adopted'] == 1]}
    for k, fr in dv2.items():
        for lab, preds in SPECS:
            o[f'amp_{k}_{lab}'] = sm.OLS(fr['log_mult'], _std(fr, preds)).fit().rsquared
        c = sm.OLS(fr['log_mult'], _std(fr, ALL_PREDS)).fit()
        o[f'amp_{k}_n'] = len(fr)
        o[f'amp_{k}_degree'] = c.params['degree']
        o[f'amp_{k}_deg_sig'] = c.pvalues['degree'] < 0.05
        o[f'b_{k}'], o[f'b_{k}_sub'], o[f'b_{k}_spans1'] = _b(fr)
        o[f'ratio_{k}'] = o[f'amp_{k}_full'] / o['ado_shipped_full']
    o['ratio_matched'] = o['amp_adopters_full'] / o['ado_ever_full']

    print(f"INFO: {os.path.basename(path)} at t={t} done", flush=True)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reduced_dir')
    ap.add_argument('--t-end', type=int, required=True)
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--cores', type=int, default=max(1, int(0.75 * os.cpu_count())))
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl')))
    with Pool(min(a.cores, len(paths))) as pool:
        rows = pool.map(_one, [(p, a.t_end) for p in paths])

    df = pd.DataFrame(rows)
    out = a.out or os.path.join(a.reduced_dir, 'two_dv_sample_robustness.csv')
    df.to_csv(out, index=False)

    def q(c, fmt='{:.4f}'):
        v = df[c].astype(float)
        return (f"{fmt.format(v.median())} [{fmt.format(v.quantile(.25))}, "
                f"{fmt.format(v.quantile(.75))}]")

    print(f"\n{len(df)} runs at t={df['t'].unique()}")
    print("\n--- DV1 adoption pseudo-R2 ---")
    for k in ('shipped', 'ever', 'noimmune'):
        print(f"  {k:9s} n={df[f'ado_{k}_n'].median():.0f} y=1:{df[f'ado_{k}_pos'].median():.0f}"
              f"  full {q(f'ado_{k}_full')}  topo {q(f'ado_{k}_topo')}"
              f"  alpha {q(f'ado_{k}_alpha', '{:+.3f}')}")
    print("\n--- DV2 amplification R2 ---")
    for k in ('shipped', 'meatonly', 'adopters'):
        print(f"  {k:9s} n={df[f'amp_{k}_n'].median():.0f}"
              f"  full {q(f'amp_{k}_full')}  topo {q(f'amp_{k}_topo')}"
              f"  degree {q(f'amp_{k}_degree', '{:+.3f}')} sig {df[f'amp_{k}_deg_sig'].mean():.0%}")
    print("\n--- b, degree-amplification exponent ---")
    for k in ('shipped', 'meatonly', 'adopters'):
        print(f"  {k:9s} {q(f'b_{k}', '{:.3f}')}  sub-linear {df[f'b_{k}_sub'].sum()}/{len(df)}"
              f"  CI spans 1 {df[f'b_{k}_spans1'].sum()}/{len(df)}")
    print("\n--- ratio (DV2 variant over DV1 shipped) ---")
    for k in ('shipped', 'meatonly', 'adopters'):
        print(f"  {k:9s} {q(f'ratio_{k}', '{:.2f}')}")
    print(f"  matched   {q('ratio_matched', '{:.2f}')}   (DV1 ever / DV2 adopters)")
    print(f"INFO: wrote {out}")


if __name__ == '__main__':
    main()
