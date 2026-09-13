"""Transient shape per topology arm: does structure change the path to the (topology-free) endpoint?

Reads any mix of
  - topology_comparison.py per-arm frames  trajectory_analysis_sample-max_<arm>_<date>.pkl
  - its combined list                      topology_comparison_smax_<date>.pkl  (all arms)
  - a reduced ensemble dir                 .../run_XX.pkl (the N=2000 record)
  - topology_transient_sim.py output dir   .../<ARM>_<seed>.pkl
and writes one row per run of crossing times and shape measures, plus a paired summary
against a reference arm. Runs are paired on seed (every script uses 42 + i); the pairing is
weak once networks differ, so Mann-Whitney is printed beside the signed-rank test.

Usage (repo root):
  python analysis/topology_transient_metrics.py prod=model_output/trajectory_analysis_sample-max_prod_20260912.pkl \
      ER=model_output/trajectory_analysis_sample-max_ER_20260912.pkl ... --ref prod --out model_output/topo_transient_20260912.csv
  python analysis/topology_transient_metrics.py all=model_output/topology_comparison_smax_20260912.pkl --ref homophilic_emp
"""
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t_end_logistic import fc_window
from results_analysis import _smooth_derivs

LEVELS = (0.10, 0.20, 0.35, 0.50, 0.65)


def load(label, path):
    """Yield (arm, seed, trajectory as float array) without holding a whole 300 MB pickle twice."""
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, '*.pkl'))):
            d = pd.read_pickle(f)
            if 'counts' in d:                                    # topology_transient_sim output
                yield d['arm'], d['seed'], d['counts'] / d['N']
            elif 'fraction_veg' in d:                            # reduced ensemble run
                yield label, d['params']['seed'], np.asarray(d['fraction_veg'], float)
        return
    d = pd.read_pickle(path)
    rows = d.to_dict('records') if isinstance(d, pd.DataFrame) else d
    for r in rows:
        yield r.get('topology', label) if label == 'all' else label, r['seed'], \
            np.asarray(r['fraction_veg_trajectory'], float)


def metrics(F):
    n = len(F)
    # crossings on a 1%-of-run running mean: F moves by 1/N per step and a raw first
    # passage is set by noise near the level
    w = max(1, n // 100)
    sm = np.convolve(F, np.ones(w) / w, 'valid')
    off = w // 2
    out = dict(steps=n - 1, F0=F[0], F_end=F[-max(1, n // 20):].mean())
    for x in LEVELS:
        hit = np.flatnonzero(sm >= x)
        out[f't_{x:.2f}'] = hit[0] + off if len(hit) else np.nan
    for q in (0.125, 0.25, 0.5):
        out[f'F_at_{q:g}run'] = F[int(q * (n - 1))]
    # early growth: doubling from 0.10 to 0.20 (the initial 0.06 sits inside the noise)
    out['t_0.10_to_0.20'] = out['t_0.20'] - out['t_0.10']
    out['t_0.20_to_0.65'] = out['t_0.65'] - out['t_0.20']
    # F_c and peak slope with the estimator of record (results_analysis.analysis_5)
    win = fc_window(n)
    r = _smooth_derivs(F, win, win)
    if r is not None:
        tix, s, d1, d2 = r
        i1 = np.argmax(d1)
        d2m = np.where(s > 0.5, 0, d2)
        i2 = np.argmax(d2m)
        out.update(peak_slope_per1k=d1[i1] * 1000, F_peak_slope=s[i1], t_peak_slope=tix[i1],
                   F_c=s[i2] if d2m[i2] > 0 else np.nan, t_c=tix[i2] if d2m[i2] > 0 else np.nan)
    return out


def summarise(df, ref):
    cols = [c for c in df.columns if c not in ('arm', 'seed')]
    q = lambda x: f'{np.nanmedian(x):.4g} [{np.nanpercentile(x, 25):.4g}, {np.nanpercentile(x, 75):.4g}]'
    rows = []
    for arm, g in df.groupby('arm'):
        for c in cols:
            row = dict(arm=arm, measure=c, n=g[c].notna().sum(), median_IQR=q(g[c]))
            if arm != ref:
                p = df[df.arm == ref].set_index('seed')[c].rename('ref').to_frame().join(
                    g.set_index('seed')[c].rename('arm'), how='inner').dropna()
                a, b = g[c].dropna(), df.loc[df.arm == ref, c].dropna()
                if len(p) >= 5 and (p.arm - p.ref).abs().sum() > 0:
                    row.update(paired_n=len(p), med_diff=np.median(p.arm - p.ref),
                               p_wilcoxon=wilcoxon(p.arm, p.ref).pvalue)
                if len(a) >= 5 and len(b) >= 5:
                    row['p_mannwhitney'] = mannwhitneyu(a, b).pvalue
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='LABEL=path (LABEL "all" keeps the topology field)')
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True, help='per-run CSV; the summary goes to *_summary.csv')
    # F_c's window is 20% of run length: a run far longer than its own t_end loses F_c (the argmax
    # lands on the mask edge), so N=385 arms are read at 100k, the length t_end_logistic pairs with N=2000 at 400k
    ap.add_argument('--truncate', type=int, default=0, help='analyse only the first T steps (0 = whole run)')
    a = ap.parse_args()
    rows = []
    for spec in a.inputs:
        label, path = spec.split('=', 1)
        for arm, seed, F in load(label, path):
            rows.append(dict(arm=arm, seed=seed, **metrics(F[:a.truncate + 1] if a.truncate else F)))
            print(f'INFO: {arm} seed {seed} F_c {rows[-1].get("F_c", np.nan):.3f} t_0.50 {rows[-1]["t_0.50"]}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    s = summarise(df, a.ref)
    s.to_csv(a.out.replace('.csv', '_summary.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(s.to_string(index=False))
