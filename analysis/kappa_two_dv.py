#!/usr/bin/env python3
"""Two-DV re-measurement (plan section 5) on the kappa=0.55 headline ensemble.

Repeats the N=385 smoketest of section 5a at the reported population size, per
run rather than on a pooled frame: adoption pseudo-R2, amplification R2, their
ratio, the topology/psychology split, the standardised coefficients, the
degree-amplification exponent b with its measurement point, and the network
statistics.

Feature extraction is the shipped `plotting.agency_predictor_analysis.
extract_features_fast`, called on an adapter that rebuilds a networkx graph and
the diet strings from the reduced per-run artefacts (reduce_ensemble.py); the
predictor set, standardisation and model specifications are the shipped ones,
so the numbers are comparable with the submitted analysis. The fits are
re-run here rather than scraped because the shipped functions print.

b is measured AT a stated snapshot time and rises through a run: never quote it
without one, and never measure it past t_end (section 5a).

Usage:
    python kappa_two_dv.py <reduced_dir> --t-end 300000 [-o out.csv]
"""
import os, sys, glob, pickle, argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
from scipy.stats import spearmanr, pointbiserialr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'plotting'))
import agency_predictor_analysis as apa
from agency_predictor_analysis import TOPO_PREDS, PSYCH_PREDS, ALL_PREDS


def rebuild_row(run, t_cut):
    """Reduced artefact -> the row shape extract_features_fast expects."""
    snaps = run['snapshots']
    times = [t for t in snaps if isinstance(t, int) and 'edges' in snaps[t]]
    t = max([x for x in times if x <= t_cut], default=max(times))
    s = snaps[t]

    G = nx.Graph()
    G.add_nodes_from(range(s['n_nodes']))
    G.add_edges_from(map(tuple, s['edges']))
    nx.set_node_attributes(G, dict(enumerate(s['node_theta'].astype(float))), 'theta')

    def snap(d, diets):
        return {'graph': G, 'reductions': d['reductions'], 'alphas': d['alphas'],
                'rhos': d['rhos'], 'immune': d['immune'].astype(bool),
                'direct_conversions': d['direct_conversions'], 'diets': diets}

    diets = np.where(s['diets'].astype(bool), 'veg', 'meat')
    return t, {'snapshots': {0: {'diets': list(run['initial_diets'])},
                             t: snap(s, diets)}}


def _std(frame, preds):
    return sm.add_constant((frame[preds] - frame[preds].mean()) / frame[preds].std())


def measure(df, t, run_id):
    meat = df[df['init_meat'] == 1].copy()
    pos = df[df['reduction_kg'] > 0].copy()
    pos['log_mult'] = np.log(pos['multiplier'])
    out = {'run': run_id, 't': t, 'n_meat': len(meat),
           'n_adopted': int(meat['adopted'].sum()), 'n_pos': len(pos)}

    for label, preds in [('psych', PSYCH_PREDS), ('topo', TOPO_PREDS), ('full', ALL_PREDS)]:
        out[f'adopt_pr2_{label}'] = sm.Logit(
            meat['adopted'], _std(meat, preds)).fit(disp=0).prsquared
        out[f'amp_r2_{label}'] = sm.OLS(
            pos['log_mult'], _std(pos, preds)).fit().rsquared
    for label in ('psych', 'topo', 'full'):
        out[f'ratio_{label}'] = out[f'amp_r2_{label}'] / out[f'adopt_pr2_{label}']

    # standardised coefficients and per-predictor marginals
    amp_full = sm.OLS(pos['log_mult'], _std(pos, ALL_PREDS)).fit()
    ado_full = sm.Logit(meat['adopted'], _std(meat, ALL_PREDS)).fit(disp=0)
    for p in ALL_PREDS:
        out[f'amp_coef_{p}'] = amp_full.params[p]
        out[f'amp_sig_{p}'] = amp_full.pvalues[p] < 0.05
        out[f'ado_coef_{p}'] = ado_full.params[p]
        out[f'ado_sig_{p}'] = ado_full.pvalues[p] < 0.05
        rho, pv = spearmanr(pos[p], pos['log_mult'])
        out[f'amp_rho_{p}'], out[f'amp_rho_sig_{p}'] = rho, pv < 0.05
        r, pv = pointbiserialr(meat['adopted'], meat[p])
        out[f'ado_rpb_{p}'], out[f'ado_rpb_sig_{p}'] = r, pv < 0.05

    # degree-amplification exponent (isolated agents with credit poison log(0))
    fit = pos[pos['degree'] > 0]
    out['n_isolated_with_credit'] = len(pos) - len(fit)
    m = sm.OLS(np.log(fit['reduction_kg']), sm.add_constant(np.log(fit['degree']))).fit()
    ci = m.conf_int().iloc[1]
    out['b'], out['b_lo'], out['b_hi'] = m.params.iloc[1], ci[0], ci[1]
    out['b_sublinear'] = ci[1] < 1
    out['b_superlinear'] = ci[0] > 1
    return out


def _one(job):
    path, t_end = job
    with open(path, 'rb') as f:
        run = pickle.load(f)
    t, row = rebuild_row(run, t_end)
    feats = apa.extract_features_fast(row)
    G = row['snapshots'][t]['graph']
    net = {'run': run['run'], 't': t,
           'assortativity': nx.degree_assortativity_coefficient(G),
           'clustering': nx.average_clustering(G),
           'transitivity': nx.transitivity(G),
           'mean_degree': np.mean([d for _, d in G.degree()])}
    print(f"INFO: {os.path.basename(path)} at t={t} done", flush=True)
    return measure(feats, t, run['run']), net


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reduced_dir')
    ap.add_argument('--t-end', type=int, required=True,
                    help='analysis cutoff; the nearest graph-bearing snapshot at or '
                         'below it is used')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--cores', type=int, default=max(1, int(0.75 * os.cpu_count())))
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl')))
    # exact betweenness at N=2000 is minutes a graph; the runs are independent
    with Pool(min(a.cores, len(paths))) as pool:
        out = pool.map(_one, [(p, a.t_end) for p in paths])
    rows, net = [r for r, _ in out], [n for _, n in out]

    df, ndf = pd.DataFrame(rows), pd.DataFrame(net)
    out = a.out or os.path.join(a.reduced_dir, 'two_dv.csv')
    df.to_csv(out, index=False)
    ndf.to_csv(out.replace('.csv', '_network.csv'), index=False)

    def q(v, fmt='{:.4g}'):
        v = np.asarray(v, dtype=float)
        return (f"{fmt.format(np.median(v))}  IQR [{fmt.format(np.percentile(v, 25))}, "
                f"{fmt.format(np.percentile(v, 75))}]")

    print(f"\n{len(df)} runs at t={df['t'].unique()}, "
          f"n_meat={df['n_meat'].median():.0f}, n_pos={df['n_pos'].median():.0f}")
    for k in ['adopt_pr2_full', 'adopt_pr2_topo', 'adopt_pr2_psych',
              'amp_r2_full', 'amp_r2_topo', 'amp_r2_psych',
              'ratio_full', 'ratio_topo', 'b']:
        print(f"{k:20s} {q(df[k])}")
    print(f"{'b sub-linear':20s} {df['b_sublinear'].sum()}/{len(df)} runs "
          f"(CI excludes 1 from below)")
    print(f"{'amp R2 > adopt pR2':20s} "
          f"{(df['amp_r2_full'] > df['adopt_pr2_full']).sum()}/{len(df)} runs")
    print("\npredictor        amp_coef (frac sig)      ado_coef (frac sig)")
    for p in ALL_PREDS:
        print(f"  {p:<14} {df[f'amp_coef_{p}'].median():>7.3f} "
              f"({df[f'amp_sig_{p}'].mean():>4.0%})        "
              f"{df[f'ado_coef_{p}'].median():>7.3f} ({df[f'ado_sig_{p}'].mean():>4.0%})")
    print(f"\nnetwork at t: assortativity {q(ndf['assortativity'])}, "
          f"clustering {q(ndf['clustering'])}, mean degree {q(ndf['mean_degree'])}")
    print(f"INFO: wrote {out}")


if __name__ == '__main__':
    main()
