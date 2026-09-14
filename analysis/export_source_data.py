"""Write the numerical source data behind every main-figure panel to source_data/.

One CSV per panel (or panel group), read straight from the artefacts the plotting
scripts consume, so the deposited numbers are the plotted numbers. Per-agent
survey parameters (theta, rho, alpha) are not exported: no panel plots them.
Fig. 5 exports binned counts, the only form of the LISS variables we may release.

Usage: python export_source_data.py    (from analysis/, venv active)
"""
import glob, json, os, pickle
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'source_data')
RED = os.path.join(ROOT, 'model_output', 'trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced')
SMALL = os.path.join(ROOT, 'model_output', 'trajectory_sample-max_20260903_kappa0p55_100k.pkl')
CASCADE = os.path.join(ROOT, 'model_output', 'cascade_overview_run_twin_N2000_kappa0p55_seed42_20260910.pkl')
DTA = os.path.join(ROOT, 'data', 'data_construction_paper')
TRUNCATE, DECIMATE = 350000, 100          # plot_config.yaml big_n.truncate_steps
SMALL_MID, SMALL_END = 30000, 56000       # plot_config.yaml small_n.mid_t / analysis_t_end
os.makedirs(OUT, exist_ok=True)


def fig1a_trajectories():
    runs = sorted(glob.glob(os.path.join(RED, 'run_*.pkl')))
    cols = {}
    for p in runs:
        r = pickle.load(open(p, 'rb'))
        cols[f"run_{r['run']:02d}"] = r['fraction_veg'][:TRUNCATE + 1:DECIMATE]
    df = pd.DataFrame(cols)
    df.insert(0, 't', np.arange(0, TRUNCATE + 1, DECIMATE))
    df.to_csv(os.path.join(OUT, 'fig1a_fveg_trajectories.csv'), index=False, float_format='%.5f')
    print(f"fig1a: {len(runs)} runs, {len(df)} rows (every {DECIMATE} steps to {TRUNCATE})")


def fig1bc_fig3_credit():
    rows = []
    for p in sorted(glob.glob(os.path.join(RED, 'vegtime_A_run_*.npz'))):
        z = np.load(p)
        run = int(p.rsplit('_', 1)[1].split('.')[0])
        rows.append(pd.DataFrame({'run': run, 'agent': np.arange(len(z['A'])), 't_end': int(z['t_end']),
                                  'A': z['A'], 'credit': z['credit'], 'own': z['own']}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(OUT, 'fig1bc_fig3_vegtime_credit_per_agent.csv'), index=False, float_format='%.6g')
    pd.read_csv(os.path.join(RED, 'vegtime_stats.csv')).to_csv(os.path.join(OUT, 'fig3_vegtime_stats_per_run.csv'), index=False)
    print(f"fig1bc/fig3: {df.run.nunique()} runs x {df.agent.nunique()} agents")


def fig1_networks():
    df = pd.read_pickle(SMALL)
    row = df[df['is_median_twin']].iloc[0] if df.get('is_median_twin', pd.Series(False)).any() else df.iloc[len(df) // 2]
    snaps = row['snapshots']
    times = sorted(t for t in snaps if isinstance(t, int))
    picks = {'initial': 0, 'mid': min(times, key=lambda t: abs(t - SMALL_MID)),
             'final': min(times, key=lambda t: abs(t - SMALL_END))}
    for label, t in picks.items():
        s = snaps[t]
        edges = s['edges'] if 'edges' in s else np.array(list(s['graph'].edges()))
        pd.DataFrame(edges, columns=['source', 'target']).to_csv(
            os.path.join(OUT, f'fig1_network_{label}_t{t}_edges.csv'), index=False)
        pd.DataFrame({'agent': np.arange(len(s['diets'])), 'diet': s['diets'],
                      'direct_conversions': s['direct_conversions'], 'immune': s['immune']}).to_csv(
            os.path.join(OUT, f'fig1_network_{label}_t{t}_nodes.csv'), index=False)
    ev = pd.DataFrame([(e[0], e[1], e[2]) for e in row['events']], columns=['kind', 't', 'agent'])
    ev.to_csv(os.path.join(OUT, 'fig1_network_run_events.csv'), index=False)
    print(f"fig1 networks: run {row['run']} snapshots {picks}, {len(ev)} events")


def fig2_cascade():
    d = pd.read_pickle(CASCADE)
    rows = []
    for k, e in enumerate(d['events']):
        if e[0] == 'rev':
            rows.append((k, 'rev', e[1], e[2], None, None, None))
        else:
            rows.append((k, 'conv', e[1], e[2], e[3], e[4], json.dumps([list(b) for b in e[5]])))
    pd.DataFrame(rows, columns=['event', 'kind', 't', 'agent', 'partner', 'partner_diet',
                                'memory_buffer_diet_source_t']).to_csv(
        os.path.join(OUT, 'fig2_cascade_run_events.csv'), index=False)
    pd.DataFrame({'agent': np.arange(len(d['initial_diets'])), 'initial_diet': d['initial_diets']}).to_csv(
        os.path.join(OUT, 'fig2_cascade_run_initial_diets.csv'), index=False)
    json.dump({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in d['params'].items()},
              open(os.path.join(OUT, 'fig2_cascade_run_params.json'), 'w'), indent=1)
    print(f"fig2: {len(rows)} events, seed {d['params'].get('seed')}")


def fig4_two_dv():
    for name in ('two_dv_vegtime.csv', 'two_dv_vegtime_network.csv'):
        p = os.path.join(RED, name)
        if os.path.exists(p):
            pd.read_csv(p).to_csv(os.path.join(OUT, 'fig4_' + name), index=False)
    print("fig4: per-run regression summaries copied")


def fig5_histograms():
    # Mirrors data/data_analysis/parameter_distributions_paper.py exactly.
    theta = pd.to_numeric(pd.read_stata(os.path.join(DTA, 'theta_diet.dta'))['theta'], errors='coerce').dropna().values
    alpha = pd.to_numeric(pd.read_stata(os.path.join(DTA, 'alpha.dta'))['alpha'], errors='coerce').dropna().values
    rho = pd.to_numeric(pd.read_stata(os.path.join(DTA, 'rho.dta'))['rho'], errors='coerce').dropna().values
    dens, edges = np.histogram(theta, bins=40, density=True)
    counts, _ = np.histogram(theta, bins=40)
    pd.DataFrame({'bin_left': edges[:-1], 'bin_right': edges[1:], 'count': counts, 'density': dens}).to_csv(
        os.path.join(OUT, 'fig5a_theta_histogram.csv'), index=False, float_format='%.6g')
    for lab, v in (('b_rho', rho), ('c_alpha', alpha)):
        u, c = np.unique(v, return_counts=True)
        pd.DataFrame({'value': u, 'count': c, 'proportion': c / c.sum()}).to_csv(
            os.path.join(OUT, f'fig5{lab}_bar.csv'), index=False, float_format='%.6g')
    best = None
    for name in ('skewnorm', 'norm'):
        dist = getattr(stats, name); prm = dist.fit(theta)
        aic = 2 * len(prm) - 2 * np.sum(dist.logpdf(theta, *prm))
        if best is None or aic < best[0]:
            best = (aic, name, prm)
    summ = {'theta': dict(n=len(theta), mean=theta.mean(), sd=theta.std(), fit=best[1], fit_params=list(map(float, best[2]))),
            'rho': dict(n=len(rho), mean=rho.mean(), sd=rho.std()),
            'alpha': dict(n=len(alpha), mean=alpha.mean(), sd=alpha.std())}
    json.dump(summ, open(os.path.join(OUT, 'fig5_summary.json'), 'w'), indent=1, default=float)
    print(f"fig5: theta n={len(theta)}, rho n={len(rho)}, alpha n={len(alpha)}, fit {best[1]}")


if __name__ == '__main__':
    fig1a_trajectories(); fig1bc_fig3_credit(); fig1_networks(); fig2_cascade(); fig4_two_dv(); fig5_histograms()
