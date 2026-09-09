"""Veg-time accounting on the kappa ensemble (the reported headline since 2026-09-09).

A_i = vegetarian-time credited to i through the event graph / i's own vegetarian-time,
both truncated at t_end (attribution_ledger unit="time"; exposure parents, no dwell
weight, lambda 0.7). Per run: rank agreement with the event-count ledger, the initial
vegetarians' share of the tail, the own-time denominator, early/late split on the first
conversion, window sensitivity, lambda sensitivity. Writes vegtime_stats.csv and one
vegtime_A_run_XX.npz per run (full-length A, credit, own) into the reduced dir; the
npz files feed publication_plots_main.plot_amplification_ensemble(multipliers_dir=...).
Record: claude_stuff/Review/amplification_accounting_final_2026-09-09.md.

Usage:
    python vegtime_accounting.py <reduced_dir> [--t-end 310000] [--cores 3]
"""
import os, sys, glob, pickle, argparse, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_ledger import replay, veg_time, _concentration
TE = 310000
EV = dict(parent='exposure', weight='none', unit='event')
VT = dict(parent='exposure', weight='none', unit='time')

def stats(A):
    return dict(mean=A.mean(), median=np.median(A), p90=np.percentile(A, 90),
                p99=np.percentile(A, 99), max=A.max(), **_concentration(A))

def one(job):
    path, TE = job
    r = pickle.load(open(path, 'rb'))
    ev, d0, p = r['events'], r['initial_diets'], r['params']
    delta = p['meat_CO2'] - p['veg_CO2']; N = len(d0); act = 2 * N
    init_veg = np.array([d == 'veg' for d in d0])
    out = {'run': r['run']}
    own = veg_time(ev, d0, TE)
    ce = replay(ev, d0, p, t_end=TE, **EV); ct = replay(ev, d0, p, t_end=TE, **VT)
    m = ct > 0
    Ae = ce[m] / delta; At = ct[m] / (delta * own[m])
    out['n_credited'] = int(m.sum())
    out['rho_rank'] = spearmanr(Ae, At)[0]
    out['r_log'] = np.corrcoef(np.log(Ae), np.log(At))[0, 1]
    k = max(1, int(round(0.1 * m.sum())))
    idx = np.where(m)[0]
    top_e = set(idx[np.argsort(-Ae)[:k]]); top_t = set(idx[np.argsort(-At)[:k]])
    out['top10_jaccard'] = len(top_e & top_t) / len(top_e | top_t)
    k1 = max(1, int(round(0.01 * m.sum())))
    for lab, A in (('event', Ae), ('time', At)):
        o = np.argsort(-A)
        out[f'initveg_top1_{lab}'] = init_veg[idx[o[:k1]]].mean()
        out[f'initveg_top10_{lab}'] = init_veg[idx[o[:k]]].mean()
        out[f'initveg_share_{lab}'] = (A * init_veg[idx]).sum() / A.sum()
        out[f'mean_conv_{lab}'] = A[~init_veg[idx]].mean()
        out[f'mean_initveg_{lab}'] = A[init_veg[idx]].mean()
        out[f'median_conv_{lab}'] = np.median(A[~init_veg[idx]])
    # own-time denominator among credited agents
    oa = own[m] / act
    out['own_med_acts'] = np.median(oa); out['own_lt1_act'] = (oa < 1).mean()
    out['own_lt5_act'] = (oa < 5).mean(); out['own_frac_window_med'] = np.median(own[m]) / TE
    out['own_conv_med_acts'] = np.median(oa[~init_veg[idx]])
    out['sys_time'] = ct.sum() / (delta * own.sum())
    A_full = np.zeros(N); A_full[m] = At
    np.savez(path.replace('run_', 'vegtime_A_run_').replace('.pkl', '.npz'),
             t_end=TE, A=A_full, credit=ct, own=own)
    # share of population veg-time held by credited converters vs never-credited
    # early/late split on first conversion before the trajectory crosses 0.5
    traj = np.asarray(r['fraction_veg'], float)
    step = p['steps'] / (len(traj) - 1) if len(traj) > 1 else 1
    cr = np.where(traj >= 0.5)[0]; t_half = cr[0] * step if len(cr) else TE
    out['t_half'] = t_half
    first = {}
    for e in ev:
        if e[1] > TE: break
        if e[0] == 'conv' and e[2] not in first: first[e[2]] = e[1]
    fc = np.array([first.get(j, -1) for j in idx])
    early = (fc >= 0) & (fc < t_half); late = fc >= t_half
    for lab, A in (('event', Ae), ('time', At)):
        out[f'early_mean_{lab}'] = A[early].mean(); out[f'late_mean_{lab}'] = A[late].mean()
        out[f'early_med_{lab}'] = np.median(A[early]); out[f'late_med_{lab}'] = np.median(A[late])
    # window sensitivity
    for te in (200000, 250000, 310000, 350000, 400000):
        o2 = veg_time(ev, d0, te)
        c2 = replay(ev, d0, p, t_end=te, **VT); m2 = c2 > 0
        A2 = c2[m2] / (delta * o2[m2])
        out[f'w{te}_mean_time'] = A2.mean(); out[f'w{te}_med_time'] = np.median(A2)
        out[f'w{te}_sys_time'] = c2.sum() / (delta * o2.sum())
        out[f'w{te}_gini_time'] = _concentration(A2)['gini']
        c3 = replay(ev, d0, p, t_end=te, **EV); A3 = c3[c3 > 0] / delta
        out[f'w{te}_mean_event'] = A3.mean()
        n_conv = sum(1 for e in ev if e[0] == 'conv' and e[1] <= te)
        out[f'w{te}_sys_event'] = c3.sum() / (delta * n_conv)
    # lambda sensitivity, veg-time
    for lam in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        c4 = replay(ev, d0, p, t_end=TE, decay=lam, **VT); m4 = c4 > 0
        A4 = c4[m4] / (delta * own[m4])
        for kk, v in stats(A4).items(): out[f'lam{lam}_{kk}'] = v
        out[f'lam{lam}_sys'] = c4.sum() / (delta * own.sum())
    print('INFO', r['run'], flush=True)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reduced_dir'); ap.add_argument('--t-end', type=int, default=TE)
    ap.add_argument('--cores', type=int, default=3)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl')))
    with Pool(a.cores) as pool:
        rows = pool.map(one, [(p, a.t_end) for p in paths])
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.reduced_dir, 'vegtime_stats.csv'), index=False)
    def q(v): return f"{np.median(v):.3g} [{np.percentile(v,25):.3g}, {np.percentile(v,75):.3g}]"
    for c in df.columns:
        if c != 'run': print(f"{c:28s} {q(df[c])}")
