#!/usr/bin/env python3
"""Adversarial diagnostics for the post-peak gamma decline (scratch, not for commit).

Same continuation as gamma_convergence.py (seeds 42+run_id, twin N=2000) but records,
at every snapshot:
  gamma_w      positive-only OLS on the DWELL-WEIGHTED ledger (reduction_out)  [reproduces]
  gamma_unw    positive-only OLS on the UNWEIGHTED ledger (reduction_out_unw)
  band_w/unw   band-mean estimator (zeros kept) -- the estimator C6 says is unrecoverable
  gamma_frz    weighted positive-only, but regressed on degree FROZEN at t_ref=150k
  gamma_sup    weighted positive-only on the SUPPORT frozen at t_ref (current degree)
  wslope       OLS slope of log10(mean dwell weight) on log10(k) over credited agents
  deg drift    mean/sd/max degree, spearman(k_t, k_ref), isolate count
"""
import os, sys, time, argparse
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
from model_runner_mp import DEFAULT_PARAMS, get_model

DIRECT = 664.0
BANDS = [(3, 4), (5, 6), (7, 8), (9, 11), (12, 15), (16, 20), (21, 30), (31, 50), (51, 400)]
T_REF = 150000


def pos_slope(A, k):
    m = (A > 0) & (k > 0)
    if m.sum() < 20:
        return np.nan
    return np.polyfit(np.log10(k[m]), np.log10(A[m]), 1)[0]


def band_slope(A, k):
    kk, mu = [], []
    for lo, hi in BANDS:
        m = (k >= lo) & (k <= hi)
        if m.sum() < 5:
            continue
        kk.append(k[m].mean()); mu.append(A[m].mean())
    kk, mu = np.array(kk), np.array(mu)
    g = mu > 0
    if g.sum() < 3:
        return np.nan
    return np.polyfit(np.log10(kk[g]), np.log10(mu[g]), 1)[0]


def run_one(params):
    import random
    rid = params['run']; seed = 42 + rid
    np.random.seed(seed); random.seed(seed); params['seed'] = seed
    model = get_model(params)
    recs, state = [], {}
    tau_p = params['M'] * 2 * params['N']

    def summarise(t):
        G = model.G1
        ags = model.agents
        aw = np.array([a.reduction_out for a in ags], float) / DIRECT
        au = np.array([a.reduction_out_unw for a in ags], float) / DIRECT
        k = np.array([G.degree(a.i) for a in ags], float)
        if t >= T_REF and 'kref' not in state:
            state['kref'] = k.copy()
            state['sup'] = (aw > 0)
        kr = state.get('kref')
        # mean dwell weight per agent, as it stands now
        ct = np.array([a.change_time if a.change_time is not None else np.nan for a in ags], float)
        w = np.where(np.isnan(ct), np.nan, 1.0 - np.exp(-(t - ct) / tau_p))
        mw = (aw > 0) & (k > 0) & ~np.isnan(w) & (w > 0)
        wslope = (np.polyfit(np.log10(k[mw]), np.log10(w[mw]), 1)[0]
                  if mw.sum() >= 20 else np.nan)
        r = {'t': t, 'f_veg': model.fraction_veg[-1] if model.fraction_veg else np.nan,
             'n_pos_w': int(((aw > 0) & (k > 0)).sum()),
             'n_pos_unw': int(((au > 0) & (k > 0)).sum()),
             'gamma_w': pos_slope(aw, k), 'gamma_unw': pos_slope(au, k),
             'band_w': band_slope(aw, k), 'band_unw': band_slope(au, k),
             'mean_A_w': aw.mean(), 'mean_A_unw': au.mean(),
             'mean_A_w_pos': aw[aw > 0].mean() if (aw > 0).any() else np.nan,
             'wslope': wslope,
             'mean_w': np.nanmean(w), 'n_attached': int((~np.isnan(w)).sum()),
             'deg_mean': k.mean(), 'deg_sd': k.std(), 'deg_max': k.max(),
             'n_iso': int((k == 0).sum()),
             'gamma_frz': pos_slope(aw, kr) if kr is not None else np.nan,
             'sp_kref': spearmanr(k, kr).statistic if kr is not None else np.nan}
        if 'sup' in state:
            s = state['sup'] & (k > 0)
            r['gamma_sup'] = (np.polyfit(np.log10(k[s]), np.log10(aw[s]), 1)[0]
                              if s.sum() >= 20 else np.nan)
        else:
            r['gamma_sup'] = np.nan
        return r

    model.record_snapshot = lambda t: (recs.append(summarise(int(t)))
                                       if isinstance(t, (int, np.integer)) else None)
    t0 = time.time()
    model.run()
    recs.append(summarise(params['steps']))
    df = pd.DataFrame(recs).drop_duplicates('t', keep='last').sort_values('t')
    df['run'] = rid
    print(f"  run={rid} done gamma_w={df.gamma_w.iloc[-1]:.3f} "
          f"gamma_unw={df.gamma_unw.iloc[-1]:.3f} band_w={df.band_w.iloc[-1]:.3f} "
          f"{time.time()-t0:.0f}s", flush=True)
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--steps', type=int, default=800000)
    ap.add_argument('--cores', type=int, default=5)
    ap.add_argument('--out', default='../model_output/gamma_diag_800k.csv')
    a = ap.parse_args()
    p = DEFAULT_PARAMS.copy()
    p.update({'agent_ini': 'twin', 'N': 2000, 'steps': a.steps, 'snapshot_dense_start': 2000})
    with Pool(a.cores) as pool:
        frames = pool.map(run_one, [{**p, 'run': i} for i in range(a.runs)])
    pd.concat(frames, ignore_index=True).to_csv(a.out, index=False)
    print("INFO: wrote", a.out)
