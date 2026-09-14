"""Depth-resolved system credit on the kappa ensemble.

Share of veg-time system credit paid at cascade depth d = 1, 2, 3, >=4, and the system
ratio with every depth, depth 1 only, and depth <= 3, at the headline window. Exposure
parents, no dwell weight, lambda 0.7 (vegtime_accounting.VT). A depth cap changes only
what is paid, not the walk, so the capped system ratio is a partial sum of the histogram;
a per-agent capped A would need a cap inside replay and is not computed here.

Usage:
    python depth_share.py <reduced_dir> [--t-end 310000] [--cores 3]
"""
import os, sys, glob, pickle, argparse, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_ledger import replay, veg_time
VT = dict(parent='exposure', weight='none', unit='time')

def one(job):
    path, te = job
    r = pickle.load(open(path, 'rb'))
    ev, d0, p = r['events'], r['initial_diets'], r['params']
    by = {}
    tot = replay(ev, d0, p, t_end=te, by_depth=by, **VT).sum()
    assert np.isclose(sum(by.values()), tot)
    own = (p['meat_CO2'] - p['veg_CO2']) * veg_time(ev, d0, te).sum()
    part = lambda lo, hi: sum(v for d, v in by.items() if lo <= d <= hi)
    return {'run': r['run'], 'max_depth': max(by),
            'share_d1': part(1, 1) / tot, 'share_d2': part(2, 2) / tot,
            'share_d3': part(3, 3) / tot, 'share_d4plus': part(4, 10**9) / tot,
            'sys': tot / own, 'sys_d1': part(1, 1) / own, 'sys_cap3': part(1, 3) / own}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('reduced_dir')
    ap.add_argument('--t-end', type=int, default=310000)
    ap.add_argument('--cores', type=int, default=3)
    a = ap.parse_args()
    jobs = [(f, a.t_end) for f in sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl')))]
    with Pool(a.cores) as pool:
        df = pd.DataFrame(pool.map(one, jobs))
    df.to_csv(os.path.join(a.reduced_dir, f'depth_share_{a.t_end}.csv'), index=False)
    s = df.drop(columns='run')
    print(pd.DataFrame({'mean': s.mean(), 'q25': s.quantile(0.25), 'median': s.median(),
                        'q75': s.quantile(0.75)}).round(4).to_string())
