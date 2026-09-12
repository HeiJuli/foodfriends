"""Early/late adopter amplification on the veg-time ledger (kappa ensemble, t = 310,000).

Re-measures manuscript-v2.tex :107/:117 (early/late means, per-cohort degree exponents,
"~1.4x per 20k steps") on the reported ledger. Inputs are vegtime_accounting.py's npz (A) and
the reduced run pickles (events for first-conversion time, the t_cut snapshot graph for degree,
as kappa_two_dv.rebuild_row). Cohorts as vegtime_accounting: first conversion before the run's
own F_veg = 0.5 crossing; initial vegetarians belong to neither cohort but are in overall b.
Timing regression as timing_topology_a4.continuous_timing, on bare A.

Usage: python early_late_vegtime.py <reduced_dir> [--t-cut 310000]
"""
import os, glob, pickle, argparse, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def run_table(path, t_cut):
    r = pickle.load(open(path, 'rb'))
    A = np.load(path.replace('run_', 'vegtime_A_run_').replace('.pkl', '.npz'))['A']
    snaps = r['snapshots']
    t = max(k for k in snaps if isinstance(k, int) and 'edges' in snaps[k] and k <= t_cut)
    s = snaps[t]
    deg = np.bincount(np.asarray(s['edges']).ravel(), minlength=s['n_nodes'])
    traj = np.asarray(r['fraction_veg'], float)
    step = r['params']['steps'] / (len(traj) - 1)
    cr = np.where(traj >= 0.5)[0]
    t_half = cr[0] * step if len(cr) else t_cut
    first = {}
    for e in r['events']:
        if e[1] > t_cut: break
        if e[0] == 'conv' and e[2] not in first: first[e[2]] = e[1]
    fc = np.array([first.get(j, -1) for j in range(len(A))], float)
    init_veg = np.array([d == 'veg' for d in r['initial_diets']])
    return pd.DataFrame(dict(run=r['run'], A=A, degree=deg, fc=fc, init_veg=init_veg,
                             early=(fc >= 0) & (fc < t_half), late=fc >= t_half,
                             t_half=t_half, snap_t=t))


def slope(g):
    g = g[(g.A > 0) & (g.degree > 0)]
    return stats.linregress(np.log(g.degree), np.log(g.A)).slope if len(g) >= 20 else np.nan


def q(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]"


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reduced_dir'); ap.add_argument('--t-cut', type=int, default=310000)
    a = ap.parse_args()
    D = pd.concat([run_table(p, a.t_cut) for p in
                   sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl')))], ignore_index=True)
    print(f"INFO: {D.run.nunique()} runs, snapshot t = {sorted(D.snap_t.unique())}, "
          f"t_half {q(D.groupby('run').t_half.first())}")
    P = D[D.A > 0]

    # levels, per run, both bare A and the reported 1 + A
    rows = []
    for rid, g in P.groupby('run'):
        e, l = g[g.early].A, g[g.late].A
        rows.append(dict(e=e.mean(), l=l.mean(), all=g.A.mean(), e1=1 + e.mean(), l1=1 + l.mean(),
                         all1=1 + g.A.mean(), em=e.median(), lm=l.median(),
                         b_all=slope(g), b_e=slope(g[g.early]), b_l=slope(g[g.late]),
                         deg_e=g[g.early].degree.mean(), deg_l=g[g.late].degree.mean(),
                         n_e=len(e), n_l=len(l)))
    R = pd.DataFrame(rows)
    print("\nLEVELS (median [IQR] over runs, credited agents)")
    for lab, c in [('bare A  early', 'e'), ('bare A  late', 'l'), ('bare A  all', 'all'),
                   ('1+A     early', 'e1'), ('1+A     late', 'l1'), ('1+A     all', 'all1'),
                   ('median A early', 'em'), ('median A late', 'lm'),
                   ('mean degree early', 'deg_e'), ('mean degree late', 'deg_l'),
                   ('n early', 'n_e'), ('n late', 'n_l')]:
        print(f"  {lab:18s} {q(R[c])}")
    g = R.e / R.l; g1 = R.e1 / R.l1
    print(f"  gap bare A         {q(g)}  (>1 in {(g > 1).sum()}/{len(g)})")
    print(f"  gap 1+A            {q(g1)}  (>1 in {(g1 > 1).sum()}/{len(g1)})")

    print("\nDEGREE EXPONENT b (per-run log-log, bare A)")
    for lab, c in [('all credited', 'b_all'), ('early', 'b_e'), ('late', 'b_l')]:
        print(f"  {lab:14s} {q(R[c])}")
    d = (R.b_e - R.b_l).dropna()
    print(f"  early - late   {q(d)}  (>0 in {(d > 0).sum()}/{len(d)}, "
          f"Wilcoxon p={stats.wilcoxon(d).pvalue:.1e})")

    # continuous switch time, converters only
    C = P[(P.fc > 0) & (P.degree > 0)].copy()
    C['ld'] = np.log(C.degree); C['la'] = np.log(C.A); C['sw'] = C.fc / 1000
    C['lex'] = np.log((a.t_cut - C.fc).clip(lower=1))
    for c in ('ld', 'sw', 'lex'): C[c + 'c'] = C[c] - C[c].mean()
    print(f"\nCONTINUOUS TIMING (converters, n={len(C)}, SEs clustered by run)")
    for f in ('la ~ ldc*swc', 'la ~ ldc*swc + lexc'):
        m = smf.ols(f, data=C).fit(cov_type='cluster', cov_kwds={'groups': C.run})
        print(f"  {f}   R2={m.rsquared:.3f}")
        for n in m.params.index:
            print(f"    {n:10s} b={m.params[n]:+.5f}  z={m.tvalues[n]:+7.1f}")
    b = smf.ols('la ~ ldc*swc', data=C).fit().params
    print("  bare-A ratio for converting 20k steps (5 sweeps) earlier:")
    for k in (4, 8, 16, 40):
        print(f"    degree {k:2d}: {np.exp(-20 * (b['swc'] + b['ldc:swc'] * (np.log(k) - C.ld.mean()))):.3f}x")
