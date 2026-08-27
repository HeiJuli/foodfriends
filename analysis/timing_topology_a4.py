#!/usr/bin/env python3
"""A4: does timing multiply with topology, or is the early/late gap accounting?

Analysis A4 of claude_stuff/Review/tipping_reframe_plan_2026-08-18.md. A1-A3 live in
the existing scripts (plotting/explore_derivatives.py, analysis/results_analysis.py
analysis_5_inflection, analysis/t_end_logistic.py compare_logistic_linear); only A4 is new.

  A4a  re-attribute with the dwell weight w == 1 and recompute the early/late gap.
       Needs 'reductions_unw' in the snapshots -- the passive second ledger added to
       model_main._cascade_attribute. It writes to no dynamical variable, so the run
       is bit-identical to the weighted one.
  A4b  degree- and betweenness-amplification log-log slopes fitted separately per cohort.
  A4c  matched-degree early/late comparison, bootstrap CIs (descriptive supplement).

Usage: python timing_topology_a4.py [pkl]
"""
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

PKL = '../model_output/trajectory_analysis_twin_20260820.pkl'
ANALYSIS_T = 139000          # like-for-like with regeneration_results_2026-08-20
DIRECT_REDUCTION_KG = 664
BTW_PIVOTS = 400             # approximate betweenness; exact is O(NM) per graph


def hdr(s):
    print(f"\n{'='*66}\n  {s}\n{'='*66}")


def _resolve_snap(snaps, t=ANALYSIS_T):
    ok = [k for k in snaps if isinstance(k, (int, np.integer)) and k <= t]
    return (snaps[max(ok)], max(ok)) if ok else (snaps['final'], 'final')


def _cohorts(row, snap):
    """Boolean early mask: switched at or before the run's own F_veg=0.5 crossing."""
    traj = np.asarray(row['fraction_veg_trajectory'], dtype=float)
    crossed = np.where(traj >= 0.5)[0]
    t_half = int(crossed[0]) if len(crossed) else len(traj)
    ct = np.array([np.nan if c is None else c for c in snap['change_times']], dtype=float)
    return (ct > 0) & (ct <= t_half), ct, t_half


def agent_table(df, t_cut=ANALYSIS_T, red_key='reductions', need_between=True):
    """One row per (run, agent): amplification, cohort, degree, betweenness.

    red_key='reductions_unw' repeats the analysis on the w==1 ledger (A4a x A4b).
    need_between=False skips betweenness, which dominates the runtime.
    """
    out = []
    for i, (_, row) in enumerate(df.iterrows()):
        snap, key = _resolve_snap(row['snapshots'], t_cut)
        reds = np.array(snap[red_key], dtype=float)
        early, ct, _ = _cohorts(row, snap)
        G = snap['graph']
        nodes = list(G.nodes())
        bc = (nx.betweenness_centrality(G, k=min(BTW_PIVOTS, len(nodes)), seed=1)
              if need_between else {n: 0.0 for n in nodes})
        out.append(pd.DataFrame(dict(
            run=i, node=nodes, red=reds, amp=reds / DIRECT_REDUCTION_KG, ct=ct,
            degree=[float(G.degree(n)) for n in nodes],
            betweenness=[bc[n] for n in nodes], early=early)))
        if (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(df)} runs (snapshot {key})")
    return pd.concat(out, ignore_index=True)


def _loglog_slope(x, y, nmin=20):
    m = (x > 0) & (y > 0)
    if m.sum() < nmin:
        return np.nan
    return stats.linregress(np.log(x[m]), np.log(y[m])).slope


def a4a_unweighted(df):
    hdr("A4a  early/late gap with the dwell weight stripped (w == 1)")
    snap, _ = _resolve_snap(df.iloc[0]['snapshots'])
    if 'reductions_unw' not in snap:
        print("  SKIP: snapshots carry no 'reductions_unw'. Rerun the ensemble on the")
        print("  current model_main.py -- the unweighted ledger is recorded alongside")
        print("  'reductions' and changes nothing dynamical.")
        return None
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        s, _ = _resolve_snap(row['snapshots'])
        early, _, _ = _cohorts(row, s)
        for tag, key in [('weighted', 'reductions'), ('unweighted (w=1)', 'reductions_unw')]:
            r = np.array(s[key], dtype=float) / DIRECT_REDUCTION_KG
            m = r * DIRECT_REDUCTION_KG > 1e-3   # same kg threshold as A4b/A4c
            rows.append(dict(run=i, ledger=tag, credited=int(m.sum()),
                             early=np.mean(r[m & early]) if (m & early).sum() else np.nan,
                             late=np.mean(r[m & ~early]) if (m & ~early).sum() else np.nan,
                             overall=np.mean(r[m])))
    d = pd.DataFrame(rows)
    d['gap'] = d.early / d.late
    for tag, g in d.groupby('ledger', sort=False):
        print(f"  {tag:18s} early={g.early.median():6.2f}x  late={g.late.median():6.2f}x  "
              f"gap={g.gap.median():5.2f}  IQR=[{g.gap.quantile(.25):.2f}, {g.gap.quantile(.75):.2f}]"
              f"  overall={g.overall.median():.2f}x  credited={g.credited.median():.0f}")
    w = d[d.ledger == 'weighted'].set_index('run')
    u = d[d.ledger != 'weighted'].set_index('run')
    diff = w.gap - u.gap
    t = stats.wilcoxon(w.gap.values, u.gap.values)
    print(f"\n  paired gap difference (weighted - unweighted): median={diff.median():+.2f}  "
          f"central 95% across runs [{np.percentile(diff.dropna(),2.5):+.2f}, "
          f"{np.percentile(diff.dropna(),97.5):+.2f}]  Wilcoxon p={t.pvalue:.2e}")
    print(f"  share of the weighted gap that survives w=1: "
          f"{(u.gap.median()-1)/(w.gap.median()-1)*100:.0f}%")
    ug = u.gap.dropna().values
    print(f"  unweighted gap > 1 in {(ug > 1).sum()}/{len(ug)} runs;  "
          f"one-sided Wilcoxon vs 1: p={stats.wilcoxon(ug - 1, alternative='greater').pvalue:.2e}")
    return d


def a4b_interaction(A, preds=('degree', 'betweenness'), label=''):
    hdr(f"A4b  timing x topology interaction (per-run log-log slopes by cohort){label}")
    pos = A[A.red > 1e-3]
    for pred in preds:
        e, l = [], []
        for _, g in pos.groupby('run'):
            ge, gl = g[g.early], g[~g.early]
            e.append(_loglog_slope(ge[pred].values, ge.amp.values))
            l.append(_loglog_slope(gl[pred].values, gl.amp.values))
        e, l = np.asarray(e, float), np.asarray(l, float)
        keep = ~(np.isnan(e) | np.isnan(l))
        e, l = e[keep], l[keep]
        d = e - l
        p = stats.wilcoxon(e, l).pvalue
        print(f"\n  {pred}")
        print(f"    b_early median={np.median(e):+.3f}  "
              f"IQR=[{np.percentile(e,25):+.3f}, {np.percentile(e,75):+.3f}]")
        print(f"    b_late  median={np.median(l):+.3f}  "
              f"IQR=[{np.percentile(l,25):+.3f}, {np.percentile(l,75):+.3f}]")
        # NB: this is the spread ACROSS runs, not a CI on the median -- a bootstrap CI
        # on the median is far tighter (48-50/50 runs positive, p~1e-15)
        boot = np.array([np.median(np.random.default_rng(k).choice(d, len(d)))
                         for k in range(2000)])
        print(f"    paired difference median={np.median(d):+.3f}  "
              f"bootstrap 95% CI on median [{np.percentile(boot,2.5):+.3f}, "
              f"{np.percentile(boot,97.5):+.3f}]")
        print(f"    central 95% across runs [{np.percentile(d,2.5):+.3f}, "
              f"{np.percentile(d,97.5):+.3f}]  n={len(d)}  Wilcoxon p={p:.2e}")
        print(f"    runs with b_early > b_late: {(d > 0).sum()}/{len(d)}")

    print("\n  Pooled OLS: log(amp) ~ log(degree) * early")
    p = pos[(pos.degree > 0) & (pos.amp > 0)].copy()
    ld, la, E = np.log(p.degree.values), np.log(p.amp.values), p.early.values.astype(float)
    X = np.column_stack([np.ones(len(p)), ld, E, ld * E])
    b = np.linalg.lstsq(X, la, rcond=None)[0]
    r = la - X @ b
    se = np.sqrt(np.diag((r @ r / (len(p) - 4)) * np.linalg.inv(X.T @ X)))
    for n, bi, si in zip(['const', 'log(degree)', 'early', 'log(degree):early'], b, se):
        tv = bi / si
        print(f"    {n:20s} b={bi:+.4f}  SE={si:.4f}  t={tv:+.2f}  p={2*stats.norm.sf(abs(tv)):.2e}")


def a4c_matched_degree(A, nboot=2000, seed=0):
    hdr("A4c  matched-degree early/late comparison (descriptive; bootstrap CIs)")
    pos = A[A.red > 1e-3].copy()
    pos['dec'] = pd.qcut(pos.degree, 10, labels=False, duplicates='drop')
    rs = np.random.default_rng(seed)
    print(f"  {'dec':>3} {'k range':>12} {'n_E':>6} {'n_L':>6} {'mean_E':>7} {'mean_L':>7} "
          f"{'ratio':>6} {'95% CI':>16} {'med_E':>6} {'med_L':>6}")
    for d, g in pos.groupby('dec'):
        ge, gl = g[g.early].amp.values, g[~g.early].amp.values
        if len(ge) < 10 or len(gl) < 10:
            continue
        bs = np.array([rs.choice(ge, len(ge)).mean() / rs.choice(gl, len(gl)).mean()
                       for _ in range(nboot)])
        print(f"  {int(d):>3} {g.degree.min():5.0f}-{g.degree.max():<6.0f} {len(ge):>6} {len(gl):>6} "
              f"{ge.mean():>7.2f} {gl.mean():>7.2f} {ge.mean()/gl.mean():>6.2f} "
              f"[{np.percentile(bs,2.5):>5.2f}, {np.percentile(bs,97.5):>5.2f}] "
              f"{np.median(ge):>6.2f} {np.median(gl):>6.2f}")
    ge, gl = pos[pos.early].amp.values, pos[~pos.early].amp.values
    print(f"\n  Unmatched: early mean={ge.mean():.2f}x (n={len(ge)}), "
          f"late mean={gl.mean():.2f}x (n={len(gl)}), ratio={ge.mean()/gl.mean():.2f}")
    print(f"  Medians:   early={np.median(ge):.2f}x  late={np.median(gl):.2f}x")
    print(f"  Mean degree: early={pos[pos.early].degree.mean():.1f}  "
          f"late={pos[~pos.early].degree.mean():.1f}")


def continuous_timing(A, t_cut=ANALYSIS_T, label=''):
    """Switch time as a continuous predictor, not dichotomised at F_veg=0.5.

    The cohort split at the trajectory's own median is an analytical cut, not a physical
    one; agents either side of it differ by a few thousand steps and look identical. Fitting
    switch time continuously is the honest test of whether timing pays, and whether it pays
    more at high degree. log(remaining exposure) is included as a control because early
    adopters have more simulation time left for downstream conversions to land.
    """
    import statsmodels.formula.api as smf
    hdr(f"Timing as a continuous predictor{label}")
    p = A[(A.red > 1e-3) & (A.degree > 0) & (A.ct > 0)].copy()
    p['ld'] = np.log(p.degree)
    p['la'] = np.log(p.amp)
    p['sw'] = p.ct / 1000.0                             # switch time, thousands of steps
    p['lex'] = np.log((t_cut - p.ct).clip(lower=1))     # remaining exposure
    for c in ('ld', 'sw', 'lex'):
        p[c + 'c'] = p[c] - p[c].mean()
    print(f"  n={len(p)}  switch time mean={p.sw.mean():.1f}k sd={p.sw.std():.1f}k  "
          f"mean degree={p.degree.mean():.1f}")
    for f, tag in [('la ~ ldc*swc', 'switch time x degree'),
                   ('la ~ ldc*swc + lexc', '+ log(exposure) control')]:
        m = smf.ols(f, data=p).fit(cov_type='cluster', cov_kwds={'groups': p.run})
        print(f"\n  {tag}   (SEs clustered by run, R2={m.rsquared:.4f})")
        for n in m.params.index:
            print(f"    {n:10s} b={m.params[n]:+.5f}  z={m.tvalues[n]:+8.2f}  "
                  f"p={m.pvalues[n]:.2e}")
    b = smf.ols('la ~ ldc*swc', data=p).fit().params
    print("\n  implied amplification ratio for switching 20k steps earlier:")
    for d in (4, 8, 16, 40):
        eff = (b['swc'] + b['ldc:swc'] * (np.log(d) - p.ld.mean())) * (-20)
        print(f"    degree={d:3d}: {np.exp(eff):.2f}x")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else PKL
    df = pd.read_pickle(path)
    print(f"Loaded {len(df)} runs from {path}")
    has_unw = 'reductions_unw' in _resolve_snap(df.iloc[0]['snapshots'])[0]
    a4a_unweighted(df)
    for key, tag in ([('reductions', 'WEIGHTED')] +
                     ([('reductions_unw', 'UNWEIGHTED w=1')] if has_unw else [])):
        print(f"\nBuilding per-agent table [{tag}]...")
        A = agent_table(df, red_key=key)
        a4b_interaction(A, label=f"  [{tag}]")
        a4c_matched_degree(A)
        continuous_timing(A, label=f"  [{tag}]")


if __name__ == '__main__':
    main()
