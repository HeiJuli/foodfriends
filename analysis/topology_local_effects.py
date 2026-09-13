"""Where does the network show up in the kappa N=2000 ensemble, if not in the endpoint?

Four on-disk measures, all replayed from the event log (every diet switch is logged) on the
snapshot network nearest below each time (rewiring moves ~15% of edges by 310k):

 1. Diet assortativity r(t) on the network. Nulls: (a) diet labels permuted within degree
    deciles (keeps F and any degree-diet link); (b) CROSS-RUN: run r+k's diets at the same t
    (or the same F level) laid on run r's network. Agents, theta/rho/alpha, immunity and
    initial diets are identical in every run, only the network and the history differ, so
    (b) keeps every attribute-driven diet pattern and removes the part a specific network's
    contagion history put there.
 2. Local reinforcement: risk-set case-control. Cases = non-immune conversions (or
    reversions); 4 controls per case = random non-immune meat-eaters (vegetarians) at the same
    step. Conditional logit per run per epoch on neighbour veg share s, closure among the veg
    neighbours (share of their pairs linked: the Centola complex-contagion structure), ego
    clustering, log degree, log time since own last switch, and own theta/rho/alpha. Null: s recomputed on k random
    non-neighbours (placebo).
 3. Spreading geography: rank correlation of each agent's first conversion time with its
    graph distance to the initial vegetarians, degree and clustering at t=0; cross-run null.
 4. Churn by epoch: reversions per conversion (churn is present from the first steps).
Plus the buffer at conversion: distinct vegetarian sources (complex vs simple contagion).

Epochs per run from the 1%-of-run running mean: early F < 0.20, rise 0.20-0.50 (F_c ~0.35
sits here), late F >= 0.50 up to t = 310,000.

Usage (repo root): python analysis/topology_local_effects.py [n_runs] [workers]
Writes topology_local_*.csv beside the run pkls and prints the summary.
"""
import sys
import glob
import bisect
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr
from statsmodels.discrete.conditional_models import ConditionalLogit

sys.path.insert(0, 'analysis')
from topology_transient_metrics import metrics

DIR = 'model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced'
T_END, N_CASE, N_CTRL, N_PERM, K_CROSS = 310000, 500, 4, 50, 5
GRID = [0, 1000, 5000] + list(range(10000, 400001, 10000))
LEVELS = ('t_0.10', 't_0.20', 't_0.35', 't_0.50', 't_0.65', 't_c')
EPOCHS = ('early', 'rise', 'late')
warnings.filterwarnings('ignore')


def assort(x, e):
    a, b = np.r_[x[e[:, 0]], x[e[:, 1]]], np.r_[x[e[:, 1]], x[e[:, 0]]]
    return np.corrcoef(a, b)[0, 1]


def graph(s, n):
    A = [set() for _ in range(n)]
    for u, v in s['edges']:
        A[u].add(v); A[v].add(u)
    tri = np.array([sum(len(A[u] & A[i]) for u in A[i]) / 2 for i in range(n)])
    k = np.array([len(a) for a in A])
    with np.errstate(invalid='ignore', divide='ignore'):
        cc = np.where(k > 1, tri / (k * (k - 1) / 2), 0.0)
    return A, cc, k


def feats(j, veg, A, cc, rng, n):
    nb = A[j]; k = len(nb)
    vn = [u for u in nb if veg[u]]; nv = len(vn); vs = set(vn)
    clos = sum(len(A[u] & vs) for u in vn) / (nv * (nv - 1)) if nv >= 2 else np.nan
    pl = rng.choice(n, k, replace=False)
    return dict(k=k, s=nv / k, nv=nv, clos=clos, cc=cc[j], s_pl=veg[pl].mean())


def fit(g):
    g = g.assign(s10=10 * g.s, cc10=10 * g.cc, logk=np.log(g.k), has2=(g.nv >= 2).astype(float),
                 clos10=10 * g.clos.fillna(0), spl10=10 * g.s_pl, lage=np.log1p(g.age))
    # lage: log steps since the agent's last switch. A recent convert has a high-s neighbourhood
    # (it converted because of it) and a short expected stint, which confounds s for reversions
    own = ['logk', 'cc10', 'lage', 'theta', 'rho', 'alpha']
    out = []
    for name, X in (('base', ['s10'] + own), ('closure', ['s10', 'clos10', 'has2'] + own),
                    ('placebo', ['spl10'] + own)):
        try:
            r = ConditionalLogit(g.case.values, g[X].values.astype(float), groups=g.set_id.values).fit(disp=0)
            for i, term in enumerate(X):
                if term in ('s10', 'clos10', 'cc10', 'logk', 'spl10', 'lage'):
                    out.append(dict(model=name, term=term, OR=np.exp(r.params[i]), p=r.pvalues[i]))
        except Exception as ex:
            print(f'WARNING: fit {name} failed: {ex}')
    return out


def one_run(path):
    rng = np.random.default_rng(1)
    d = pd.read_pickle(path)
    S, run = d['snapshots'], d['run']
    ts = sorted(k for k in S if isinstance(k, int) and 'edges' in S[k])   # the reducer kept edges every 10k only
    n = S[0]['n_nodes']
    F = np.asarray(d['fraction_veg'], float)
    m = metrics(F)
    imm = np.asarray(S[0]['immune'], bool)
    own = dict(theta=np.asarray(S[0]['node_theta'], float), rho=np.asarray(S[0]['rhos'], float),
               alpha=np.asarray(S[0]['alphas'], float))
    d0 = np.array(d['initial_diets']) == 'veg'
    cache = {}

    def G(t):
        key = ts[bisect.bisect_right(ts, t) - 1]
        if key not in cache:
            cache[key] = graph(S[key], n)
        return key, cache[key]

    def epoch(t):
        return 'early' if t < m['t_0.20'] else 'rise' if t < m['t_0.50'] else 'late'

    ev = d['events']
    # pre-select cases per type x epoch
    pools = {}
    for idx, e in enumerate(ev):
        if 1000 <= e[1] <= T_END and not imm[e[2]]:
            pools.setdefault((e[0], epoch(e[1])), []).append(idx)
    chosen = {idx: key for key, ids in pools.items()
              for idx in rng.choice(ids, min(N_CASE, len(ids)), replace=False)}

    stimes = sorted(set(GRID) | {int(m[l]) for l in LEVELS if np.isfinite(m[l])})
    states, gi = {}, 0
    veg = d0.copy()
    rows, buf, first, last = [], [], np.full(n, np.inf), np.zeros(n)
    nonimm = np.flatnonzero(~imm)
    for idx, e in enumerate(ev):
        t = e[1]
        while gi < len(stimes) and stimes[gi] <= t:
            states[stimes[gi]] = veg.copy(); gi += 1
        if e[0] == 'conv' and not imm[e[2]] and 1000 <= t <= T_END:
            vs = [src for dd, src, _ in e[5] if dd == 'veg']
            buf.append(dict(epoch=epoch(t), n_veg_entries=len(vs), distinct_veg=len(set(vs))))
            first[e[2]] = min(first[e[2]], t)
        if idx in chosen:
            typ, ep = chosen[idx]
            _, (A, cc, _) = G(t)
            want = typ == 'rev'                       # controls share the case's current diet
            ctrl, tries = [], 0
            while len(ctrl) < N_CTRL and tries < 20000:
                c = nonimm[rng.integers(len(nonimm))]; tries += 1
                if c != e[2] and veg[c] == want and len(A[c]) and c not in ctrl:
                    ctrl.append(c)
            if len(A[e[2]]) and ctrl:
                for j, case in [(e[2], 1)] + [(c, 0) for c in ctrl]:
                    rows.append(dict(type=typ, epoch=ep, set_id=idx, case=case, t=t, age=t - last[j],
                                     theta=own['theta'][j], rho=own['rho'][j], alpha=own['alpha'][j],
                                     **feats(j, veg, A, cc, rng, n)))
        veg[e[2]] = e[0] == 'conv'; last[e[2]] = t
    for st in stimes[gi:]:
        states[st] = veg.copy()

    cc_rows = []
    rdf = pd.DataFrame(rows)
    for (typ, ep), g in rdf.groupby(['type', 'epoch']):
        desc = g.groupby('case')[['s', 'clos', 'cc', 'k']].mean()
        for r in fit(g):
            cc_rows.append(dict(run=run, type=typ, epoch=ep, n_sets=g.set_id.nunique(),
                                s_case=desc.loc[1, 's'], s_ctrl=desc.loc[0, 's'],
                                clos_case=desc.loc[1, 'clos'], clos_ctrl=desc.loc[0, 'clos'], **r))

    # assortativity with degree-decile permutation null; edges kept for the cross-run null
    edges, ass = {}, []
    for st in stimes:
        key, (A, cc, k) = G(st)
        e = np.asarray(S[key]['edges'])
        edges[key] = e
        x = states[st].astype(float)
        bins = np.searchsorted(np.quantile(k, np.linspace(0, 1, 11)[1:-1]), k, side='right')
        perm = []
        for _ in range(N_PERM):
            xp = x.copy()
            for b in np.unique(bins):
                ii = np.flatnonzero(bins == b); xp[ii] = x[rng.permutation(ii)]
            perm.append(assort(xp, e))
        ass.append(dict(run=run, t=st, gkey=key, F=x.mean(), r_obs=assort(x, e),
                        r_perm_mean=np.mean(perm), r_perm_sd=np.std(perm),
                        level=next((l for l in LEVELS if np.isfinite(m[l]) and int(m[l]) == st), '')))

    # geography: graph distance from the initial vegetarians on the t=0 network
    A0, cc0, k0 = graph(S[0], n)
    dist = np.full(n, np.inf); dist[d0] = 0; front = list(np.flatnonzero(d0)); lvl = 0
    while front:
        lvl += 1
        nxt = [v for u in front for v in A0[u] if dist[v] == np.inf]
        for v in nxt:
            dist[v] = lvl
        front = list(set(nxt))

    # churn by epoch: reversions per conversion (all agents; immune agents never switch)
    onset = {}
    for ep in EPOCHS:
        nc = sum(1 for e in ev if e[0] == 'conv' and 1000 <= e[1] <= T_END and epoch(e[1]) == ep)
        nr = sum(1 for e in ev if e[0] == 'rev' and 1000 <= e[1] <= T_END and epoch(e[1]) == ep)
        onset[f'rev_per_conv_{ep}'] = nr / max(nc, 1)
        onset[f'conv_{ep}'] = nc

    return dict(run=run, metrics=m, cc=cc_rows, ass=ass, states=states, edges=edges,
                buf=pd.DataFrame(buf).assign(run=run), first=first, dist=dist, deg0=k0, cc0=cc0,
                d0=d0, imm=imm, onset=onset, n_edges_tc=len(S[0]['edges']))


def pooled(OR):
    # runs are independent replicates (own network and history): mean log OR, 95% CI over runs
    L = np.log(np.asarray(OR, float)); L = L[np.isfinite(L)]
    se = L.std(ddof=1) / np.sqrt(len(L)) if len(L) > 1 else np.nan
    return f'{np.exp(L.mean()):.3f} ({np.exp(L.mean() - 1.96 * se):.3f}-{np.exp(L.mean() + 1.96 * se):.3f})'


def q(x):
    x = np.asarray(x, float)
    return f'{np.nanmedian(x):.3f} [{np.nanpercentile(x, 25):.3f}, {np.nanpercentile(x, 75):.3f}]'


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    paths = sorted(glob.glob(f'{DIR}/run_*.pkl'))[:n_runs]
    with Pool(workers) as p:
        R = p.map(one_run, paths)
    R.sort(key=lambda r: r['run'])
    nR = len(R)

    # 1. assortativity, with the cross-run null at the same t (grid) or the same F level
    ass = pd.DataFrame([a for r in R for a in r['ass']])
    lvl_t = [{a['level']: a['t'] for a in r['ass'] if a['level']} for r in R]
    xr = []
    for i, r in enumerate(R):
        for a in r['ass']:
            vals = []
            for kk in range(1, K_CROSS + 1):
                o = (i + kk) % nR
                st = lvl_t[o].get(a['level']) if a['level'] else a['t']
                if st is not None and st in R[o]['states']:
                    vals.append(assort(R[o]['states'][st].astype(float), r['edges'][a['gkey']]))
            xr.append(np.mean(vals) if vals else np.nan)
    ass['r_cross_mean'] = xr
    ass.to_csv(f'{DIR}/topology_local_assortativity.csv', index=False)

    print('\n== 1. Diet assortativity on the network (median [IQR] over runs) ==')
    print(f'{"point":<10}{"F":>22}{"r_obs":>26}{"r_perm(degree)":>26}{"r_cross-run":>26}{"obs-cross":>26}')
    ass['z'] = (ass.r_obs - ass.r_perm_mean) / ass.r_perm_sd
    ass['excess_cross'] = ass.r_obs - ass.r_cross_mean
    for key, g in list(ass[ass.level != ''].groupby('level')) + \
            [(f't={t // 1000}k', ass[(ass.level == '') & (ass.t == t)]) for t in (0, 10000, 50000, 100000, 200000, 310000 - 10000, 400000)]:
        if len(g):
            print(f'{key:<10}{q(g.F):>22}{q(g.r_obs):>26}{q(g.r_perm_mean):>26}{q(g.r_cross_mean):>26}{q(g.excess_cross):>26}')
    grid = ass[ass.level == '']
    pk = grid.loc[grid.groupby('run').excess_cross.idxmax()]
    print(f'   peak obs-cross per run: {q(pk.excess_cross)} at t {q(pk.t)}, F {q(pk.F)}; '
          f'obs-cross at t=310k {q(ass[(ass.t == 310000)].excess_cross)}')

    # 2. case-control
    cc = pd.DataFrame([c for r in R for c in r['cc']])
    cc.to_csv(f'{DIR}/topology_local_casecontrol.csv', index=False)
    print('\n== 2. Conditional logit per run: OR [IQR], share of runs p<0.05 ==')
    for (typ, model, term), g in cc.groupby(['type', 'model', 'term']):
        if term in ('s10', 'clos10', 'spl10', 'cc10', 'logk', 'lage'):
            line = '  '.join(f'{ep}: {q(g[g.epoch == ep].OR)} ({(g[g.epoch == ep].p < 0.05).mean():.2f}) '
                             f'pooled {pooled(g[g.epoch == ep].OR)}' for ep in EPOCHS)
            print(f'{typ:<5}{model:<8}{term:<7}{line}')
    print('   mean s (neighbour veg share) and closure, cases vs controls:')
    base = cc[(cc.model == 'base') & (cc.term == 's10')]
    for (typ, ep), g in base.groupby(['type', 'epoch']):
        print(f'   {typ} {ep:<6} s {q(g.s_case)} vs {q(g.s_ctrl)}; closure {q(g.clos_case)} vs {q(g.clos_ctrl)}; sets {q(g.n_sets)}')

    buf = pd.concat([r['buf'] for r in R])
    print('\n== buffer at conversion (non-immune, t<=310k): distinct veg sources, share with <=1 ==')
    bb = buf.groupby(['run', 'epoch']).agg(md=('distinct_veg', 'mean'), le1=('distinct_veg', lambda x: (x <= 1).mean()),
                                           nv=('n_veg_entries', 'mean')).reset_index()
    bb.to_csv(f'{DIR}/topology_local_buffer.csv', index=False)
    for ep in EPOCHS:
        g = bb[bb.epoch == ep]
        print(f'   {ep:<6} mean distinct veg sources {q(g.md)}; veg entries of 9 {q(g.nv)}; share <=1 distinct {q(g.le1)}')

    # 3. geography, cross-run null
    print('\n== 3. First-conversion time vs t=0 network position (Spearman, non-immune initial meat-eaters; never = censored) ==')
    geo = []
    degcorr = np.nanmedian([spearmanr(R[i]['deg0'], R[(i + 1) % nR]['deg0'])[0] for i in range(nR)])
    for i, r in enumerate(R):
        pop = ~r['imm'] & ~r['d0']
        row = dict(run=r['run'])
        for name, x in (('dist', r['dist']), ('deg', r['deg0']), ('cc', r['cc0'])):
            row[f'rho_{name}'] = spearmanr(r['first'][pop], x[pop])[0]
            row[f'rho_{name}_cross'] = np.mean([spearmanr(R[(i + kk) % nR]['first'][pop], x[pop])[0] for kk in range(1, K_CROSS + 1)])
        order = np.argsort(np.where(pop, r['first'], np.inf))[:200]
        row['first200_dist1'] = (r['dist'][order] == 1).mean()
        row['pop_dist1'] = (r['dist'][pop] == 1).mean()
        row['first200_dist1_cross'] = np.mean([(r['dist'][np.argsort(np.where(pop, R[(i + kk) % nR]['first'], np.inf))[:200]] == 1).mean()
                                               for kk in range(1, K_CROSS + 1)])
        row.update(r['onset']); row.update({k: r['metrics'][k] for k in ('t_c', 'F_c', 't_0.20', 't_0.35', 't_0.50')})
        geo.append(row)
    geo = pd.DataFrame(geo)
    geo.to_csv(f'{DIR}/topology_local_geography_churn.csv', index=False)
    print(f'   (degree rank correlation of the same agent across runs: {degcorr:.3f})')
    for name in ('dist', 'deg', 'cc'):
        print(f'   {name:<5} obs {q(geo[f"rho_{name}"])}  cross-run {q(geo[f"rho_{name}_cross"])}  '
              f'paired diff {q(geo[f"rho_{name}"] - geo[f"rho_{name}_cross"])}')
    print(f'   first 200 converters adjacent to an initial vegetarian {q(geo.first200_dist1)}; '
          f'cross-run {q(geo.first200_dist1_cross)}; all non-immune meat-eaters {q(geo.pop_dist1)}')

    print('\n== 4. Churn by epoch (t in [1000, 310000]) ==')
    for c in [f'{a}_{ep}' for ep in EPOCHS for a in ('rev_per_conv', 'conv')] + ['t_0.20', 't_c', 't_0.35', 't_0.50', 'F_c']:
        print(f'   {c:<20} {q(geo[c])}')


if __name__ == '__main__':
    main()
