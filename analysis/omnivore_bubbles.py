"""Do immune omnivores shelter surviving omnivores? Tests the manuscript :95 clustering claim.

Hypothesis: the ~25% omnivore remainder at t_end persists because non-immune omnivores sit in
bubbles around immune omnivores (who never switch), not because of their own theta/rho.
Reads the reduced kappa=0.55 N=2000 twin ensemble; t_end = snapshot at 310,000.

Note that theta, rho and the immune set are identical in every run (the sampler reseeds
at 42), only the network differs, so runs are replicate networks over one population.

Usage: python analysis/omnivore_bubbles.py [n_runs]   (from repo root)
Writes omnivore_bubbles_per_run.csv beside the run pkls; prints the summary table.
"""
import sys
import glob
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import statsmodels.api as sm

DIR = 'model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced'
T_END, GATE_C, N_PERM = 310000, 0.35, 200
rng = np.random.default_rng(0)


def graph(s):
    G = nx.Graph()
    G.add_nodes_from(range(s['n_nodes']))
    G.add_edges_from(map(tuple, s['edges']))
    return G


def adj(G):
    return sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes())))


def nb_share(A, x):
    # fraction of neighbours with x; NaN for isolates
    deg = np.asarray(A.sum(1)).ravel()
    with np.errstate(invalid='ignore', divide='ignore'):
        return (A @ x.astype(float)) / np.where(deg > 0, deg, np.nan)


def group_clustering(G, cl, mask):
    # (a) mean full-graph local clustering over the group; (b) clustering inside the group subgraph
    idx = np.flatnonzero(mask)
    return np.mean([cl[i] for i in idx]), nx.average_clustering(G.subgraph(idx.tolist()))


def perm_null(A, pool, k, focal):
    # immune-omnivore labels reshuffled over initial omnivores outside the focal set; network fixed
    L = np.zeros((A.shape[0], N_PERM))
    for r in range(N_PERM):
        L[rng.choice(pool, k, replace=False), r] = 1
    cnt = (A @ L)[focal]
    deg = np.asarray(A.sum(1)).ravel()[focal][:, None]
    return (cnt > 0).mean(), np.nanmean(cnt / np.where(deg > 0, deg, np.nan))


def one_run(path):
    d = pd.read_pickle(path)
    S = d['snapshots']
    s0, s1 = S[0], S[T_END]
    G0, G1 = graph(s0), graph(s1)
    A0, A1 = adj(G0), adj(G1)
    theta, rho = s0['node_theta'].astype(float), np.asarray(s0['rhos'], float)
    imm = s0['immune'].astype(bool)
    veg0 = np.array(d['initial_diets']) == 'veg'
    veg1 = s1['diets'].astype(bool)
    omni1 = ~veg1
    imm_o = imm & ~veg0                      # immune omnivores (never switch)
    foc = omni1 & ~imm                        # surviving non-immune omnivores
    conv = veg1 & ~veg0 & ~imm                # converted, non-immune
    N = len(veg1)

    r = dict(run=d['run'], t=T_END, F_veg=veg1.mean(), n_omni=omni1.sum(),
             n_omni_immune=(omni1 & imm).sum(), n_imm_omni_hi_tail=(imm_o & ((theta + rho) / 2 > np.median((theta + rho) / 2))).sum(),
             n_imm_veg=(imm & veg0).sum(), nonimm_omni_share_N=foc.sum() / N)

    # 2. adjacency to immune omnivores, observed vs label-permutation null
    for tag, A in (('t0', A0), ('t1', A1)):
        cnt = A @ imm_o.astype(float)
        sh = nb_share(A, imm_o)
        for grp, m in (('omni', foc), ('conv', conv)):
            obs_any, obs_sh = (cnt[m] > 0).mean(), np.nanmean(sh[m])
            pool = np.flatnonzero(~veg0 & ~m)
            null_any, null_sh = perm_null(A, pool, imm_o.sum(), np.flatnonzero(m))
            r.update({f'{grp}_any_immO_{tag}': obs_any, f'{grp}_shr_immO_{tag}': obs_sh,
                      f'{grp}_any_ratio_{tag}': obs_any / null_any, f'{grp}_shr_ratio_{tag}': obs_sh / null_sh})

    # 3. omnivore-induced subgraph at t_end: bubbles or scattered?
    comps = list(nx.connected_components(G1.subgraph(np.flatnonzero(omni1).tolist())))
    size = np.zeros(N, int)
    has_imm = np.zeros(N, bool)
    for c in comps:
        c = list(c)
        size[c] = len(c)
        has_imm[c] = imm_o[c].any()
    r.update(n_comp=len(comps), largest_comp_frac_omni=max(map(len, comps)) / omni1.sum(),
             foc_in_comp_with_immO=has_imm[foc].mean(), foc_singleton=(size[foc] == 1).mean(),
             foc_median_comp_size=np.median(size[foc]))
    # null: same number of non-immune omnivores placed at random among non-immune agents,
    # immune agents where they are; asks whether the giant component is just percolation
    nulls = []
    for _ in range(20):
        o = imm_o.copy()
        o[rng.choice(np.flatnonzero(~imm), foc.sum(), replace=False)] = True
        cs = list(nx.connected_components(G1.subgraph(np.flatnonzero(o).tolist())))
        hi = np.zeros(N, bool)
        for c in cs:
            c = list(c)
            hi[c] = imm_o[c].any()
        nulls.append((max(map(len, cs)) / o.sum(), hi[o & ~imm].mean()))
    r['null_largest_comp_frac'], r['null_foc_in_comp_with_immO'] = np.mean(nulls, 0)

    # persistence: churn vs stable bubbles, over the dense snapshots 280k-398k (2000 steps apart)
    dense = [t for t in S if isinstance(t, int) and t >= 280000]
    meat = np.array([~S[t]['diets'].astype(bool) for t in dense])
    r.update(n_dense=len(dense), foc_omni_all_dense=meat.all(0)[foc].mean(),
             foc_time_omni=meat.mean(0)[foc].mean(), nonimm_time_omni=meat.mean(0)[~imm].mean())

    # 5. neighbourhood veg share at t_end
    vs = nb_share(A1, veg1)
    r.update(foc_nb_veg=np.nanmean(vs[foc]), veg_nb_veg=np.nanmean(vs[veg1]),
             conv_nb_veg=np.nanmean(vs[conv]), foc_below_c=np.nanmean(vs[foc] < GATE_C),
             foc_nb_omni_all=np.nanmean(1 - vs[foc]))

    # 6. per-group clustering, both definitions; local clustering near immune omnivores
    cl0, cl1 = nx.clustering(G0), nx.clustering(G1)
    r['C_all_t0'], r['C_all_t1'] = np.mean(list(cl0.values())), np.mean(list(cl1.values()))
    r['C_o_t0'], r['C_o_sub_t0'] = group_clustering(G0, cl0, ~veg0)
    r['C_v_t0'], r['C_v_sub_t0'] = group_clustering(G0, cl0, veg0)
    r['C_o_t1'], r['C_o_sub_t1'] = group_clustering(G1, cl1, omni1)
    r['C_v_t1'], r['C_v_sub_t1'] = group_clustering(G1, cl1, veg1)
    near = (A1 @ imm_o.astype(float)) > 0
    c1 = np.array([cl1[i] for i in range(N)])
    r.update(foc_cl_near_immO=c1[foc & near].mean(), foc_cl_far_immO=c1[foc & ~near].mean())

    # 4. regression frame: initial non-immune omnivores
    m = ~veg0 & ~imm
    deg0, deg1 = np.asarray(A0.sum(1)).ravel(), np.asarray(A1.sum(1)).ravel()
    reg = pd.DataFrame(dict(run=d['run'], agent=np.flatnonzero(m), y=omni1[m].astype(int),
                            theta=theta[m], rho=rho[m], deg0=deg0[m], deg1=deg1[m],
                            imm0=np.nan_to_num(nb_share(A0, imm_o)[m]),
                            imm1=np.nan_to_num(nb_share(A1, imm_o)[m])))
    return r, reg


def logit(df, X, groups=None):
    Z = (df[X] - df[X].mean()) / df[X].std()
    if groups is not None:
        Z = pd.concat([Z, pd.get_dummies(df['run'], prefix='run', drop_first=True, dtype=float)], axis=1)
    f = sm.Logit(df['y'], sm.add_constant(Z)).fit(
        disp=0, **({'cov_type': 'cluster', 'cov_kwds': {'groups': df[groups]}} if groups else {}))
    return f


def q(x):
    x = np.asarray(x, float)
    return f'{np.nanmedian(x):.3f} [{np.nanpercentile(x, 25):.3f}, {np.nanpercentile(x, 75):.3f}]'


if __name__ == '__main__':
    paths = sorted(glob.glob(f'{DIR}/run_*.pkl'))[:int(sys.argv[1]) if len(sys.argv) > 1 else None]
    out = [one_run(p) for p in paths]
    runs = pd.DataFrame([o[0] for o in out])
    reg = pd.concat([o[1] for o in out], ignore_index=True)

    PSY, NET0, NET1 = ['theta', 'rho'], ['deg0', 'imm0'], ['deg1', 'imm1']
    per_run = {}
    for tag, X in (('t0', PSY + NET0), ('t1', PSY + NET1)):
        fits = [logit(g, X) for _, g in reg.groupby('run')]
        for v in X:
            per_run[f'b_{v}_{tag}'] = [f.params[v] for f in fits]
        per_run[f'r2_full_{tag}'] = [f.prsquared for f in fits]
    per_run['r2_psy'] = [logit(g, PSY).prsquared for _, g in reg.groupby('run')]
    per_run['r2_net0'] = [logit(g, NET0).prsquared for _, g in reg.groupby('run')]
    per_run['r2_net1'] = [logit(g, NET1).prsquared for _, g in reg.groupby('run')]
    runs = runs.assign(**per_run)
    runs.to_csv(f'{DIR}/omnivore_bubbles_per_run.csv', index=False)

    print(f'INFO: {len(runs)} runs, t_end snapshot t = {T_END}')
    for c in runs.columns.drop(['run', 't']):
        print(f'{c:28s} {q(runs[c])}')

    # pooled with run fixed effects, SEs clustered by agent (same agents in every run)
    for tag, X in (('t0 network', PSY + NET0), ('t_end network (partly endogenous)', PSY + NET1)):
        f = logit(reg, X, groups='agent')
        print(f'\nPOOLED logit, run FE, agent-clustered SE, {tag}: n={int(f.nobs)}, McFadden R2={f.prsquared:.3f}')
        print(pd.DataFrame({'b': f.params[X], 'se': f.bse[X], 'OR': np.exp(f.params[X])}).round(3))
    print(f"\nPOOLED y mean (still omnivore | initial non-immune omnivore): {reg['y'].mean():.3f}")

    # same agents in every run: how much of the t_end state is a fixed property of the agent at all?
    # share of Bernoulli variance that is between agents, minus the binomial floor p(1-p)/n_runs
    p = reg.groupby('agent')['y'].mean()
    pb, n = reg['y'].mean(), reg['run'].nunique()
    between = (p.var() - pb * (1 - pb) / n) / (pb * (1 - pb))
    fit = logit(reg, PSY)
    print(f'AGENT-LEVEL: between-agent share of outcome variance {between:.3f} (n_runs={n}); '
          f'corr(p_agent, theta/rho logit fit) {np.corrcoef(p, fit.predict().reshape(n, -1).mean(0))[0, 1]:.3f}')
