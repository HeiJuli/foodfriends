import pickle, sys, time
import numpy as np, networkx as nx
from scipy.stats import spearmanr, rankdata
import statsmodels.api as sm
sys.path.insert(0, 'plotting')
from agency_predictor_analysis import compute_complex_centrality

D = 'model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced/run_%02d.pkl'
T = 310000; T_CONV = 173000
rows = []
for r in range(5):
    d = pickle.load(open(D % r, 'rb'))
    s = d['snapshots'][T]
    G = nx.Graph(); G.add_nodes_from(range(s['n_nodes'])); G.add_edges_from(map(tuple, s['edges']))
    veg = s['diets'].astype(bool)
    nodes = list(range(s['n_nodes']))
    deg = np.array([G.degree(v) for v in nodes], float)
    bet = nx.betweenness_centrality(G)
    bet = np.array([bet[v] for v in nodes])
    clu = nx.clustering(G); clu = np.array([clu[v] for v in nodes])
    r1 = spearmanr(deg, bet).statistic
    r2 = spearmanr(clu, deg).statistic
    r3 = nx.degree_assortativity_coefficient(G)
    com = nx.community.louvain_communities(G, seed=0)
    Q = nx.community.modularity(G, com)
    ER = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), seed=0)
    comE = nx.community.louvain_communities(ER, seed=0)
    QE = nx.community.modularity(ER, comE)
    t0 = time.time()
    cc = compute_complex_centrality(G); cc = np.array([cc[v] for v in nodes])
    tcc = time.time() - t0
    nb = np.array([veg[list(G.neighbors(v))].mean() if deg[v] > 0 else np.nan for v in nodes])
    ok = ~np.isnan(nb)
    r5m = spearmanr(cc[ok], nb[ok]).statistic
    # partial: residualise ranks of CC and nb on rank of degree (Spearman partial), and on log degree
    def resid(y, x):
        X = sm.add_constant(x); return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rd = rankdata(deg[ok]); rc = rankdata(cc[ok]); rn = rankdata(nb[ok])
    r5p_rank = np.corrcoef(resid(rc, rd), resid(rn, rd))[0, 1]
    ld = np.log(deg[ok])
    r5p_log = spearmanr(resid(cc[ok], ld), resid(nb[ok], ld)).statistic
    # 6: logistic conversion before 173k on std CC and std log degree, omnivores at t0
    init = np.array(d['initial_diets']); omn = init != 'veg'
    conv = np.zeros(len(nodes), bool)
    for e in d['events']:
        if e[0] == 'conv' and e[1] < T_CONV: conv[e[2]] = True
    m = omn & (deg > 0)
    z = lambda a: (a - a.mean()) / a.std()
    X = sm.add_constant(np.column_stack([z(cc[m]), z(np.log(deg[m]))]))
    fit = sm.Logit(conv[m].astype(float), X).fit(disp=0)
    OR = np.exp(fit.params[1]); ORd = np.exp(fit.params[2])
    rows.append((r, r1, r2, r3, Q, len(com), QE, len(comE), r5m, r5p_rank, r5p_log, OR, ORd, conv[m].mean(), tcc, veg.mean()))
    print(f"run{r}: rs(deg,bet)={r1:.3f} rs(clu,deg)={r2:.3f} assort={r3:.3f} Q={Q:.3f}({len(com)}) QER={QE:.3f}({len(comE)}) "
          f"rs(CC,nb)={r5m:.3f} partial_rank={r5p_rank:.3f} partial_log={r5p_log:.3f} OR_CC={OR:.3f} OR_logdeg={ORd:.3f} "
          f"conv_share={conv[m].mean():.3f} Fveg={veg.mean():.3f} CC_time={tcc:.0f}s", flush=True)
A = np.array(rows)
med = np.median(A[:, 1:], axis=0)
print("MEDIAN: rs(deg,bet)=%.3f rs(clu,deg)=%.3f assort=%.3f Q=%.3f(%g) QER=%.3f(%g) rs(CC,nb)=%.3f partial_rank=%.3f partial_log=%.3f OR_CC=%.3f OR_logdeg=%.3f conv_share=%.3f" % tuple(med[:12]))
