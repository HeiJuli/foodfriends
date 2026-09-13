"""Does network structure change the TRANSIENT, not just the endpoint? Local paired-arm simulation.

Arms, all twin N=2000, kappa 0.55, model_runn.DEFAULT_PARAMS, same seed per replicate:
  EMP  homophilic_emp network (the reported topology)
  DPR  the same EMP graph after 10*m degree-preserving double-edge swaps: degree sequence
       kept, clustering and any demographic/attribute homophily destroyed
  ER   G(N, m) with the EMP edge count: degree heterogeneity destroyed as well
Rewiring is frozen (p_rewire = 0) in every arm, otherwise triadic closure would regrow
clustering in DPR/ER and blur the contrast; frozen vs default moved F_veg by +0.0015
(prewire_and_topology_results_2026-09-12.md s.1), so this is not a different model.

Dynamics are model_main.Agent.step unchanged. Only per-step O(N) bookkeeping the dynamics
never read is skipped (system_C, harmonise_netIn, steady-state check, record_fraction, which
is replaced by an incremental count), and scipy's lognormal emission draw is stubbed out:
C is not read by the dynamics, it only costs ~50 us a step. After the graph is final, the
RNGs are reseeded and memories re-seeded from neighbours identically in every arm.

Usage (repo root):
  python analysis/topology_transient_sim.py --seeds 30 --steps 400000 --workers 12 --arms EMP,DPR,ER
  --mode sample-max --N 385 --steps 200000 matches the 2026-09-12 topology arms' population instead.
Writes model_output/topology_transient_sim_<tag>/<arm>_<seed>.pkl (existing files are skipped,
so a killed campaign resumes).
"""
import os
import sys
import time
import pickle
import random
from types import SimpleNamespace
from multiprocessing import Pool
import numpy as np
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import argparse
os.chdir(os.path.join(ROOT, 'model_src'))           # model paths are ../data/...
sys.path[:0] = [os.path.join(ROOT, 'model_src'), ROOT]
import model_main
import model_runn

_real_st = model_main.st
model_main.st = SimpleNamespace(lognorm=SimpleNamespace(rvs=lambda s, scale: scale),
                                truncnorm=_real_st.truncnorm, truncweibull_min=_real_st.truncweibull_min)
PMF = model_runn.load_pmf_tables()


def run(job):
    arm, seed, steps, mode, N, prw, OUT = job
    path = os.path.join(OUT, f'{arm}_{seed}.pkl')
    if os.path.exists(path):
        return path
    t0 = time.time()
    np.random.seed(seed); random.seed(seed)
    p = dict(model_runn.DEFAULT_PARAMS, N=N, agent_ini=mode, steps=steps, seed=seed,
             topology='homophilic_emp', p_rewire=prw, snapshot_dense_start=0)
    m = model_main.Model(p, pmf_tables=PMF if mode == 'twin' else None)
    m.agent_ini()                                    # builds the EMP graph from the seed
    G = m_emp_G = m.G1
    if arm == 'DPR':
        G = G.copy()
        nx.double_edge_swap(G, nswap=10 * G.number_of_edges(), max_tries=200 * G.number_of_edges(), seed=seed)
    elif arm == 'ER':
        G = nx.gnm_random_graph(p['N'], m.G1.number_of_edges(), seed=seed)
    m.G1 = G
    np.random.seed(seed + 7919); random.seed(seed + 7919)
    for a in m.agents:
        a.memory = []
        a.initialize_memory_from_neighbours(G, m.agents)

    agents, N = m.agents, len(m.agents)
    nveg = sum(a.diet == 'veg' for a in agents)
    counts = np.empty(steps + 1, np.uint16); counts[0] = nveg
    ev = []
    for t in range(steps):
        i = np.random.choice(N)
        if m.flip(0.50):
            e = agents[i].step(G, agents, t)
            if e is not None:
                if e[0] == 'conv':
                    buf = e[5]
                    vs = [s for d, s, _ in buf if d == 'veg']
                    ev.append(('conv', t, i, e[3], e[4], len(vs), len(set(vs))))
                    nveg += 1
                else:
                    ev.append(e); nveg -= 1
            m.rewire(agents[i])                      # at p_rewire = 0 consumes one draw, no-op
        counts[t + 1] = nveg

    deg = np.array([d for _, d in G.degree()])
    res = dict(arm=arm, p_rewire=prw, mode=mode, N=N, clustering_emp_t0=nx.average_clustering(m_emp_G), seed=seed, steps=steps, counts=counts, events=ev,
               edges=np.array(G.edges(), dtype=np.int32),
               initial_diets=None, immune=np.array([a.immune for a in agents]),
               theta=np.array([a.theta for a in agents]), rho=np.array([a.rho for a in agents]),
               alpha=np.array([a.alpha for a in agents]),
               net=dict(clustering=nx.average_clustering(G), transitivity=nx.transitivity(G),
                        deg_mean=deg.mean(), deg_cv=deg.std() / deg.mean(), deg_max=int(deg.max()),
                        deg_assort=nx.degree_assortativity_coefficient(G),
                        n_isolates=int((deg == 0).sum())),
               runtime_s=time.time() - t0)
    # initial diets: undo the event log from the final state
    d = np.array([a.diet == 'veg' for a in agents])
    for e in reversed(ev):
        d[e[2]] = e[0] == 'rev'
    res['initial_diets'] = d
    os.makedirs(OUT, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(res, f)
    print(f'INFO: {arm} seed {seed} F_end {nveg / N:.3f} CC {res["net"]["clustering"]:.3f} '
          f'{res["runtime_s"]:.0f}s', flush=True)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--steps', type=int, default=400000)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--arms', default='EMP,DPR,ER')
    ap.add_argument('--mode', default='twin', choices=['twin', 'sample-max'])
    ap.add_argument('--N', type=int, default=2000)
    ap.add_argument('--p-rewire', type=float, default=0.0)
    ap.add_argument('--tag', default='20260913_N2000_frozen')
    a = ap.parse_args()
    out = os.path.join(ROOT, 'model_output', f'topology_transient_sim_{a.tag}')
    # interleave arms so a stopped campaign still has balanced, paired seeds
    jobs = [(arm, 42 + s, a.steps, a.mode, a.N, a.p_rewire, out)
            for s in range(a.seeds) for arm in a.arms.split(',')]
    with Pool(a.workers) as pool:
        for _ in pool.imap_unordered(run, jobs):
            pass
