"""Mean-field self-consistency for the omnivore remainder (MS :95, 2026-09-13).
No network: each buffer is M i.i.d. draws at the global veg share, sources drawn with
replacement from the agent's own degree so gamma's diminishing returns are kept. Per-agent
stationary occupancy is k_conv / (k_conv + k_rev) with the model's own hamiltonian(); immune
agents keep their initial diet. Iterated to a fixed point for the default and for immune_n = 0.
Result 2026-09-13: 0.756 vs ensemble 0.750; 0.809 vs the OAT immune_n = 0 point 0.812.
Usage: python analysis/meanfield_remainder.py   (from repo root; reads run_00 of the reduced
kappa N=2000 ensemble for theta, rho, alpha, immune set and degrees)."""
import sys, math, numpy as np, pandas as pd, networkx as nx
sys.path[:0] = ['model_src', '.']
from model_main import hamiltonian, boltzmann_prob
P = dict(M=9, gamma=0.3, beta=13, kappa=0.55, c=0.35, k=35)
d = pd.read_pickle('model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced/run_00.pkl')
s0 = d['snapshots'][0]
theta = s0['node_theta'].astype(float); rho = np.asarray(s0['rhos'], float)
w = 1 - np.asarray(s0['alphas'], float); imm = s0['immune'].astype(bool)
veg0 = np.array(d['initial_diets']) == 'veg'
G = nx.Graph(); G.add_nodes_from(range(s0['n_nodes'])); G.add_edges_from(map(tuple, s0['edges']))
deg = np.array([max(G.degree(i), 1) for i in range(len(theta))])
rng = np.random.default_rng(0)
NB = 300
def rates(i, F):
    """per-activation P(switch) from omni and from veg, averaged over NB buffers"""
    out = []
    for diet, s_stay in (('meat', 0.0), ('veg', 1.0)):
        ps = []
        for _ in range(NB):
            src = rng.integers(0, deg[i], P['M'])
            mem = [('veg' if rng.random() < F else 'meat', int(s), 0) for s in src]
            r = P['kappa'] * rho[i]
            Hs = hamiltonian(s_stay, theta[i], w[i], mem, P['M'], r, diet, P['c'], P['k'], P['gamma'])
            Hw = hamiltonian(1 - s_stay, theta[i], w[i], mem, P['M'], r, diet, P['c'], P['k'], P['gamma'])
            ps.append(boltzmann_prob(Hw, Hs, P['beta']))
        out.append(np.mean(ps))
    return out  # k_conv, k_rev
def occupancy(F, mask):
    p = np.where(imm, veg0, 0.0).astype(float)
    kc = np.zeros(len(theta)); kr = np.zeros(len(theta))
    for i in np.where(mask)[0]:
        kc[i], kr[i] = rates(i, F)
        p[i] = kc[i] / (kc[i] + kr[i])
    return p, kc, kr
for label, mask in (('default (immune fixed)', ~imm), ('immune_n = 0 (all switchable)', np.ones(len(theta), bool))):
    F = 0.5
    for it in range(12):
        p, kc, kr = occupancy(F, mask); Fn = p.mean()
        print(f'{label}: iter {it} F={Fn:.4f}')
        if abs(Fn - F) < 2e-3: break
        F = Fn
    sw = mask
    print(f'  fixed point F_veg = {Fn:.3f}; switchable agents: mean k_conv {kc[sw].mean():.3f}, '
          f'mean k_rev {kr[sw].mean():.4f}, p_veg median {np.median(p[sw]):.3f} IQR '
          f'[{np.percentile(p[sw],25):.3f}, {np.percentile(p[sw],75):.3f}], '
          f'share with p_veg<0.5: {(p[sw]<0.5).mean():.3f}')
# who is pulled back: closed-gate reference point kappa*rho vs 0.5
print(f'share of switchable agents with kappa*rho < 0.5: {(P["kappa"]*rho[~imm] < 0.5).mean():.3f}; '
      f'median kappa*rho {np.median(P["kappa"]*rho[~imm]):.3f}; median (theta+1)/2 {np.median((theta[~imm]+1)/2):.3f}; median w {np.median(w[~imm]):.3f}')
