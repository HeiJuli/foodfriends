#!/usr/bin/env python3
"""System-size scaling analysis (corrected).

Tests how key observables scale with N using:
  - steps = UPDATES_PER_AGENT * N  (fair per-agent equilibration)
  - modular network: K = N/COMMUNITY_SIZE communities, each using
    homophilic_emp, with random inter-community weak ties (mu=0.20)
  - KDE synthetic agents for N > 5602 (no cloning)
  - random demographic partition across communities

Corrects artifacts in prior version (2026-03-22):
  1. Steps capped at 150k -> agents at large N never equilibrated
  2. tau_persistence ~ N was correct but steps didn't keep pace
  3. Holme-Kim network had no community structure at any scale
  4. Agent cloning above N=5602 reduced parameter heterogeneity
  See: claude_stuff/system_size_scaling_artifacts_2026-03-23.md

N values: 2000, 4000, 6000, 10000, 20000
  - N=2000 is the baseline (single community, matches validated model)
  - N<=20000 keeps runtime tractable (~50 updates/agent)
  - Larger N possible but expect multi-day runtimes

Usage:
  python test_system_size_scaling.py              # all sizes
  python test_system_size_scaling.py 2000 4000    # specific sizes only
"""
import os, sys, time, pickle, random
import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool
from datetime import date
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import model_main
from auxillary.homophily_network_v2 import generate_homophily_network_v2
from auxillary.sampling_utils import stratified_sample_agents

DIRECT_REDUCTION_KG = 664
N_RUNS = 10
COMMUNITY_SIZE = 2000     # validated model scale
UPDATES_PER_AGENT = 50    # match N=2000 baseline: 100k steps / 2000
MU = 0.20                 # inter-community mixing; Q~0.5 (Newman 2006)
ATTR_WEIGHTS = np.array([0.20, 0.35, 0.18, 0.32, 0.05])

ALL_SIZES = [2000, 4000, 6000, 10000, 20000, 100000]

BASE_PARAMS = {
    "veg_CO2": 1390, "vegan_CO2": 1054, "meat_CO2": 2054,
    "erdos_p": 3, "k": 8, "immune_n": 0.10, "M": 9,
    "veg_f": 0.5, "meat_f": 0.5,
    "p_rewire": 0.01, "rewire_h": 0.1, "tc": 0.7,
    "topology": "prebuilt",  # we inject the modular graph
    "beta": 13, "alpha": 0.35, "rho": 0.45, "theta": 0,
    "agent_ini": "twin",
    "survey_file": "../data/hierarchical_agents.csv",
    "adjust_veg_fraction": True, "target_veg_fraction": 0.06,
    "tau": 0.035, "theta_gate_c": 0.35, "theta_gate_k": 35,
    "alpha_min": 0.05, "alpha_max": 0.80,
    "mu": 0.2, "gamma": 0.3,
    "tau_persistence": None,  # auto: M*2*N (scales correctly when steps=50*N)
    "snapshot_dense_start": 0,
}


# ---------------------------------------------------------------------------
#  Synthetic agent generation (KDE, no cloning)
# ---------------------------------------------------------------------------

def generate_synthetic_agents(empirical_df, n_target, random_state=42):
    """Generate n_target agents preserving multivariate structure.

    Uses Gaussian KDE on complete cases for (theta, rho, alpha),
    samples demographics from empirical marginals.
    All agents returned as complete (has_rho=True, has_alpha=True).
    """
    rng = np.random.RandomState(random_state)
    if n_target <= len(empirical_df):
        return stratified_sample_agents(
            empirical_df, n_target,
            strata_cols=['gender', 'age_group', 'incquart', 'educlevel'],
            random_state=random_state, verbose=False
        ).reset_index(drop=True)

    base = empirical_df.copy()
    n_extra = n_target - len(base)

    # Fit KDE on complete cases for continuous params
    complete = base[base['has_alpha'] & base['has_rho']]
    continuous = complete[['theta', 'rho', 'alpha']].values.T
    kde = gaussian_kde(continuous)
    synth_cont = kde.resample(n_extra, seed=random_state).T

    # Clip to valid ranges
    synth_cont[:, 0] = np.clip(synth_cont[:, 0], -1, 1)   # theta
    synth_cont[:, 1] = np.clip(synth_cont[:, 1], 0, 1)    # rho
    synth_cont[:, 2] = np.clip(synth_cont[:, 2], 0, 1)    # alpha

    # Demographics: stratified resample from full empirical set
    demo_cols = ['gender', 'age_group', 'incquart', 'educlevel', 'diet']
    demo_sample = base[demo_cols].sample(
        n=n_extra, replace=True, random_state=random_state
    ).reset_index(drop=True)

    synthetic = demo_sample.copy()
    synthetic['theta'] = synth_cont[:, 0]
    synthetic['rho'] = synth_cont[:, 1]
    synthetic['alpha'] = synth_cont[:, 2]
    synthetic['has_rho'] = True
    synthetic['has_alpha'] = True
    synthetic['nomem_encr'] = [f'synth_{i}' for i in range(n_extra)]

    # Assign diet consistent with theta: high theta -> more likely veg
    # (preserves empirical theta-diet relationship)
    theta_sorted = synthetic.sort_values('theta', ascending=False)
    n_veg_target = int(0.06 * n_extra)
    synthetic.loc[theta_sorted.index[:n_veg_target], 'diet'] = 'veg'
    synthetic.loc[theta_sorted.index[n_veg_target:], 'diet'] = 'meat'

    result = pd.concat([base, synthetic], ignore_index=True)
    print(f"INFO: {len(base)} empirical + {n_extra} KDE-synthetic = {len(result)} agents")
    return result


# ---------------------------------------------------------------------------
#  Modular network generation
# ---------------------------------------------------------------------------

def generate_modular_network(N, agents_df, community_size=COMMUNITY_SIZE,
                              mu=MU, seed=42):
    """Generate modular network: K communities of ~community_size each,
    connected by sparse random inter-community weak ties.

    Intra-community: homophilic_emp (Holme-Kim + homophily), avg_degree=8.
    Inter-community: random edges, ~mu/(1-mu) * intra_edges total.

    Args:
        N: total agents
        agents_df: DataFrame with demographics + theta (len == N)
        community_size: target size per community (default 2000)
        mu: mixing parameter — fraction of edges that are inter-community
            (Newman 2006: Q~0.5 for social networks -> mu~0.20)
        seed: random seed

    Returns:
        G: nx.Graph with N nodes
        community_labels: np.array of community assignments
        n_communities: int
    """
    rng = np.random.RandomState(seed)

    # Partition into K communities (random, stratified by demographics)
    K = max(1, N // community_size)
    indices = np.arange(N)
    rng.shuffle(indices)
    communities = [indices[i * (N // K): (i + 1) * (N // K)] for i in range(K)]
    # Distribute remainder
    remainder = indices[K * (N // K):]
    for i, idx in enumerate(remainder):
        communities[i % K] = np.append(communities[i % K], idx)

    community_labels = np.zeros(N, dtype=int)
    for c, members in enumerate(communities):
        community_labels[members] = c

    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Generate intra-community networks
    total_intra = 0
    for c, members in enumerate(communities):
        size = len(members)
        subset_df = agents_df.iloc[members].reset_index(drop=True)
        G_sub, _ = generate_homophily_network_v2(
            N=size, avg_degree=8, agents_df=subset_df,
            attribute_weights=ATTR_WEIGHTS,
            seed=seed + c, tc=0.7
        )
        # Remap node IDs to global indices
        member_list = list(members)
        for u, v in G_sub.edges():
            G.add_edge(member_list[u], member_list[v])
        total_intra += G_sub.number_of_edges()

    # Inter-community weak ties (random, Granovetter 1973)
    # Target: mu = inter / (inter + intra) -> inter = mu/(1-mu) * intra
    inter_added = 0
    if K > 1:
        n_inter_target = int(mu / (1 - mu) * total_intra)
        max_attempts = n_inter_target * 5
        attempts = 0
        while inter_added < n_inter_target and attempts < max_attempts:
            c1, c2 = rng.choice(K, size=2, replace=False)
            u = rng.choice(communities[c1])
            v = rng.choice(communities[c2])
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                inter_added += 1
            attempts += 1

    total_edges = G.number_of_edges()
    actual_mu = inter_added / total_edges if total_edges > 0 else 0
    degrees = [d for _, d in G.degree()]
    print(f"INFO: Modular network: {K} communities, {total_intra} intra + "
          f"{inter_added} inter edges (mu={actual_mu:.3f}), "
          f"avg_degree={np.mean(degrees):.1f}")

    return G, community_labels, K


# ---------------------------------------------------------------------------
#  PMF tables loader
# ---------------------------------------------------------------------------

def load_pmf_tables():
    path = os.path.join('..', 'data', 'demographic_pmfs.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
#  Single run worker
# ---------------------------------------------------------------------------

def run_single(args):
    """Worker: run one model, return summary stats."""
    N, run_id, steps = args
    seed = 42 + run_id * 1000 + N  # unique per (N, run), no collisions
    np.random.seed(seed)
    random.seed(seed)

    params = BASE_PARAMS.copy()
    params['N'] = N
    params['steps'] = steps
    params['run'] = run_id

    # Load empirical data + generate synthetic agents if needed
    empirical_df = pd.read_csv(params['survey_file'])
    agents_df = generate_synthetic_agents(empirical_df, N, random_state=seed)

    # Build modular network
    G_mod, comm_labels, n_communities = generate_modular_network(
        N, agents_df, seed=seed
    )

    # Write agents to temp CSV for model to load (twin mode reads CSV)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir='/tmp')
    agents_df.to_csv(tmp.name, index=False)
    params['survey_file'] = tmp.name

    pmf_tables = load_pmf_tables()
    model = model_main.Model(params, pmf_tables=pmf_tables)

    # Inject pre-built modular network (replaces empty placeholder)
    model.G1 = G_mod

    t0 = time.time()
    model.run()
    elapsed = time.time() - t0

    # Clean up temp file
    os.unlink(tmp.name)

    # Extract observables (same as original)
    snap_final = model.snapshots.get('final', model.snapshots[max(
        k for k in model.snapshots if isinstance(k, int))])
    snap_init = model.snapshots[0]
    G = snap_final['graph']
    nodes = list(G.nodes())
    reds = np.array(snap_final['reductions'])

    pos = reds[reds > 0]
    mults = pos / DIRECT_REDUCTION_KG if len(pos) > 0 else np.array([0])
    degrees = np.array([G.degree(n) for n in nodes])

    # 1. Degree-amplification log-log slope (gamma)
    mask = reds > 0
    k_pos = degrees[mask].astype(float)
    A_pos = reds[mask] / DIRECT_REDUCTION_KG
    valid = k_pos > 0
    gamma, r2_gamma = np.nan, np.nan
    if valid.sum() > 10:
        lk, lA = np.log10(k_pos[valid]), np.log10(A_pos[valid])
        coeffs = np.polyfit(lk, lA, 1)
        gamma = coeffs[0]
        ss_res = np.sum((lA - (gamma * lk + coeffs[1]))**2)
        ss_tot = np.sum((lA - np.mean(lA))**2)
        r2_gamma = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 2-3. Amplification stats
    mean_mult = np.mean(mults) if len(mults) > 0 else 0
    max_mult = np.max(mults) if len(mults) > 0 else 0
    p90_mult = np.percentile(mults, 90) if len(mults) > 0 else 0

    # 4. Gini
    gini = np.nan
    if len(pos) > 1:
        n = len(pos)
        sorted_r = np.sort(pos)
        gini = (2 * np.sum(np.arange(1, n+1) * sorted_r) / (n * np.sum(sorted_r))) - (n + 1) / n

    # 5. Critical fraction (max d2F/dt2, F<0.5)
    traj = np.array(model.fraction_veg, dtype=float)
    fc = np.nan
    win = min(10001, len(traj) // 3)
    if win % 2 == 0: win -= 1
    if win >= 5 and len(traj) > win * 2:
        smoothed = savgol_filter(traj, window_length=win, polyorder=3)
        d2 = savgol_filter(traj, window_length=win, polyorder=3, deriv=2)
        burnin = min(5000, len(traj) // 10)
        d2[:burnin] = 0
        d2_masked = d2.copy()
        d2_masked[smoothed > 0.5] = 0
        idx = np.argmax(d2_masked)
        if d2_masked[idx] > 0:
            fc = smoothed[idx]

    # 6. CCDF tail slope
    alpha_ccdf = np.nan
    if len(pos) > 20:
        pos_t = pos / 1000
        sorted_x = np.sort(pos_t)
        ccdf_y = 1.0 - np.arange(1, len(sorted_x) + 1) / len(sorted_x)
        m = ccdf_y > 0
        lx, ly = np.log10(sorted_x[m]), np.log10(ccdf_y[m])
        tail = lx > np.median(lx)
        if tail.sum() > 5:
            c = np.polyfit(lx[tail], ly[tail], 1)
            alpha_ccdf = -c[0]

    # 7. Direct conversions scaling
    dc_slope = np.nan
    if 'direct_conversions' in snap_final:
        dc = np.array(snap_final['direct_conversions'])
        dc_pos_mask = (dc > 0) & (degrees > 0)
        if dc_pos_mask.sum() > 10:
            lk_dc = np.log10(degrees[dc_pos_mask].astype(float))
            l_dc = np.log10(dc[dc_pos_mask].astype(float))
            dc_slope = np.polyfit(lk_dc, l_dc, 1)[0]

    # Network stats
    f_veg = snap_final['veg_fraction']
    avg_deg = np.mean(degrees)
    r_assort = nx.degree_assortativity_coefficient(G)

    print(f"  N={N:>6d} run={run_id:>2d}  F_veg={f_veg:.3f}  gamma={gamma:.2f}  "
          f"mean_A={mean_mult:.1f}x  max_A={max_mult:.0f}x  Gini={gini:.3f}  "
          f"F_c={fc:.3f}  comms={n_communities}  elapsed={elapsed:.0f}s")

    return {
        'N': N, 'run': run_id, 'steps': steps,
        'n_communities': n_communities,
        'f_veg': f_veg, 'avg_degree': avg_deg, 'r_assort': r_assort,
        'gamma': gamma, 'r2_gamma': r2_gamma,
        'mean_mult': mean_mult, 'max_mult': max_mult, 'p90_mult': p90_mult,
        'gini': gini, 'fc': fc, 'alpha_ccdf': alpha_ccdf,
        'dc_slope': dc_slope,
        'n_positive': len(pos), 'n_agents': len(reds),
        'elapsed_s': elapsed,
    }


# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

def summarize(df):
    """Print scaling summary table."""
    print(f"\n{'='*90}")
    print(f"  SYSTEM-SIZE SCALING SUMMARY (corrected)")
    print(f"{'='*90}")
    print(f"{'N':>7s} {'n':>3s} {'K':>3s} {'F_veg':>6s} {'gamma':>7s} {'mean_A':>7s} "
          f"{'max_A':>7s} {'Gini':>6s} {'F_c':>6s} {'CCDF_a':>7s} {'dc_slope':>8s}")
    print(f"{'-'*90}")
    for N, grp in df.groupby('N'):
        print(f"{N:>7d} {len(grp):>3d} "
              f"{int(grp['n_communities'].median()):>3d} "
              f"{grp['f_veg'].median():>6.3f} "
              f"{grp['gamma'].median():>7.2f} ({grp['gamma'].std():>4.2f}) "
              f"{grp['mean_mult'].median():>5.1f}x "
              f"{grp['max_mult'].median():>5.0f}x "
              f"{grp['gini'].median():>6.3f} "
              f"{grp['fc'].median():>6.3f} "
              f"{grp['alpha_ccdf'].median():>7.2f} "
              f"{grp['dc_slope'].median():>8.2f}")

    # Log-log regression: max_mult ~ N
    grouped = df.groupby('N').agg({'max_mult': 'median', 'mean_mult': 'median'}).reset_index()
    lN = np.log10(grouped['N'].values.astype(float))
    for col in ['max_mult', 'mean_mult']:
        lY = np.log10(grouped[col].values)
        valid = np.isfinite(lY)
        if valid.sum() > 2:
            c = np.polyfit(lN[valid], lY[valid], 1)
            print(f"\n  {col} ~ N^{c[0]:.3f}  (log-log slope)")

    # Gamma stability
    gammas = df.groupby('N')['gamma'].agg(['median', 'std']).reset_index()
    print(f"\n  Degree-amplification exponent (gamma) across N:")
    for _, r in gammas.iterrows():
        print(f"    N={r['N']:>6.0f}:  gamma = {r['median']:.3f} +/- {r['std']:.3f}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1:]]
    else:
        sizes = ALL_SIZES

    def steps_for_N(N):
        return UPDATES_PER_AGENT * N

    n_cores = max(1, int(0.75 * os.cpu_count()))
    runs_per = {N: (3 if N >= 100000 else N_RUNS) for N in sizes}
    total_tasks = sum(runs_per[N] for N in sizes)
    print(f"System-size scaling sweep (corrected)")
    print(f"  N values: {sizes}")
    print(f"  Runs per N: {dict(runs_per)}")
    print(f"  Total sims: {total_tasks}")
    print(f"  Updates/agent: {UPDATES_PER_AGENT}")
    print(f"  Community size: {COMMUNITY_SIZE}")
    print(f"  Inter-community mu: {MU}")
    print(f"  Total agent-steps: {sum(N * steps_for_N(N) * runs_per[N] for N in sizes):,.0f}")
    print(f"  Cores: {n_cores}")
    print()

    t0 = time.time()
    results = []
    for N in sizes:
        n_runs = 3 if N >= 100000 else N_RUNS
        tasks = [(N, run, steps_for_N(N)) for run in range(n_runs)]
        # Cap concurrency for large N to prevent OOM
        n_workers = max(1, min(n_cores, 1 if N >= 100000 else 2 if N >= 20000 else 4 if N >= 10000 else n_cores))
        print(f"  Running N={N} ({n_runs} runs, {n_workers} workers, "
              f"{steps_for_N(N):,} steps)...")
        with Pool(n_workers) as pool:
            results.extend(pool.map(run_single, tasks))

    df = pd.DataFrame(results)
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {int(elapsed/60)}m {elapsed%60:.0f}s")

    summarize(df)

    # Save
    outdir = os.path.join('..', 'model_output')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f'system_size_scaling_{date.today().strftime("%Y%m%d")}.pkl')
    df.to_pickle(outfile)
    print(f"\nSaved: {outfile}")
