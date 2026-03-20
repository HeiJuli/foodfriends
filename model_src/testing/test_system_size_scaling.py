#!/usr/bin/env python3
"""System-size scaling analysis.

Tests how key observables scale with N:
  1. Degree-amplification exponent (gamma): A ~ k^gamma
  2. Max amplification factor
  3. Mean amplification factor
  4. Gini coefficient (inequality stability)
  5. Critical fraction F_c (onset of acceleration)
  6. CCDF tail exponent
  7. Direct conversions vs cascade credit scaling divergence

N values: 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000
(385 and 2000 from existing runs used as anchors)
10 runs each, twin mode.

Usage:
  python test_system_size_scaling.py              # all sizes
  python test_system_size_scaling.py 500 1000     # specific sizes only
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from datetime import date
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import model_main

DIRECT_REDUCTION_KG = 664
N_RUNS = 10
# Geometric spacing: 500, 1k, 2k, 5k, 10k, 20k, 50k, 100k
ALL_SIZES = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

BASE_PARAMS = {
    "veg_CO2": 1390, "vegan_CO2": 1054, "meat_CO2": 2054,
    "erdos_p": 3, "k": 8, "immune_n": 0.10, "M": 9,
    "veg_f": 0.5, "meat_f": 0.5,
    "p_rewire": 0.01, "rewire_h": 0.1, "tc": 0.7,
    "topology": "homophilic_emp",
    "beta": 13, "alpha": 0.35, "rho": 0.45, "theta": 0,
    "agent_ini": "twin",
    "survey_file": "../data/hierarchical_agents.csv",
    "adjust_veg_fraction": True, "target_veg_fraction": 0.06,
    "tau": 0.035, "theta_gate_c": 0.35, "theta_gate_k": 35,
    "alpha_min": 0.05, "alpha_max": 0.80,
    "mu": 0.2, "gamma": 0.3,
    "tau_persistence": None,
    "snapshot_dense_start": 0,
}


def load_pmf_tables():
    import pickle
    path = os.path.join('..', 'data', 'demographic_pmfs.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_single(args):
    """Worker: run one model, return summary stats."""
    N, run_id, steps = args
    seed = 42 + run_id + N  # unique per (N, run)
    np.random.seed(seed)
    import random; random.seed(seed)

    params = BASE_PARAMS.copy()
    params['N'] = N
    # Scale steps with N: larger systems need more time per agent
    # but cap at 150k to keep runtime sane
    params['steps'] = steps
    params['run'] = run_id

    pmf_tables = load_pmf_tables()
    model = model_main.Model(params, pmf_tables=pmf_tables)

    t0 = time.time()
    model.run()
    elapsed = time.time() - t0

    # Extract observables
    import networkx as nx
    snap_final = model.snapshots.get('final', model.snapshots[max(k for k in model.snapshots if isinstance(k, int))])
    snap_init = model.snapshots[0]
    G = snap_final['graph']
    nodes = list(G.nodes())
    reds = np.array(snap_final['reductions'])
    init_diets = snap_init['diets']
    final_diets = snap_final['diets']

    pos = reds[reds > 0]
    mults = pos / DIRECT_REDUCTION_KG if len(pos) > 0 else np.array([0])
    degrees = np.array([G.degree(n) for n in nodes])

    # 1. Degree-amplification log-log slope
    mask = reds > 0
    k_pos = degrees[mask].astype(float)
    A_pos = (reds[mask] / DIRECT_REDUCTION_KG)
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
    win = min(5001, len(traj) // 3)
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
          f"F_c={fc:.3f}  elapsed={elapsed:.0f}s")

    return {
        'N': N, 'run': run_id, 'steps': steps,
        'f_veg': f_veg, 'avg_degree': avg_deg, 'r_assort': r_assort,
        'gamma': gamma, 'r2_gamma': r2_gamma,
        'mean_mult': mean_mult, 'max_mult': max_mult, 'p90_mult': p90_mult,
        'gini': gini, 'fc': fc, 'alpha_ccdf': alpha_ccdf,
        'dc_slope': dc_slope,
        'n_positive': len(pos), 'n_agents': len(reds),
        'elapsed_s': elapsed,
    }


def summarize(df):
    """Print scaling summary table."""
    print(f"\n{'='*90}")
    print(f"  SYSTEM-SIZE SCALING SUMMARY")
    print(f"{'='*90}")
    print(f"{'N':>7s} {'n':>3s} {'F_veg':>6s} {'gamma':>7s} {'mean_A':>7s} "
          f"{'max_A':>7s} {'Gini':>6s} {'F_c':>6s} {'CCDF_a':>7s} {'dc_slope':>8s}")
    print(f"{'-'*90}")
    for N, grp in df.groupby('N'):
        print(f"{N:>7d} {len(grp):>3d} "
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


if __name__ == '__main__':
    # Parse optional N values from CLI
    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1:]]
    else:
        sizes = ALL_SIZES

    # Scale steps with N: small N converges faster
    def steps_for_N(N):
        if N <= 1000: return 50000
        if N <= 5000: return 100000
        if N <= 20000: return 150000
        return 150000  # cap

    tasks = [(N, run, steps_for_N(N)) for N in sizes for run in range(N_RUNS)]

    n_cores = max(1, int(0.75 * os.cpu_count()))
    total_agents = sum(N * N_RUNS for N in sizes)
    print(f"System-size scaling sweep")
    print(f"  N values: {sizes}")
    print(f"  Runs per N: {N_RUNS}")
    print(f"  Total sims: {len(tasks)}")
    print(f"  Total agent-steps: {sum(N * steps_for_N(N) * N_RUNS for N in sizes):,.0f}")
    print(f"  Cores: {n_cores}")
    print()

    t0 = time.time()
    with Pool(n_cores) as pool:
        results = pool.map(run_single, tasks)

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
