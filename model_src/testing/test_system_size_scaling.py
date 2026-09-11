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
  See: claude_stuff/Infrastructure/system_size_scaling_artifacts_2026-03-23.md

Amplification is replayed from the event log by analysis/attribution_ledger.py
(exposure-proportional parents, no dwell weight) at each run's own t_end, on two
ledgers: veg-time, the reported unit since 2026-09-09 (*_time columns; the reported
factor is 1 + A over credited agents, as in analysis/vegtime_accounting.py), and
event count (mean_mult, max_mult, p90_mult, gamma, gini, alpha_ccdf), kept for
continuity with the sensitivity and null scripts. The in-simulation ledger
(last-draw, dwell-weighted) is kept as mean_mult_sub / max_mult_sub / n_positive_sub.
Per-agent values are saved (*_agents columns), so a new statistic is a re-score of
the pickle, not another sweep.

N values: 2000, 4000, 6000, 10000, 20000
  - N=2000 is the baseline (single community, matches validated model)
  - N<=20000 keeps runtime tractable (adaptive stop, 210-270 updates/agent)
  - Larger N possible but expect multi-day runtimes

Usage:
  python test_system_size_scaling.py              # all sizes
  python test_system_size_scaling.py 2000 4000    # specific sizes only
  python test_system_size_scaling.py --updates 500 --runs 5 --no-stop 2000
                                                  # calibration: fixed length, no stop

Run length is adaptive (see the stop constants below): each run ends when its own
logistic t_end has settled, with UPDATES_PER_AGENT as the ceiling.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../analysis'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import model_main
from attribution_ledger import replay, veg_time
from t_end_logistic import estimate_t_end, t_end_with_status, fc_window, fit_params
from auxillary.homophily_network_v2 import generate_homophily_network_v2
from auxillary.sampling_utils import stratified_sample_agents

DIRECT_REDUCTION_KG = 664
N_RUNS = 10
COMMUNITY_SIZE = 2000     # validated model scale
# Run length is adaptive, not a constant: no fixed number of updates/agent serves
# both ends of a sweep that spans a factor of ten in N (the 2026-09-07 pilot's
# unclamped requirement was a median of 218 at N=2000 and 228 at N=4000, worst seed
# 295 and 306, and rising with N). Each run stops when its own logistic t_end has
# settled, and the constant below is only the ceiling. Criterion and its validation
# against the 2026-09-08 calibration: server_runbook_kappa s.2.3.
UPDATES_PER_AGENT = 350   # CEILING, not a length. Job 1 of the calibration ran to 500
                          # and found t_end(prefix) converged to within ~1% by 350 at
                          # both sizes, so 350 gives up nothing and saves 30% of the
                          # sweep. A run that hits it is recorded stop_reason="ceiling".
STOP_FLOOR_UPDATES = 100  # no run stops before this; the fit is still moving fast below it
STOP_CHECK_UPDATES = 10   # check cadence, in updates/agent -- NOT a fixed step count. 50k
                          # steps would be 25 updates/agent of resolution at N=2000 and 2.5
                          # at N=20000, i.e. a size-dependent stopping bias inside a
                          # scaling measurement.
STOP_MARGIN = 1.30        # stop at t >= 1.30 * t_end(prefix). 1.10 was adopted on the
                          # argument that at margin 1.0 the estimate is still falling where it
                          # crosses the prefix; the margin test (measurement 5, 2026-09-08,
                          # one N=2000 seed replayed at each margin) then measured what the
                          # choice is worth and 1.10 did not survive it. Against the same run
                          # left to 500 updates/agent, mean amplification comes out +18.9% at
                          # margin 1.0, +14.9% at 1.1 and +7.6% at 1.3, and gamma is 0.997 /
                          # 0.985 / 0.969 against a converged 0.969 -- monotone, so a later
                          # stop is not merely different but closer. 1.0 -> 1.3 moves mean_A by
                          # 1.78 seed-sd, which is the pre-committed threshold for paying the
                          # 12.5% extra compute (270 vs 240 updates/agent).
STOP_R2_MIN = 0.99        # guard, not a test: satisfied from ~60 updates/agent onward.
STOP_ASYMPTOTE_MAX = 1.0  # this one does the work. F_veg <= 1 is a hard bound, and the
                          # fitted asymptote sits above 1.0 across the whole 100-190 band,
                          # dropping below it only at ~250 updates/agent.
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
    "kappa": 0.55,            # intention-behaviour discount (Webb & Sheeran 2006)
    "tau_persistence": None,  # auto: M*2*N (scales correctly when steps=UPDATES_PER_AGENT*N)
    "snapshot_dense_start": 0,
}

# This script builds its own BASE_PARAMS rather than importing the runner's, so an
# absent key falls through to model_main's .get("kappa", 1.0) and the whole sweep
# runs at face value while everything else runs at 0.55. Same failure class as the
# tau = 11,700 incident; assert on it.
assert BASE_PARAMS.get("kappa") == 0.55, "ERROR: kappa missing from BASE_PARAMS"


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
#  Adaptive stop
# ---------------------------------------------------------------------------

def make_stop_check(N):
    """Stop once the fitted t_end has settled: t >= STOP_MARGIN * t_end(prefix).

    The estimator being watched is the one that later sets the credit window, so
    the stop and the recorded t_end are the same number measured twice; using a
    cheaper proxy here would put the two out of step. Validated offline against the
    20 pilot trajectories and the 10 calibration runs: it never fires early (below
    ~250 updates/agent the fit clamps to the prefix, and 1.10 x prefix > prefix
    declines to stop), and the asymptote guard is what holds it back, not the margin.

    Cost is a savgol pass plus a curve_fit over the prefix on each check: measured
    49 s of fitting for a full 100-350 grid at N=2000, i.e. 13% of a run that goes
    all the way to the ceiling and ~8% of one that stops at 230. It falls as 1/N --
    the fit is linear in run length, the run itself is N^2.
    """
    floor = STOP_FLOOR_UPDATES * N

    def stop_check(model, t):
        if t < floor:
            return False
        fit = fit_params(model.fraction_veg)
        return (fit is not None
                and fit['r2'] >= STOP_R2_MIN
                and fit['asymptote'] <= STOP_ASYMPTOTE_MAX
                and t >= STOP_MARGIN * fit['t_end'])

    return stop_check


# ---------------------------------------------------------------------------
#  Single run worker
# ---------------------------------------------------------------------------

def run_single(args):
    """Worker: run one model, return summary stats."""
    N, run_id, steps, adaptive = args
    seed = 42 + run_id * 1000 + N  # unique per (N, run), no collisions
    np.random.seed(seed)
    random.seed(seed)

    params = BASE_PARAMS.copy()
    params['N'] = N
    params['steps'] = steps
    # 200 degree samples per run, so the grid follows run length instead of a
    # constant: 0.5% of the run is far finer than the drift gamma is sensitive to,
    # and the memory stays bounded at the large N (16 MB/run at N=20000).
    params['degree_sample_every'] = max(1, steps // 200)
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
    model.run(stop_check=make_stop_check(N) if adaptive else None,
              stop_every=STOP_CHECK_UPDATES * N if adaptive else 0)
    elapsed = time.time() - t0

    # The run may have stopped short of the ceiling, and params["steps"] is what
    # replay() takes as its default t_end and what a later re-analysis reads as the
    # run length -- leaving the ceiling there credits to a length never reached.
    steps_run = len(model.fraction_veg) - 1
    model.params['steps'] = steps_run
    stop_reason = 'converged' if steps_run < steps else 'ceiling'

    # Clean up temp file
    os.unlink(tmp.name)

    # Extract observables (same as original)
    snap_final = model.snapshots.get('final', model.snapshots[max(
        k for k in model.snapshots if isinstance(k, int))])
    snap_init = model.snapshots[0]
    G = snap_final['graph']
    nodes = list(G.nodes())
    # Primary credit convention (exposure-proportional parents, event count, no
    # dwell weight), replayed from the event log; snap_final['reductions'] is the
    # submitted convention and is kept only for the _sub comparison below.
    # Amplification is credited at each run's OWN fitted t_end, matching
    # sensitivity_campaign and the headline ensemble. Credit accrues for as long
    # as the run lasts and churn continues at the plateau, so replaying to
    # t = steps compares sizes at whatever post-saturation tail each happens to
    # have -- the fixed-window problem removed from the OAT sweep in cf5a50b.
    traj_full = np.asarray(model.fraction_veg, dtype=float)
    t_end_fit, t_end_status, t_end_r2 = t_end_with_status(traj_full)
    reds = replay(model.events, model.snapshots[0]['diets'], model.params,
                  parent="exposure", weight="none", unit="event", t_end=t_end_fit)
    reds_sub = np.array(snap_final['reductions'])

    # Veg-time is the reported unit since 2026-09-09: A_i = credited vegetarian-time
    # / own vegetarian-time. It is NOT a rescaling of the event-count ledger -- the
    # denominator is per agent -- and gamma does not survive the switch (+0.126 at
    # N=2000, same sign in 50/50 runs, analysis/gamma_ledger_check.py). Both units are
    # therefore scored here; the event-count columns stay for continuity with the
    # sensitivity and null scripts, which are still on that unit.
    own = veg_time(model.events, model.snapshots[0]['diets'], t_end_fit)
    reds_t = replay(model.events, model.snapshots[0]['diets'], model.params,
                    parent="exposure", weight="none", unit="time", t_end=t_end_fit)
    A_time = np.divide(reds_t, DIRECT_REDUCTION_KG * own,
                       out=np.zeros_like(reds_t, dtype=float), where=own > 0)

    pos = reds[reds > 0]
    mults = pos / DIRECT_REDUCTION_KG if len(pos) > 0 else np.array([0])
    # Degrees must be read where the credit window closes. Rewiring runs at
    # 0.005/step throughout, so the final network's degrees are partly decorrelated
    # from the ones that earned the credit, and the log-log slope is attenuated by
    # that correlation (0.785 at 500 upd/agent, verified 2026-09-08). t_deg is
    # recorded so a later reader can see how far the two times were apart.
    if model.degree_history:
        t_deg, degrees = min(model.degree_history, key=lambda th: abs(th[0] - t_end_fit))
    else:
        t_deg, degrees = steps_run, np.array([G.degree(n) for n in nodes])

    # 1. Degree-amplification log-log slope (gamma)
    mask = reds > 0
    k_pos = degrees[mask].astype(float)
    A_pos = reds[mask] / DIRECT_REDUCTION_KG
    valid = k_pos > 0

    def _loglog(k, A):
        v = (k > 0) & (A > 0)
        if v.sum() <= 10:
            return np.nan, np.nan
        lk, lA = np.log10(k[v]), np.log10(A[v])
        g, c = np.polyfit(lk, lA, 1)
        ss_res = np.sum((lA - (g * lk + c))**2)
        ss_tot = np.sum((lA - np.mean(lA))**2)
        return g, (1 - ss_res / ss_tot if ss_tot > 0 else 0)

    gamma, r2_gamma = _loglog(k_pos, A_pos)
    gamma_time, r2_gamma_time = _loglog(degrees.astype(float), A_time)

    # 2-3. Amplification stats
    mean_mult = np.mean(mults) if len(mults) > 0 else 0
    max_mult = np.max(mults) if len(mults) > 0 else 0
    p90_mult = np.percentile(mults, 90) if len(mults) > 0 else 0
    pos_sub = reds_sub[reds_sub > 0]
    mults_sub = pos_sub / DIRECT_REDUCTION_KG if len(pos_sub) > 0 else np.array([0])
    mean_mult_sub = np.mean(mults_sub)
    max_mult_sub = np.max(mults_sub)
    pos_t = A_time[A_time > 0]
    mean_A_time = np.mean(pos_t) if len(pos_t) else 0
    max_A_time = np.max(pos_t) if len(pos_t) else 0
    p90_A_time = np.percentile(pos_t, 90) if len(pos_t) else 0
    median_A_time = np.median(pos_t) if len(pos_t) else 0
    sys_A_time = (reds_t.sum() / (DIRECT_REDUCTION_KG * own.sum())
                  if own.sum() > 0 else np.nan)

    # 4. Gini
    def _gini(x):
        if len(x) <= 1 or np.sum(x) <= 0:
            return np.nan
        n = len(x); sx = np.sort(x)
        return (2 * np.sum(np.arange(1, n+1) * sx) / (n * np.sum(sx))) - (n + 1) / n

    gini = _gini(pos)
    gini_time = _gini(pos_t)

    # 5. Critical fraction (max d2F/dt2, F<0.5). Kept in the pickle, NOT reported:
    # fc_window is 20% of the run, so F_c is incomparable across the variable-length
    # runs an adaptive stop produces, and at these lengths it is degenerate besides
    # -- 20% of a 350-update run is a 100-update kernel searching a ~50-update band,
    # which returned 8.5e-21 at N=4000 in the calibration. Headline F_c comes from
    # the kappa ensemble, not from here.
    traj = traj_full
    fc = np.nan
    win = fc_window(len(traj))          # 20% of the run, the shared convention
    if win >= 5 and len(traj) > win * 2:
        smoothed = savgol_filter(traj, window_length=win, polyorder=3)
        d2 = savgol_filter(traj, window_length=win, polyorder=3, deriv=2)
        d2[:win] = 0                    # mask a whole kernel, not a fixed 5000
        d2_masked = d2.copy()
        d2_masked[smoothed > 0.5] = 0
        idx = np.argmax(d2_masked)
        if d2_masked[idx] > 0:
            fc = smoothed[idx]

    # 6. CCDF tail slope, OLS on the log-log CCDF above the median, both ledgers.
    # Veg-time is fitted on 1 + A over credited agents, the quantity Fig. 1B plots
    # (publication_plots_main.py:687): the delta scale does not move a slope, the +1 does.
    def _ccdf_tail(x):
        if len(x) <= 20:
            return np.nan
        sx = np.sort(x)
        ccdf_y = 1.0 - np.arange(1, len(sx) + 1) / len(sx)
        m = ccdf_y > 0
        lx, ly = np.log10(sx[m]), np.log10(ccdf_y[m])
        tail = lx > np.median(lx)
        return -np.polyfit(lx[tail], ly[tail], 1)[0] if tail.sum() > 5 else np.nan

    alpha_ccdf = _ccdf_tail(pos / 1000)
    alpha_ccdf_time = _ccdf_tail(1.0 + pos_t)

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
          f"g_t={gamma_time:.2f}  meanA_t={mean_A_time:.2f}  sys1+A_t={1 + sys_A_time:.2f}  "
          f"upd={steps_run/N:.0f}/agent ({stop_reason})  comms={n_communities}  "
          f"elapsed={elapsed:.0f}s")

    return {
        'N': N, 'run': run_id, 'steps': steps_run,
        'steps_ceiling': steps, 'stop_reason': stop_reason,
        'upd_per_agent': steps_run / N,
        'kappa': params.get('kappa', 1.0),
        't_end_fit': t_end_fit, 't_end_status': t_end_status,
        't_end_r2': t_end_r2, 'fc_win': win, 't_deg': t_deg,
        # decimated trajectory, so a changed window or estimator can be
        # re-scored offline rather than forcing another 92 core-hour sweep
        # (stride 100 costs <= 35 steps in 300k on the t_end refit)
        'traj_ds': traj_full[::100].astype(np.float32),
        'n_communities': n_communities,
        'f_veg': f_veg, 'avg_degree': avg_deg, 'r_assort': r_assort,
        'gamma': gamma, 'r2_gamma': r2_gamma,
        'gamma_time': gamma_time, 'r2_gamma_time': r2_gamma_time,
        'mean_A_time': mean_A_time, 'max_A_time': max_A_time,
        'p90_A_time': p90_A_time, 'sys_A_time': sys_A_time,
        'median_A_time': median_A_time, 'alpha_ccdf_time': alpha_ccdf_time,
        'gini_time': gini_time, 'n_positive_time': int((A_time > 0).sum()),
        # per-agent values at the credit window, index-aligned: any distribution
        # statistic is then a re-score of this pickle (~5 MB for the sweep)
        'A_time_agents': A_time.astype(np.float32),
        'own_time_agents': own.astype(np.float32),
        'reds_event_agents': np.asarray(reds, dtype=np.float32),
        'degree_agents': np.asarray(degrees, dtype=np.int32),
        'init_veg_agents': np.array([d == 'veg' for d in model.snapshots[0]['diets']]),
        'mean_mult': mean_mult, 'max_mult': max_mult, 'p90_mult': p90_mult,
        'mean_mult_sub': mean_mult_sub, 'max_mult_sub': max_mult_sub,
        'n_positive_sub': len(pos_sub),
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
    print(f"{'N':>7s} {'n':>3s} {'cens':>4s} {'ceil':>4s} {'upd':>4s} {'K':>3s} "
          f"{'F_veg':>6s} {'gamma':>7s} {'mean_A':>7s} {'max_A':>7s} {'Gini':>6s} "
          f"{'CCDF_a':>7s} {'dc_slope':>8s} {'g_time':>7s} {'meanA_t':>8s}")
    print(f"{'-'*90}")
    for N, grp in df.groupby('N'):
        # A clamped t_end means the logistic could not place t_95 inside the run, so
        # amplification was credited over the whole run rather than to t_end. That is
        # not the same as "did not saturate" -- the fit inflates the asymptote on a
        # slow tail -- but it does mean the credit window differs across runs, so the
        # count has to be visible next to the medians it distorts.
        n_cens = (int((grp['t_end_status'] == 'beyond_run').sum())
                  if 't_end_status' in grp else -1)   # -1: pre-2026-09-07 pkl, no status
        # Under the adaptive stop cens should be 0 by construction; a ceiling hit is
        # the thing to watch, since that run is the one whose length was imposed
        # rather than measured.
        n_ceil = (int((grp['stop_reason'] == 'ceiling').sum())
                  if 'stop_reason' in grp else -1)
        upd = grp['upd_per_agent'].median() if 'upd_per_agent' in grp else np.nan
        print(f"{N:>7d} {len(grp):>3d} {n_cens:>4d} {n_ceil:>4d} {upd:>4.0f} "
              f"{int(grp['n_communities'].median()):>3d} "
              f"{grp['f_veg'].median():>6.3f} "
              f"{grp['gamma'].median():>7.2f} ({grp['gamma'].std():>4.2f}) "
              f"{grp['mean_mult'].median():>5.1f}x "
              f"{grp['max_mult'].median():>5.0f}x "
              f"{grp['gini'].median():>6.3f} "
              f"{grp['alpha_ccdf'].median():>7.2f} "
              f"{grp['dc_slope'].median():>8.2f} "
              f"{grp['gamma_time'].median():>7.2f} "
              f"{grp['mean_A_time'].median():>8.2f}")

    # The reported unit. The table above is event-count apart from g_time/meanA_t.
    print(f"\n  Veg-time, reported unit (1 + A over credited agents; max_A bare):")
    print(f"{'N':>7s} {'sys':>6s} {'mean':>6s} {'median':>6s} {'p90':>6s} {'max_A':>6s} "
          f"{'Gini':>6s} {'CCDF_a':>7s} {'gamma':>6s}")
    for N, grp in df.groupby('N'):
        m = grp.median(numeric_only=True)
        print(f"{N:>7d} {1 + m['sys_A_time']:>6.3f} {1 + m['mean_A_time']:>6.3f} "
              f"{1 + m['median_A_time']:>6.3f} {1 + m['p90_A_time']:>6.3f} "
              f"{m['max_A_time']:>6.1f} {m['gini_time']:>6.3f} "
              f"{m['alpha_ccdf_time']:>7.2f} {m['gamma_time']:>6.3f}")

    # Log-log slopes on N of the per-size medians; veg-time levels as 1 + A
    med = df.groupby('N').median(numeric_only=True)
    for col in ['sys_A_time', 'mean_A_time', 'median_A_time']:
        med[f'1+{col}'] = 1 + med[col]
    lN = np.log10(med.index.values.astype(float))
    print()
    for col in ['max_mult', 'mean_mult', '1+sys_A_time', '1+mean_A_time',
                '1+median_A_time', 'max_A_time', 'gini_time']:
        lY = np.log10(med[col].values.astype(float))
        valid = np.isfinite(lY)
        if valid.sum() > 2:
            c = np.polyfit(lN[valid], lY[valid], 1)
            print(f"  {col} ~ N^{c[0]:.3f}  (log-log slope)")

    # Gamma stability, both units. The event-count column is the historical one;
    # veg-time is the reported ledger, and the two differ by a fixed offset only if
    # the offset is flat in N -- which is the whole point of printing them together.
    g = df.groupby('N')[['gamma', 'gamma_time']].agg(['median', 'std'])
    print(f"\n  Degree-amplification exponent (gamma) across N:")
    print(f"    {'N':>6s}  {'event-count':>18s}  {'veg-time':>18s}  {'offset':>7s}")
    for N, r in g.iterrows():
        d = r[('gamma_time', 'median')] - r[('gamma', 'median')]
        print(f"    {N:>6.0f}  {r[('gamma', 'median')]:>9.3f} +/- {r[('gamma', 'std')]:<6.3f}"
              f"  {r[('gamma_time', 'median')]:>9.3f} +/- {r[('gamma_time', 'std')]:<6.3f}"
              f"  {d:>+7.3f}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --updates/--runs override the constants for a calibration run without
    # editing them; positional args are the sizes, as before.
    args = sys.argv[1:]
    # --no-stop reproduces a calibration run: plain fixed-length runs, no stop logic,
    # which is what the criterion above was settled from and what re-settling it would
    # need again. With it, --updates is a length; without it, --updates is the ceiling.
    adaptive = "--no-stop" not in args
    if not adaptive:
        args.remove("--no-stop")
    opts = {}
    for flag in ("--updates", "--runs"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = int(args[i + 1])
            del args[i:i + 2]
    # --tag keeps a same-day relaunch from overwriting the date-tagged pickle
    tag = ""
    if "--tag" in args:
        i = args.index("--tag")
        tag = "_" + args[i + 1]
        del args[i:i + 2]
    updates = opts.get("--updates", UPDATES_PER_AGENT)
    N_RUNS = opts.get("--runs", N_RUNS)
    sizes = [int(x) for x in args] if args else ALL_SIZES

    def steps_for_N(N):
        return updates * N

    n_cores = max(1, int(0.75 * os.cpu_count()))
    runs_per = {N: (3 if N >= 100000 else N_RUNS) for N in sizes}
    total_tasks = sum(runs_per[N] for N in sizes)
    print(f"System-size scaling sweep (corrected)")
    print(f"  N values: {sizes}")
    print(f"  Runs per N: {dict(runs_per)}")
    print(f"  Total sims: {total_tasks}")
    print(f"  Updates/agent: {updates} ({'ceiling, adaptive stop' if adaptive else 'fixed, no stop'})")
    if adaptive:
        print(f"  Stop: t >= {STOP_MARGIN} * t_end(prefix), r2 >= {STOP_R2_MIN}, "
              f"asymptote <= {STOP_ASYMPTOTE_MAX}; "
              f"floor {STOP_FLOOR_UPDATES}, check every {STOP_CHECK_UPDATES} upd/agent")
    print(f"  kappa: {BASE_PARAMS['kappa']}")
    print(f"  Community size: {COMMUNITY_SIZE}")
    print(f"  Inter-community mu: {MU}")
    print(f"  Total agent-steps: {sum(N * steps_for_N(N) * runs_per[N] for N in sizes):,.0f}"
          f"{' (upper bound)' if adaptive else ''}")
    print(f"  Cores: {n_cores}")
    print()

    t0 = time.time()
    results = []
    for N in sizes:
        n_runs = 3 if N >= 100000 else N_RUNS
        tasks = [(N, run, steps_for_N(N), adaptive) for run in range(n_runs)]
        # Memory is not the constraint these caps assumed. A live N=10000 worker at
        # 350 updates/agent measures 0.41 GB RSS (wegc203106, 2026-09-08) against
        # 453 GB free, and the terms that grow -- trajectory, degree grid, graph --
        # put N=20000 near 1 GB. The old caps (4 workers at N >= 10000, 2 at
        # N >= 20000) split the largest sizes into five sequential batches for no
        # measured reason: at N=20000 that is ~38 h of wall clock against ~8 h for
        # identical core-hours. N=100000 keeps a guard because nothing has measured it.
        n_workers = max(1, min(n_cores, 2 if N >= 100000 else n_runs))
        print(f"  Running N={N} ({n_runs} runs, {n_workers} workers, "
              f"{steps_for_N(N):,} steps {'max' if adaptive else ''})...")
        with Pool(n_workers) as pool:
            results.extend(pool.map(run_single, tasks))

    df = pd.DataFrame(results)
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {int(elapsed/60)}m {elapsed%60:.0f}s")

    summarize(df)

    # Save
    outdir = os.path.join('..', 'model_output')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f'system_size_scaling_{date.today().strftime("%Y%m%d")}{tag}.pkl')
    df.to_pickle(outfile)
    print(f"\nSaved: {outfile}")
