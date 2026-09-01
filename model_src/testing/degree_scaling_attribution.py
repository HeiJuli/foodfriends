#!/usr/bin/env python3
"""Paired-seed A/B: did the veg-partner credit rule cause the degree-slope drop?

WHAT THIS ANSWERS
-----------------
On 2026-08-20 the regenerated ensembles showed the degree-amplification log-log
slope fall from 1.02-1.04 (linear) to ~0.70-0.72 (sub-linear). Two changes landed
in the same commit (51f238d) and either could be responsible:

  (a) cascade credit is booked only when the sampled partner is vegetarian
      (Agent.step, model_src/model_main.py)
  (b) an arrival-order shuffle in the agent loaders (model_main.py,
      auxillary/sampling_utils.py) that removed an age/degree confound

This script isolates (a). Both arms run the SAME seeds through the SAME current
code -- including the shuffle -- and differ only in the credit rule. Whatever slope
difference survives is attributable to (a); anything left over belongs to (b) or
to something else.

THE EXACT DIFFERENCE (git show 51f238d^:model_src/model_main.py vs current)
--------------------------------------------------------------------------
PRE (51f238d^, lines 303-308) -- credit every meat->veg switch, whatever the
partner was eating:

    if old_diet == "meat" and self.diet == "veg":
        # meat -> veg: credit the influence chain
        self.influence_parent = other_agent.i
        self.change_time = t
        other_agent.influenced_agents.add(self.i)
        self._cascade_attribute(delta, other_agent, agents, t)

POST (current, lines 307-316) -- credit only when the sampled partner was veg:

    if old_diet == "meat" and self.diet == "veg":
        self.change_time = t
        if other_agent.diet == "veg":
            # meat -> veg with veg partner: credit the influence chain
            self.influence_parent = other_agent.i
            other_agent.influenced_agents.add(self.i)
            self._cascade_attribute(delta, other_agent, agents, t)
        # meat partner cannot supply veg exposure -> no influencer booked.
        # NOT "spontaneous": h_soc and the theta gate are still buffer-
        # conditioned; only the trigger draw was meat.

That is the whole change. `change_time` is set on every meat->veg switch in both
arms; the veg->meat detach branch is byte-identical. Nothing in the credit path
(`_cascade_attribute`) draws a random number, and neither `prob_calc` nor `rewire`
reads `influence_parent`, `influenced_agents` or `reduction_out` -- so the two arms
are dynamically identical by construction, and the script asserts that.

The credit branch is inline in `Agent.step()` and cannot be swapped without
duplicating the method, so `_step_pre` below is a minimal verbatim copy of the
PRE body, installed at runtime with `model_main.Agent.step = _step_pre`.
model_main.py itself is never touched.

HOW TO RUN
----------
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate foodfriends
    cd model_src/testing && python degree_scaling_attribution.py

    --runs N     paired seeds, default 8 (seeds 42..42+N-1)
    --steps N    timesteps, default 30000
    --t-end T    force the analysis snapshot; default is the logistic 95% estimate
    --cores N    multiprocessing; default sequential

Roughly 15 s per run per arm: 8 seeds x 2 arms is a few minutes.

HOW TO READ THE RESULT
----------------------
The headline is the paired per-seed difference in log-log slope, post minus pre.

  POSITIVE RESULT (the credit rule did it): median difference near -0.3, Wilcoxon
  p small, and the pre-arm median slope back near 1.0. The sub-linearity is then a
  property of the corrected attribution rule, not of the network -- report it as
  such and leave the shuffle out of it.

  NULL RESULT (the credit rule is not the cause): median difference near 0 with a
  CI straddling it, both arms sub-linear at ~0.70. The slope drop then comes from
  the arrival-order shuffle (b) or from something not varied here, and the next
  test is the loader, not the attribution.

  Anything in between is a partial explanation; report the median difference with
  its per-seed spread rather than picking a side.

Watch the WARNING lines. If the arms' trajectories diverge for any seed the
pairing is broken and the difference is not interpretable.
"""
import os, sys, random, pickle, argparse, hashlib, contextlib, io
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'model_src'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, 'model_src'))   # relative survey_file / model_output paths

import model_main
import model_runn
from t_end_logistic import estimate_t_end

# model_runn.DEFAULT_PARAMS is the source of truth (the path the reported
# ensembles ran through), NOT model_main.params.
BASE_PARAMS = dict(model_runn.DEFAULT_PARAMS)
BASE_PARAMS.update({
    "agent_ini": "sample-max",          # N is overridden to 385 complete cases
    "N": 385,
    "topology": "homophilic_emp",
    "survey_file": "../data/hierarchical_agents.csv",
})

_STEP_POST = model_main.Agent.step      # current code, unmodified


def _step_pre(self, G, agents, t):
    """Agent.step as of 51f238d^ -- credit booked on every meat->veg switch."""
    neighbours = [agents[n] for n in G.neighbors(self.i)]
    if not neighbours:
        return
    other_agent = random.choice(neighbours)
    self.memory.append((other_agent.diet, other_agent.i))

    if not self.immune and self.flip(self.prob_calc(other_agent)):
        old_diet = self.diet
        self.diet = "meat" if self.diet == "veg" else "veg"
        self.C = self.diet_emissions(self.diet)
        delta = self.C_base["meat"] - self.C_base["veg"]

        if old_diet == "meat" and self.diet == "veg":
            self.influence_parent = other_agent.i
            self.change_time = t
            other_agent.influenced_agents.add(self.i)
            self._cascade_attribute(delta, other_agent, agents, t)

        elif old_diet == "veg" and self.diet == "meat":
            if self.influence_parent is not None:
                agents[self.influence_parent].influenced_agents.discard(self.i)
                self.influence_parent = None
                self.change_time = None
    else:
        self.C = st.lognorm.rvs(s=0.20, scale=self.C_base[self.diet])


# ---------------------------------------------------------------------------
# Snapshot resolution and metric
# ---------------------------------------------------------------------------
def _resolve_key(snapshots, traj, t_end):
    """Explicit t_end > logistic 95% estimate > 'final'.

    Mirrors plotting/agency_predictor_analysis.py:_resolve_snap: map the target
    to the nearest positive integer snapshot key. Returns (key, t_end_used).
    """
    if t_end is None:
        t_end = estimate_t_end(traj)
    if t_end is not None:
        int_ts = sorted(t for t in snapshots if isinstance(t, int) and t > 0)
        if int_ts:
            return min(int_ts, key=lambda t: abs(t - t_end)), t_end
    return 'final', t_end


def _degree_scaling(snap):
    """Identical estimator to agency_predictor_analysis.py:analyze_degree_scaling:
    OLS of log(reduction_kg) on log(degree) over agents with reduction_kg > 0.
    Degree-0 agents are dropped as well (log(0) is undefined; the reference
    function has no such agents to trip over)."""
    G = snap['graph']
    nodes = list(G.nodes())
    red = np.asarray(snap['reductions'], float)
    deg = np.array([G.degree(n) for n in nodes], float)
    parents = snap['influence_parents']

    keep = (red > 0) & (deg > 0)
    n_pos, n_keep = int((red > 0).sum()), int(keep.sum())
    out = {
        "n_credited": n_pos,
        "n_dropped_deg0": n_pos - n_keep,
        "n_with_parent": int(sum(p is not None for p in parents)),
        "total_credit_kg": float(red.sum()),
    }
    if n_keep < 10:
        out.update(slope=np.nan, ci_lo=np.nan, ci_hi=np.nan, r2=np.nan,
                   rho_s=np.nan, p_s=np.nan)
        return out

    X = sm.add_constant(np.log(deg[keep]))
    fit = sm.OLS(np.log(red[keep]), X).fit()
    ci = fit.conf_int()[1]
    rs, ps = spearmanr(deg[keep], red[keep])
    out.update(slope=float(fit.params[1]), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
               r2=float(fit.rsquared), rho_s=float(rs), p_s=float(ps))
    return out


def verdict(ci_lo, ci_hi):
    if not np.isfinite(ci_lo):
        return "n/a"
    return "SUPER-LINEAR" if ci_lo > 1 else "LINEAR" if ci_hi > 1 else "SUB-LINEAR"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _run_one(job):
    arm, seed, steps, t_end = job
    model_main.Agent.step = _step_pre if arm == "pre" else _STEP_POST

    # seeding convention copied from model_src/sensitivity_campaign.py:_run_one
    np.random.seed(seed); random.seed(seed)
    p = BASE_PARAMS.copy()
    p.update({"seed": seed, "steps": steps, "tau_persistence": None})
    with contextlib.redirect_stdout(io.StringIO()):
        m = model_main.Model(p, pmf_tables=None)
        m.run()

    traj = np.asarray(m.fraction_veg, float)
    key, t_used = _resolve_key(m.snapshots, traj, t_end)
    row = {"arm": arm, "seed": seed, "snap_key": key, "t_end": t_used,
           "F_veg_final": float(traj[-1]), "steady_state_t": m.steady_state_t,
           "traj_hash": hashlib.sha1(traj.tobytes()).hexdigest(),
           "traj": traj.astype(np.float32)}
    row.update(_degree_scaling(m.snapshots[key]))
    return row


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def check_pairing(df):
    """The credit rule must not touch dynamics. If it does, nothing below means
    anything."""
    ok = True
    for seed, g in df.groupby("seed"):
        h = g.set_index("arm")
        if h.loc["pre", "traj_hash"] != h.loc["post", "traj_hash"]:
            a, b = np.asarray(h.loc["pre", "traj"]), np.asarray(h.loc["post", "traj"])
            i = int(np.argmax(a != b)) if len(a) == len(b) else -1
            print(f"WARNING: seed {seed} -- arms DIVERGED (first differing step {i}); "
                  f"F_veg_final pre={h.loc['pre','F_veg_final']:.4f} "
                  f"post={h.loc['post','F_veg_final']:.4f}. Pairing broken.")
            ok = False
        if h.loc["pre", "snap_key"] != h.loc["post", "snap_key"]:
            print(f"WARNING: seed {seed} -- arms resolved different snapshots "
                  f"({h.loc['pre','snap_key']} vs {h.loc['post','snap_key']}).")
            ok = False
    print("INFO: pairing verified -- arms are dynamically identical on every seed."
          if ok else "WARNING: pairing FAILED -- results below are not interpretable.")
    return ok


def report(df):
    print("\n" + "=" * 78)
    print("PER-SEED RESULTS")
    print("=" * 78)
    print(f"{'seed':>5} {'arm':>5} {'snap':>7} {'slope':>7} {'95% CI':>17} "
          f"{'R2':>6} {'rho_s':>7} {'n_cred':>7} {'n_par':>6} {'credit_kg':>11}")
    for seed, g in df.groupby("seed"):
        for arm in ("pre", "post"):
            r = g[g.arm == arm].iloc[0]
            print(f"{seed:>5} {arm:>5} {str(r.snap_key):>7} {r.slope:>7.3f} "
                  f"[{r.ci_lo:>7.3f},{r.ci_hi:>7.3f}] {r.r2:>6.3f} {r.rho_s:>7.3f} "
                  f"{r.n_credited:>7d} {r.n_with_parent:>6d} {r.total_credit_kg:>11.0f}")
        print()

    print("=" * 78)
    print("PAIRED SUMMARY")
    print("=" * 78)
    piv = df.pivot(index="seed", columns="arm", values="slope")
    for arm in ("pre", "post"):
        s = df[df.arm == arm]
        lo, hi = np.percentile(s.slope.dropna(), [25, 75])
        print(f"  {arm:>4}: median slope {s.slope.median():.3f}  IQR [{lo:.3f}, {hi:.3f}]"
              f"   median n_credited {s.n_credited.median():.0f}"
              f"   median credit {s.total_credit_kg.median():.0f} kg")

    d = (piv["post"] - piv["pre"]).dropna()
    lo, hi = np.percentile(d, [25, 75]) if len(d) else (np.nan, np.nan)
    print(f"\n  median per-seed difference (post - pre): {d.median():+.3f} "
          f"IQR [{lo:+.3f}, {hi:+.3f}]  over {len(d)} seeds")
    if len(d) >= 6 and np.any(d != 0):
        stat, p = wilcoxon(d)
        print(f"  Wilcoxon signed-rank: W={stat:.1f}, p={p:.4g}")
        if len(d) < 10:
            print(f"  NOTE: n={len(d)} seeds -- the smallest attainable p is "
                  f"{2.0 / 2 ** len(d):.4f}. Read the effect size, not the p.")
    else:
        print("  WARNING: too few non-zero differences for a Wilcoxon test.")

    for arm in ("pre", "post"):
        s = df[df.arm == arm]
        print(f"  {arm:>4} verdict at median: "
              f"{verdict(s.ci_lo.median(), s.ci_hi.median())}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=8, help="paired seeds (default 8)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--t-end", type=int, default=None,
                    help="force analysis snapshot; default logistic 95% estimate")
    ap.add_argument("--cores", type=int, default=1)
    args = ap.parse_args()

    seeds = [42 + i for i in range(args.runs)]
    jobs = [(arm, s, args.steps, args.t_end) for s in seeds for arm in ("post", "pre")]
    print(f"INFO: {len(jobs)} runs -- {args.runs} seeds x 2 arms, N=385, "
          f"steps={args.steps}, topology=homophilic_emp, cores={args.cores}")
    print(f"INFO: snapshot = {args.t_end if args.t_end else 'logistic 95% estimate'}")

    if args.cores > 1:
        with Pool(args.cores) as pool:
            rows = pool.map(_run_one, jobs)
    else:
        rows = []
        for n, job in enumerate(jobs, 1):
            print(f"INFO: run {n}/{len(jobs)}  arm={job[0]} seed={job[1]}")
            rows.append(_run_one(job))

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "model_output",
                       f"degree_scaling_attribution_{date.today().strftime('%Y%m%d')}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_pickle(out)
    print(f"\nINFO: raw rows -> {out}")

    check_pairing(df)
    report(df)
    print("\nINFO: done.")


if __name__ == "__main__":
    main()
