#!/usr/bin/env python3
"""Churn check: does a longer memory or a sharper decision rule tame the churn?

Under kappa = 0.55 the median converter converts about 5 times and a median
vegetarian stint lasts 1.55 of that agent's own activations (2N = 4000 steps,
because the activation coin is p = 0.5). That is buffer-noise flicker, not diet
change, and it sits under every credit denominator (open decision 11) and under
the churn factor 6.22 that the amplification ratios are quoted with. Open
decision 12 asks whether the configuration can be defended as it stands.

This sweeps the two parameters that could plausibly damp the flicker without
adding a mechanism -- memory length M (buffer-share sd falls as 1/sqrt(M)) and
inverse temperature beta (a sharper rule flips fewer marginal agents) -- one at a
time from the default configuration, plus kappa = 1.0 as the pre-discount
reference on the current code and data. If none of them brings the median
converter to <= 2 conversions and the median stint to >= 10 activations while
holding F_end near 0.75 and keeping the sigmoid, the answer to decision 12 is a
persistence term or a disclosure, not a parameter choice.

Design (claude_stuff/Review/handover_2026-09-08_churn_check.md, part A):
  default  M = 9, beta = 13, kappa = 0.55   reference; must reproduce churn 6.22
  M        12, 15
  beta     20, 30
  kappa    1.0
Five paired seeds per point, 30 runs at 400k. Pairing, parameter inheritance and
worker seeding are sensitivity_campaign's -- BASE_PARAMS and the seed set are
imported, not restated, so this sweeps the configuration the headline numbers
come from (twin, N = 2000) and the arms differ by the parameter, not the draw.

The sensitivity pickle cannot answer this offline: it stores observables, not the
event log. So each run is saved whole (events, initial_diets, params, decimated
trajectory, fitted t_end) and every statistic is computed afterwards from the
pickles by --measure, which can therefore be re-run against a changed definition
without touching the model.

Every statistic is reported twice: at the run's own fitted t_end, and at the fixed
310000 of the kappa N=2000 ensemble so the default point is comparable with the
numbers in churn_realism_2026-09-08.md. F_c is deliberately absent -- its window
is a fraction of run length, so it is not comparable across points whose t_end
differs (sensitivity_campaign, OBS block).

Usage (run on a compute server; ~2 h for 30 runs at 400k on 6 cores):
  python testing/churn_check.py --cores 6              # run, then measure
  python testing/churn_check.py --measure              # re-measure from pickles
  python testing/churn_check.py --steps 4000 --seeds 1 --cores 2   # smoke test
"""
import os, sys, io, time, random, pickle, argparse, contextlib
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../analysis'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import model_main
import sensitivity_campaign as sc
from attribution_ledger import summarise
from t_end_logistic import t_end_with_status

TAG = "20260908"
OUTDIR = f"../model_output/churn_check_{TAG}"
SUMMARY = f"../model_output/churn_check_summary_{TAG}.csv"
STEPS = 400000
SEEDS = 5
FIXED_T = 310000      # analysis_t_end of the kappa N=2000 ensemble
TRAJ_STRIDE = 100     # as sensitivity_campaign; refits within 0.01% of undecimated

# (param, value); "baseline" is the default configuration, run once and read as the
# reference for all three arms. Values must not be the baseline value themselves.
POINTS = [("baseline", np.nan), ("M", 12), ("M", 15),
          ("beta", 20), ("beta", 30), ("kappa", 1.0)]


def _tag(param, value):
    return param if param == "baseline" else f"{param}{value:g}".replace(".", "p")


def _path(param, value, seed):
    return f"{OUTDIR}/run_{_tag(param, value)}_seed{seed}.pkl"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _run_one(job):
    """One run, saved whole. Mirrors sensitivity_campaign._run_one exactly on the
    seeding and parameter path -- same network realisation and initial condition
    per seed across points -- and differs only in what it keeps."""
    param, value, seed, steps = job
    out = _path(param, value, seed)
    if os.path.exists(out):
        return out
    np.random.seed(seed); random.seed(seed)
    p = sc.BASE_PARAMS.copy()
    p.update({"seed": seed, "steps": steps, "tau_persistence": None})
    if param != "baseline":
        p[param] = value
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = model_main.Model(p, pmf_tables=sc.pmf_tables())
        m.run()
    traj = np.asarray(m.fraction_veg, float)
    t_end, status, r2 = t_end_with_status(traj)
    row = {"param": param, "value": value, "seed": seed, "steps": steps,
           "events": m.events,
           "initial_diets": m.snapshots[0]['diets'],
           "params": m.params,
           "traj_ds": traj[::TRAJ_STRIDE].astype(np.float32),
           "t_end_fit": t_end, "t_end_status": status, "t_end_r2": r2}
    with open(out, "wb") as f:
        pickle.dump(row, f)
    print(f"INFO: {os.path.basename(out)}  t_end={t_end} ({status}) "
          f"F_end={traj[-1]:.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def run(seeds, steps, cores):
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = [(prm, val, 42 + i, steps) for prm, val in POINTS for i in range(seeds)]
    todo = [j for j in jobs if not os.path.exists(_path(*j[:3]))]
    print(f"INFO: {len(jobs)} runs, {len(jobs) - len(todo)} already on disk, "
          f"{len(todo)} to run at {steps} steps on {cores} cores")
    if todo:
        with Pool(cores) as pool:
            pool.map(_run_one, todo)
    return [_path(*j[:3]) for j in jobs]


# ---------------------------------------------------------------------------
# Churn statistics
# ---------------------------------------------------------------------------
def churn_stats(events, N, t_end):
    """Conversion/reversion statistics up to t_end.

    This is the implementation that produced churn_realism_2026-09-08.md s.1;
    it is reused verbatim so the default point is comparable with it. Stints are
    the ones that ENDED inside the window -- an open stint has no length yet, and
    counting the truncation as a length would make every point look calmer the
    later its conversions fall.
    """
    ev = [e for e in events if e[1] <= t_end]
    nconv = Counter(e[2] for e in ev if e[0] == 'conv')
    nrev = Counter(e[2] for e in ev if e[0] == 'rev')
    if not nconv:
        return {}
    c = np.array(list(nconv.values()))
    sw = np.array([nconv[a] + nrev.get(a, 0) for a in nconv])
    start, L = {}, []
    for e in sorted(ev, key=lambda e: e[1]):
        if e[0] == 'conv':
            start[e[2]] = e[1]
        elif e[0] == 'rev' and e[2] in start:
            L.append(e[1] - start.pop(e[2]))
    L = np.array(L, float)
    return {"churn": c.sum() / len(c),
            "n_conv": int(c.sum()), "n_converters": int(len(c)),
            "conv_median": float(np.median(c)),
            "conv_p75": float(np.percentile(c, 75)),
            "conv_max": int(c.max()),
            "share_once": float((c == 1).mean()),
            "share_ge5": float((c >= 5).mean()),
            "share_ge10": float((c >= 10).mean()),
            "switches_median": float(np.median(sw)),
            "share_stints_ended": len(L) / c.sum(),
            "stint_median_steps": float(np.median(L)) if len(L) else np.nan,
            "stint_median_acts": float(np.median(L)) / (2 * N) if len(L) else np.nan}


def measure_run(row, t_end, N):
    """Churn, trajectory and primary-ledger statistics at one window."""
    out = churn_stats(row["events"], N, t_end)
    traj = row["traj_ds"]
    i = min(int(t_end // TRAJ_STRIDE), len(traj) - 1)
    out["F_at_window"] = float(traj[i])
    s = summarise(row, t_end=t_end)
    prim = s["PRIMARY    (exposure, none, event)"]
    out.update({"net_adopters": s["net_adopters"],
                "sys_amp": prim["sys_amp"],
                "sys_amp_adopter": prim["sys_amp_adopter"],
                "amp_mean": prim["mean"], "amp_max": prim["max"],
                "n_credited": prim["n_credited"]})
    return out


def measure(paths):
    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path, "rb") as f:
            row = pickle.load(f)
        N = len(row["initial_diets"])
        meta = {k: row[k] for k in ("param", "value", "seed", "steps",
                                    "t_end_fit", "t_end_status", "t_end_r2")}
        meta["F_end"] = float(row["traj_ds"][-1])
        for window, t in (("t_end", row["t_end_fit"]), ("fixed", FIXED_T)):
            if t > row["steps"]:
                continue
            rows.append({**meta, "window": window, "t_window": t,
                         **measure_run(row, t, N)})
        print(f"INFO: measured {os.path.basename(path)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY, index=False)
    print(f"INFO: Saved -> {SUMMARY}")
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
REPORT = ["churn", "conv_median", "conv_p75", "conv_max", "share_once",
          "share_ge5", "share_ge10", "switches_median", "share_stints_ended",
          "stint_median_steps", "stint_median_acts", "F_at_window", "t_end_fit", "t_end_r2",
          "sys_amp", "sys_amp_adopter"]


def report(df):
    """Median over seeds with the IQR, one block per window."""
    for window, g in df.groupby("window"):
        print(f"\n=== window: {window} " + "=" * 40)
        agg = g.groupby(["param", "value"], dropna=False)[REPORT].agg(
            ["median", lambda x: np.percentile(x, 75) - np.percentile(x, 25)])
        agg.columns = [f"{a}_{'iqr' if 'lambda' in b else b}" for a, b in agg.columns]
        with pd.option_context("display.width", 200, "display.max_columns", 100):
            print(agg.round(3).to_string())
        print("\nt_end status: ", g.groupby(["param", "value"], dropna=False)
              .t_end_status.value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--cores", type=int, default=6)
    ap.add_argument("--measure", action="store_true",
                    help="skip the model runs; measure the pickles already on disk")
    a = ap.parse_args()
    if a.measure:
        paths = [_path(prm, val, 42 + i) for prm, val in POINTS
                 for i in range(a.seeds)]
    else:
        paths = run(a.seeds, a.steps, a.cores)
    report(measure(paths))


if __name__ == "__main__":
    main()
