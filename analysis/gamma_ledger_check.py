"""Does the degree-amplification exponent survive the event-count -> veg-time switch?

SI S4's scale-invariance claim rests on gamma, the log-log slope of credited
amplification against degree, and the scaling sweep scores it on the event-count
ledger (test_system_size_scaling.py:380). Veg-time divides each agent's credit by
its OWN vegetarian time, which is agent-specific, so the two units are not a common
rescaling and gamma need not carry over. This replays the same event log under both
units on the kappa N=2000 ensemble and fits the sweep's estimator to each.

gamma_time_same fits the veg-time values on the event-count agent set, separating a
change in which agents are credited from a change in their values.

Usage:
    python gamma_ledger_check.py <reduced_dir> [--t-end 310000] [--cores 4]
"""
import os, sys, glob, pickle, argparse, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_ledger import replay, veg_time

EV = dict(parent="exposure", weight="none", unit="event")
VT = dict(parent="exposure", weight="none", unit="time")


def _degrees_at(snapshots, t_target):
    """Degrees from the snapshot nearest t_target (the sweep reads them where the
    credit window closes; rewiring decorrelates the final network from the earning one)."""
    t = min((k for k in snapshots if isinstance(k, int)), key=lambda x: abs(x - t_target))
    snap = snapshots[t]
    k = np.zeros(snap["n_nodes"], float)
    for a, b in snap["edges"]:
        k[a] += 1; k[b] += 1
    return t, k


def _slope(k, A):
    v = (k > 0) & (A > 0)
    if v.sum() <= 10:
        return np.nan, np.nan, 0
    lk, lA = np.log10(k[v]), np.log10(A[v])
    g, c = np.polyfit(lk, lA, 1)
    ss_res = np.sum((lA - (g * lk + c)) ** 2)
    ss_tot = np.sum((lA - lA.mean()) ** 2)
    return g, (1 - ss_res / ss_tot if ss_tot > 0 else 0.0), int(v.sum())


def one(job):
    path, TE = job
    r = pickle.load(open(path, "rb"))
    ev, d0, p = r["events"], r["initial_diets"], r["params"]
    delta = p["meat_CO2"] - p["veg_CO2"]
    own = veg_time(ev, d0, TE)
    ce = replay(ev, d0, p, t_end=TE, **EV) / delta
    ct = replay(ev, d0, p, t_end=TE, **VT) / delta
    At = np.where(own > 0, ct / np.where(own > 0, own, 1), 0.0)
    t_deg, k = _degrees_at(r["snapshots"], TE)

    ge, r2e, ne = _slope(k, ce)
    gt, r2t, nt = _slope(k, At)
    m = ce > 0                                    # event-count agent set
    gs, r2s, ns = _slope(k[m], At[m])
    return dict(run=r["run"], t_deg=t_deg, gamma_event=ge, r2_event=r2e, n_event=ne,
                gamma_time=gt, r2_time=r2t, n_time=nt,
                gamma_time_same=gs, r2_time_same=r2s, n_same=ns,
                mean_A_event=ce[ce > 0].mean(), mean_A_time=At[At > 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reduced_dir")
    ap.add_argument("--t-end", type=int, default=310000)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N runs only (pilot)")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.reduced_dir, "run_*.pkl")))
    if a.limit:
        paths = paths[:a.limit]
    print(f"INFO: {len(paths)} runs, t_end={a.t_end}, {a.cores} cores")
    with Pool(a.cores) as pool:
        rows = pool.map(one, [(p, a.t_end) for p in paths])

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("run")
    out = os.path.join(a.reduced_dir, "gamma_ledger_check.csv")
    df.to_csv(out, index=False)
    print(df[["run", "t_deg", "gamma_event", "gamma_time", "gamma_time_same",
              "n_event", "n_time", "mean_A_event", "mean_A_time"]].to_string(index=False))
    print("\n  median (sd) across runs")
    for c in ("gamma_event", "gamma_time", "gamma_time_same",
              "r2_event", "r2_time", "mean_A_event", "mean_A_time"):
        print(f"    {c:<18s} {df[c].median():>8.3f}  ({df[c].std():.3f})")
    d = df.gamma_time - df.gamma_event
    print(f"\n  paired gamma_time - gamma_event: median {d.median():+.3f}, "
          f"sd {d.std():.3f}, same sign in {(np.sign(df.gamma_time) == np.sign(df.gamma_event)).sum()}/{len(df)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
