"""Offline cascade-attribution ledgers, replayed from the conversion event log.

The model logs every diet switch (model_main.Agent.step): a meat->veg conversion
carries the sampled partner and the M-entry memory buffer that prob_calc saw; a
veg->meat reversion carries only the agent. Every credit convention is a replay of
that log, so the simulation never has to be rerun to change one:

  parent  "last"      the sampled partner, if vegetarian (the submitted rule)
          "exposure"  the vegetarian sources in the buffer, weighted by their
                      n**gamma share of h_soc (what the Methods describe)
  weight  "none"      tenure-neutral
          "dwell"     omega = 1 - exp(-dur / tau_persistence) on each ancestor
  unit    "event"     one delta per conversion event (the submitted unit)
          "time"      delta per step for as long as the stint lasts (veg-time)
  decay   lambda, geometric attenuation per cascade depth

Chain bookkeeping follows the simulation: a parent link is set at conversion and
cleared at reversion (cumulative-only, no debits), credit propagates up the links as
they stand at conversion time, and a visited set stops cycles. The configuration
(parent="last", weight="dwell", unit="event") reproduces reduction_out bit for bit
and ("last", "none", "event") reproduces reduction_out_unw -- run --validate.

Usage:
  python attribution_ledger.py ../model_output/<ensemble>.pkl [--t-end T] [--validate]
Prints per-convention amplification summaries averaged over the runs in the pickle.
"""
import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd


def _exposure_parents(buffer, j, gamma):
    """(source, share) for the vegetarian sources in the buffer, self excluded."""
    veg = Counter(src for d, src in buffer if d == "veg" and src != j)
    if not veg:
        return []
    eff = {src: n ** gamma for src, n in veg.items()}
    tot = sum(eff.values())
    return [(src, e / tot) for src, e in eff.items()]


def replay(events, initial_diets, params, parent="last", weight="none",
           unit="event", decay=None, t_end=None):
    """Return credit per agent in kg CO2 (same units as reduction_out).

    events        list of ("conv", t, i, partner, partner_diet, buffer) / ("rev", t, i)
    initial_diets diets at t=0 (snapshot 0)
    params        the run's params dict (M, gamma, decay, tau_persistence, steps, CO2)
    t_end         ignore events after this step; stints are truncated here
    """
    N = len(initial_diets)
    decay = params.get("decay", 0.7) if decay is None else decay
    gamma = params.get("gamma", 0.5)
    tau_p = params["tau_persistence"]
    delta = params["meat_CO2"] - params["veg_CO2"]
    t_end = params["steps"] if t_end is None else t_end

    stint = _stints(events, t_end) if unit == "time" else None

    parents = [[] for _ in range(N)]
    change_time = [None] * N
    credit = np.zeros(N)

    def propagate(p, depth, mult, amount, t, visited):
        if p in visited:
            return
        visited.add(p)
        if weight == "dwell":
            dur = (t - change_time[p]) if change_time[p] is not None else t
            w = 1.0 - np.exp(-dur / tau_p)
            c = amount * w * (decay ** (depth - 1))
        else:
            c = amount * (decay ** (depth - 1))
        credit[p] += c if mult == 1.0 else c * mult
        for q, share in parents[p]:
            propagate(q, depth + 1, mult * share, amount, t, visited)

    for k, ev in enumerate(events):
        t = ev[1]
        if t > t_end:
            break
        if ev[0] == "conv":
            _, _, j, partner, pdiet, buffer = ev
            change_time[j] = t
            if parent == "last":
                links = [(partner, 1.0)] if pdiet == "veg" else []
            else:
                links = _exposure_parents(buffer, j, gamma)
            if links:                       # simulation leaves the link untouched otherwise
                parents[j] = links
            amount = delta if unit == "event" else delta * stint[k]
            visited = set()
            for q, share in links:
                propagate(q, 1, share, amount, t, visited)
        else:
            _, _, j = ev
            if parents[j]:                  # simulation clears change_time only with a link
                parents[j] = []
                change_time[j] = None
    return credit


def _stints(events, t_end):
    """Stint length (steps) for each conversion event index, truncated at t_end."""
    stint, open_conv = {}, {}
    for k, ev in enumerate(events):
        if ev[1] > t_end:
            break
        if ev[0] == "conv":
            open_conv[ev[2]] = k
        elif ev[2] in open_conv:
            k0 = open_conv.pop(ev[2])
            stint[k0] = ev[1] - events[k0][1]
    for j, k0 in open_conv.items():
        stint[k0] = t_end - events[k0][1]
    return stint


def veg_time(events, initial_diets, t_end):
    """Steps spent vegetarian per agent up to t_end (own contribution for unit='time')."""
    N = len(initial_diets)
    own = np.zeros(N)
    since = [0 if d == "veg" else None for d in initial_diets]
    for ev in events:
        if ev[1] > t_end:
            break
        j = ev[2]
        if ev[0] == "conv":
            since[j] = ev[1]
        elif since[j] is not None:
            own[j] += ev[1] - since[j]
            since[j] = None
    for j, s in enumerate(since):
        if s is not None:
            own[j] += t_end - s
    return own


def amplification(credit, unit, params, own=None):
    """Dimensionless amplification: credit / one delta (event) or credit / own veg-time
    in delta units (time). Agents with no credit are excluded, as in the paper."""
    delta = params["meat_CO2"] - params["veg_CO2"]
    if unit == "event":
        amp = credit / delta
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            amp = np.where(own > 0, credit / (delta * own), 0.0)
    return amp[credit > 0]


CONVENTIONS = {
    "submitted  (last, dwell, event)":     dict(parent="last", weight="dwell", unit="event"),
    "no dwell   (last, none, event)":      dict(parent="last", weight="none", unit="event"),
    "PRIMARY    (exposure, none, event)":  dict(parent="exposure", weight="none", unit="event"),
    "veg-time   (exposure, none, time)":   dict(parent="exposure", weight="none", unit="time"),
}


def summarise(row, t_end=None, decay=None):
    """Per-convention summary for one ensemble row."""
    ev, d0, p = row["events"], row["initial_diets"], row["params"]
    te = p["steps"] if t_end is None else t_end
    own = veg_time(ev, d0, te)
    n_conv = sum(1 for e in ev if e[0] == "conv" and e[1] <= te)
    n_conv_agents = len({e[2] for e in ev if e[0] == "conv" and e[1] <= te})
    out = {"churn": n_conv / n_conv_agents if n_conv_agents else np.nan}
    for name, cfg in CONVENTIONS.items():
        credit = replay(ev, d0, p, decay=decay, t_end=te, **cfg)
        amp = amplification(credit, cfg["unit"], p, own)
        out[name] = dict(mean=amp.mean(), p90=np.percentile(amp, 90), max=amp.max(),
                         n_credited=int((credit > 0).sum()))
    return out


def validate(row, atol=0.0):
    """Replay must reproduce the in-simulation ledgers exactly."""
    ev, d0, p = row["events"], row["initial_diets"], row["params"]
    sim_w = np.asarray(row["individual_reductions"], dtype=float)
    sim_u = np.asarray(row["individual_reductions_unw"], dtype=float)
    rep_w = replay(ev, d0, p, parent="last", weight="dwell", unit="event")
    rep_u = replay(ev, d0, p, parent="last", weight="none", unit="event")
    dw, du = np.abs(rep_w - sim_w).max(), np.abs(rep_u - sim_u).max()
    ok = dw <= atol and du <= atol
    print(f"{'OK' if ok else 'ERROR'}: replay vs simulation, max abs diff "
          f"weighted {dw:.3e}, unweighted {du:.3e} (sum {sim_w.sum():.6e})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pkl")
    ap.add_argument("--t-end", type=int, default=None)
    ap.add_argument("--decay", type=float, default=None)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    df = pd.read_pickle(a.pkl)
    if "events" not in df.columns:
        sys.exit("ERROR: no event log in this pickle (generated before 2026-09-02)")
    if a.validate:
        if not all(validate(r) for _, r in df.iterrows()):
            sys.exit(1)
    rows = [summarise(r, a.t_end, a.decay) for _, r in df.iterrows()]
    print(f"{len(rows)} runs, t_end={a.t_end or 'full'}, lambda={a.decay or 'run default'}, "
          f"churn {np.mean([r['churn'] for r in rows]):.2f} conversions per converter")
    print(f"{'convention':38s} {'mean':>7s} {'p90':>7s} {'max':>8s} {'credited':>9s}")
    for name in CONVENTIONS:
        m = {k: np.mean([r[name][k] for r in rows]) for k in ("mean", "p90", "max", "n_credited")}
        print(f"{name:38s} {m['mean']:7.2f} {m['p90']:7.2f} {m['max']:8.1f} {m['n_credited']:9.0f}")


if __name__ == "__main__":
    main()
