"""Offline cascade-attribution ledgers, replayed from the conversion event log.

The model logs every diet switch (model_main.Agent.step): a meat->veg conversion
carries the sampled partner and the M-entry memory buffer that prob_calc saw, each
entry (diet, source, t_sampled); a veg->meat reversion carries only the agent. Every
credit convention is a replay of that log, so the simulation never has to be rerun to
change one:

  parent  "last"      the sampled partner, if vegetarian (the submitted rule)
          "exposure"  the vegetarian sources in the buffer, weighted by their
                      n**gamma share of h_soc (the PRIMARY rule, fixed 2026-09-02)
  weight  "none"      tenure-neutral
          "dwell"     omega = 1 - exp(-dur / tau_persistence) on each ancestor
  unit    "event"     one delta per conversion event (the submitted unit)
          "time"      delta per step for as long as the stint lasts (veg-time)
  decay   lambda, geometric attenuation per cascade depth
  gamma   exponent of the exposure shares; defaults to the run's own. Set it to
          something else to see the ledger's gamma-sensitivity at fixed dynamics.

Propagation (decided 2026-09-02; claude_stuff/Review/ledger_convention_decision_2026-09-02.md).
Credit walks the graph of conversion EVENTS. A conversion's shares are read as the
probability that each source was its cause, and an ancestor's credit is its expected
credit under the single-cause chain rule: the sum over paths of (product of shares) x
lambda^(d-1), Katz-style (Wallinga & Teunis 2004; Katz 1953; Goyal et al. 2011). An
ancestor's own links are followed only if its current vegetarian stint had begun when the
child sampled it (ltime[ancestor] <= t_sampled). A stint that has ended forwards nothing --
the Methods' "detached on reversion" -- while the agent keeps what it was paid and is still
paid through descendants converted during that stint. One agent converts per step, so the
graph is acyclic and the walk terminates without truncation. A source that reverted between
being sampled and the child's conversion is still paid: the entry is what h_soc saw. The
one surviving form of self-credit (j's earlier stint influenced p, p later brings j back, j
is paid at depth 2) is kept; it is about 2% of credit.

The simulation's in-run ledger (reduction_out, reduction_out_unw) instead walks the
agent-indexed parent links with a visited-set cycle guard, which re-attaches a reconverted
agent's old descendants under its new parent and pays about 5% more under last-draw. It
is reproduced bit for bit by cycle="visited"; --validate uses that as the test of the walk
machinery. Every REPORTED convention, the two last-draw rows included, uses the event-graph
walk. Buffers logged before D7 as (diet, source) pairs are accepted: t_sampled then
defaults to the child's conversion time, a slightly looser stint test.

Usage:
  python attribution_ledger.py ../model_output/<ensemble>.pkl [--t-end T] [--decay L] [--gamma G] [--validate]
Prints per-convention amplification summaries averaged over the runs in the pickle.
"""
import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd


def _exposure_parents(buffer, j, gamma, t_conv):
    """(source, share, t_sampled) for the vegetarian sources in the buffer, self excluded.
    t_sampled is the most recent sampling of that source; two-field entries get t_conv."""
    veg, seen = Counter(), {}
    for e in buffer:
        d, src = e[0], e[1]
        if d != "veg" or src == j:
            continue
        veg[src] += 1
        ts = e[2] if len(e) > 2 else t_conv
        seen[src] = max(seen.get(src, ts), ts)
    if not veg:
        return []
    eff = {src: n ** gamma for src, n in veg.items()}
    tot = sum(eff.values())
    return [(src, e / tot, seen[src]) for src, e in eff.items()]


def replay(events, initial_diets, params, parent="last", weight="none", unit="event",
           decay=None, t_end=None, gamma=None, cycle="stint"):
    """Return credit per agent in kg CO2 (same units as reduction_out).

    events        list of ("conv", t, i, partner, partner_diet, buffer) / ("rev", t, i)
    initial_diets diets at t=0 (snapshot 0)
    params        the run's params dict (M, gamma, decay, tau_persistence, steps, CO2)
    t_end         ignore events after this step; stints are truncated here
    cycle         "stint" (event graph, reported) or "visited" (the simulation's own
                  chain walk, parent="last" only, for --validate)
    """
    if cycle == "visited" and parent != "last":
        raise ValueError("cycle='visited' reproduces the simulation and exists for parent='last' only")
    N = len(initial_diets)
    decay = params.get("decay", 0.7) if decay is None else decay
    gamma = params.get("gamma", 0.3) if gamma is None else gamma
    tau_p = params["tau_persistence"]
    delta = params["meat_CO2"] - params["veg_CO2"]
    t_end = params["steps"] if t_end is None else t_end

    stint = _stints(events, t_end) if unit == "time" else None

    parents = [[] for _ in range(N)]     # (source, share, t_sampled) links set at conversion
    ltime = [None] * N                   # step the current links were set: start of the stint
    change_time = [None] * N
    credit = np.zeros(N)

    def pay(p, depth, mass, amount, t):
        if weight == "dwell":
            dur = (t - change_time[p]) if change_time[p] is not None else t
            mass = mass * (1.0 - np.exp(-dur / tau_p))
        credit[p] += amount * mass * (decay ** (depth - 1))

    def walk_events(links, amount, t):
        """Level-synchronous walk over the event graph, additive over paths. A frontier
        entry holds [mass to pay, mass that travels on]; mass travels on only through a
        source whose current stint had begun when the child sampled it."""
        frontier = {}
        for q, share, ts in links:
            f = frontier.setdefault(q, [0.0, 0.0])
            f[0] += share
            if parents[q] and ltime[q] <= ts:
                f[1] += share
        depth = 1
        while frontier:
            nxt = {}
            for q, (mass, live) in frontier.items():
                pay(q, depth, mass, amount, t)
                for r, share, ts in (parents[q] if live else ()):
                    f = nxt.setdefault(r, [0.0, 0.0])
                    f[0] += live * share
                    if parents[r] and ltime[r] <= ts:
                        f[1] += live * share
            frontier, depth = nxt, depth + 1

    def walk_sim(links, amount, t):
        """The simulation's chain walk: single parent, each agent paid once, visited set
        as the only cycle guard, links followed whenever they were set."""
        p, depth, visited = links[0][0], 1, set()
        while p not in visited:
            visited.add(p)
            pay(p, depth, 1.0, amount, t)
            if not parents[p]:
                break
            p, depth = parents[p][0][0], depth + 1

    walk = walk_sim if cycle == "visited" else walk_events

    for k, ev in enumerate(events):
        t = ev[1]
        if t > t_end:
            break
        if ev[0] == "conv":
            _, _, j, partner, pdiet, buffer = ev
            change_time[j] = t
            if parent == "last":
                links = [(partner, 1.0, t)] if pdiet == "veg" else []
            else:
                links = _exposure_parents(buffer, j, gamma, t)
            if links:                       # simulation leaves the link untouched otherwise
                parents[j], ltime[j] = links, t
                walk(links, delta if unit == "event" else delta * stint[k], t)
        else:
            _, _, j = ev
            if parents[j]:                  # simulation clears change_time only with a link
                parents[j], ltime[j] = [], None
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
    "PRIMARY    (exposure, none, event)":  dict(parent="exposure", weight="none", unit="event"),
    "no dwell   (last, none, event)":      dict(parent="last", weight="none", unit="event"),
    "submitted  (last, dwell, event)":     dict(parent="last", weight="dwell", unit="event"),
    "veg-time   (exposure, none, time)":   dict(parent="exposure", weight="none", unit="time"),
}


def summarise(row, t_end=None, decay=None, gamma=None):
    """Per-convention summary for one ensemble row (event-graph walk throughout)."""
    ev, d0, p = row["events"], row["initial_diets"], row["params"]
    te = p["steps"] if t_end is None else t_end
    own = veg_time(ev, d0, te)
    n_conv = sum(1 for e in ev if e[0] == "conv" and e[1] <= te)
    n_conv_agents = len({e[2] for e in ev if e[0] == "conv" and e[1] <= te})
    out = {"churn": n_conv / n_conv_agents if n_conv_agents else np.nan}
    for name, cfg in CONVENTIONS.items():
        credit = replay(ev, d0, p, decay=decay, t_end=te, gamma=gamma, **cfg)
        amp = amplification(credit, cfg["unit"], p, own)
        out[name] = dict(mean=amp.mean(), p90=np.percentile(amp, 90), max=amp.max(),
                         n_credited=int((credit > 0).sum()))
    return out


def validate(row, atol=0.0):
    """The simulation-semantics replay must reproduce the in-simulation ledgers exactly;
    also reports how far the reported event-graph walk sits from them."""
    ev, d0, p = row["events"], row["initial_diets"], row["params"]
    sim_w = np.asarray(row["individual_reductions"], dtype=float)
    sim_u = np.asarray(row["individual_reductions_unw"], dtype=float)
    rep_w = replay(ev, d0, p, parent="last", weight="dwell", unit="event", cycle="visited")
    rep_u = replay(ev, d0, p, parent="last", weight="none", unit="event", cycle="visited")
    dw, du = np.abs(rep_w - sim_w).max(), np.abs(rep_u - sim_u).max()
    ok = dw <= atol and du <= atol
    print(f"{'OK' if ok else 'ERROR'}: replay (simulation semantics) vs simulation, max abs diff "
          f"weighted {dw:.3e}, unweighted {du:.3e} (sum {sim_w.sum():.6e})")
    ev_u = replay(ev, d0, p, parent="last", weight="none", unit="event")
    print(f"INFO: event-graph walk (reported) pays {ev_u.sum() / sim_u.sum() * 100:.2f}% of the "
          f"in-run unweighted ledger")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pkl")
    ap.add_argument("--t-end", type=int, default=None)
    ap.add_argument("--decay", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None, help="ledger share exponent (default: the run's)")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    df = pd.read_pickle(a.pkl)
    if "events" not in df.columns:
        sys.exit("ERROR: no event log in this pickle (generated before 2026-09-02)")
    if a.validate:
        if not all(validate(r) for _, r in df.iterrows()):
            sys.exit(1)
    rows = [summarise(r, a.t_end, a.decay, a.gamma) for _, r in df.iterrows()]
    print(f"{len(rows)} runs, t_end={a.t_end or 'full'}, lambda={a.decay or 'run default'}, "
          f"gamma={a.gamma or 'run default'}, "
          f"churn {np.mean([r['churn'] for r in rows]):.2f} conversions per converter")
    print(f"{'convention':38s} {'mean':>7s} {'p90':>7s} {'max':>8s} {'credited':>9s}")
    for name in CONVENTIONS:
        m = {k: np.mean([r[name][k] for r in rows]) for k in ("mean", "p90", "max", "n_credited")}
        print(f"{name:38s} {m['mean']:7.2f} {m['p90']:7.2f} {m['max']:8.1f} {m['n_credited']:9.0f}")


if __name__ == "__main__":
    main()
