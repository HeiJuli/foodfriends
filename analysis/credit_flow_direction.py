#!/usr/bin/env python3
"""Per-edge direction of credit flow in the primary attribution ledger.

replay(..., flow=d) decomposes the ledger exactly by the hop each unit of credit
last travelled: d[(source, child)] = [depth-1 credit, credit at every depth]. A
hop is always a network edge, because buffer sources are sampled from the child's
neighbour list, so the ledger can be read as a flow field on the graph.

Sign convention (Tschofenig & Guilbeault): flow(i->j) is INFLUENCE moving from
source i to child j. For an unordered pair {i, j}, Delta S = flow(i->j) - flow(j->i)
and Delta k = k_j - k_i, so rho(Delta S, Delta k) > 0 means influence runs up the
degree gradient -- periphery to core. The correlation is the uncentred Pearson
sum(dS dk)/sqrt(sum dS^2 sum dk^2), which is what the ordinary Pearson over the
edge list mirrored in both orientations reduces to, and is therefore invariant to
how each pair happens to be oriented.

Degrees are k0 (t=0 snapshot, the structural predictor used elsewhere) with kT
(snapshot nearest t_end) as a check. The comparator is the topology-only "nb"
permutation null: sources redrawn from the child's vegetarian neighbours at that
step, so the direction that survives is the one set by who is adjacent and
vegetarian rather than by whose diet the buffer actually held.

Usage:
    python credit_flow_direction.py <reduced_dir> [--t-end 310000]
"""
import os, sys, glob, pickle, argparse, csv, time

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_ledger import replay, permute_buffers, _exposure_parents
from kappa_ledger import _adjacency, _edge_snapshots

PRIMARY = dict(parent="exposure", weight="none", unit="event")
RTOL = 1e-9


# ------------------------------------------------------------------ flow field

def _pairs(flow, idx):
    """{(a, b), a < b: [flow a->b, flow b->a]} for component idx (0 direct, 1 total)."""
    out = {}
    for (s, c), v in flow.items():
        key, o = ((s, c), 0) if s < c else ((c, s), 1)
        p = out.setdefault(key, [0.0, 0.0])
        p[o] += v[idx]
    return out


def _rho(pairs, k, keys):
    """Uncentred Pearson of (Delta S, Delta k) over `keys`; missing pairs carry no flow."""
    if not keys:
        return np.nan
    a = np.fromiter((p[0] for p in keys), int, len(keys))
    b = np.fromiter((p[1] for p in keys), int, len(keys))
    z = (0.0, 0.0)
    dS = np.fromiter((pairs.get(p, z)[0] - pairs.get(p, z)[1] for p in keys), float, len(keys))
    dk = k[b] - k[a]
    den = np.sqrt((dS ** 2).sum() * (dk ** 2).sum())
    return float((dS * dk).sum() / den) if den > 0 else np.nan


def _up_frac(pairs, k, tot):
    """Net share of influence flowing up the degree gradient; degree ties contribute 0."""
    if tot <= 0:
        return np.nan
    num = 0.0
    for (a, b), (ab, ba) in pairs.items():
        if k[a] < k[b]:
            num += ab - ba
        elif k[a] > k[b]:
            num += ba - ab
    return num / tot


def _edge_stats(flow, k0, kT, t0_keys, tag):
    out = {}
    for idx, name in ((0, "direct"), (1, "total")):
        pairs = _pairs(flow, idx)
        tot = sum(v[idx] for v in flow.values())
        out[f"rho_dS_dk_{name}_k0{tag}"] = _rho(pairs, k0, list(pairs))
        out[f"rho_dS_dk_{name}_kT{tag}"] = _rho(pairs, kT, list(pairs))
        out[f"rho_dS_dk_alledges_{name}{tag}"] = _rho(pairs, k0, t0_keys)
        out[f"up_frac_{name}{tag}"] = _up_frac(pairs, k0, tot)
    pairs = _pairs(flow, 1)
    out[f"n_flow_edges{tag}"] = len(pairs)
    out[f"both_ways_frac{tag}"] = (sum(1 for v in pairs.values() if v[0] > 0 and v[1] > 0)
                                   / len(pairs) if pairs else np.nan)
    return out


# ------------------------------------------------------------------- geometry

def _link_times(events, params, t_end):
    """(source, child) -> first step the link was set, and the number of linked
    conversions (invariant 2's denominator)."""
    g = params.get("gamma", 0.3)
    out, n = {}, 0
    for ev in events:
        if ev[1] > t_end:
            break
        if ev[0] != "conv":
            continue
        _, t, j, _partner, _pdiet, buf = ev
        links = _exposure_parents(buf, j, g, t)
        n += bool(links)
        for q, _s, _ts in links:
            out.setdefault((q, j), t)
    return out, n


def _edge_sets(run, t_end):
    """snapshot step -> frozenset of unordered edges, for the snapshots at or below t_end
    plus the first one above (the nearest-snapshot lookup may reach it)."""
    snaps, ts = run["snapshots"], _edge_snapshots(run)
    keep = [t for t in ts if t <= t_end]
    above = [t for t in ts if t > t_end]
    if above:
        keep.append(above[0])
    return {t: frozenset(map(lambda e: (min(e), max(e)), map(tuple, snaps[t]["edges"])))
            for t in keep}, keep


def _staleness(flow, ltimes, esets, ts):
    """Fractions of flow-bearing (source, child) pairs present in the t=0 graph and in the
    graph nearest the child's conversion. Rewiring is 0.005/step: a staleness check."""
    if not flow:
        return np.nan, np.nan
    e0, n0, nn = esets[0], 0, 0
    for s, c in flow:
        p = (min(s, c), max(s, c))
        n0 += p in e0
        t = ltimes.get((s, c))
        if t is not None:
            nn += p in esets[min(ts, key=lambda x: abs(x - t))]
    return n0 / len(flow), nn / len(flow)


# ---------------------------------------------------------------- node level

def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else np.nan


def _spearman(x, y):
    return _pearson(rankdata(x), rankdata(y))


def _node_stats(credit, delta, k):
    A, m = credit / delta, credit > 0
    mk = m & (k > 0)
    r = A[mk] / k[mk]
    return dict(n_credited=int(m.sum()), n_credited_k=int(mk.sum()),
                pearson_A_k=_pearson(A[m], k[m]), spearman_A_k=_spearman(A[m], k[m]),
                pearson_Aok_k=_pearson(r, k[mk]), spearman_Aok_k=_spearman(r, k[mk]))


# --------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reduced_dir")
    ap.add_argument("--t-end", type=int, default=310000)
    a = ap.parse_args()
    te, t0 = a.t_end, time.time()

    rows = []
    for seed, path in enumerate(sorted(glob.glob(os.path.join(a.reduced_dir, "run_*.pkl")))):
        nm = os.path.basename(path)[:-4]
        with open(path, "rb") as f:
            run = pickle.load(f)
        ev, d0, p = run["events"], run["initial_diets"], run["params"]
        delta = p["meat_CO2"] - p["veg_CO2"]
        snaps = run["snapshots"]
        esets, ts = _edge_sets(run, te)
        n_nodes = snaps[0]["n_nodes"]
        k0 = np.bincount(snaps[0]["edges"].ravel(), minlength=n_nodes).astype(float)
        kt = min(_edge_snapshots(run), key=lambda s: abs(s - te))
        kT = np.bincount(snaps[kt]["edges"].ravel(), minlength=n_nodes).astype(float)
        t0_keys = sorted(esets[0])

        flow = {}
        credit = replay(ev, d0, p, t_end=te, flow=flow, **PRIMARY)
        ltimes, n_linked = _link_times(ev, p, te)

        # invariants: the flow field is the credit vector, and depth-1 flow is one delta
        # per linked conversion because the exposure shares sum to 1.
        tot, dir_tot = (sum(v[i] for v in flow.values()) for i in (1, 0))
        assert abs(tot - credit.sum()) <= RTOL * credit.sum(), (nm, tot, credit.sum())
        assert abs(dir_tot - delta * n_linked) <= RTOL * delta * n_linked, (nm, dir_tot, n_linked)
        assert set(flow) == set(ltimes), nm

        row = dict(run=nm, t_end=te, k_snapshot=kt, n_linked_conv=n_linked)
        row.update(_edge_stats(flow, k0, kT, t0_keys, ""))
        f_t0, f_near = _staleness(flow, ltimes, esets, ts)
        row.update(in_t0_frac=f_t0, in_nearest_frac=f_near)
        row.update(_node_stats(credit, delta, k0))

        nflow = {}
        pev, _ = permute_buffers(ev, d0, "nb", np.random.default_rng(seed),
                                 adjacency=_adjacency(run), t_end=te)
        replay(pev, d0, p, t_end=te, flow=nflow, **PRIMARY)
        nlt, _ = _link_times(pev, p, te)
        row.update(_edge_stats(nflow, k0, kT, t0_keys, "_null_nb"))
        f_t0, f_near = _staleness(nflow, nlt, esets, ts)
        row.update(in_t0_frac_null_nb=f_t0, in_nearest_frac_null_nb=f_near)

        rows.append(row)
        print(f"INFO: {nm} done ({len(flow)} flow edges, kT from snapshot {kt})", flush=True)

    # the tag suffix keeps the two arms adjacent in code; the CSV wants a prefix
    def col(k):
        return "null_nb_" + k[:-8] if k.endswith("_null_nb") else k

    out = os.path.join(a.reduced_dir, "credit_flow_per_run.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([col(k) for k in rows[0]])
        w.writerows([list(r.values()) for r in rows])
    print(f"INFO: wrote {out}")

    _summary(rows, te)
    print(f"INFO: {len(rows)} runs in {time.time() - t0:.1f} s")


def _summary(rows, te):
    def q(key):
        x = np.array([r[key] for r in rows], float)
        x = x[np.isfinite(x)]
        if not len(x):
            return "  n/a" + " " * 27
        return (f"{np.median(x):9.4f} [{np.percentile(x, 25):8.4f},{np.percentile(x, 75):8.4f}]"
                f" {int((x > 0).sum()):3d}/{len(x)}")

    paired = [k for k in rows[0] if k + "_null_nb" in rows[0]]
    solo = [k for k in rows[0] if k not in ("run", "t_end") and not k.endswith("_null_nb")
            and k not in paired]
    w = max(len(k) for k in paired + solo)
    print(f"\n{len(rows)} runs, t_end={te}, primary convention (exposure, none, event, "
          f"lambda 0.7)")
    print(f"{'':{w}}   {'MODEL: median [p25, p75]  n>0':<40}  {'NULL-nb: median [p25, p75]  n>0'}")
    print("-" * (w + 84))
    for k in paired:
        print(f"{k:{w}}   {q(k)}  {q(k + '_null_nb')}")
    print("-" * (w + 84))
    for k in solo:
        print(f"{k:{w}}   {q(k)}")


if __name__ == "__main__":
    main()
