#!/usr/bin/env python3
"""Attribution ledger replay across the kappa=0.55 headline ensemble.

Runs attribution_ledger.summarise over the reduced per-run artefacts -- same
code path as `attribution_ledger.py <pkl>`, which cannot load this ensemble.
The primary convention (exposure-proportional parents, event-count credit, no
dwell weight, lambda 0.7) is the reported one; the other three ride along for
the SI sensitivity table. The convention is fixed and is not revisited in
response to the output.

Amplification accumulates over the run, so --t-end is not optional in practice:
quote the window and the churn factor with every multiplier.

--null runs the permutation null of null_models_reassessment_2026-09-08.md s.3
instead: the primary convention replayed on event logs whose vegetarian buffer
sources have been redrawn from the vegetarians at that step (mf) or from the
converting agent's vegetarian neighbours (nb). F_veg(t), churn, arrival order
and chain termination are held exactly; only who is credited moves. Both
denominators of open decision 11 are reported -- (a) credit/delta as now, and
(b1) credit/(delta x max(own conversions, 1)).

Neighbours come from the nearest snapshot carrying an edge list (every 10000
steps, plus t=0). Rewiring is 0.005/step, so in the sparse early region that
graph is up to 50k steps stale, ~250 rewires on ~8000 edges: adequate for a
null whose statistic is the credit distribution, not any single edge.

Usage:
    python kappa_ledger.py <reduced_dir> --t-end 310000 [--validate] [--npz]
    python kappa_ledger.py <reduced_dir> --t-end 310000 --null mf nb [--reps 3]
"""
import os, sys, glob, pickle, argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_ledger import (CONVENTIONS, summarise, replay, permute_buffers,
                                conv_counts, _concentration, _veg_count)
from naive_counterfactuals import BANDS          # superseded script, band scheme stands

PRIMARY = dict(parent="exposure", weight="none", unit="event")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'visualisations_output')


def short(name):
    """Convention label for the CSV: 'no dwell', not 'no' (fixed 2026-09-08)."""
    return name.split('(')[0].strip()


# ----------------------------------------------------------------- null models

def _edge_snapshots(run):
    s = run['snapshots']
    return sorted(t for t in s if isinstance(t, int) and 'edges' in s[t])


def _adjacency(run):
    """t -> neighbour arrays by agent, from the nearest snapshot with an edge list."""
    snaps, ts, cache = run['snapshots'], _edge_snapshots(run), {}

    def adj(t):
        k = min(ts, key=lambda s: abs(s - t))
        if k not in cache:
            nb = [[] for _ in range(snaps[k]['n_nodes'])]
            for u, v in snaps[k]['edges']:
                nb[u].append(v); nb[v].append(u)
            cache[k] = [np.array(x, dtype=np.int32) for x in nb]
        return cache[k]
    return adj


def _degrees(run, t_end):
    snaps = run['snapshots']
    k = min(_edge_snapshots(run), key=lambda s: abs(s - t_end))
    e = snaps[k]['edges']
    return np.bincount(e.ravel(), minlength=snaps[k]['n_nodes']).astype(float), k


def _stats(credit, params, nconv, n_conv, n_net, roots):
    """The amplification_per_run.csv statistics under both denominators."""
    delta = params['meat_CO2'] - params['veg_CO2']
    pos = credit > 0
    out = {}
    for denom, div in (('a', 1.0), ('b1', np.maximum(nconv, 1))):
        amp = credit / (delta * div)
        a = amp[pos]
        c = _concentration(a)
        k1 = max(1, int(round(0.01 * len(a))))
        top = np.argsort(a)[::-1][:k1]
        out[denom] = dict(mean=a.mean(), median=np.median(a), p90=np.percentile(a, 90),
                          p99=np.percentile(a, 99), max=a.max(), n_credited=int(pos.sum()),
                          sys_amp=credit.sum() / (delta * n_conv),
                          sys_amp_adopter=credit.sum() / (delta * n_net) if n_net else np.nan,
                          root_top1=roots[pos][top].mean(),
                          root_credit=credit[roots].sum() / credit.sum(), **c)
    return out


def _bands(credit, params, deg):
    """Mean A per degree band over ALL agents in the band, zeros kept."""
    delta = params['meat_CO2'] - params['veg_CO2']
    A = credit / delta
    out = {}
    for lo, hi in BANDS:
        m = (deg >= lo) & (deg <= hi)
        out[(lo, hi)] = (m.sum(), deg[m].mean() if m.sum() else np.nan,
                         A[m].mean() if m.sum() else np.nan)
    return out


def run_null(a):
    """Model vs permutation nulls: per-run statistics, degree bands and one figure."""
    import matplotlib.pyplot as plt

    te = a.t_end
    labels = ['model'] + [f'null-{m}' for m in a.null]
    rows, bands, pooled = [], {L: [] for L in labels}, {L: [] for L in labels}
    diag = {m: {} for m in a.null}
    for path in sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl'))):
        nm = os.path.basename(path)[:-4]
        with open(path, 'rb') as f:
            run = pickle.load(f)
        ev, d0, p = run['events'], run['initial_diets'], run['params']
        nconv = conv_counts(ev, d0, te)
        n_conv = int(nconv.sum())
        n_net = _veg_count(ev, d0, te) - sum(1 for d in d0 if d == 'veg')
        roots = np.array([d == 'veg' for d in d0])
        deg, kdeg = _degrees(run, te)
        adj = _adjacency(run)
        rng = np.random.default_rng(a.seed + int(run['run']))

        for lab in labels:
            reps = 1 if lab == 'model' else a.reps
            for rep in range(reps):
                if lab == 'model':
                    events = ev
                else:
                    mode = lab.split('-')[1]
                    events, d = permute_buffers(ev, d0, mode, rng, adj, te)
                    for k, v in d.items():
                        diag[mode][k] = diag[mode].get(k, 0) + v
                credit = replay(events, d0, p, t_end=te, **PRIMARY)
                st = _stats(credit, p, nconv, n_conv, n_net, roots)
                for denom in ('a', 'b1'):
                    rows.append(dict(run=nm, t_end=te, model=lab, rep=rep, denom=denom,
                                     churn=n_conv / int((nconv > 0).sum()), n_conv=n_conv,
                                     n_converters=int((nconv > 0).sum()), net_adopters=n_net,
                                     **st[denom]))
                bands[lab].append(_bands(credit, p, deg))
                pooled[lab].append(credit / (p['meat_CO2'] - p['veg_CO2']))
        print(f"INFO: {nm} done (degrees from snapshot {kdeg})", flush=True)

    stat_keys = [k for k in rows[0] if k not in
                 ('run', 't_end', 'model', 'rep', 'denom')]
    out = os.path.join(a.reduced_dir, 'null_per_run.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, ['run', 't_end', 'model', 'rep', 'denom'] + stat_keys)
        w.writeheader(); w.writerows(rows)
    print(f"INFO: wrote {out}")

    bout = os.path.join(a.reduced_dir, 'null_bands.csv')
    with open(bout, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('model', 'k_lo', 'k_hi', 'n', 'k_mean', 'A_mean', 'A_over_k'))
        for lab in labels:
            for b in BANDS:
                v = [x[b] for x in bands[lab] if np.isfinite(x[b][2])]
                if not v:
                    continue
                n, k, A = (np.mean([x[i] for x in v]) for i in range(3))
                w.writerow((lab, b[0], b[1], f"{n:.0f}", f"{k:.2f}", f"{A:.4f}",
                            f"{A / k:.4f}"))
    print(f"INFO: wrote {bout}")

    # ---- table
    print(f"\n{len(rows) // (2 * (1 + a.reps * len(a.null)))} runs, t_end={te}, "
          f"{a.reps} permutations per run, primary convention")
    for mode, d in diag.items():
        print(f"  null-{mode}: {d.get('permuted', 0)} conversions permuted, "
              f"{d.get('small_pool', 0)} with a pool smaller than the source count, "
              f"{d.get('empty_pool', 0)} with no vegetarian in the pool")
    for denom in ('a', 'b1'):
        print(f"\ndenominator ({denom})  "
              f"{'credit / delta' if denom == 'a' else 'credit / (delta x own conversions)'}")
        print(f"{'':10s}{'mean':>8}{'median':>8}{'p90':>8}{'p99':>8}{'max':>8}"
              f"{'gini':>8}{'top1':>8}{'top10':>8}{'credited':>9}{'roots@top1':>11}")
        for lab in labels:
            r = [x for x in rows if x['model'] == lab and x['denom'] == denom]
            m = {k: np.median([x[k] for x in r]) for k in
                 ('mean', 'median', 'p90', 'p99', 'max', 'gini', 'top1', 'top10',
                  'n_credited', 'root_top1')}
            print(f"{lab:10s}{m['mean']:8.2f}{m['median']:8.2f}{m['p90']:8.2f}"
                  f"{m['p99']:8.1f}{m['max']:8.1f}{m['gini']:8.3f}{m['top1']:8.3f}"
                  f"{m['top10']:8.3f}{m['n_credited']:9.0f}{m['root_top1']:11.3f}")

    # ---- figure
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for lab, col in zip(labels, ('C0', 'C3', 'C2')):
        ls = '--' if lab == 'null-nb' else '-'      # it sits on top of the model
        v = np.sort(np.concatenate([x[x > 0] for x in pooled[lab]]))[::-1]
        ax.plot(np.arange(1, len(v) + 1) / len(v) * 100, v, lw=1.2, ls=ls, color=col,
                label=lab)
        kk = np.array([np.mean([x[b][1] for x in bands[lab]]) for b in BANDS])
        mu = np.array([np.mean([x[b][2] for x in bands[lab]]) for b in BANDS])
        m = np.isfinite(kk) & (mu > 0)
        bx.plot(kk[m], mu[m], 'o-', ms=3, lw=1.0, ls=ls, color=col,
                label=f'{lab}, $E[A|k]$')
    ax.set(xlabel='Credited agents [%]', ylabel='Amplification factor $A$',
           xlim=(0, 100), yscale='log')
    ax.set_title('a  credit distribution vs permutation nulls', fontsize=8, loc='left')
    ax.legend(fontsize=6, frameon=False)
    bx.set(xscale='log', yscale='log', xlabel='Degree $k$',
           ylabel='Mean amplification $E[A|k]$')
    bx.set_title('b  degree scaling against the topology-only null', fontsize=8, loc='left')
    bx.legend(fontsize=6, frameon=False)
    for a_ in (ax, bx):
        for sp in ('top', 'right'):
            a_.spines[sp].set_visible(False)
        a_.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/permutation_null_{te}.{ext}', dpi=300, bbox_inches='tight')
    print(f"\nINFO: wrote {OUT}/permutation_null_{te}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reduced_dir')
    ap.add_argument('--t-end', type=int, default=None)
    ap.add_argument('--validate', action='store_true',
                    help='replay simulation semantics against the in-run ledger')
    ap.add_argument('--csv', nargs='?', const='amplification_per_run.csv', default=None,
                    help='write one row per run per convention into the reduced dir')
    ap.add_argument('--npz', action='store_true',
                    help='save each run\'s per-agent credit vectors as credit_run_XX.npz')
    ap.add_argument('--null', nargs='+', choices=('mf', 'nb'), default=None,
                    help='permutation null instead of the convention table')
    ap.add_argument('--reps', type=int, default=3, help='permutations per run')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    if a.null:
        if a.t_end is None:
            sys.exit("ERROR: --null needs an explicit --t-end")
        return run_null(a)

    rows, worst, names = [], (0.0, 0.0), []
    for path in sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl'))):
        names.append(os.path.basename(path)[:-4])
        with open(path, 'rb') as f:
            run = pickle.load(f)
        if a.validate:
            sim = np.asarray(run['individual_reductions_unw'], float)
            rep = replay(run['events'], run['initial_diets'], run['params'],
                         parent='last', weight='none', unit='event', cycle='visited')
            d = np.abs(rep - sim).max()
            worst = (max(worst[0], d), max(worst[1], d / max(sim.max(), 1.0)))
        credits = {} if a.npz else None
        rows.append(summarise(run, a.t_end, credits=credits))
        if a.npz:
            dest = os.path.join(a.reduced_dir, f"credit_{names[-1]}.npz")
            np.savez_compressed(dest, t_end=a.t_end or run['params']['steps'],
                                **{short(k): v for k, v in credits.items()})
        print(f"INFO: {os.path.basename(path)} done", flush=True)

    if a.validate:
        print(f"\nvalidate (unweighted, simulation semantics): max abs diff over runs "
              f"{worst[0]:.3e}, relative to the largest credit {worst[1]:.3e}")

    print(f"\n{len(rows)} runs, t_end={a.t_end or 'full run'}")
    ch = [r['churn'] for r in rows]
    print(f"churn: median {np.median(ch):.2f} conversions per converter, "
          f"IQR [{np.percentile(ch, 25):.2f}, {np.percentile(ch, 75):.2f}]")
    print(f"\n{'convention':38s} {'mean':>7s} {'p90':>7s} {'max':>8s} {'credited':>9s}")
    for name in CONVENTIONS:
        m = {k: np.median([r[name][k] for r in rows])
             for k in ('mean', 'p90', 'max', 'n_credited')}
        q = np.percentile([r[name]['mean'] for r in rows], [25, 75])
        print(f"{name:38s} {m['mean']:7.2f} {m['p90']:7.2f} {m['max']:8.1f} "
              f"{m['n_credited']:9.0f}   IQR(mean) [{q[0]:.2f}, {q[1]:.2f}]")

    if a.csv:
        out = a.csv if os.path.isabs(a.csv) else os.path.join(a.reduced_dir, a.csv)
        stats = sorted(rows[0][next(iter(CONVENTIONS))])
        shared = ('churn', 'n_conv', 'n_converters', 'net_adopters')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(('run', 't_end', 'convention') + shared + tuple(stats))
            for nm, r in zip(names, rows):
                for conv in CONVENTIONS:
                    w.writerow((nm, a.t_end or '', short(conv)) + tuple(r[k] for k in shared)
                               + tuple(r[conv][k] for k in stats))
        print(f"\nINFO: wrote {out}")


if __name__ == '__main__':
    main()
