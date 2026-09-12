#!/usr/bin/env python3
"""Dynamics nulls for the amplification distribution: naive contagion, same ledger.

Rebuilds the multi-arm counterfactual figure that `naive_counterfactuals.py` produced
before 2026-09-08, on a footing that survives the primary ledger. The retired CF1-CF3
credited a single sampled partner in a reversion-free tree and were compared against an
in-run ledger that spreads credit over exposure-proportional parents on an event graph:
not a like-for-like comparison (null_models_reassessment_2026-09-08.md s.1). The fix is to
make the nulls nulls on the DYNAMICS rather than on the attribution, then score every arm
through the same replay:

  CF2'  ER at the run's own edge count, naive contagion with an M-entry buffer
  CF3'  the run's own empirical network at t=0, same dynamics

Naive means: convert when the buffer's vegetarian share exceeds 0.5, no theta, no rho, no
alpha, no Boltzmann draw, no kappa, no reversion, topology held fixed. Immunity and the
initial diets are kept from the paired model run, so the arms share the pool of convertible
agents and can reach the same F_veg. The buffer is the model's own M = 9 FIFO of sampled
partners, logged as (diet, source, t_sampled), so `attribution_ledger.replay` sees exactly
the object it sees in a model run and the primary convention applies unchanged.

Each arm stops at the model's own F_veg at t_end and is scored there; with no reversion
the ledger is complete at that point. The model arm is the kappa = 0.55 headline ensemble
replayed at --t-end, one null run paired to each of its 50 runs.

CF1 stays what it is: the analytic uniform-random-recursive-tree rank law, a reference
curve for a single-parent ledger, not an arm.

Read the gap with the churn factor quoted. The model converts ~6.2 times per converter and
the ledger takes no debits, so under credit/delta it is paid for repeat conversions that a
no-reversion null cannot make.

--unit time (default since 2026-09-11) scores on the reported veg-time ledger: A = downstream
vegetarian-time / own vegetarian-time, both truncated at each arm's stop, outputs suffixed
_vegtime, panel a showing the amplification factor A over agents ever vegetarian. Nulls stop at the model's F_veg,
not at a common time, so a slow arm (CF1', ~2.4M steps) accrues its initial vegetarians' own
time over a longer window; read its level with that in mind. --unit event reproduces the
2026-09-08 event-count figure.

Usage:
    python dynamics_null.py <reduced_dir> --t-end 310000 [--runs 50] [--seed 0] [--unit time]
"""
import os, sys, glob, pickle, argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_ledger import replay, veg_time, _concentration, _veg_count, conv_counts
from naive_counterfactuals import BANDS, rrt_rank_law
from kappa_ledger import PRIMARY, _degrees, OUT

# submitted sub-linear exponent (event-count era); veg-time b in the revised text (:117)
REPORTED_B = {'event': 0.78, 'time': 1.17}


# ------------------------------------------------------------------ simulation
def _neighbours(edges, n):
    nb = [[] for _ in range(n)]
    for u, v in edges:
        nb[u].append(v); nb[v].append(u)
    return [np.array(x, np.int32) for x in nb]


_COMPLETE = {}


def _complete_neighbours(n):
    """K_n as neighbour arrays. Degree CV is exactly zero, so this arm carries no degree
    scaling and is left out of panel b; it is here for the concentration comparison."""
    if n not in _COMPLETE:
        _COMPLETE[n] = [np.concatenate([np.arange(i), np.arange(i + 1, n)]).astype(np.int32)
                        for i in range(n)]
    return _COMPLETE[n]


def _er_neighbours(n, m, rng):
    """G(n, m) as neighbour arrays: distinct undirected pairs, no self-loops."""
    seen = set()
    while len(seen) < m:
        u, v = rng.integers(n, size=2)
        if u != v:
            seen.add((min(u, v), max(u, v)))
    return _neighbours(np.array(sorted(seen)), n)


def _seed_memory(nbr, veg, M, rng):
    """Model_main.Agent.initialize_memory_from_neighbours: M entries drawn to match the
    neighbour diet mix, so the null starts from the same exposure as the model does."""
    mem = []
    for i, k in enumerate(nbr):
        if len(k) == 0:
            mem.append([("veg" if veg[i] else "meat", i, 0)] * M); continue
        vs, ms = k[veg[k]], k[~veg[k]]
        nv = int(round(M * len(vs) / len(k)))
        m = [("veg", int(rng.choice(vs)) if len(vs) else i, 0) for _ in range(nv)]
        m += [("meat", int(rng.choice(ms)) if len(ms) else i, 0) for _ in range(M - nv)]
        rng.shuffle(m)
        mem.append(m)
    return mem


def simulate(nbr, initial_diets, immune, M, target_f, rng, ceiling):
    """Naive contagion on a fixed graph. One agent per step behind a p = 0.5 coin and a
    uniform neighbour draw, as model_main.Model.run; the buffer is appended to on every
    activation whether or not the agent converts. Returns (events, t_stop, F_veg)."""
    n = len(initial_diets)
    veg = np.array([d == "veg" for d in initial_diets])
    mem, events, nveg = _seed_memory(nbr, veg, M, rng), [], int(veg.sum())
    for t in range(ceiling):
        if not rng.random() < 0.5:
            continue
        i = int(rng.integers(n))
        k = nbr[i]
        if len(k) == 0:
            continue
        j = int(k[rng.integers(len(k))])
        pdiet = "veg" if veg[j] else "meat"
        mem[i].append((pdiet, j, t))
        if veg[i] or immune[i]:
            continue
        buf = tuple(mem[i][-M:])
        if sum(1 for e in buf if e[0] == "veg") / M > 0.5:
            veg[i] = True; nveg += 1
            events.append(("conv", t, i, j, pdiet, buf))
            if nveg / n >= target_f:
                return events, t, nveg / n
    return events, ceiling, nveg / n


# --------------------------------------------------------------------- scoring
def _score(events, d0, p, t, unit, n_conv):
    """(A per agent, credited mask, panel-a pool mask, system ratio). Event unit: A =
    credit/delta, system = total credit per conversion event, which is what the level rests
    on once the model's repeat conversions are divided out. Time unit (the reported ledger):
    A = downstream / own vegetarian-time, system = total credit / total own time, as
    vegtime_accounting.py."""
    delta = p['meat_CO2'] - p['veg_CO2']
    c = replay(events, d0, p, t_end=t, **dict(PRIMARY, unit=unit))
    if unit == 'event':
        return c / delta, c > 0, np.ones(len(c), bool), c.sum() / (delta * n_conv)
    own = veg_time(events, d0, t)
    A = np.divide(c, delta * own, out=np.zeros_like(c), where=own > 0)
    return A, c > 0, own > 0, c.sum() / (delta * own.sum())


def _stats(A, cred, sys_amp, label, extra):
    """Per-agent distribution over credited agents, plus the system ratio."""
    a = A[cred]
    return dict(model=label, mean=a.mean(), median=np.median(a),
                p90=np.percentile(a, 90), p99=np.percentile(a, 99), max=a.max(),
                n_credited=len(a), sys_amp=sys_amp, **_concentration(a), **extra)


def _bands(A, deg):
    out = {}
    for lo, hi in BANDS:
        m = (deg >= lo) & (deg <= hi)
        out[(lo, hi)] = (m.sum(), deg[m].mean() if m.sum() else np.nan,
                         A[m].mean() if m.sum() else np.nan)
    return out


LABELS = ("model", "CF3' emp. net, naive", "CF2' ER, naive", "CF1' complete, naive")
COLOURS = ("C0", "C2", "C3", "C4")
NO_DEGREE = ("CF1' complete, naive",)     # every degree is n-1, outside BANDS
DISPLAY = {"model": "model ($\\kappa = 0.55$)"}   # figure only; CSV keeps LABELS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reduced_dir')
    ap.add_argument('--t-end', type=int, required=True)
    ap.add_argument('--runs', type=int, default=None, help='cap the number of runs')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ceiling', type=int, default=12_000_000,
                    help='step ceiling for a null run that never reaches the target')
    ap.add_argument('--unit', choices=('event', 'time'), default='time',
                    help="ledger unit: 'time' = reported veg-time ledger (outputs _vegtime), "
                         "'event' = the 2026-09-08 event-count scoring")
    a = ap.parse_args()
    sfx = '' if a.unit == 'event' else '_vegtime'

    te = a.t_end
    rows, bands = [], {L: [] for L in LABELS}
    pooled, adopters = {L: [] for L in LABELS}, []   # adopters: the nulls' cascade size
    paths = sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl')))[:a.runs]
    if not paths:
        sys.exit(f"ERROR: no run_*.pkl in {a.reduced_dir}")

    for path in paths:
        nm = os.path.basename(path)[:-4]
        with open(path, 'rb') as f:
            run = pickle.load(f)
        ev, d0, p = run['events'], run['initial_diets'], run['params']
        delta, M, n = p['meat_CO2'] - p['veg_CO2'], p['M'], len(d0)
        s0 = run['snapshots'][0]
        immune = np.asarray(s0['immune'], bool)
        edges = np.asarray(s0['edges'])
        target = _veg_count(ev, d0, te) / n
        rng = np.random.default_rng(a.seed + int(run['run']))

        nconv = conv_counts(ev, d0, te)
        n_conv, n_cvt = int(nconv.sum()), int((nconv > 0).sum())
        deg, kdeg = _degrees(run, te)
        A, cred, pool, sa = _score(ev, d0, p, te, a.unit, n_conv)
        rows.append(_stats(A, cred, sa, LABELS[0],
                           dict(run=nm, t_end=te, f_veg=target, n_conv=n_conv,
                                churn=n_conv / n_cvt, n_adopters=n_cvt)))
        bands[LABELS[0]].append(_bands(A, deg))
        pooled[LABELS[0]].append(A if a.unit == 'event' else 1 + A[pool])

        for label, nbr in ((LABELS[1], _neighbours(edges, n)),
                           (LABELS[2], _er_neighbours(n, len(edges), rng)),
                           (LABELS[3], _complete_neighbours(n))):
            events, t_stop, f_end = simulate(nbr, d0, immune, M, target, rng, a.ceiling)
            if f_end < target - 1e-9:
                print(f"WARNING: {nm} {label} stopped at F_veg={f_end:.3f} < {target:.3f} "
                      f"after {t_stop} steps", flush=True)
            nd = np.array([len(x) for x in nbr], float)
            A, cred, pool, sa = _score(events, d0, p, t_stop, a.unit, len(events))
            rows.append(_stats(A, cred, sa, label,
                               dict(run=nm, t_end=t_stop, f_veg=f_end, n_conv=len(events),
                                    churn=1.0, n_adopters=len(events))))
            bands[label].append(_bands(A, nd))
            pooled[label].append(A if a.unit == 'event' else 1 + A[pool])
            if label == LABELS[1]:
                adopters.append(len(events))
        print(f"INFO: {nm} done (model degrees from snapshot {kdeg}, target "
              f"F_veg={target:.3f})", flush=True)

    keys = [k for k in rows[0] if k not in ('run', 'model')]
    out = os.path.join(a.reduced_dir, f'dynamics_null_per_run{sfx}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, ['run', 'model'] + keys)
        w.writeheader(); w.writerows(rows)
    print(f"INFO: wrote {out}")

    bout = os.path.join(a.reduced_dir, f'dynamics_null_bands{sfx}.csv')
    with open(bout, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('model', 'k_lo', 'k_hi', 'n', 'k_mean', 'A_mean', 'A_over_k'))
        for L in LABELS:
            for b in BANDS:
                v = [x[b] for x in bands[L] if np.isfinite(x[b][2])]
                if not v:
                    continue
                nn, kk, AA = (np.mean([x[i] for x in v]) for i in range(3))
                w.writerow((L, b[0], b[1], f"{nn:.0f}", f"{kk:.2f}", f"{AA:.4f}",
                            f"{AA / kk:.4f}"))
    print(f"INFO: wrote {bout}")

    # ---- table
    print(f"\n{len(paths)} runs, model scored at t_end={te}, nulls at their own F_veg "
          f"stop, exposure parents, no dwell, {a.unit} unit, lambda 0.7; bare A")
    print(f"{'':22}{'mean':>8}{'median':>8}{'p90':>8}{'p99':>8}{'max':>8}{'gini':>8}"
          f"{'top1':>8}{'top10':>8}{'credited':>9}{'churn':>7}{'system':>9}")
    for L in LABELS:
        r = [x for x in rows if x['model'] == L]
        m = {k: np.median([x[k] for x in r]) for k in
             ('mean', 'median', 'p90', 'p99', 'max', 'gini', 'top1', 'top10',
              'n_credited', 'churn', 'sys_amp')}
        print(f"{L:22}{m['mean']:8.2f}{m['median']:8.2f}{m['p90']:8.2f}{m['p99']:8.1f}"
              f"{m['max']:8.1f}{m['gini']:8.3f}{m['top1']:8.3f}{m['top10']:8.3f}"
              f"{m['n_credited']:9.0f}{m['churn']:7.2f}{m['sys_amp']:9.2f}")

    # ---- figure
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for L, col in zip(LABELS, COLOURS):
        v = np.sort(np.concatenate(pooled[L]))[::-1]
        ax.plot(np.arange(1, len(v) + 1) / len(v) * 100, v, lw=1.2, color=col,
                label=DISPLAY.get(L, L))
    if a.unit == 'event':                       # the rank law is an event-count reference
        Ma = int(round(np.mean(adopters)))      # reversion-free cascade size, as CF1's
        ana = rrt_rank_law(Ma)
        ax.plot(np.arange(1, Ma + 1) / Ma * 100, ana, 'k--', lw=1.0,
                label='CF1 analytic $E[A|k]$')
    ch = np.median([x['churn'] for x in rows if x['model'] == LABELS[0]])
    ax.text(0.97, 0.72, f'model: {ch:.1f} conversions/converter\nnulls: 1 (no reversion)',
            transform=ax.transAxes, fontsize=5.5, ha='right', va='top', color='#777')
    ax.set(xlabel='Agent rank [%]',
           ylabel='Amplification factor $A$',
           xlim=(0, 100), ylim=(1e-2 if a.unit == 'event' else 0.9, None), yscale='log')
    ax.set_title('a  distribution vs naive-dynamics nulls', fontsize=8, loc='left')
    ax.legend(fontsize=6, frameon=False)

    kk_m = mu_m = None
    for L, col in zip(LABELS, COLOURS):
        if L in NO_DEGREE:
            continue
        kk = np.array([np.mean([x[b][1] for x in bands[L]]) for b in BANDS])
        mu = np.array([np.mean([x[b][2] for x in bands[L]]) for b in BANDS])
        m = np.isfinite(kk) & np.isfinite(mu) & (mu > 0)
        kk, mu = kk[m], mu[m]
        if len(kk) < 2:
            print(f"WARNING: {L} has {len(kk)} usable degree bands, left out of panel b")
            continue
        b0 = np.polyfit(np.log10(kk), np.log10(mu), 1)
        bx.plot(kk, mu, 'o-', ms=3, lw=1.0, color=col,
                label=f'{DISPLAY.get(L, L)}, $b = {b0[0]:.2f}$')
        bx.plot(kk, 10 ** np.polyval(b0, np.log10(kk)), '-', lw=0.6, color=col, alpha=0.5)
        print(f"  {L:22} band-mean slope b = {b0[0]:.3f}")
        if kk_m is None:
            kk_m, mu_m = kk, mu
    k0 = np.array([kk_m[0], kk_m[-1]], float)
    sym = 'A' if a.unit == 'event' else 'A-1'   # veg-time: bands hold the downstream-only ratio
    bx.plot(k0, mu_m[0] * (k0 / kk_m[0]), 'k--', lw=0.9, label=f'linear, ${sym} \\propto k$')
    rb = REPORTED_B[a.unit]
    bx.plot(k0, mu_m[0] * (k0 / kk_m[0]) ** rb, ':', color='#c33', lw=1.1,
            label=f'reported ${sym} \\propto k^{{{rb}}}$')
    bx.set(xscale='log', yscale='log', xlabel='Degree $k$',
           ylabel=f'Mean ${sym}$ by degree, $E[{sym}\\,|\\,k]$')
    bx.set_title('b  degree scaling, zeros kept', fontsize=8, loc='left')
    bx.legend(fontsize=6, frameon=False)
    for a_ in (ax, bx):
        for sp in ('top', 'right'):
            a_.spines[sp].set_visible(False)
        a_.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/naive_counterfactuals{sfx}.{ext}', dpi=300, bbox_inches='tight')
    print(f"\nINFO: wrote {OUT}/naive_counterfactuals{sfx}.pdf")


if __name__ == '__main__':
    main()
