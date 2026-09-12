"""SI lambda figure on the reported ledger: CCDF of the amplification factor A
(veg-time, exposure parents, no dwell weight) at lambda 0.5-1.0, pooled over the kappa
ensemble at t = 310,000. A ledger-only replay with the dynamics fixed, so every curve is
the same 50 runs. Replaces the pre-kappa, event-count sweep figure drawn by
sensitivity_campaign.fig_lambda. Numbers: claude_stuff/Review/
amplification_accounting_final_2026-09-09.md s.3b (reported there as the bare
downstream ratio A - 1; here the amplification factor A = 1 + (A - 1), the quantity
Fig. 2 shows and the symbol the paper uses since 2026-09-12).

Usage:
    python vegtime_lambda_ccdf.py <reduced_dir> [--t-end 310000] [--cores 3]
"""
import os, sys, glob, pickle, argparse, numpy as np, matplotlib.pyplot as plt
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'plotting'))
from attribution_ledger import replay, veg_time
from plot_styles import set_publication_style, apply_axis_style, COLORS
LAMS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
VT = dict(parent='exposure', weight='none', unit='time')

def one(job):
    path, te = job
    r = pickle.load(open(path, 'rb'))
    ev, d0, p = r['events'], r['initial_diets'], r['params']
    delta = p['meat_CO2'] - p['veg_CO2']
    own = veg_time(ev, d0, te)
    out = {}
    for lam in LAMS:
        c = replay(ev, d0, p, t_end=te, decay=lam, **VT); m = c > 0
        out[lam] = 1.0 + c[m] / (delta * own[m])
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reduced_dir'); ap.add_argument('--t-end', type=int, default=310000)
    ap.add_argument('--cores', type=int, default=3)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.reduced_dir, 'run_*.pkl')))
    with Pool(a.cores) as pool:
        runs = pool.map(one, [(p, a.t_end) for p in paths])

    set_publication_style()
    cmap = plt.get_cmap('viridis')
    col = lambda v: cmap(0.12 + 0.76 * (v - LAMS[0]) / (LAMS[-1] - LAMS[0]))
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for v in LAMS:
        pool_ = np.sort(np.concatenate([r[v] for r in runs]))
        ccdf = 1.0 - np.arange(1, len(pool_) + 1) / len(pool_)
        base = v == 0.7
        ax.plot(pool_, ccdf, color=col(v), lw=2.4 if base else 1.4,
                ls='-' if base else '--', zorder=3 if base else 2,
                label=rf"$\lambda={v:g}$" + (" (default)" if base else ""))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("amplification factor, $A$")
    ax.set_ylabel(r"CCDF  $P(X>x)$")
    ax.legend(frameon=False, fontsize=7, loc='lower left', handlelength=1.8,
              borderpad=0.2, labelspacing=0.35)
    apply_axis_style(ax)

    # inset: per-run statistic, median over runs with the IQR
    ins = ax.inset_axes((0.60, 0.62, 0.38, 0.35))
    x = np.array(LAMS)
    print(f"{'lambda':>7}{'mean':>16}{'p90':>16}{'max':>16}   (median [IQR] over {len(runs)} runs)")
    stats = {s: np.array([[f(r[v]) for v in LAMS] for r in runs])
             for s, f in (("mean", np.mean), ("p90", lambda z: np.percentile(z, 90)),
                          ("max", np.max))}
    for s, c in (("max", COLORS['meat']), ("p90", COLORS['vegetation']),
                 ("mean", COLORS['primary'])):
        q25, med, q75 = np.percentile(stats[s], [25, 50, 75], axis=0)
        ins.fill_between(x, q25, q75, color=c, alpha=0.16)
        ins.plot(x, med, 'o-', color=c, ms=3, lw=1.3)
        ins.annotate(s, (x[-1], med[-1]), textcoords="offset points",
                     xytext=(4, 0), va='center', fontsize=6.5, color=c)
    for i, v in enumerate(LAMS):
        print(f"{v:>7g}" + "".join(
            f"{np.median(stats[s][:, i]):>7.2f} [{np.percentile(stats[s][:, i], 25):.2f},"
            f"{np.percentile(stats[s][:, i], 75):.2f}]" for s in ("mean", "p90", "max")))
    ins.axvline(0.7, color=COLORS['highlight'], ls='--', lw=1.0)
    ins.set_yscale('log')
    ins.set_yticks([2, 4, 10, 20, 40]); ins.set_yticklabels(['2', '4', '10', '20', '40'])
    ins.yaxis.set_minor_formatter(plt.NullFormatter())
    ins.set_xlim(x[0] - 0.02, x[-1] + 0.12)
    ins.set_xticks(x); ins.set_xticklabels([f"{v:g}" for v in x])
    ins.set_xlabel(r"$\lambda$", fontsize=7, labelpad=1)
    ins.tick_params(labelsize=6, pad=1.5)
    apply_axis_style(ins)

    fig.tight_layout()
    out = os.path.join(HERE, '..', 'visualisations_output',
                       'sensitivity_lambda_vegtime_kappa0p55_N2000.pdf')
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")
