"""Old-vs-new trajectory comparison across the 2026-09-02 rho sign correction.

Every ensemble generated before 2026-09-02 ran on inverted rho. This compares a
pre-fix ensemble against its post-fix twin and reports what moved in the transient:
the equilibration jump (the known ~1 -> 6-8% artefact, driven by the meat-eaters at
rho = 1, whose membership changed with the sign), the half-rise time, the logistic
95% t_end, the peak slope and its timing, and the endpoint.

Endpoints are expected to be near-invariant; the transient is where the sign shows up.

  python rho_transient_comparison.py --mode sample-max
  python rho_transient_comparison.py --mode twin          # awaits the N=2000 rerun

With --mode twin and no post-fix pickle on disk the script reports what it is waiting
for and exits 1, so it can be re-run unchanged once the cluster job lands.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t_end_logistic import estimate_t_end

OUT = "../model_output"
VIS = "../visualisations_output"

# (old pre-rho-fix ensemble, post-fix glob stem, transient zoom width in steps)
MODES = {
    "sample-max": (f"{OUT}/trajectory_analysis_sample-max_20260820.pkl",
                   f"{OUT}/trajectory_analysis_sample-max", 3000),
    "twin":       (f"{OUT}/trajectory_analysis_twin_20260820.pkl",
                   f"{OUT}/trajectory_analysis_twin", 10000),
}
RHO_FIX_DATE = 20260902   # ensembles dated before this ran on inverted rho
JUMP_T = 1000      # equilibration window, per the accepted-quirk note
STRIDE = 100       # decimate before savgol; see the derivative trap note


def _latest(stem):
    """Newest <stem>_<date>.pkl dated on or after the rho fix.

    The datestamp must be the segment straight after the stem, so topology and
    ledger variants (..._theta_<date>, ..._20260820_unwledger) cannot be mistaken
    for the post-fix ensemble.
    """
    d, base = os.path.dirname(stem), os.path.basename(stem)
    hits = []
    for f in os.listdir(d):
        if not (f.startswith(base + "_") and f.endswith(".pkl")):
            continue
        tag = f[len(base) + 1:-4]
        if tag.isdigit() and len(tag) == 8 and int(tag) >= RHO_FIX_DATE:
            hits.append(f)
    return os.path.join(d, max(hits)) if hits else None


def _trajectories(path, n_steps=None):
    df = pd.read_pickle(path)
    tr = [np.asarray(t, float) for t in df["fraction_veg_trajectory"]]
    n = min(len(t) for t in tr) if n_steps is None else n_steps
    return np.vstack([t[:n] for t in tr])


def _metrics(traj):
    """Per-run transient metrics for one ensemble (rows = runs)."""
    rows = []
    for t in traj:
        f0, fend = t[0], t[-1]
        rise = fend - f0
        half = np.argmax(t >= f0 + 0.5 * rise) if rise > 0 else np.nan
        # decimate first: savgol on the raw 150k-step series is ~100x costlier and
        # the window has to scale with the stride to stay the same span in steps
        dec = t[::STRIDE]
        win = max(5, (len(dec) // 20) | 1)
        slope = np.gradient(savgol_filter(dec, win, 3)) / STRIDE
        burn = win  # burn-in scales with the window, not a fixed count
        pk = burn + int(np.argmax(slope[burn:]))
        rows.append({
            "F_0": f0,
            "jump_1k": t[min(JUMP_T, len(t) - 1)] - f0,
            "t_50": half,
            "t_end": estimate_t_end(t),
            "peak_slope": slope[pk],
            "t_peak": pk * STRIDE,
            "F_end": fend,
        })
    return pd.DataFrame(rows)


def _fmt(s):
    q1, q3 = np.nanpercentile(s, [25, 75])
    return f"{np.nanmedian(s):>10.4g}  [{q1:.4g}, {q3:.4g}]"


def report(old, new, label_old, label_new):
    o, n = _metrics(old), _metrics(new)
    print(f"\n{'metric':<12}{'pre-fix  median [IQR]':<34}{'post-fix  median [IQR]':<34}{'change':>10}")
    print("-" * 90)
    for k in o.columns:
        mo, mn = np.nanmedian(o[k]), np.nanmedian(n[k])
        rel = "n/a" if mo == 0 or np.isnan(mo) else f"{(mn - mo) / abs(mo) * 100:+.1f}%"
        print(f"{k:<12}{_fmt(o[k]):<34}{_fmt(n[k]):<34}{rel:>10}")
    print(f"\nruns: {len(o)} pre-fix ({label_old}), {len(n)} post-fix ({label_new})")
    print("Endpoints are expected to be near-invariant. Read t_50, t_peak and jump_1k "
          "as the transient result.")
    return o, n


def figure(old, new, zoom, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plotting"))
    try:
        from plot_styles import set_publication_style, apply_axis_style, COLORS
        set_publication_style()
        c_old, c_new = COLORS.get("meat", "#B4635A"), COLORS.get("primary", "#4A7B6F")
    except Exception:                                  # styles are optional here
        apply_axis_style, c_old, c_new = None, "#B4635A", "#4A7B6F"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, upto, title in ((axes[0], old.shape[1], "Full trajectory"),
                            (axes[1], zoom, f"Transient, first {zoom:,} steps")):
        for arr, col, lab in ((old, c_old, "pre-fix (inverted rho)"),
                              (new, c_new, "post-fix")):
            x = np.arange(upto)
            med = np.median(arr[:, :upto], axis=0)
            q1, q3 = np.percentile(arr[:, :upto], [25, 75], axis=0)
            ax.fill_between(x, q1, q3, color=col, alpha=0.18, linewidth=0)
            ax.plot(x, med, color=col, lw=1.6, label=lab)
        ax.set_xlabel("step")
        ax.set_ylabel("vegetarian fraction")
        ax.set_title(title)
        if apply_axis_style:
            apply_axis_style(ax)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"INFO: wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(MODES), default="sample-max")
    ap.add_argument("--old", help="pre-fix pickle (default: the 2026-08-20 ensemble)")
    ap.add_argument("--new", help="post-fix pickle (default: newest matching the mode)")
    ap.add_argument("--out", help="figure path (default: visualisations_output/rho_transient_<mode>.pdf)")
    a = ap.parse_args()

    base, stem, zoom = MODES[a.mode]
    old = a.old or base
    new = a.new or _latest(stem)

    if not os.path.exists(old):
        sys.exit(f"ERROR: pre-fix ensemble not found: {old}")
    if new is None or not os.path.exists(new):
        print(f"WAITING: no post-fix {a.mode} ensemble yet.")
        print(f"  expected: {stem}_<date>.pkl  (anything newer than the baseline)")
        print(f"  baseline: {old}")
        print("  re-run this command unchanged once the job lands.")
        sys.exit(1)

    print(f"INFO: pre-fix  {old}")
    print(f"INFO: post-fix {new}")
    o_tr, n_tr = _trajectories(old), _trajectories(new)
    n = min(o_tr.shape[1], n_tr.shape[1])
    if o_tr.shape[1] != n_tr.shape[1]:
        print(f"WARNING: run lengths differ ({o_tr.shape[1]} vs {n_tr.shape[1]}), "
              f"truncating both to {n}")
    o_tr, n_tr = o_tr[:, :n], n_tr[:, :n]

    report(o_tr, n_tr, os.path.basename(old), os.path.basename(new))
    figure(o_tr, n_tr, min(zoom, n), a.out or f"{VIS}/rho_transient_{a.mode}.pdf")


if __name__ == "__main__":
    main()
