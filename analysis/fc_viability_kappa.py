#!/usr/bin/env python3
"""F_c viability measurement on the kappa=0.55 headline ensemble.

Review-gate question: does the adopted configuration (kappa=0.55, N=2000,
400k steps, 50 twin runs) support a *sharp threshold* claim in the F_veg time
series? The criterion is a tight, window-stable F_c (F_veg at max smoothed
acceleration below F=0.5), not the mere existence of an argmax.

Measurements:
  1. Integrity check on the raw ensemble pickle (graphs dropped at unpickle
     time -- the snapshots column embeds ~65 networkx graphs per row and a
     naive read_pickle gets OOM-killed on this laptop).
  2. F_c per run at five Savitzky-Golay windows (old default 10001 plus
     2/5/10/20% of the 400k run, rounded to odd), using the exact estimator
     from model_src/sensitivity_campaign.py:177 (_inflection, copied verbatim
     below), with burnin = win so the masked region is at least one kernel
     width (cf. analysis/results_analysis.py:_smooth_derivs, which masks
     idx < max(burnin, win)).
  3. t_50 / t_90 / t_end (fitted 50/90/95% of asymptotic change) and fitted
     asymptote F_end per run, same logistic fit as
     analysis/t_end_logistic.py:estimate_t_end, extended to return popt.
  4. Burn-in jump F_veg(1000) - F_veg(0) per run (the kappa=1 equilibration
     artefact was 0.120).
  5. Second-derivative peak height and FWHM at the 5% reference window.

F_c across windows is computed with the undecimated estimator (exact copy).
To keep 400k x win=80k convolutions affordable the per-run work is fanned
out over a process pool; each worker holds only one trajectory.

Usage:
    python fc_viability_kappa.py            # full run (loads pickle once,
                                            # caches arrays to npz)
    python fc_viability_kappa.py --cached   # skip pickle load, reuse npz

Outputs:
    model_output/fc_viability_20260903_kappa0p55_N2000.csv   per-run numbers
    model_output/fc_viability_cache_kappa0p55_N2000.npz      array cache
"""
import sys, os, gc, json, pickle
import numpy as np
import pandas as pd

sys.path.append('../analysis')

PKL = '../model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000.pkl'
CACHE = '../model_output/fc_viability_cache_kappa0p55_N2000.npz'
INTEGRITY = '../model_output/fc_viability_integrity_kappa0p55_N2000.json'
OUT_CSV = '../model_output/fc_viability_20260903_kappa0p55_N2000.csv'

EXPECTED = {"kappa": 0.55, "steps": 400000, "tau_persistence": 36000}
REQUIRED_COLS = ['events', 'initial_diets', 'individual_reductions',
                 'individual_reductions_unw', 'params', 'snapshots']
STEPS = 400000

# Old absolute default plus fractions of the 400k run, each rounded to odd.
WINDOWS = {
    'old_default_10001': 10001,
    '2pct':  int(round(0.02 * STEPS)) | 1,
    '5pct':  int(round(0.05 * STEPS)) | 1,
    '10pct': int(round(0.10 * STEPS)) | 1,
    '20pct': int(round(0.20 * STEPS)) | 1,
}
REF_WIN = WINDOWS['5pct']   # reference window for the d2 peak/FWHM

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Load (graphs dropped)
# ---------------------------------------------------------------------------
class _Stub:
    def __setstate__(self, state): pass


_DROPPED = object()   # memo placeholder for evicted entries


class _LeakyMemo(list):
    """List-backed pickle memo (memo indices are sequential, so a list works
    and costs 8 B/entry instead of ~100 B for a dict -- with ~25 M memoized
    objects in this file the memo itself would otherwise approach 2 GB).

    Retains only objects that are plausibly re-referenced later by BINGET:
    strings (dict keys, interned values like 'veg'/'meat'), types, callables,
    ndarrays and large bytes (array payloads). Everything else -- per-element
    np.float64 scalars and their reduce-arg tuples/bytes (the trajectory
    columns are lists of np.float64: ~3 memoized objects per element, ~8 GB
    of memo garbage over 50 x 400k steps -- this is what OOM-kills the naive
    DropGraphs), container shells, graph internals -- becomes a sentinel.

    A later BINGET against an evicted entry yields the sentinel, so genuine
    cross-references to evicted containers arrive corrupted-but-visible (e.g.
    snapshots[0]['diets'] shares its list object with the initial_diets
    column and therefore reads back as the sentinel; none of the columns this
    analysis uses are affected, and the integrity gate fails loudly if params
    or a trajectory ever comes back as the sentinel)."""
    def __setitem__(self, k, v):
        if k == len(self):
            self.append(None)
        if (isinstance(v, (str, type, np.ndarray, np.dtype)) or callable(v)
                or (isinstance(v, bytes) and len(v) > 1024)):
            list.__setitem__(self, k, v)
        else:
            list.__setitem__(self, k, _DROPPED)


class DropGraphs(pickle._Unpickler):
    """Pure-Python unpickler (the C accelerator does not expose its memo) with
    networkx classes stubbed and the leaky memo installed."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.memo = _LeakyMemo()

    def find_class(self, module, name):
        if module.startswith('networkx'):
            return _Stub
        return super().find_class(module, name)


def _watch_progress(fobj, stop, unpickler=None):
    import threading, time
    def run():
        while not stop.is_set():
            msg = f"INFO: unpickle at {fobj.tell()/1e9:.2f} / 1.77 GB"
            if unpickler is not None:
                try:
                    from collections import Counter
                    c = Counter(type(v).__name__ for v in unpickler.memo)
                    top = ", ".join(f"{k}:{n}" for k, n in c.most_common(6))
                    msg += (f" | memo={len(unpickler.memo)} ({top})"
                            f" | stack={len(unpickler.stack)}")
                except Exception:
                    pass
            print(msg, flush=True)
            stop.wait(60)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def load_and_check():
    """Unpickle with graphs dropped, run the integrity gate, cache arrays."""
    import threading
    print(f"INFO: loading {PKL} (graphs dropped, leaky memo) ...", flush=True)
    stop = threading.Event()
    with open(PKL, 'rb') as f:
        u = DropGraphs(f)
        _watch_progress(f, stop, u)
        df = u.load()
    stop.set()
    print(f"INFO: loaded {len(df)} rows", flush=True)

    rep = {"path": PKL, "n_rows": int(len(df)),
           "columns": list(df.columns),
           "required_cols_present": all(c in df.columns for c in REQUIRED_COLS),
           "params_ok": True, "param_failures": [], "dropped_memo_hits": 0,
           "snapshots_nonnull": True, "traj_lengths": {}}

    if 'fraction_veg_trajectory' not in df.columns:
        raise SystemExit(f"ERROR: no fraction_veg_trajectory column; "
                         f"columns are {list(df.columns)}")

    seeds, trajs = [], []
    for i, row in df.iterrows():
        p = row['params']
        tj = row['fraction_veg_trajectory']
        if tj is _DROPPED or not hasattr(tj, '__len__'):
            rep["dropped_memo_hits"] += 1
            rep["params_ok"] = False
            rep["param_failures"].append({"row": int(i), "key": "trajectory",
                                          "got": "MEMO_SENTINEL", "expected": "list"})
            trajs.append(np.full(STEPS + 1, np.nan))
            seeds.append(i)
            continue
        if p is _DROPPED or not isinstance(p, dict):
            # a container re-referenced after eviction from the leaky memo --
            # the data cannot be trusted as loaded; fail the gate loudly
            rep["dropped_memo_hits"] += 1
            rep["params_ok"] = False
            rep["param_failures"].append({"row": int(i), "key": "__row__",
                                          "got": "MEMO_SENTINEL", "expected": "dict"})
            seeds.append(i)
            trajs.append(np.asarray(row['fraction_veg_trajectory'],
                                    dtype=np.float64))
            continue
        for k, v in EXPECTED.items():
            if p.get(k) != v:
                rep["params_ok"] = False
                rep["param_failures"].append({"row": int(i), "key": k,
                                              "got": p.get(k), "expected": v})
        snap = row['snapshots']
        if snap is _DROPPED:
            rep["dropped_memo_hits"] += 1
        if snap is None or snap is _DROPPED or len(snap) == 0:
            rep["snapshots_nonnull"] = False
        t = np.asarray(row['fraction_veg_trajectory'], dtype=np.float64)
        trajs.append(t)
        rep["traj_lengths"][str(i)] = int(len(t))
        seeds.append(int(p.get('seed', i)))

    if len({len(t) for t in trajs}) == 1:
        traj_arr = np.vstack(trajs)
    else:   # inconsistent lengths: keep as object array, integrity gate will fail
        traj_arr = np.empty(len(trajs), dtype=object)
        traj_arr[:] = trajs
    del df, trajs
    gc.collect()

    rep["traj_length_unique"] = sorted(set(rep["traj_lengths"].values()))
    rep["traj_consistent_with_400k"] = all(
        abs(n - STEPS) <= 1 for n in rep["traj_length_unique"])
    rep["passed"] = (rep["n_rows"] == 50 and rep["required_cols_present"]
                     and rep["params_ok"] and rep["snapshots_nonnull"]
                     and rep["traj_consistent_with_400k"]
                     and rep["dropped_memo_hits"] == 0)

    np.savez_compressed(CACHE, trajectories=traj_arr,
                        seeds=np.array(seeds, dtype=int))
    with open(INTEGRITY, 'w') as f:
        json.dump(rep, f, indent=2)
    print(f"INFO: cached trajectories -> {CACHE}")
    return traj_arr, np.array(seeds), rep


def load_cached():
    z = np.load(CACHE)
    with open(INTEGRITY) as f:
        rep = json.load(f)
    return z['trajectories'].astype(np.float64), z['seeds'], rep


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def _inflection(traj, win=10001, burnin=5000):
    """F_c (max d2F/dt2 below F=0.5) and inflection (max dF/dt). Mirrors
    analysis/results_analysis.py:analysis_5_inflection so the campaign and the
    main analysis report the same estimator.

    VERBATIM COPY of model_src/sensitivity_campaign.py:177 -- do not "improve".
    """
    traj = np.asarray(traj, float)
    if len(traj) < burnin + 2 * win:
        return np.nan, np.nan, np.nan
    sm = savgol_filter(traj, win, 3)
    d1 = savgol_filter(traj, win, 3, deriv=1); d1[:burnin] = 0
    d2 = savgol_filter(traj, win, 3, deriv=2); d2[:burnin] = 0
    i1 = np.argmax(d1)
    d2m = d2.copy(); d2m[sm > 0.5] = 0
    i2 = np.argmax(d2m)
    F_c = sm[i2] if d2m[i2] > 0 else np.nan
    return F_c, sm[i1], i1 / 1000.0


def _d2_peak(traj, win, burnin):
    """Masked-d2 peak height and full width at half maximum (steps).

    Same masking as _inflection (burnin zeroed, d2 zeroed where smoothed
    F > 0.5). FWHM is the contiguous above-half-peak region around the
    argmax; the zeroed regions lie below half peak and bound it naturally.
    """
    sm = savgol_filter(traj, win, 3)
    d2 = savgol_filter(traj, win, 3, deriv=2)
    d2[:burnin] = 0
    d2m = d2.copy(); d2m[sm > 0.5] = 0
    i = int(np.argmax(d2m))
    peak = float(d2m[i])
    if peak <= 0:
        return np.nan, np.nan, i
    above = d2m >= peak / 2
    l = i
    while l > 0 and above[l - 1]:
        l -= 1
    r = i
    while r < len(above) - 1 and above[r + 1]:
        r += 1
    return peak, float(r - l), i


def _logistic(t, L, k, t0, b):
    return L / (1 + np.exp(-k * (t - t0))) + b


def fit_logistic(traj, smooth_window=5001):
    """estimate_t_end from analysis/t_end_logistic.py, extended to return the
    fitted parameters so t_50/t_90/F_end come from the same fit."""
    traj = np.asarray(traj, dtype=float)
    n = len(traj)
    win = min(smooth_window, n // 2 * 2 - 1)
    smooth = savgol_filter(traj, win, 3)
    tt = np.arange(n)
    p0 = [traj[-1] - traj[0], 1e-4, n * 0.1, traj[0]]
    bounds = ([0, 0, 0, 0], [1, 1e-2, n * 2, 0.5])
    try:
        popt, _ = curve_fit(_logistic, tt, smooth, p0=p0, bounds=bounds,
                            maxfev=50000)
    except (RuntimeError, ValueError):
        return None
    L, k, t0, b = popt
    t_at = lambda pct: t0 - np.log((1 - pct) / pct) / k
    return {"t_50": float(t_at(0.50)), "t_90": float(t_at(0.90)),
            "t_end": float(t_at(0.95)), "F_end": float(b + L),
            "L": float(L), "b": float(b), "r": float(k)}


# ---------------------------------------------------------------------------
# Per-run worker
# ---------------------------------------------------------------------------
def analyse_run(args):
    seed, traj = args
    row = {"seed": int(seed)}
    for name, win in WINDOWS.items():
        F_c, _, _ = _inflection(traj, win=win, burnin=win)
        row[f"F_c_{name}"] = F_c
    peak, fwhm, i_pk = _d2_peak(traj, REF_WIN, REF_WIN)
    row["d2_peak_5pct"] = peak
    row["d2_fwhm_steps_5pct"] = fwhm
    row["d2_peak_time_5pct"] = i_pk
    fit = fit_logistic(traj)
    if fit:
        row.update(fit)
    row["burnin_jump"] = float(traj[1000] - traj[0])
    row["F_veg_final"] = float(traj[-1])
    return row


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _q(v):
    v = np.asarray(v, float)
    return (np.nanmedian(v), np.nanpercentile(v, 25), np.nanpercentile(v, 75),
            np.nanmin(v), np.nanmax(v))


def main():
    if '--cached' in sys.argv and os.path.exists(CACHE):
        print(f"INFO: using cached arrays {CACHE}")
        trajs, seeds, rep = load_cached()
    else:
        trajs, seeds, rep = load_and_check()

    print(f"\n{'='*72}\n STEP 1: INTEGRITY CHECK\n{'='*72}")
    print(f"  rows: {rep['n_rows']} (expect 50)")
    print(f"  required columns present: {rep['required_cols_present']}"
          f"  {REQUIRED_COLS}")
    print(f"  params kappa/steps/tau_persistence == "
          f"{EXPECTED['kappa']}/{EXPECTED['steps']}/{EXPECTED['tau_persistence']} "
          f"for all rows: {rep['params_ok']}")
    if rep['param_failures']:
        print(f"  FAILURES: {rep['param_failures']}")
    print(f"  snapshots non-null in every row: {rep['snapshots_nonnull']}")
    print(f"  leaky-memo sentinel hits in data columns: {rep['dropped_memo_hits']} "
          f"(must be 0 -- a hit would mean a container was shared across rows)")
    print(f"  trajectory lengths (unique): {rep['traj_length_unique']} "
          f"(consistent with {STEPS} steps: {rep['traj_consistent_with_400k']})")
    print(f"  GATE: {'PASSED' if rep['passed'] else 'FAILED'}")
    if not rep['passed']:
        print("  -> refusing to trust downstream numbers; exiting.")
        sys.exit(1)

    n_runs, n_steps = trajs.shape
    jobs = [(seeds[i], trajs[i]) for i in range(n_runs)]
    from multiprocessing import Pool
    nproc = min(6, os.cpu_count())
    print(f"\nINFO: analysing {n_runs} runs on {nproc} workers "
          f"(windows {sorted(WINDOWS.values())}, undecimated savgol) ...")
    with Pool(nproc) as pool:
        rows = pool.map(analyse_run, jobs)
    res = pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)

    print(f"\n{'='*72}\n STEP 2: F_c PER RUN AT FIVE SAVGOL WINDOWS (n={n_runs})\n{'='*72}")
    fc_cols = [f"F_c_{n}" for n in WINDOWS]
    for name, col in zip(WINDOWS, fc_cols):
        v = res[col].values
        n_nan = int(np.isnan(v).sum())
        med, q1, q3, lo, hi = _q(v)
        print(f"  win={WINDOWS[name]:>6} ({name:>18}): median = {med:.3f}  "
              f"IQR = [{q1:.3f}, {q3:.3f}]  range = [{lo:.3f}, {hi:.3f}]"
              + (f"  NaN: {n_nan}/{n_runs}" if n_nan else ""))
    fc_mat = res[fc_cols].values
    swing = np.nanmax(fc_mat, axis=1) - np.nanmin(fc_mat, axis=1)
    nan_rows = int((np.isnan(fc_mat).any(axis=1)).sum())
    med, q1, q3, lo, hi = _q(swing)
    print(f"\n  Per-seed window swing (max-min F_c across the 5 windows):")
    print(f"    median = {med:.3f}  IQR = [{q1:.3f}, {q3:.3f}]  "
          f"range = [{lo:.3f}, {hi:.3f}]")
    print(f"    seeds with swing > 0.10: {(swing > 0.10).sum()}/{n_runs}, "
          f"> 0.20: {(swing > 0.20).sum()}/{n_runs}")
    if nan_rows:
        print(f"    seeds with NaN at >=1 window (excluded from swing via nan-aware min/max): {nan_rows}")
    ref = res['F_c_old_default_10001'].values
    print(f"\n  ~0.53 cluster check at the old default window: "
          f"{np.sum((ref >= 0.50) & (ref <= 0.56))}/{np.isfinite(ref).sum()} "
          f"finite runs within [0.50, 0.56]")
    print(f"  Reference (kappa=1, old ensemble): F_c IQR was [0.280, 0.424] "
          f"(width {0.424-0.280:.3f})")

    print(f"\n{'='*72}\n STEP 3a: LOGISTIC TIMING (fitted fraction of asymptotic change)\n{'='*72}")
    for c, lbl in [('t_50', '50%'), ('t_90', '90%'), ('t_end', '95% (t_end)')]:
        v = res[c].values
        med, q1, q3, lo, hi = _q(v)
        print(f"  t @ {lbl:>12}: median = {med:,.0f}  IQR = [{q1:,.0f}, {q3:,.0f}]  "
              f"range = [{lo:,.0f}, {hi:,.0f}]  "
              f"(n={np.isfinite(v).sum()}/{n_runs} fits)")
    v = res['F_end'].values
    med, q1, q3, lo, hi = _q(v)
    print(f"  F_end (fitted asymptote b+L): median = {med:.3f}  "
          f"IQR = [{q1:.3f}, {q3:.3f}]  range = [{lo:.3f}, {hi:.3f}]")

    print(f"\n{'='*72}\n STEP 3b: BURN-IN JUMP  F_veg(1000) - F_veg(0)\n{'='*72}")
    v = res['burnin_jump'].values
    med, q1, q3, lo, hi = _q(v)
    print(f"  median = {med:.4f}  IQR = [{q1:.4f}, {q3:.4f}]  "
          f"range = [{lo:.4f}, {hi:.4f}]")
    print(f"  reference: kappa=1 value was 0.120; pilot (4 seeds) median 0.0045, "
          f"range [0.0020, 0.0050]")

    print(f"\n{'='*72}\n STEP 3c: MASKED d2 PEAK AT REFERENCE WINDOW win={REF_WIN} (5%)\n{'='*72}")
    for c, fmt in [('d2_peak_5pct', '{:.3e}'), ('d2_fwhm_steps_5pct', '{:,.0f}')]:
        v = res[c].values
        med, q1, q3, lo, hi = _q(v)
        print(f"  {c:>22}: median = {fmt.format(med)}  "
              f"IQR = [{fmt.format(q1)}, {fmt.format(q3)}]  "
              f"range = [{fmt.format(lo)}, {fmt.format(hi)}]")
    # Normalisation is not uniquely defined in the review plan; report two
    # natural dimensionless forms alongside the raw numbers.
    res['d2_peak_norm'] = res['d2_peak_5pct'] * res['t_end']**2 / res['F_end']
    res['d2_fwhm_frac_tend'] = res['d2_fwhm_steps_5pct'] / res['t_end']
    for c, fmt in [('d2_peak_norm', '{:.3f}'), ('d2_fwhm_frac_tend', '{:.3f}')]:
        v = res[c].values
        med, q1, q3, lo, hi = _q(v)
        print(f"  {c:>22}: median = {fmt.format(med)}  "
              f"IQR = [{fmt.format(q1)}, {fmt.format(q3)}]  "
              f"range = [{fmt.format(lo)}, {fmt.format(hi)}]")
    print("  (d2_peak_norm = d2_peak * t_end^2 / F_end, dimensionless; "
          "d2_fwhm_frac_tend = FWHM / t_end)")

    res.to_csv(OUT_CSV, index=False)
    print(f"\nINFO: per-run numbers -> {OUT_CSV}")

    # --- verdict-relevant summaries (numbers only; wording left to the user)
    print(f"\n{'='*72}\n STEP 4: VERDICT INPUTS\n{'='*72}")
    for name, col in zip(WINDOWS, fc_cols):
        v = res[col].values
        q1, q3 = np.nanpercentile(v, [25, 75])
        print(f"  win={WINDOWS[name]:>6}: IQR width = {q3-q1:.3f}  "
              f"full range width = {np.nanmax(v)-np.nanmin(v):.3f}")


if __name__ == '__main__':
    main()
