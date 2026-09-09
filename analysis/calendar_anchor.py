"""Calendar anchor for the model clock (R1.5): stint distribution and the years-per-sweep range.

The model has no clock. The native unit is the sweep (2N steps: with the p=0.5 activation
coin, 2N steps activate every agent once in expectation). This script measures the model's
vegetarian-stint distribution from the kappa=0.55 event logs and matches it to empirical
lapse rates, which fixes the sweep in years. Two matching rules are used -- median stint and
per-sweep leaving hazard -- and the spread across rules and sources is the reported range.

Nothing here calibrates the dynamics: a uniform rescaling of steps changes no dimensionless
quantity, so the pin is a labelling of the unit.

Run from the repo root:  python analysis/calendar_anchor.py
"""
import glob
import math
import pickle
from collections import Counter

import numpy as np

RUNS = "model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced/run_*.pkl"
T_END = 310000          # ensemble median t_end (logistic 95%), ensemble_measures.csv
N = 2000
SWEEP = 2 * N           # steps per sweep
T50 = 178013            # ensemble median t_50, ensemble_measures.csv
STEPS = 400000          # default run length

# Empirical anchors. Each value is verified at the cited source; see
# claude_stuff/Review/calendar_anchor_results_2026-09-09.md for the quotations.
EMPIRICAL = {
    "milfont_leave_nomeat_yr": (
        0.212,
        "Milfont, Satherley, Osborne, Wilson & Sibley 2021, Appetite 166:105584, "
        "Fig. 2 p.5: P(omnivore | no-meat) over one year, NZAVS T9 (2017) -> T10 "
        "(2018), N = 12,259",
    ),
    "milfont_leave_veg_yr": (
        0.226,
        "same, Fig. 1 p.6: P(omnivore | vegetarian); P(vegetarian | vegetarian) "
        "= 0.743, P(omnivore | omnivore) = 0.988",
    ),
    "faunalytics_median_stint_yr": (
        1.0,
        "Faunalytics 2014 (Asher, Green et al.), Study of Current and Former "
        "Vegetarians and Vegans, Initial Findings: 53% of former veg*ns adhered "
        "less than one year, so the median completed stint is about a year",
    ),
    "faunalytics_q3mo": (
        (0.34, 0.25),
        "same: 34% of former veg*ns kept the diet three months or less",
    ),
    "wur_pp_per_year": (
        1.0,
        "Protein Monitor 2024, Wageningen Social & Economic Research for the "
        "Ministry of LVVN: 'In 2024, the Dutch population consumed only one "
        "percentage point more plant-based protein than the year before'; the "
        "split is 40% plant / 60% animal",
    ),
    "mollenhorst_turnover_7yr": (
        0.529,
        "Mollenhorst, Volker & Flap 2014, Social Networks 37:65-80, p.72, as "
        "quoted in Small, Pamphile & McMahan 2015, Social Networks 40:90-102, "
        "p.94: '52.9 percent of the relationships were discontinued because of a "
        "lack of meeting opportunities' -- a reason-specific subset of turnover, "
        "not verified at the primary source (paywalled)",
    ),
}

P_REWIRE = 0.01     # model_main.DEFAULT_PARAMS; rewire() is called inside the p=0.5 coin
MEAN_DEG = 7.99     # snapshots[0]['edges'], twin N = 2000


# ---------------------------------------------------------------- model side

def stints(run, t_end=T_END):
    """Vegetarian stints (start, end, censored) up to t_end, plus veg exposure in steps.

    Initial vegetarians open a stint at t = 0. A 'conv' event opens one, a 'rev' closes it.
    Stints still open at t_end are right-censored at t_end.
    """
    ev = sorted((e for e in run["events"] if e[1] <= t_end), key=lambda e: e[1])
    open_at = {i: 0 for i, d in enumerate(run["initial_diets"]) if d == "veg"}
    init = set(open_at)
    out = []
    for e in ev:
        if e[0] == "conv":
            open_at.setdefault(e[2], e[1])
        elif e[0] == "rev" and e[2] in open_at:
            out.append((open_at.pop(e[2]), e[1], False, e[2] in init))
    out += [(t0, t_end, True, i in init) for i, t0 in open_at.items()]
    return out


def km(dur, obs, grid):
    """Kaplan-Meier survival on `grid` from durations `dur` with event indicator `obs`."""
    order = np.argsort(dur)
    dur, obs = np.asarray(dur)[order], np.asarray(obs)[order]
    n, s, curve = len(dur), 1.0, []
    times, surv = [], []
    for k, (t, o) in enumerate(zip(dur, obs)):
        at_risk = n - k
        if o:
            s *= 1 - 1 / at_risk
        times.append(t)
        surv.append(s)
    times, surv = np.asarray(times), np.asarray(surv)
    for g in grid:
        j = np.searchsorted(times, g, side="right") - 1
        curve.append(surv[j] if j >= 0 else 1.0)
    med = times[np.argmax(surv <= 0.5)] if (surv <= 0.5).any() else np.nan
    q25 = times[np.argmax(surv <= 0.75)] if (surv <= 0.75).any() else np.nan
    q75 = times[np.argmax(surv <= 0.25)] if (surv <= 0.25).any() else np.nan
    return np.array(curve), med, q25, q75


def per_run(run, t_end=T_END):
    st = stints(run, t_end)
    conv = [s for s in st if not s[3]]                      # stints of converters only
    dur = np.array([b - a for a, b, _, _ in st], float)
    obs = np.array([not c for _, _, c, _ in st])
    dur_c = np.array([b - a for a, b, _, _ in conv], float)
    obs_c = np.array([not c for _, _, c, _ in conv])
    grid = np.array([1, 2, 4, 8]) * SWEEP
    s_all, med, q25, q75 = km(dur, obs, grid)
    s_c, med_c, _, _ = km(dur_c, obs_c, grid)
    exposure = dur.sum() / SWEEP                            # veg-sweeps of exposure
    n_rev = int(obs.sum())
    nconv = Counter(e[2] for e in run["events"] if e[0] == "conv" and e[1] <= t_end)
    comp = np.sort(dur[obs]) / SWEEP                        # completed stints only
    # sweeps at which KM survival first falls to the empirical one-year retention
    s_emp = 1 - EMPIRICAL["milfont_leave_nomeat_yr"][0]
    fine = np.arange(1, 4 * SWEEP) / SWEEP
    curve, *_ = km(dur, obs, fine * SWEEP)
    t_at_s_emp = fine[np.argmax(curve <= s_emp)] if (curve <= s_emp).any() else np.nan
    return dict(
        n_stints=len(st), n_censored=int((~obs).sum()),
        med=med / SWEEP, q25=q25 / SWEEP, q75=q75 / SWEEP,
        med_conv=med_c / SWEEP,
        med_raw=float(np.median(comp)),
        p34_raw=float(np.percentile(comp, 34)),
        s1=s_all[0], s2=s_all[1], s4=s_all[2], s8=s_all[3],
        s1_conv=s_c[0], s2_conv=s_c[1], s4_conv=s_c[2], s8_conv=s_c[3],
        t_at_s_emp=t_at_s_emp,
        hazard=n_rev / exposure,
        churn=sum(nconv.values()) / max(len(nconv), 1),
    )


def model_side():
    files = sorted(glob.glob(RUNS))
    rows = []
    for f in files:
        with open(f, "rb") as fh:
            rows.append(per_run(pickle.load(fh)))
    keys = list(rows[0])
    q = {k: np.percentile([r[k] for r in rows], [25, 50, 75]) for k in keys}
    print(f"INFO: {len(files)} runs, window t <= {T_END} ({T_END / SWEEP:.1f} sweeps), "
          f"sweep = {SWEEP} steps")
    print(f"{'statistic':34s} {'median':>9s}  [run-to-run IQR]")
    for k in keys:
        lo, m, hi = q[k]
        print(f"{k:34s} {m:9.3f}  [{lo:.3f}, {hi:.3f}]")
    return q


# ------------------------------------------------------- matching and conversion

def years_per_sweep(q):
    """Four matching rules. Each fixes the sweep in years; the spread is the range."""
    out = {}
    med_yr = EMPIRICAL["faunalytics_median_stint_yr"][0]
    frac, q_yr = EMPIRICAL["faunalytics_q3mo"][0]
    p_nm = EMPIRICAL["milfont_leave_nomeat_yr"][0]
    p_vg = EMPIRICAL["milfont_leave_veg_yr"][0]

    print("\nMatching rules (model median over runs)")
    out["A completed-stint median vs Faunalytics"] = med_yr / q["med_raw"][1]
    print(f"  A  completed-stint median {q['med_raw'][1]:.2f} sweeps = {med_yr:.2f} yr "
          f"(Faunalytics 53% under a year)          -> {out[list(out)[-1]]:.2f} yr/sweep")

    out["B completed-stint 34th pct vs Faunalytics 3 months"] = q_yr / q["p34_raw"][1]
    print(f"  B  34th pct of completed stints {q['p34_raw'][1]:.2f} sweeps = {q_yr:.2f} yr "
          f"(34% within 3 months)         -> {out[list(out)[-1]]:.2f} yr/sweep")

    for lab, p in (("no-meat", p_nm), ("vegetarian", p_vg)):
        h = -math.log(1 - p)
        out[f"C exposure hazard vs Milfont ({lab})"] = q["hazard"][1] / h
        print(f"  C  hazard {q['hazard'][1]:.3f}/sweep vs Milfont {lab} {p:.3f}/yr "
              f"(h = {h:.3f})              -> {out[list(out)[-1]]:.2f} yr/sweep")

    out["D survival at one year vs Milfont"] = 1.0 / q["t_at_s_emp"][1]
    print(f"  D  S_model = {1 - p_nm:.3f} at {q['t_at_s_emp'][1]:.2f} sweeps = 1 yr "
          f"(Milfont one-year retention)   -> {out[list(out)[-1]]:.2f} yr/sweep")
    return out


def diffusion_route(q):
    """Secondary, curve-derived: match the model's own mean slope to the observed rate."""
    pp, _ = EMPIRICAL["wur_pp_per_year"]
    F0, Fend, sweeps = 0.06, 0.749, T_END / SWEEP
    slope = (Fend - F0) * 100 / sweeps               # percentage points per sweep
    print(f"\nSecondary (diffusion): model slope {slope:.2f} pp/sweep vs WUR {pp:.1f} pp/yr"
          f"  -> {slope / pp:.2f} yr/sweep, run = {slope / pp * sweeps:.0f} yr")


def network_clock(years_lo, years_hi):
    """Rewiring turnover as a clock, for the disclosure paragraph."""
    rate = 0.5 * P_REWIRE                            # rewire() sits inside the p=0.5 coin
    events = rate * T_END
    own = 2 * events / N                             # ego loses one tie and gains one
    total = 4 * events / N                           # ego 2, new alter +1, dropped alter -1
    turn_lo, turn_hi = own / MEAN_DEG, total / MEAN_DEG
    tw, _ = EMPIRICAL["mollenhorst_turnover_7yr"]
    yr_lo = turn_lo / tw * 7 / (T_END / SWEEP)
    yr_hi = turn_hi / tw * 7 / (T_END / SWEEP)
    print(f"\nNetwork clock (disclosure only)")
    print(f"  realised rewire rate {rate:.4f}/step -> {events:.0f} events per run to t_end")
    print(f"  tie changes per agent: {own:.2f} own-initiated, {total:.2f} counting all roles")
    print(f"  ego-network turnover per run: {turn_lo:.0%} - {turn_hi:.0%} of {MEAN_DEG:.1f} ties")
    print(f"  matched to {tw:.1%} in 7 yr -> {yr_lo * 365:.0f} - {yr_hi * 365:.0f} days/sweep"
          f"  ({yr_lo:.3f} - {yr_hi:.3f} yr/sweep)")
    print(f"  slower than the lapse pin by a factor {years_lo / yr_hi:.0f} - {years_hi / yr_lo:.0f}")


def explain_back(q, lo, hi):
    print(f"\nExplained back at {lo}-{hi} yr/sweep")
    rows = [
        ("t_end (77.6 sweeps)", T_END / SWEEP, "yr"),
        ("t_50 (44.5 sweeps)", T50 / SWEEP, "yr"),
        ("full 400k run", STEPS / SWEEP, "yr"),
        ("buffer span M = 9 sweeps", 9, "yr"),
        ("median completed stint", q["med_raw"][1], "yr"),
        ("median stint (KM, censoring)", q["med"][1], "yr"),
    ]
    for lab, s, u in rows:
        print(f"  {lab:34s} {s * lo:8.2f} - {s * hi:8.2f} {u}")
    print(f"  {'median completed stint':34s} {q['med_raw'][1] * lo * 12:8.1f} - "
          f"{q['med_raw'][1] * hi * 12:8.1f} months")
    print(f"  {'decision occasions / agent / yr':34s} {1 / hi:8.2f} - {1 / lo:8.2f}")
    print(f"  {'conversions per converter':34s} {q['churn'][1]:8.2f} over the transition")


# ----------------------------------------------------------- structural checks

def structural_checks():
    """B2: decision-band width in h_eff at beta = 13 vs the buffer-share noise.

    prob_calc/hamiltonian give P(switch) = sigmoid(beta (2 h_eff - 1)) with
    h_eff = (1 - w) h_ind + w h_soc, so the 10-90% band in h_eff is ln(9)/beta.
    """
    beta, M, gamma = 13, 9, 0.3
    band = math.log(9) / beta
    sd_binom = math.sqrt(0.25 / M)
    print(f"  10-90% decision band in h_eff at beta = {beta}: {band:.3f}")
    print(f"  binomial sd of a {M}-draw share at F = 0.5: {sd_binom:.3f}  "
          f"(band / sd = {band / sd_binom:.2f})")
    rng = np.random.default_rng(0)
    # h_soc is not the plain share: buffer entries are (diet, source) and each distinct
    # source contributes n**gamma. M draws with replacement from an ego network of degree k.
    sds = {}
    for k in (8, 12):
        s = np.empty(100000)
        for r in range(s.size):
            src = rng.integers(0, k, M)
            veg = rng.random(k) < 0.5
            cnt = Counter(src)
            s[r] = (sum(n ** gamma for j, n in cnt.items() if veg[j])
                    / sum(n ** gamma for n in cnt.values()))
        sds[k] = s.std()
        print(f"  sd(h_soc) at F = 0.5, gamma = {gamma}, degree {k}: {sds[k]:.3f}")
    with open(sorted(glob.glob(RUNS))[0], "rb") as fh:
        w = 1 - pickle.load(fh)["snapshots"][0]["alphas"]
    noise = w * sds[8]
    print(f"  w = 1 - alpha: median {np.median(w):.3f}, IQR "
          f"[{np.percentile(w, 25):.3f}, {np.percentile(w, 75):.3f}]")
    print(f"  buffer noise carried into h_eff (w sd(h_soc)): median {np.median(noise):.3f}, "
          f"share of agents above the band {np.mean(noise > band):.0%}")


def liss_check():
    """Is there a Dutch leave rate on disk? Only one meat-frequency wave, so no.

    su19a046 (May 2019) is the only meat-frequency item under data/. The 2018 diet
    item (oi18a016, in hierarchical_agents.csv) is a binary veg/meat category, a
    different construct, so the cross-tabulation below is not a like-for-like
    transition and n is 29. Printed for the record, not used as a constant.
    """
    import pandas as pd
    h = pd.read_csv("data/hierarchical_agents.csv")
    s = pd.read_stata("data/data_construction_paper/su19a_EN_1.0p.dta",
                      convert_categoricals=False)[["nomem_encr", "su19a046"]]
    d = h.merge(s, on="nomem_encr")
    d = d[d.su19a046.notna()]
    v18 = (d.diet == "veg").values
    never19 = (d.su19a046 == 6).values
    rare19 = (d.su19a046 >= 5).values
    print(f"\nLISS (one meat-frequency wave only; not used as a constant), n = {len(d)}")
    print(f"  veg in 2018 (oi18a016): {v18.sum()}; of those eating meat in 2019: "
          f"{(v18 & ~never19).sum()} ({(v18 & ~never19).mean() / v18.mean():.0%})")
    print(f"  same with 'less than weekly or never' as the 2019 state: "
          f"{(v18 & ~rare19).sum()} ({(v18 & ~rare19).mean() / v18.mean():.0%})")
    print(f"  meat in 2018 -> never eats meat 2019: {(~v18 & never19).sum()} of {(~v18).sum()}")


if __name__ == "__main__":
    q = model_side()
    routes = years_per_sweep(q)
    diffusion_route(q)
    liss_check()
    lo, hi = min(routes.values()), max(routes.values())
    print(f"\nRange across matching rules: {lo:.2f} - {hi:.2f} yr/sweep")
    explain_back(q, 0.5, 1.0)
    network_clock(0.5, 1.0)
    print("\nStructural checks")
    structural_checks()
