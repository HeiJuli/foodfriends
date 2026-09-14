"""Merge the 2026-09-14 veg-time sensitivity arms and compare S against the published
event-count campaign. Run from analysis/ with the venv active.

Merge follows run_extension's rule: non-baseline rows from the shorter frames, the
longest frame whole, so the shared baseline is the 2M one. That choice is immaterial
for the reported columns -- baseline's t_end (~347k) resolves inside 400k, and a longer
run is a prefix-extension of the shorter, so its _tend amplification is identical.

An arm whose pickle is absent is skipped with a warning: the sweeps it carries are then
truncated and their S is a lower bound on the span, not the span.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, '../model_src')
sys.path.insert(0, '../plotting')
import sensitivity_campaign as sc

M = "../model_output/sensitivity_campaign_20260914_N2000_%s.pkl"
ARMS = ["main", "ext800k", "c045_2M"]        # ascending run length; last one merges whole
# The points held back from `main` and run straight at their longer length.
EXTENDED = "M=12,M=15,beta=20,beta=30,gamma=1.0,immune_n=0.15,immune_n=0.20,kappa=0.40," \
           "theta_gate_c=0.40,theta_gate_c=0.45,theta_gate_k=45,theta_gate_k=55"

frames = {}
for arm in ARMS:
    if os.path.exists(M % arm):
        frames[arm] = pd.read_pickle(M % arm)
    else:
        print("WARNING: arm %s not found, sweeps carrying its points are truncated" % arm)
if not frames:
    sys.exit("ERROR: no arms found")

isbase = lambda d: np.array([v == sc.BASELINE[p] for p, v in zip(d.param, d.value)])
longest = [a for a in ARMS if a in frames][-1]
merged = pd.concat([d if a == longest else d[~isbase(d)] for a, d in frames.items()],
                   ignore_index=True)
print("merged rows %d, points %d, arms %s"
      % (len(merged), merged.groupby(['param', 'value']).ngroups, ",".join(frames)))
print(merged.groupby('steps').size().to_string(), "\n")

summary = sc.summarise(merged)
sc.OBS = sc.OBS + ['amp_mean_tend', 'amp_p90_tend', 'amp_max_tend']   # score both units
sens = sc.sensitivity_index(summary)

if set(frames) == set(ARMS):
    merged.to_pickle(M % "merged")
    summary.to_csv("../model_output/sensitivity_summary_20260914_N2000_merged.csv",
                   index=False)
    print("INFO: wrote merged pickle and summary")
else:
    print("INFO: partial merge, nothing written")

incomplete = {p.split('=')[0] for p in EXTENDED.split(',')} if 'ext800k' not in frames \
    else set()

# published event-count campaign, same S formula on its own observable names
old = pd.read_csv("../model_output/sensitivity_summary_20260907_N2000_ext800k_c045at2M.csv")


def S_old(prm, ob, restrict=None):
    """S as sensitivity_index defines it, on the published frame. `restrict` limits the
    sweep to the points the new frame actually has, so the spans are comparable."""
    sub = old[old.param == prm].sort_values("value")
    if restrict is not None:
        sub = sub[sub.value.isin(restrict)]
    y = sub[f"{ob}_mean"].values
    yb = old[(old.param == prm) & old.is_baseline][f"{ob}_mean"].values
    if not len(yb) or not np.isfinite(y).any() or len(y) < 2:
        return np.nan
    sign = np.sign(spearmanr(sub.value.values, y, nan_policy="omit").statistic)
    return sign * (np.nanmax(y) - np.nanmin(y)) / abs(yb[0])


rows = []
for prm in sc.SWEEPS:
    have = sorted(summary[summary.param == prm].value.unique())
    get = lambda ob: sens[(sens.param == prm) & (sens.obs == ob)].S
    rows.append({"param": prm,
                 "complete": prm not in incomplete,
                 "n_pts": len(have),
                 "S_old_ec": S_old(prm, "amp_mean_tend", restrict=have),
                 "S_new_ec": get("amp_mean_tend").values[0],
                 "S_new_vt": get("amp_vt_mean_tend").values[0]})
cmp = pd.DataFrame(rows)
cmp["rank_new_vt"] = cmp.S_new_vt.abs().rank(ascending=False)
cmp["rank_new_ec"] = cmp.S_new_ec.abs().rank(ascending=False)
cmp["unit_shift"] = cmp.rank_new_vt - cmp.rank_new_ec
print("MEAN AMPLIFICATION -- S_old_ec: published campaign, event count, same points")
print("                      S_new_ec / S_new_vt: this campaign, both units, one event log")
print(cmp.sort_values("rank_new_vt").to_string(index=False, float_format="%.3f"))

print("\nF_veg_final -- ledger-independent, should reproduce the published campaign")
for prm in sc.SWEEPS:
    have = sorted(summary[summary.param == prm].value.unique())
    n = sens[(sens.param == prm) & (sens.obs == "F_veg_final")].S
    print("  %-14s old %+.3f   new %+.3f%s"
          % (prm, S_old(prm, "F_veg_final", restrict=have), n.values[0],
             "" if prm not in incomplete else "   (truncated)"))

new_s = summary.set_index(['param', 'value']).F_veg_final_mean
old_s = old.set_index(['param', 'value']).F_veg_final_mean
both = pd.concat({'new': new_s, 'old': old_s}, axis=1).dropna()
print("\nper-point F_veg_final: max |delta| = %.5f over %d shared points"
      % ((both.new - both.old).abs().max(), len(both)))

print("\nt_end_fit per arm: which points genuinely needed extending")
for arm, d in frames.items():
    if arm == "main":
        continue
    g = d[~isbase(d)].copy()
    g["censored"] = g.t_end_fit >= g.steps - 2
    print(" ", arm)
    print(g.groupby(["param", "value"]).agg(
        cens=("censored", "mean"), t_end_med=("t_end_fit", "median")).to_string())
