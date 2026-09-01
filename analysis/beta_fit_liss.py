"""Fit beta (inverse temperature) to the LISS panel with the model's own switching rule.

Same-data lower anchor for beta (R4.17). Each complete-case respondent (theta, rho, alpha
observed) is a two-state chain under hamiltonian() with mean-field social field h_soc = F,
the 2018 vegan fraction. Diet observed twice: oi18a016 (2018, vegan yes/no) and su19a046
(2019, meat frequency; 6 = never). Because the 2018 item is "vegan", most 2018 non-vegans
who never eat meat in 2019 were vegetarian already; the data are closer to a cross-section
than to transitions, so the stationary-limit row is the one to quote.

beta and the number of activations between waves are not jointly identified (the R1.5
timescale problem), so beta is reported under fixed conventions: one update per wave
(Galesic et al. 2021's fitting convention) and the stationary limit.

Run from the repo root:  python analysis/beta_fit_liss.py
"""
import numpy as np
import pandas as pd
from scipy.stats import binom

M, GATE_C, GATE_K, A_MIN, A_MAX = 9, 0.35, 35, 0.05, 0.80  # model_runn.DEFAULT_PARAMS
BETAS = np.exp(np.linspace(np.log(0.5), np.log(300), 400))
LISS = "data/data_construction_paper/su19a_EN_1.0p.dta"


def sig(x):
    return 1 / (1 + np.exp(-x))


def expected_gate(F_opp):
    """E[gate] over M binomial draws when the opposite diet has population fraction F_opp."""
    p_opp = np.arange(M + 1) / M
    return (binom.pmf(np.arange(M + 1), M, F_opp) * sig(GATE_K * (p_opp - GATE_C))).sum()


def rates(beta, rho, t, w, F):
    """Per-activation switch probabilities for a meat-eater (m->v) and a vegetarian (v->m)."""
    g_m, g_v = expected_gate(F), expected_gate(1 - F)
    hind_m, hind_v = (1 - g_m) * rho + g_m * t, (1 - g_v) * rho + g_v * t
    dH_m = (1 - w) * (1 - 2 * hind_m) + w * (1 - 2 * F)   # H_veg - H_meat
    dH_v = (1 - w) * (2 * hind_v - 1) + w * (2 * F - 1)   # H_meat - H_veg
    return sig(-beta * dH_m), sig(-beta * dH_v)


def loglik(beta, n, y0, y1, rho, t, w, F):
    """n activations (int) or None for the stationary limit; exact for a two-state chain."""
    pmv, pvm = rates(beta, rho, t, w, F)
    pi = pmv / (pmv + pvm)
    mix = 1.0 if n is None else 1 - (1 - pmv - pvm) ** n
    p = np.clip(np.where(y0 == 1, 1 - (1 - pi) * mix, pi * mix), 1e-12, 1 - 1e-12)
    return (y1 * np.log(p) + (1 - y1) * np.log(1 - p)).sum()


def fit(n, y0, y1, rho, t, w, F):
    L = np.array([loglik(b, n, y0, y1, rho, t, w, F) for b in BETAS])
    ci = BETAS[L >= L.max() - 1.92]
    return BETAS[L.argmax()], ci.min(), ci.max()


def main():
    h = pd.read_csv("data/hierarchical_agents.csv")
    F = (h.diet == "veg").mean()
    s = pd.read_stata(LISS, convert_categoricals=False)[["nomem_encr", "su19a046"]]
    d = h.merge(s, on="nomem_encr")
    d = d[d.has_rho & d.has_alpha & d.su19a046.notna()]
    w = 1 - (A_MIN + (A_MAX - A_MIN) * d.alpha.values)
    t = (d.theta.values + 1) / 2
    y0 = (d.diet == "veg").values.astype(int)
    y1 = (d.su19a046 == 6).values.astype(int)
    print(f"INFO: n={len(d)} complete cases, F_2018={F:.4f}, "
          f"m->v={((y0 == 0) & (y1 == 1)).sum()}, v->m={((y0 == 1) & (y1 == 0)).sum()}")
    print(f"{'convention':22s} {'rho coding':12s} {'beta_hat':>8s}  95% profile CI")
    # rho in the CSV is the corrected coding (1 = "Yes, definitely"; fixed 2026-09-02);
    # the second row reproduces the pre-correction (inverted) coding for the record.
    for rlab, rho in [("corrected", d.rho.values), ("old inverted", 1 - d.rho.values)]:
        for nlab, n in [("one update per wave", 1), ("stationary limit", None)]:
            b, lo, hi = fit(n, y0, y1, rho, t, w, F)
            print(f"{nlab:22s} {rlab:12s} {b:8.1f}  [{lo:.1f}, {hi:.1f}]")
    print("NOTE: our beta scale; Galesic-equivalent = beta/2 (see boltzmann_model.md).")


if __name__ == "__main__":
    main()
