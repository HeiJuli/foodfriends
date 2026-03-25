# Data, Sampling, and PMF Imputation

**Last updated:** 2026-03-05
**Files:** `auxillary/sampling_utils.py`, `auxillary/create_pmf_tables.py`, `data/hierarchical_agents.csv`, `data/demographic_pmfs.pkl`

---

## 1. Survey Data Composition

- Total participants: 5602
- Complete cases (theta, rho, alpha): 1298 (23.2%)
- Partial cases requiring imputation: 4304 (76.8%)
  - Has rho only: 1093 (19.5%)
  - Has alpha only: 1475 (26.3%)
  - Has neither: 1736 (31.0%)

**Complete-cases-only (N=1298) has severe age bias**: age 70+: 33% vs 19% in full survey; age 18-29: 4% vs 15%. Not usable as standalone population.

---

## 2. Optimal N: Recommendation N=2000

**Trade-off**: finite-size effects (need large N) vs empirical grounding (need low imputation fraction).

CV from finite-size: `CV ≈ 1/sqrt(N)`. Publication standard: CV < 2.5%.

| N | %Imputed | CV | Assessment |
|---|----------|-----|---|
| 1298 | 0% | 2.8% | Age-biased |
| **2000** | **35%** | **2.2%** | **Optimal** |
| 3000 | 57% | 1.8% | High imputation |
| 5602 | 77% | 1.3% | Mostly synthetic |

**Why N=2000:**
- CV=2.2% meets publication standard
- 65% of agents use complete empirical parameters
- PMF sampling preserves correlations: theta-rho=-0.341 (empirical) vs -0.299 (simulated, Δ=0.042); theta-alpha=+0.136 vs +0.143
- ~3x faster than N=5602
- Sufficient for network effects, phase transitions, cascade dynamics

For publication: ensemble of 10-20 runs at N=2000 > single run at N=5602 (better uncertainty quantification, stronger empirical grounding per run).

---

## 3. Stratified Sampling (implemented 2025-01-27)

**Problem**: Previous code used `survey_data.sample(n=N, random_state=42)` — simple random sampling. Could introduce ~1% demographic bias (gender ±0.68%, age ±1.11%, income ±1.15%, education ±0.98%).

**Fix**: `stratified_sample_agents()` in `auxillary/sampling_utils.py`. Stratifies across 4 demographic dimensions (gender, age_group, incquart, educlevel) — 188 unique strata. Maximum deviation: **±0.21%**.

**Improvement**: 78-97% better demographic preservation across all dimensions.

**Where it runs**: Base `Model.__init__()` in `model_src/model_main.py`. Automatic when N < 5602 in twin mode. All code paths (direct instantiation, runner scripts, testing scripts) inherit this.

---

## 4. PMF Imputation Method

For agents missing alpha or rho: **random hot-deck imputation within adjustment cells** (Andridge & Little 2010).

### Formulation

$$\hat{P}(\alpha = a \mid \mathbf{d}, b) = \frac{n(\alpha = a,\; \mathbf{d},\; b)}{n(\mathbf{d},\; b)}$$

where d = demographics (gender, age, income, education) and b = theta bin.

**Theta bins:** (-1.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]

**Fallback hierarchy:**
1. Full key (demographics + theta bin)
2. Demographics only
3. Global pool (uniform draw)

### Why theta-stratified?

Conditioning on theta bin preserves empirical correlations: theta-rho (r=-0.30), theta-alpha (r=+0.14), rho-alpha (r=-0.06). Simple demographic-only PMFs break these.

### Relation to other methods

- **vs bootstrapping**: hot-deck imputes missing values; bootstrapping resamples complete observations for inference
- **vs MLE**: hot-deck makes no distributional assumptions (uses raw empirical frequencies)
- **vs multiple imputation**: single imputation per agent per run; ensemble of runs captures imputation uncertainty implicitly

### Manuscript description

"For agents lacking a directly observed alpha (or rho), we impute by sampling from the empirical conditional distribution P(alpha | d, theta_bin), estimated nonparametrically from relative frequencies within each demographic-theta stratum. This random hot-deck approach within adjustment cells (Andridge & Little 2010) preserves the empirical correlations between parameters (r_theta,rho=-0.30, r_theta,alpha=+0.14) without imposing distributional assumptions."

---

## 5. References

- Andridge, R.R. & Little, R.J. (2010). A review of hot deck imputation. International Statistical Review 78(1):40-64.
- Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.
- Watts, D.J. & Strogatz, S.H. (1998). Nature 393:440.
- Barabási, A.L. & Albert, R. (1999). Science 286:509.
- Centola, D. & Macy, M. (2007). American Journal of Sociology 113(3):702-734.
