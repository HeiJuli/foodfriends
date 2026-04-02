# Alpha: Matched vs Unmatched Respondents
*Do alpha respondents who also answered the theta survey differ from those who did not?*

---

## Background

The alpha survey was answered by 4,944 respondents. Of these, 2,728 are also present in the theta survey ("matched") and 2,216 are not ("unmatched"). This matters because the current PMF imputation approach builds alpha distributions from the matched sample only (via inner join with theta). If matched and unmatched respondents differ systematically in their alpha values, the PMF tables are biased.

---

## Overall

| Group | Mean alpha | SD | n |
|---|---|---|---|
| Full sample | 0.677 | 0.244 | 4,944 |
| Matched (has theta) | 0.677 | 0.243 | 2,728 |
| Unmatched (no theta) | 0.678 | 0.247 | 2,216 |

At the overall level, matched and unmatched are essentially identical.

---

## By demographic group

### Gender
| Group | Matched | Unmatched | Diff | Cohen's d | p (FDR) |
|---|---|---|---|---|---|
| Female | 0.681 (n=1,415) | 0.686 (n=1,239) | −0.004 | −0.017 | 0.977 |
| Male | 0.672 (n=1,313) | 0.667 (n=977) | +0.005 | +0.020 | 0.977 |

No significant differences. Cohen's d < 0.02 — entirely negligible.

### Age group
| Group | Matched | Unmatched | Diff | Cohen's d | p (FDR) |
|---|---|---|---|---|---|
| 18–29 | 0.660 (n=269) | 0.665 (n=389) | −0.004 | −0.021 | 0.977 |
| 30–39 | 0.666 (n=346) | 0.680 (n=337) | −0.014 | −0.060 | 0.954 |
| 40–49 | 0.670 (n=507) | 0.682 (n=356) | −0.012 | −0.053 | 0.954 |
| 50–59 | 0.697 (n=689) | 0.700 (n=358) | −0.003 | −0.012 | 0.977 |
| 60–69 | 0.669 (n=659) | 0.686 (n=391) | −0.017 | −0.064 | 0.954 |
| 70+ | 0.689 (n=258) | 0.655 (n=385) | +0.034 | +0.125 | 0.954 |

No significant differences after FDR correction. The 70+ difference (+0.034) looks notable but has Cohen's d=0.125 — below the threshold for even a small effect (0.2) — and is not statistically significant.

### Income quartile
| Group | Matched | Unmatched | Diff | Cohen's d | p (FDR) |
|---|---|---|---|---|---|
| Q1 (lowest) | 0.657 (n=605) | 0.656 (n=531) | +0.000 | +0.002 | 0.989 |
| Q2 | 0.675 (n=665) | 0.673 (n=602) | +0.003 | +0.010 | 0.977 |
| Q3 | 0.682 (n=677) | 0.678 (n=560) | +0.004 | +0.016 | 0.977 |
| Q4 (highest) | 0.690 (n=781) | 0.704 (n=523) | −0.014 | −0.064 | 0.954 |

No significant differences. All Cohen's d < 0.07 — negligible.

### Education level
| Group | Matched | Unmatched | Diff | Cohen's d | p (FDR) |
|---|---|---|---|---|---|
| Low | 0.677 (n=207) | 0.629 (n=241) | +0.049 | +0.164 | 0.954 |
| Mid-low | 0.662 (n=1,000) | 0.670 (n=883) | −0.008 | −0.033 | 0.954 |
| Mid-high | 0.674 (n=641) | 0.674 (n=461) | −0.000 | −0.001 | 0.989 |
| High | 0.696 (n=880) | 0.709 (n=631) | −0.013 | −0.058 | 0.954 |

The low education group shows the largest raw difference (+0.049, Cohen's d=0.164), but this does not survive statistical testing (p_fdr=0.954). While it is the closest to a meaningful effect, it falls below the conventional small-effect threshold (d=0.2) and is not significant after correcting for multiple comparisons. The small matched cell size (n=207) means there is considerable uncertainty around the estimate.

---

## Summary

Independent samples t-tests with Benjamini-Hochberg FDR correction across 16 tests (4 demographic variables × groups).

| Variable | Max raw diff | Cohen's d | p (FDR) | Significant? |
|---|---|---|---|---|
| Gender | 0.005 | 0.020 | 0.977 | No |
| Age group | 0.034 (70+) | 0.125 | 0.954 | No |
| Income quartile | 0.014 | 0.064 | 0.954 | No |
| Education level | 0.049 (Low) | 0.164 | 0.954 | No |

**No differences are statistically significant after FDR correction.** All Cohen's d values are below 0.2 (the conventional threshold for a small effect), meaning the matched and unmatched groups are statistically indistinguishable in their alpha values across all demographic groups.

The raw differences that initially looked concerning (low education +0.049, 70+ +0.034) fall within normal sampling variation given the within-group standard deviation of ~0.24.

---

## Implication for imputation

The matched sample (those who also answered the theta survey) is **not systematically biased** in alpha values relative to the full alpha sample. This means the original concern — that building alpha PMFs from the matched sample would produce biased imputation — is not supported by the data.

The argument for switching alpha to **demographics-only PMFs from the full 4,944-person sample** therefore rests on statistical efficiency rather than bias:
- Larger cells → more stable PMF estimates
- No data wasted from discarding 2,216 respondents who answered alpha but not theta
- Simpler and more transparent imputation procedure

The theta-alpha correlation (r=+0.14) is real but weak, and the additional complexity of theta-stratified cells for alpha is unlikely to meaningfully improve imputation quality given the negligible selection differences shown here.
