# Demographic Distribution of Model Parameters
*Based on `hierarchical_agents.csv` — 5,602 survey participants (Netherlands)*

---

## Parameter definitions

| Parameter | Survey question / derivation | Range | What a high value means |
|---|---|---|---|
| **θ (theta)** | Personal preference for vegetarian diet | −1 to +1 | Strongly prefers veg food |
| **ρ (rho)** | Behavioral intention to change diet | 0 to 1 | Strong intention to act |
| **α (alpha)** | "My personal identity, independent of others, is very important to me" | 0 to 1 | Strong sense of independent personal identity |

**Coverage:** theta for all 5,602 participants; rho for 2,391 (from merged file — only 106 rho respondents have no theta match, negligible); alpha for 4,944 (from the original alpha survey directly — using the merged file would exclude ~2,328 alpha respondents who never answered the theta survey). Means and SDs reported; error bars in figures are standard error.

**Important note on alpha:** Alpha is a general measure of individual identity strength — not dietary identity specifically. It captures how much someone sees themselves as an autonomous individual, independent of social context. In the model this is used as a weight on how strongly an agent's own preferences resist social influence.

---

## Gender

| Group | θ mean (SD) | n | ρ mean (SD) | n | α mean (SD) | n |
|---|---|---|---|---|---|---|
| Female | 0.508 (0.307) | 3,015 | 0.436 (0.306) | 1,220 | 0.683 (0.244) | 2,654 |
| Male | 0.379 (0.354) | 2,587 | 0.534 (0.304) | 1,171 | 0.670 (0.244) | 2,290 |

**Theta:** Women have substantially higher veg food preference than men (0.51 vs 0.38). This aligns with existing literature on gender differences in dietary attitudes.

**Rho:** Despite preferring veg food more, women report *lower* behavioral intention to change (0.44 vs 0.53). This is a meaningful tension: women are more aligned with plant-based food in preference but less ready to act on it. This could reflect greater perceived social or practical barriers to dietary change, or a higher baseline satisfaction with current diet.

**Alpha:** Essentially identical across genders (0.68 vs 0.67). Sense of independent personal identity does not differ by gender in this sample.

---

## Age group

| Age | θ mean (SD) | n | ρ mean (SD) | n | α mean (SD) | n |
|---|---|---|---|---|---|---|
| 18–29 | 0.370 (0.336) | 813 | 0.450 (0.349) | 220 | 0.663 (0.216) | 658 |
| 30–39 | 0.397 (0.346) | 783 | 0.501 (0.310) | 274 | 0.673 (0.231) | 683 |
| 40–49 | 0.407 (0.357) | 813 | 0.490 (0.314) | 302 | 0.675 (0.232) | 863 |
| 50–59 | 0.450 (0.334) | 987 | 0.475 (0.311) | 378 | 0.698 (0.235) | 1,047 |
| 60–69 | 0.509 (0.318) | 1,167 | 0.452 (0.294) | 575 | 0.676 (0.267) | 1,050 |
| 70+   | 0.510 (0.306) | 1,039 | 0.520 (0.300) | 642 | 0.668 (0.275) | 643 |

**Theta:** A clear positive trend with age — older participants prefer veg food more. The 60–69 and 70+ groups (~0.51) score notably higher than 18–29 (~0.37). This is counterintuitive if one expects younger generations to be more environmentally conscious, but may reflect cohort-level dietary patterns, health motivations among older adults, or the Dutch context specifically.

**Rho:** Relatively flat across age groups, with a slight dip in the middle age groups (50–59: 0.475) and the highest values among 30–39 (0.501) and 70+ (0.520). There is no strong monotonic age trend in behavioral intention, despite the clear preference trend. The 70+ group combining high theta and high rho is notable — this group is both the most pro-veg and among the most ready to act.

**Alpha:** Slightly higher among 60–69 (0.698) and 50–59 (0.681), lower among younger groups (~0.66). Older adults report a marginally stronger sense of independent personal identity. The differences are small and may not be substantively meaningful.

---

## Income quartile

| Income | θ mean (SD) | n | ρ mean (SD) | n | α mean (SD) | n |
|---|---|---|---|---|---|---|
| Q1 (lowest)  | 0.441 (0.336) | 1,199 | 0.459 (0.329) | 396 | 0.657 (0.245) | 1,136 |
| Q2           | 0.490 (0.326) | 1,546 | 0.510 (0.307) | 676 | 0.674 (0.259) | 1,267 |
| Q3           | 0.444 (0.332) | 1,351 | 0.477 (0.306) | 619 | 0.680 (0.244) | 1,237 |
| Q4 (highest) | 0.415 (0.344) | 1,506 | 0.480 (0.300) | 700 | 0.695 (0.227) | 1,304 |

**Theta:** No clear monotonic trend with income. Q2 is highest (0.49), Q4 lowest (0.42). Higher income does not translate to stronger veg food preference in this sample.

**Rho:** Q2 is again highest (0.51), Q1 lowest (0.46). The pattern is non-monotonic. The lowest income group has both lower theta and lower rho, suggesting potential resource or access barriers to dietary change that are worth considering in policy contexts.

**Alpha:** Q4 (0.696) and Q2 (0.685) are slightly higher than Q1 (0.654) and Q3 (0.663). Wealthier individuals have a marginally stronger sense of independent personal identity, though the differences are small.

---

## Education level

| Education | θ mean (SD) | n | ρ mean (SD) | n | α mean (SD) | n |
|---|---|---|---|---|---|---|
| Low      | 0.474 (0.340) | 355 | 0.526 (0.308) | 166 | 0.651 (0.298) | 448 |
| Mid-low  | 0.464 (0.335) | 1,739 | 0.509 (0.300) | 755 | 0.666 (0.252) | 1,883 |
| Mid-high | 0.425 (0.340) | 1,348 | 0.525 (0.313) | 533 | 0.674 (0.239) | 1,102 |
| High     | 0.446 (0.332) | 2,160 | 0.433 (0.308) | 937 | 0.702 (0.218) | 1,511 |

**Theta:** Small variation with no clear trend. Mid-high education is slightly lowest (0.425), others are 0.44–0.47. Education does not strongly predict veg food preference.

**Rho:** The most striking education pattern: low, mid-low, and mid-high education groups all have similar and relatively high rho (~0.51–0.53), but the highly educated group is notably lower (0.433). Highly educated people report *less* intention to change their diet despite having moderate veg preference. This could reflect: greater scrutiny of dietary advice, stronger identity investment in their current behaviour, or more critical evaluation of behavioural change (the "intention-action gap" being more pronounced among more analytical thinkers).

**Alpha:** Increases with education: low (0.659) to high (0.697). More educated individuals report a stronger sense of independent personal identity. This is consistent with education fostering individualism and autonomous self-concept.

---

## Summary of key patterns

| Finding | Parameters | Direction |
|---|---|---|
| Women prefer veg more but are less ready to act | θ, ρ | θ: F>M; ρ: M>F |
| Older adults are more pro-veg (counterintuitive) | θ | Increases with age |
| No group is strongly income-patterned for veg preference | θ | Flat/non-monotonic |
| Highly educated people have the lowest behavioral intention | ρ | Low education > High education |
| Independent identity is uniform across gender and age | α | Largely flat |
| Highly educated individuals have the strongest independent identity | α | Increases with education |

---

## Implications for the model

- The **gender gap in rho vs theta** (women: high preference, low intention; men: low preference, high intention) means the model's starting conditions differ meaningfully by gender — worth considering in targeted intervention scenarios.
- The **older = more pro-veg** finding suggests that if the network is age-homophilic (which it is, by design), vegetarian clusters may form more readily among older age groups.
- The **flat alpha across most groups** supports the methodological choice to impute alpha from demographics-only PMFs — there is limited demographic signal in alpha to exploit.
- The **high-education low-rho** pattern is important: highly educated agents are harder to move despite moderate preference, which could dampen adoption in high-education clusters.

---

*Files in this folder:*
- `demographic_parameter_distributions.py` — script to reproduce figure and CSV
- `demographic_parameter_summary.csv` — numerical summary table
- `demographic_parameter_distributions.png` — figure (bar charts by group)
- `demographic_parameter_findings.md` — this document
