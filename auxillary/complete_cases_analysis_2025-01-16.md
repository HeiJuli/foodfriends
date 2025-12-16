# Complete Cases Analysis - 2025-01-16

## Question
Can we obtain 800+ agents with all three parameters (theta, rho, alpha) measured empirically while maintaining demographic representativeness?

## Summary
**NO** - Complete cases (n=1298) are NOT demographically representative due to severe age bias.

## Key Findings

### Sample Size
- Full theta survey: 5714 respondents
- Complete cases (theta + rho + alpha): **1298** (22.7% retention)
- **Exceeds** 800 minimum threshold ✓

### Demographic Representativeness (using +/- 5% threshold)

#### Age Distribution - CRITICAL ISSUES
| Age Group | Full Survey | Complete Cases | Difference |
|-----------|-------------|----------------|------------|
| 18-29     | 14.3%       | 4.3%          | **-9.9%** ✗ |
| 30-39     | 13.8%       | 6.9%          | **-6.8%** ✗ |
| 40-49     | 14.2%       | 11.2%         | -3.1% ✓ |
| 50-59     | 17.3%       | 17.1%         | -0.2% ✓ |
| 60-69     | 20.4%       | 27.2%         | **+6.8%** ✗ |
| 70+       | 18.2%       | 33.3%         | **+15.1%** ✗ |

**Pattern**: Complete cases severely **underrepresent young adults** (18-39) and **overrepresent older adults** (60+), particularly 70+ group.

#### Income Distribution
| Quartile | Full Survey | Complete Cases | Difference |
|----------|-------------|----------------|------------|
| Q1       | 22.9%       | 15.1%         | **-7.8%** ✗ |
| Q2       | 27.1%       | 29.5%         | +2.4% ✓ |
| Q3       | 23.6%       | 26.3%         | +2.7% ✓ |
| Q4       | 26.4%       | 29.0%         | +2.7% ✓ |

**Pattern**: Underrepresents lowest income quartile.

#### Gender Distribution
| Gender | Full Survey | Complete Cases | Difference |
|--------|-------------|----------------|------------|
| Female | 54.0%       | 48.1%         | **-5.9%** ✗ |
| Male   | 46.0%       | 51.9%         | **+5.9%** ✗ |

**Pattern**: Marginal bias (just above 5% threshold).

#### Education Distribution - ACCEPTABLE
All categories within +/- 5% threshold ✓

## Stratified Sampling Feasibility

### Target: n=800 with Age Stratification
| Age Group | Target % | Available N | Need | Status |
|-----------|----------|-------------|------|--------|
| 18-29     | 14.5%    | 56          | 116  | **✗ 48% shortfall** |
| 30-39     | 14.0%    | 90          | 112  | **✗ 20% shortfall** |
| 40-49     | 14.5%    | 145         | 116  | ✓ |
| 50-59     | 17.6%    | 222         | 140  | ✓ |
| 60-69     | 20.8%    | 353         | 166  | ✓ |
| 70+       | 18.5%    | 432         | 148  | ✓ |

**Result**: INFEASIBLE - Insufficient young respondents

### Maximum Feasible Sample
With **perfect age stratification**: **385 agents**
- Below 800 threshold ✗

## Options

### Option 1: Post-hoc Weighting (n=1298 complete cases)
Use all complete cases, apply demographic weights in analysis
- **Pros**: Fully parametrized, no imputation, meets n>800
- **Cons**: Need weighting infrastructure, residual bias

### Option 2: Accept Smaller Sample (n=385)
Use maximally representative stratified subsample
- **Pros**: Perfect demographic matching
- **Cons**: Below 800 threshold

### Option 3: Continue PMF Approach (n=5602) **CURRENT**
Theta-stratified imputation for missing alpha/rho values
- **Pros**:
  - Demographically representative
  - Large sample size
  - Correlations preserved (diff < 0.04)
- **Cons**:
  - ~50-75% of alpha/rho values are sampled
  - Not fully empirical

### Option 4: Hybrid Approach
Use complete cases (1298) + minimal imputation to reach 800+ representative sample
- **Pros**: Maximizes empirical data
- **Cons**: Still requires imputation, complex implementation

## Validation of Current PMF Approach

From `validate_theta_stratification.py` (Dec 2025):

### Correlation Preservation
| Correlation | Empirical | Simulated | Difference |
|-------------|-----------|-----------|------------|
| theta-rho   | -0.341    | -0.304    | 0.037 ✓ |
| theta-alpha | +0.136    | +0.138    | 0.002 ✓ |
| rho-alpha   | -0.062    | -0.056    | 0.006 ✓ |

All correlations preserved within < 0.04 difference ✓

### Parameter Coverage
- Total agents: 5602
- Has alpha (empirical): 2773 (49.5%)
- Has rho (empirical): 2391 (42.7%)
- Has both: 1298 (23.2%)

## Conclusion

**Cannot achieve 800+ demographically representative complete-cases sample.**

The current **PMF imputation approach** (Option 3) is the best solution:
- Maintains demographic representativeness
- Preserves parameter correlations
- Large sample size (n=5602)
- Validated performance

The complete-cases-only approach fails due to systematic age bias in survey completion - younger respondents were less likely to complete all three parameter surveys.
