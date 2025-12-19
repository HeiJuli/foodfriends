# Auxillary Folder

This folder contains utilities for parameter sampling, validation, and network analysis for the FoodFriends model.

## Core Production Scripts

### `create_hierarchical_agents.py`
Creates the hierarchical agent dataset from raw survey data.

**Purpose**: Combines theta, rho, and alpha surveys into a single dataset with hierarchical parameter availability (complete cases prioritized over partial cases).

**Input**:
- `../data/theta_diet_demographics.xlsx`
- `../data/rho_demographics.xlsx`
- `../data/alpha_demographics.xlsx`

**Output**:
- `../data/hierarchical_agents.csv`

**When to run**: When raw survey data is updated or when recreating the agent population.

### `create_pmf_tables.py`
Generates theta-stratified probability mass function (PMF) tables for parameter imputation.

**Purpose**: Creates demographic PMF tables stratified by theta to preserve parameter correlations when imputing missing alpha/rho values.

**Input**:
- `../data/theta_diet_demographics.xlsx`
- `../data/rho_demographics.xlsx`
- `../data/alpha_demographics.xlsx`

**Output**:
- `../data/demographic_pmfs.pkl`

**When to run**: After updating survey data or when modifying theta stratification bins.

**Key feature**: Alpha and rho PMFs are stratified by theta bins to preserve empirical correlations.

### `sampling_utils.py`
**STRATIFIED SAMPLING UTILITY** (NEW: 2025-01-27)

Provides demographic-preserving sampling for agent initialization when N < 5602.

**Purpose**: Ensures population samples maintain demographic representativeness across gender, age, income, and education.

**Key function**: `stratified_sample_agents(df, n_target, strata_cols, random_state, verbose)`

**Performance**: Preserves demographic distributions within ±0.21% maximum deviation (vs ~1% for simple random sampling).

**Usage**: Automatically used by model when `agent_ini="twin"` and N < 5602. Can also be imported for manual use.

**Why it matters**: Previous simple random sampling could introduce demographic bias. Stratified sampling provides:
- 89.7% improvement in gender preservation
- 96.7% improvement in age preservation
- 87.3% improvement in income preservation
- 78.9% improvement in education preservation

### `analyze_sample_size.py`
**OPTIMAL N ANALYSIS TOOL** (NEW: 2025-01-27)

Analyzes trade-off between finite-size effects and parameter imputation accuracy.

**Purpose**: Determines optimal population size N for balancing statistical noise (CV ~ 1/√N) vs empirical grounding (fraction of complete cases).

**What it analyzes**:
1. Complete vs partial case composition
2. Finite-size coefficient of variation at different N
3. Imputation fraction vs N
4. Effective threshold distributions

**Output**:
- Comprehensive analysis report
- Comparison plots: `../visualisations_output/optimal_sample_size_analysis.png`
- Recommendation: N=2000 optimal for most use cases

**Key findings**:
- N=2000: CV=2.2%, 35% imputation, publication quality
- N=5602: CV=1.3%, 77% imputation, mostly synthetic

**When to run**: When questioning optimal N or preparing publication justification.

### `stratified_sampling.py`
**SAMPLING COMPARISON TOOL** (NEW: 2025-01-27)

Demonstrates improvement of stratified sampling over random sampling.

**Purpose**: Provides validation and comparison analysis of sampling methods.

**What it does**:
1. Compares random vs stratified sampling across multiple trials
2. Quantifies demographic preservation quality
3. Generates comparison statistics

**When to run**: For validation or when documenting sampling approach for publication.

## Validation Scripts

### `validate_theta_stratification.py`
**PRIMARY VALIDATION SCRIPT**

Validates that the theta-stratified PMF sampling approach preserves parameter correlations and produces stable model dynamics.

**What it tests**:
1. Correlation preservation (theta-rho, theta-alpha, rho-alpha)
2. Effective thresholds for meat eaters
3. Distribution matching

**When to run**: After creating/updating PMF tables or when changing sampling approach.

**Expected output**:
- Correlations preserved within 0.05 difference
- Effective threshold > 0.20 for stability
- Validation plots in `../visualisations_output/`

### `parameter_diagnostics.py`
**CONSOLIDATED DIAGNOSTICS**

Comprehensive diagnostic suite for troubleshooting parameter sampling issues.

**What it analyzes**:
1. Hierarchical agent dataset composition
2. PMF table statistics
3. Parameter correlations in empirical data
4. Conditional effective thresholds by theta bin
5. Complete cases demographic representativeness

**When to run**: When investigating rapid uptake, strange model behavior, or demographic bias.

## Network Analysis Utilities

### `network_stats.py`
Network topology analysis functions.

**Functions**:
- Edge type counting (veg-veg, meat-meat, mixed)
- Homophily measures
- Network statistics calculation

**Usage**: Import functions into other scripts for network analysis.

### `test_homophilly.py`
Test script for network homophily measures.

**Purpose**: Validates network generation and homophily calculations.

## Workflow

### Standard Model Run Workflow
1. **One-time setup** (already completed):
   ```bash
   python create_hierarchical_agents.py  # Creates agent dataset
   python create_pmf_tables.py            # Creates PMF tables
   python validate_theta_stratification.py  # Validates approach
   ```

2. **Model runtime** (automatic):
   - Loads `hierarchical_agents.csv`
   - If N < 5602: applies **stratified sampling** to preserve demographics
   - If N = 5602: uses all participants
   - Loads `demographic_pmfs.pkl`
   - Samples missing parameters using theta-stratified PMFs

3. **Recommended N**: Use N=2000 for optimal balance (see `analyze_sample_size.py`)

### Troubleshooting Workflow
1. Run `parameter_diagnostics.py` for comprehensive analysis
2. Check effective thresholds by theta bin
3. Examine complete cases vs imputed cases
4. Review validation plots

### Optimal N Selection Workflow
1. Run `python analyze_sample_size.py` to see full trade-off analysis
2. Review output plots in `../visualisations_output/optimal_sample_size_analysis.png`
3. For most use cases: N=2000 recommended (CV=2.2%, 35% imputation)
4. For sensitivity analysis: test N ∈ {2000, 3000, 5000}

## Archive

The `old/` folder contains superseded validation scripts from earlier iterations:
- `validation_full.py` (Sep 2025)
- `validation_testing.py` (Sep 2025)
- `demographic_validation.py` (Sep 2025)
- `parameter_pairwise_validation.py` (Sep 2025)
- And others...

These are kept for reference but have been replaced by the current validation suite.

## Key Findings Documented

### Optimal Sample Size (2025-01-27)
**See**: `../claude_stuff/optimal_sample_size_analysis_2025-01-27.md`

Key findings:
- **N=2000 recommended** for optimal balance of finite-size effects vs empirical grounding
- Complete cases: only 1298 (23.2%) of 5602 total participants
- **Stratified sampling essential**: preserves demographics within ±0.21% (vs ~1% for random)
- Finite-size CV at N=2000: 2.2% (publication quality, < 2.5% threshold)
- Imputation fraction at N=2000: 35% (vs 77% at N=5602)
- **CRITICAL FIX**: Replaced simple random sampling with stratified sampling in model code

### Complete Cases Analysis (2025-01-16)
**See**: `complete_cases_analysis_2025-01-16.md`

Key findings:
- Why complete-cases-only approach is infeasible (age bias: 70+ overrepresented 33% vs 19%)
- Stratified sampling feasibility assessment
- Comparison of different parametrization approaches
- Recommendation to continue with PMF imputation approach
