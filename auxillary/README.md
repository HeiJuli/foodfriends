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
Generates conditional PMF tables for parameter imputation.

**Purpose**: Creates demographic PMF tables for imputing missing alpha/rho values.
- **Alpha**: demographics only (n=4944). Theta stratification removed (2026-03-25) due to selection bias -- only 54.6% of alpha respondents have theta, with biased subgroups in 70+ and low-education cells. Weak alpha-theta correlation (r=0.14) does not justify the data loss.
- **Rho**: demographics + theta bins (n=2391, 96.3% theta overlap, r=+0.34; sign corrected 2026-09-02, was quoted as -0.30 from inverted rho).

**Input**:
- `../data/theta_diet_demographics.xlsx`
- `../data/rho_demographics.xlsx`
- `../data/alpha_demographics.xlsx`

**Output**:
- `../data/demographic_pmfs.pkl`

**When to run**: After updating survey data or when modifying imputation strategy.

### `sampling_utils.py`

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

Analyzes trade-off between finite-size effects and parameter imputation accuracy.

**Purpose**: Determines optimal population size N for balancing statistical noise (CV ~ 1/√N) vs empirical grounding (fraction of complete cases).

**What it analyzes**:
1. Complete vs partial case composition
2. Finite-size coefficient of variation at different N
3. Imputation fraction vs N

**Output**:
- Comprehensive analysis report
- Comparison plots: `../visualisations_output/optimal_sample_size_analysis.png`
- Recommendation: N=2000 optimal for most use cases

**Key findings**:
- N=2000: CV=2.2%, 35% imputation, publication quality
- N=5602: CV=1.3%, 77% imputation, mostly synthetic

**When to run**: When questioning optimal N or preparing publication justification.

## Validation Scripts

### `validate_theta_stratification.py`

Validates that the PMF sampling approach preserves parameter correlations. Rho uses theta-stratified PMFs; alpha uses demographics-only PMFs.

**What it tests**:
1. Correlation preservation (theta-rho, theta-alpha, rho-alpha)
2. Distribution matching

**When to run**: After creating/updating PMF tables or when changing sampling approach.

**Expected output**:
- Correlations preserved within 0.05 difference
- Validation plots in `../visualisations_output/`

### `parameter_diagnostics.py`

Comprehensive diagnostic suite for troubleshooting parameter sampling issues.

**What it analyzes**:
1. Hierarchical agent dataset composition
2. PMF table statistics
3. Parameter correlations in empirical data
4. Rho and alpha by theta bin (meat eaters)
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
   - Samples missing parameters using conditional PMFs (alpha: demographics-only; rho: theta-stratified)

3. **Recommended N**: Use N=2000 for optimal balance (see `analyze_sample_size.py`)

### Troubleshooting Workflow
1. Run `parameter_diagnostics.py` for comprehensive analysis
2. Check rho by theta bin
3. Examine complete cases vs imputed cases
4. Review validation plots

### Optimal N Selection Workflow
1. Run `python analyze_sample_size.py` to see full trade-off analysis
2. Review output plots in `../visualisations_output/optimal_sample_size_analysis.png`
3. For most use cases: N=2000 recommended (CV=2.2%, 35% imputation)
4. For sensitivity analysis: test N ∈ {2000, 3000, 5000}

## Archive

The repository's top-level `old/` folder holds superseded validation scripts from
earlier iterations, kept for reference only.

## Design decisions

### Sample size
- N=2000 balances finite-size effects against empirical grounding.
- Complete cases: 1298 (23.2%) of 5602 participants.
- Stratified sampling preserves demographics within +-0.21%, against ~1% for random.
- Finite-size CV at N=2000: 2.2%. Imputation fraction at N=2000: 35%, against 77% at N=5602.

### Complete cases
- A complete-cases-only parametrisation is infeasible: the 70+ group is overrepresented
  (33% against 19% in the population).
- PMF imputation is used instead, conditioned on demographics.
