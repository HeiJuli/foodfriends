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
1. **One-time setup**:
   ```bash
   python create_hierarchical_agents.py  # Creates agent dataset
   python create_pmf_tables.py            # Creates PMF tables
   python validate_theta_stratification.py  # Validates approach
   ```

2. **Model uses**:
   - Loads `hierarchical_agents.csv`
   - Loads `demographic_pmfs.pkl`
   - Samples missing parameters using theta-stratified PMFs

### Troubleshooting Workflow
1. Run `parameter_diagnostics.py` for comprehensive analysis
2. Check effective thresholds by theta bin
3. Examine complete cases vs imputed cases
4. Review validation plots

## Archive

The `old/` folder contains superseded validation scripts from earlier iterations:
- `validation_full.py` (Sep 2025)
- `validation_testing.py` (Sep 2025)
- `demographic_validation.py` (Sep 2025)
- `parameter_pairwise_validation.py` (Sep 2025)
- And others...

These are kept for reference but have been replaced by the current validation suite.

## Key Findings Documented

See `complete_cases_analysis_2025-01-16.md` for detailed analysis of:
- Why complete-cases-only approach is infeasible (age bias)
- Stratified sampling feasibility assessment
- Comparison of different parametrization approaches
- Recommendation to continue with PMF imputation approach
