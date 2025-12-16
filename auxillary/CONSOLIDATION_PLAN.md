# Auxillary Folder Consolidation Plan
**Date**: 2025-01-16

## Current State Analysis

### File Categories

#### **CORE PRODUCTION FILES** (Keep - Recent & Essential)
1. `create_pmf_tables.py` (Dec 11) - Creates theta-stratified PMF tables for parameter imputation
2. `create_hierarchical_agents.py` (Aug 29) - Creates hierarchical agent dataset from surveys
3. `network_stats.py` (Jun 27) - Network topology analysis utilities
4. `test_homophilly.py` (Jun 27) - Network homophily testing

#### **RECENT VALIDATION SCRIPTS** (Dec 2025 - Consolidate)
5. `validate_theta_stratification.py` (Dec 12) - **PRIMARY VALIDATION**
   - Validates theta-stratified PMF approach
   - Tests correlation preservation
   - Checks effective thresholds
   - Generates validation plots

6. `diagnose_parameter_sampling.py` (Dec 12) - **DIAGNOSTIC**
   - Analyzes parameter sampling strategy
   - Checks for rapid uptake indicators
   - Examines PMF tables
   - Overlaps with #5 significantly

7. `analyze_conditional_thresholds.py` (Dec 12) - **THRESHOLD ANALYSIS**
   - Analyzes effective thresholds by theta bins
   - Tests if low-theta agents have higher thresholds
   - Creates threshold distribution plots
   - Specialized analysis, partially redundant with #5

8. `check_complete_cases_representativeness.py` (Dec 16) - **NEW ANALYSIS**
   - Checks demographic representativeness of complete cases
   - One-off analysis, can be removed after documentation

9. `create_stratified_representative_sample.py` (Dec 16) - **NEW ANALYSIS**
   - Tests feasibility of stratified sampling
   - One-off analysis, can be removed after documentation

#### **OLDER VALIDATION SCRIPTS** (Sep 2025 - Redundant)
10. `validation_full.py` (Sep 2) - **SUPERSEDED**
    - Tests demographic predictive power
    - Tests PMF cell differences
    - Tests sampling behavior
    - Validation methodology tests
    - **REPLACED BY** `validate_theta_stratification.py`

11. `validation_testing.py` (Sep 2) - **SUPERSEDED**
    - PMF sampling validation
    - Demographic consistency checks
    - Overlaps analysis
    - **REPLACED BY** `diagnose_parameter_sampling.py`

12. `demographic_validation.py` (Sep 2) - **SUPERSEDED**
    - Rho-theta bias analysis
    - Dual correction weights
    - **REPLACED BY** conditional threshold analysis

13. `parameter_pairwise_validation.py` (Sep 2) - **SUPERSEDED**
    - Pairwise correlations
    - Three-way relationships
    - Prediction approach comparisons
    - **REPLACED BY** correlation checks in recent scripts

#### **TEST SCRIPTS** (Nov 2025 - Remove)
14. `test_reload.py` (Nov 10) - Module reload testing, not relevant
15. `test_cascade.py` (Nov 10) - Cascade tracking test for model, not relevant to parameter sampling

## Consolidation Actions

### KEEP (6 files)
```
create_pmf_tables.py              # Production: PMF table generation
create_hierarchical_agents.py     # Production: Agent dataset creation
network_stats.py                  # Utility: Network analysis
test_homophilly.py                # Utility: Network homophily
validate_theta_stratification.py  # Primary validation script
complete_cases_analysis_2025-01-16.md  # Documentation
```

### CREATE NEW (1 consolidated script)
```
parameter_diagnostics.py  # Consolidated from diagnose + conditional threshold analysis
```

### ARCHIVE (Move to old/)
```
validation_full.py
validation_testing.py
demographic_validation.py
parameter_pairwise_validation.py
diagnose_parameter_sampling.py
analyze_conditional_thresholds.py
check_complete_cases_representativeness.py
create_stratified_representative_sample.py
test_reload.py
test_cascade.py
```

## Proposed New Structure

```
auxillary/
├── README.md                              # NEW: Explains folder purpose
├── create_hierarchical_agents.py          # KEEP: Agent dataset creation
├── create_pmf_tables.py                   # KEEP: PMF table generation
├── validate_theta_stratification.py       # KEEP: Primary validation
├── parameter_diagnostics.py               # NEW: Consolidated diagnostics
├── network_stats.py                       # KEEP: Network utilities
├── test_homophilly.py                     # KEEP: Network testing
├── complete_cases_analysis_2025-01-16.md  # KEEP: Documentation
├── old/                                   # Archive folder
│   └── [10 archived scripts]
└── __pycache__/
```

## New Consolidated Script: `parameter_diagnostics.py`

### Purpose
Single script for diagnosing parameter sampling issues, combining:
- Parameter sampling diagnostics (from diagnose_parameter_sampling.py)
- Conditional threshold analysis (from analyze_conditional_thresholds.py)
- Complete cases analysis (from check_complete_cases_representativeness.py)

### Functions
```python
# From diagnose_parameter_sampling.py
- analyze_hierarchical_csv()
- analyze_pmf_tables()
- analyze_parameter_correlations()
- simulate_agent_initialization()
- check_rapid_uptake_indicators()

# From analyze_conditional_thresholds.py
- simulate_single_population()
- analyze_conditional_thresholds()
- create_threshold_distribution_plot()

# From check_complete_cases_representativeness.py
- check_complete_cases_representativeness()
```

## Documentation: README.md

### Content
- Purpose of auxiliary folder
- Description of each production script
- Workflow: hierarchical_agents → PMF tables → validation
- When to run each script
- Archive folder explanation

## Benefits

1. **Reduced redundancy**: 16 → 7 active files (56% reduction)
2. **Clear organization**: Production, validation, utilities, docs
3. **Preserved history**: All old scripts archived, not deleted
4. **Single source of truth**: One validation script, one diagnostics script
5. **Better maintainability**: Less confusion about which script to run
6. **Documentation**: README and analysis docs explain everything

## Implementation Steps

1. Create `old/` subdirectory
2. Create consolidated `parameter_diagnostics.py`
3. Create `README.md`
4. Move 10 files to `old/`
5. Test that remaining scripts work
6. Update CLAUDE.md if needed
