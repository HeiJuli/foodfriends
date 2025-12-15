# Testing Directory

Consolidated test scripts for model development and validation.

## Test Scripts

### test_initial_dynamics.py
Consolidated testing for initial dynamics and uptake behavior in the utility model.

**Tests included:**
- Memory (M) parameter sweep (5-10)
- Initialization method comparison (100% current diet vs neighbor-based)
- Initial inertia strategy (utility model with inertia penalties)

**Usage:**
```bash
cd model_src/testing
python test_initial_dynamics.py
```

Select test when prompted: 1, 2, 3, or 'all'

**Outputs:**
- `visualisations_output/sweep_memory_M.png`
- `visualisations_output/initialization_comparison.png`
- `visualisations_output/initial_inertia_utility.png`

**Consolidated from:**
- sweep_sigmoid_social.py
- test_initial_uptake.py
- test_initial_inertia.py

---

### test_threshold_variants.py
Consolidated testing for threshold model variants and modifications.

**Tests included:**
- Initial inertia strategy (threshold model with inertia boosts)
- Slowdown strategies (dissonance scaling + threshold floor)

**Usage:**
```bash
cd model_src/testing
python test_threshold_variants.py
```

Select test when prompted: 1, 2, or 'all'

**Outputs:**
- `visualisations_output/threshold_initial_inertia.png`
- `visualisations_output/threshold_slowdown_strategies.png`

**Consolidated from:**
- test_initial_inertia_threshold.py
- test_threshold_slowdown.py

---

### test_threshold_model.py
Current w_i parameter sweep for threshold model (LEAVE AS IS).

Tests steepness parameter k (w_i) values 5-14 for threshold model dynamics.

---

### diagnose_initial_state.py
Diagnostic script for analyzing initial network state and conditions.

Analyzes neighborhood structure, memory composition, utility distributions, and switching probabilities at t=0.

**Usage:**
```bash
cd model_src/testing
python diagnose_initial_state.py
```

**Output:**
- `visualisations_output/initial_state_diagnostics.png`
- Terminal output with detailed metrics

---

## Consolidation Summary

**Removed redundant scripts:**
- sweep_sigmoid_social.py -> test_initial_dynamics.py
- test_initial_uptake.py -> test_initial_dynamics.py
- test_initial_inertia.py -> test_initial_dynamics.py
- test_initial_inertia_threshold.py -> test_threshold_variants.py
- test_threshold_slowdown.py -> test_threshold_variants.py

**Retained scripts:**
- test_threshold_model.py (current w_i sweep)
- diagnose_initial_state.py (diagnostic utility)
