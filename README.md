# Foodfriends

An **agent-based model** of how vegetarian dietary behavior spreads through a
social network, grounded in survey data from 5,602 Dutch participants. The
model tracks adoption dynamics, CO2 reduction, and how influence cascades
attribute "credit" for triggering chains of dietary change.

Originally started at the CSH Winterschool. Manuscript currently in submission

---

## Repository layout

```
.
├── data/                 Survey inputs and processed parameter tables
├── model_src/            Core simulation engine and runners
│   └── testing/          Validation tests
├── auxillary/            Data preparation, sampling, network utilities
├── analysis/             Post-run statistical analysis
├── plotting/             Figure generation for the paper
├── viz/                  LaTeX schematic for the cascade attribution diagram
├── model_output/         Simulation results (.pkl)
├── visualisations_output/  Generated figures
└── old/                  Archived legacy code (gitignored, kept for posterity)
```

---

## The model (`model_src/`)

### `model_main.py` — core engine

Defines two classes:

- **`Agent`** — one person in the simulation. Holds:
  - `diet`: `'veg'` or `'omnivore'`
  - `theta` ∈ [-1, 1]: intrinsic vegetarian preference (survey-derived)
  - `rho` ∈ [0, 1]: behavioral intention
  - `alpha` ∈ [0.05, 0.80]: self-identity weight (compressed for social
    desirability correction)
  - `beta`: social-influence weight (= 1 − alpha)
  - `memory`: rolling buffer (length M=9) of recent contacts' diets

  Each step the agent samples a neighbor, updates memory, and stochastically
  switches diet by Boltzmann probability over a Hamiltonian energy:

  ```
  H(s) = (1 − w)·(s − h_ind)² + w·(s − h_soc)² − tau·s
  ```

  with `h_ind` shifting toward theta on dissonance, `h_soc` from neighbor diet
  shares (with diminishing-returns gamma), `tau` an external pro-veg field.

- **`Model`** — population, network, and simulation loop. Builds the network
  (default `homophilic_emp`, empirically calibrated homophily), loads agents
  from survey data, runs the dynamics, detects steady state, and computes
  cascade attribution (cumulative-activation accounting with depth decay and
  dwell-time weighting).

### Runners

- **`model_runner_mp.py`** — main multiprocessing CLI runner for parameter
  sweeps and batch ensembles. Output as `.pkl` in `model_output/`.
  ```bash
  python model_runner_mp.py --analysis trajectory --agent_ini twin --runs 50
  ```
- **`model_runn.py`** — lighter single-process runner used by some testing
  scripts. Stores `steady_state_t` alongside snapshots.
- **`extended_model_runner.py`** — helper functions for emissions analysis,
  vegetarian-fraction studies, and topology comparisons.

### `testing/`

- `test_system_size_scaling.py` — verifies scale invariance of key
  observables across N=2k–20k with modular networks and KDE-synthetic agents.

---

## Data preparation (`auxillary/`)

Run once before simulating:

1. **`create_hierarchical_agents.py`** — merges the three raw survey files
   (`theta_diet_demographics.xlsx`, `rho_demographics.xlsx`,
   `alpha_demographics.xlsx`) into `data/hierarchical_agents.csv` (5,602
   participants, 1,298 with all three parameters measured).
2. **`create_pmf_tables.py`** — builds conditional PMFs for hot-deck
   imputation of missing alpha/rho. Alpha conditioned on demographics only;
   rho stratified by demographics × theta-bin (preserves the empirical
   theta–rho correlation r=−0.30). Output: `data/demographic_pmfs.pkl`.
3. **`validate_theta_stratification.py`** — confirms that PMF imputation
   preserves the empirical correlations.
4. **`analyze_sample_size.py`** — finite-size vs. imputation trade-off; lands
   on **N=2000** (CV=2.2%, ±0.21% demographic deviation).

Other utilities:

- **`sampling_utils.py`** — stratified sampler used automatically when N <
  5602 to preserve gender/age/income/education distributions.
- **`homophily_network_v2.py`** — homophilic network generator (Blend of preferential attachment and demographic similarity across
  age/gender/income/education/theta, with triadic closure).
- **`network_stats.py`** — homophily and topology measures.
- **`parameter_diagnostics.py`**, **`test_homophilly.py`**,
  **`visualize_homophilic_emp_network.py`** — diagnostics.

See `auxillary/README.md` for more.

---

## Analysis (`analysis/`)

- **`results_analysis.py`** — power-law fits, Gini, network assortativity,
  degree–amplification scaling, inflection detection. Snapshot priority:
  explicit `t_cutoff` (CLI arg) > steady-state snapshot > logistic 95%
  asymptote > final snapshot.
- **`verify_descendants_scaling.py`** — sanity-checks influence-tree size
  against system size.

---

## Plotting (`plotting/`)

- **`publication_plots_main.py`** — main 7-panel paper figure: network
  snapshots over time, trajectory ensemble, CCDF of CO2 reductions, Lorenz
  curve. All non-trajectory panels share a single ensemble-fixed analysis
  cutoff (`analysis_t_end` from `plot_config.yaml`) for temporal consistency.
- **`publication_plots_supp.py`** — supplementary figures.
- **`agency_predictor_analysis.py`** — regression of cascade amplification on
  structural and psychological predictors.
- **`trajectory_t_end_facet.py`** — visual QA: per-run trajectory + logistic
  fit + t_end marker.
- **`plot_styles.py`**, **`plot_config.yaml`** — shared style and inputs.
- **`utility_probability_plot.py`**, **`compare_social_functions.py`**,
  **`demo_patch_network.py`**, **`explore_derivatives.py`** — exploratory.

---

## Setup

```bash
conda env create -f environment_foodfriends.yml
conda activate foodfriends
```

Or `pip install -r requirements_foodfriends.txt`.

## Typical workflow

```bash
# Data prep (one-time)
cd auxillary
python create_hierarchical_agents.py
python create_pmf_tables.py

# Simulation
cd ../model_src
python model_runner_mp.py --analysis trajectory --agent_ini twin --runs 50

# Analysis + figures
cd ../analysis && python results_analysis.py
cd ../plotting && python publication_plots_main.py
```

## Key parameters (defaults)

| Param            | Default          | Meaning                                          |
|------------------|------------------|--------------------------------------------------|
| `N`              | 2000             | Population size                                  |
| `steps`          | 150,000          | Interaction steps                                |
| `beta`           | 1 − alpha        | Social weight (per-agent)                        |
| `alpha`          | survey/imputed   | Self-identity weight, compressed [0.05, 0.80]    |
| `rho`            | survey/imputed   | Behavioral intention                             |
| `theta`          | survey           | Intrinsic veg preference                         |
| `gamma`          | 0.45             | Diminishing returns on repeat contacts           |                           |
| `M`              | 9                | Memory buffer length                             |
| `decay`          | 0.7              | Cascade-credit depth decay                       |
| `tau_persistence`| M·2·N            | Dwell-time weighting timescale                   |
| `immune_n`       | 0.10–0.15        | Fraction of immune (extreme-conviction) agents   |
| `topology`       | `homophilic_emp` | Network type                                     |
| `agent_ini`      | `sample-max`     | Agent initialization mode                        |

Topology options: `homophilic_emp`, `BA`, `complete`, `WS`, `CSF`, `PATCH`,
`prebuilt`. Initialization modes: `twin`, `sample-max`, `synthetic`,
`parameterized`.

For deeper notes on parameter choices, see `CLAUDE.md` and `claude_stuff/`.

---

## License

Source code: BSD 3-clause (see `LICENSE.md`). Manuscript text is not open
source — rights reserved by the authors.
