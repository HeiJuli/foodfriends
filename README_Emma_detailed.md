# Foodfriends — Detailed Project Guide

---

## What is this project?

**Foodfriends** is an **agent-based model (ABM)** that simulates how vegetarian/vegan dietary behavior spreads through a social network. The key idea is: people influence each other's food choices. When someone switches to a plant-based diet, their friends are more likely to follow. The model tracks not just how many people change, but also how much CO₂ is saved as a result of that spread — and importantly, *who* gets "credit" for triggering a chain of changes.

The model is grounded in **real survey data** from the Netherlands (5,602 participants) about people's dietary preferences, self-identity, and behavioral intentions. It was originally started as a group project at the CSH Winterschool.

---

## Folder & File Overview

```
Foodfriends-fresh/
│
├── README.md                         ← Short project overview
├── README_Emma_detailed.md           ← This detailed guide
├── data_sampling.md                  ← Technical documentation on sampling & PMF imputation
├── environment_foodfriends.yml       ← Conda environment (list of Python packages)
├── requirements_foodfriends.txt      ← Alternative pip package list
│
├── data/                             ← Input data (survey files) and data analysis
├── model_src/                        ← Core simulation code
├── auxillary/                        ← Data prep & utility scripts
├── analysis/                         ← Post-run analysis scripts
├── plotting/                         ← Figure generation scripts
└── viz/                              ← A LaTeX/PDF diagram schematic
```

---

## 1. `data/` — The Inputs

This folder contains all the raw and processed data the model needs, along with supporting analysis and the raw data construction files.

### Survey & model input files

| File | What it is |
|------|-----------|
| `theta_diet_demographics.xlsx` | Main survey: 5,602 people with their current diet and food preference score (theta) |
| `rho_demographics.xlsx` | Survey subset (~1,500 people) with "behavioral intention" (rho) — how ready they are to change |
| `alpha_demographics.xlsx` | Survey subset (~1,600 people) with "self-identity" (alpha) — how much they see themselves as a "meat-eater" or "veggie" |
| `hierarchical_agents.csv` | **Processed output** — all three surveys merged into one table (created by `auxillary/create_hierarchical_agents.py`) |
| `demographic_pmfs.pkl` | **Processed output** — statistical tables for filling in missing data (created by `auxillary/create_pmf_tables.py`) |
| `stratified_sample_n2000.csv` | A pre-drawn sample of 2,000 agents used for model runs |
| `final_data_parameters.csv` | Final processed parameter table |
| `alpha_empirical_pmfs_by_group.csv` | PMF tables exported as CSV for inspection |
| `rho_empirical_pmfs_by_group.csv` | Same, for rho |
| `DataForFoodProject.xlsx` | Raw project data file |

**Why is data preparation necessary?**
Only 23.2% of the 5,602 survey participants have *all three* parameters (theta + rho + alpha) measured. For the rest, missing values are filled in ("imputed") using demographic-matched statistical tables — this is what the PMF files are for. See `data_sampling.md` for full technical details.

### `data/data_analysis/` — Parameter distribution analysis

Scripts and outputs examining the empirical parameter distributions used to characterize and report the agent population.

| File | What it is |
|------|-----------|
| `parameter_distributions_paper.py` | Generates the parameter distribution figure for the paper |
| `parameter_distributions_ABM.py` | Generates parameter distributions for the ABM population |
| `demographic_parameter_distributions.py` | Analyses parameter distributions broken down by demographics |
| `alpha_differences.py` | Analyses differences in alpha across subgroups |
| `parameter_distributions_paper.pdf/.png` | Output figures |
| `parameter_distributions_ABM.pdf/.png` | Output figures |
| `demographic_parameter_distributions.png` | Output figure |
| `alpha_differences.png` | Output figure |
| `demographic_parameter_findings.md` | Written summary of demographic parameter findings |
| `demographic_parameter_summary.csv` | Summary statistics table |
| `alpha_differences_summary.csv` | Summary statistics for alpha differences |
| `alpha_differences.md` | Written summary of alpha differences |

### `data/data_construction_paper/` — Raw data and Stata pipeline

Contains the original Stata `.dta` survey files and the do-file (`paper_dofile.do`) used to construct the clean survey datasets from raw LISS panel waves. These are the upstream inputs to the Python data pipeline.

---

## 2. `model_src/` — The Heart of the Project

This is where the actual simulation lives.

### `model_main.py` — THE CORE FILE

This is the most important file in the project. It defines two Python classes:

#### `Agent` — a single person in the simulation

Each agent has:
- **`diet`** — current diet: `'veg'` (vegetarian/vegan) or `'meat'` (meat-eater)
- **`theta`** — their *intrinsic preference* for vegetarian food (from survey data). Range: [-1, 1], where positive = prefers veg.
- **`rho`** — their *behavioral intention*: how ready they are to change. Range: [0, 1].
- **`alpha`** — their *self-identity weight*: how strongly their sense of self is tied to their diet. Range: [0.05, 0.80].
- **`beta`** — *social influence weight*: how much they are swayed by friends. Default: 13.
- **`memory`** — a short list (length M=9) of recent social contacts: who they talked to and what those people eat.

At each simulation step, an agent:
1. Picks a random neighbor in the network
2. Observes that neighbor's diet
3. Updates their memory
4. Calculates a probability of switching diet using the **Hamiltonian formula** (see below)
5. Switches diet (or not) based on that probability

#### `Model` — the whole simulation

The model holds all agents, the social network (a graph), and the simulation parameters. Key things it does:

- **`_generate_network()`** — builds the social network. Default type: `homophilic_emp`, meaning people are more likely to be connected to demographically similar others (matching real-world homophily).
- **`agent_ini()`** — loads agents from the survey data.
- **`run()`** — runs 150,000 simulation steps (each step = one random agent has one interaction).
- **`rewire()`** — at each step, the network can also change: edges are added/removed based on triadic closure (friends of friends become friends) or homophily.
- **`cascade_statistics()`** — after the simulation, figures out *who influenced whom* and calculates CO₂ reduction attribution.

#### The Decision Formula (Hamiltonian)

At each step, an agent compares the "energy cost" of being a meat-eater vs. being vegetarian:

```
H(s) = (1 - w) × (s - h_ind)² + w × (s - h_soc)² - tau × s
```

Where:
- `s` = the diet option being evaluated (0 = meat, 1 = veg)
- `h_ind` = personal preference (normally = rho, but shifts toward theta if exposed to opposite-diet contacts)
- `h_soc` = social signal (fraction of veg contacts, adjusted for diminishing returns from the same person)
- `w` = social weight (beta/alpha blend)
- `tau` = external pressure pushing toward veg (e.g., media, policy)

The agent switches to whichever diet has the lower energy, with some randomness (controlled by beta — higher beta = more deterministic).

---

### `model_runner_mp.py` — Running Many Simulations in Parallel

This script is how you actually run the model in bulk. It uses multiple CPU cores to run many simulations simultaneously.

**How to use it from the command line:**
```bash
python model_runner_mp.py --analysis trajectory --agent_ini twin --runs 50
```

**Analysis modes:**
| Mode | What it does |
|------|-------------|
| `trajectory` | Runs 50 simulations with the full N=2000 survey population, saves how veg fraction and CO₂ evolve over time |
| `emissions` | Varies initial veg fraction, measures final CO₂ per person |
| `parameter` | Sweeps across combinations of alpha/beta/theta to find tipping-point conditions |
| `veg_growth` | Runs sigmoid growth analysis: given X% initial veg, what % do you end up with? |
| `param_trajectories` | Parameter sweep with full trajectory output |

**Outputs** are saved as `.pkl` files (Python pickle format — like a saved table) in `model_output/`.

---

### `extended_model_runner.py` — Supplementary Analysis Helpers

Not run directly from the command line. Contains helper functions for:
- Comparing different network topologies (random vs. scale-free vs. homophilic)
- Testing targeted interventions (what if you seed veg behavior in high-degree nodes?)
- Analyzing critical dynamics near tipping points
- Studying how vegetarian clusters form

---

### `model_runn.py` — Variant Runner

An earlier variant of the model runner, retained for reference.

---

### `model_src/testing/` — Development & Validation Tests

These scripts were used during model development to test specific behaviors. They are not required for standard use but are useful for understanding model dynamics and validating specific components.

| Script | What it tests |
|--------|--------------|
| `test_initial_dynamics.py` | How the model behaves right at the start |
| `test_threshold_variants.py` | Different ways agents can have inertia |
| `test_threshold_model.py` | Threshold-based decision rule variant |
| `diagnose_initial_state.py` | Checks what the network and agents look like at t=0 |
| `test_system_size_scaling.py` | What happens if you run with N=500 vs N=5000? |
| `test_seeding_experiment.py` | If you start with 5% vs 15% veg, how does it change outcomes? |
| `beta_sweep.py` | How sensitive is the model to the beta parameter? |
| `gamma_sweep_complex_cent.py` | Tests the diminishing-returns parameter |
| `sweep_wt_m_tc.py` | Grid sweep over social weight, memory length, and triadic closure |
| `test_contagion_transition.py` | Phase transition: at what point does adoption accelerate? |
| `test_phase2_2d_sweep.py` | Two-parameter grid sweep |
| `test_adoption_complex_centrality.py` | How network centrality affects adoption via complex contagion |
| `test_alpha_complex_centrality.py` | Interaction between self-identity (alpha) and centrality |
| `test_memory_complex_centrality.py` | Memory buffer effects in high-centrality nodes |

---

### `model_src/old/` — Archived Model Versions

Earlier versions of the model kept for reference. These are **not used** in the current analysis but document the evolution of the model:
- `model_main_boltzmann.py` — early Boltzmann version
- `model_main_threshold.py` — threshold-based decision variant
- `model_main_no_complex.py` — version without complex contagion
- `model_main_single.py`, `model_main_single_utility.py` — single-agent utility variants

---

## 3. `auxillary/` — Data Preparation & Utility Scripts

These scripts are run **once** before doing any simulations, to prepare the data.

### Setup Pipeline (run in this order)

**Step 1: `create_hierarchical_agents.py`**
- Reads the three raw Excel survey files
- Merges them into a single table (`hierarchical_agents.csv`)
- Marks which agents have empirical rho, alpha, or both
- Result: 5,602 agents, 1,298 with complete data (23.2%)

**Step 2: `create_pmf_tables.py`**
- Takes the merged agent table
- Builds probability tables (PMFs = Probability Mass Functions) for each demographic group
- **Key innovation**: tables for alpha and rho are split by theta bins — so imputed values still match the empirical correlation between dietary preference and behavioral intention
- Saves everything to `demographic_pmfs.pkl`

**Step 3: `validate_theta_stratification.py`**
- Runs 10 simulated agent populations using the PMF imputation
- Checks that the correlations between theta, rho, and alpha are preserved (target: <0.05 difference from empirical)
- Produces a validation plot

**Step 4: `analyze_sample_size.py`**
- Analyzes the trade-off between statistical noise (need large N) and empirical grounding (larger N = more imputed agents, less real survey data)
- Recommendation: **N=2000** — coefficient of variation = 2.2% (publication quality), only 35% imputed

---

### Network Utilities

**`homophily_network_v2.py`** — builds the social network
- People are connected to others who are demographically similar (homophily)
- Similarity is computed across 5 dimensions: gender, age, income, education, theta (veg preference)
- Network grows using the Holme-Kim model: new nodes arrive and preferentially attach to similar, already-connected nodes
- Triadic closure (tc=0.7): 70% chance a new connection forms through a mutual friend
- Returns a NetworkX graph object

**`network_stats.py`** — analyzes network structure
- Computes homophily (how often same-group nodes connect)
- Infers homophily from edge type counts
- Exports network to DataFrames for analysis

**`sampling_utils.py`** — demographic-preserving sampling
- `stratified_sample_agents()`: Draws N agents from the full 5,602 while preserving gender, age, income, and education proportions
- Maximum demographic deviation after sampling: ±0.21% (vs ~1% for random sampling)

**`stratified_sampling.py`** — demonstration script showing how much better stratified sampling is vs random

---

### Validation & Diagnostics

**`parameter_diagnostics.py`** — comprehensive checks on the data pipeline:
- How complete is the agent data?
- Are the PMF tables well-populated?
- Are parameter correlations realistic?
- Is the "effective threshold" (rho - alpha × dissonance) above 0.20 for meat-eaters? (It needs to be for the initial state to be stable)

**`test_homophilly.py`** — validates the homophily network generation

**`visualize_homophilic_emp_network.py`** — creates a visual of the generated network to inspect it

**`auxillary/old/`** — archived development scripts, not used in current pipeline

---

### Documentation inside `auxillary/`

| File | Content |
|------|---------|
| `README.md` | Detailed guide to all auxillary utilities and the setup workflow |
| `CONSOLIDATION_PLAN.md` | Plan for consolidating older test scripts |
| `CONSOLIDATION_SUMMARY.md` | What was consolidated and why |
| `complete_cases_analysis_2025-01-16.md` | Analysis of the 1,298 complete-case agents (age bias, demographics) |

---

## 4. `analysis/` — After the Simulations

These scripts analyze the results after you've run the model.

**`results_analysis.py`** — the main post-processing script. Does 5 things:
1. **Heavy-tail analysis**: Fits a power-law to the distribution of CO₂ reductions per agent — most agents contribute little, a few contribute a huge amount
2. **Gini coefficient**: Measures inequality in CO₂ reduction attribution (typically 0.65–0.75, similar to income inequality)
3. **Network assortativity**: Measures whether vegetarians tend to cluster together (they do)
4. **Degree-amplification scaling**: Highly-connected agents have disproportionately large cascade impacts (superlinear: A ~ k^γ where γ=1–1.5)
5. **Inflection point**: Finds the moment when adoption starts accelerating (the "tipping point")

**`verify_descendants_scaling.py`** — checks that the influence trees scale correctly with network size.

---

## 5. `plotting/` — Making Figures

These scripts generate all the figures for the paper.

**`publication_plots_main.py`** — the main 7-panel publication figure:
- **Top row**: Three snapshots of the social network at t=0, t=mid, t=end (showing how vegetarians cluster over time)
- **Bottom row**:
  1. Trajectory of vegetarian fraction over time (with ensemble runs)
  2. CCDF (log-log plot of CO₂ reduction distribution — shows the power-law tail)
  3. Lorenz curve (inequality: top 10% reducers account for ~65–75% of total reductions)

**`publication_plots_supp.py`** — supplementary figures

**`plot_styles.py`** — sets consistent visual style for all figures:
- Color scheme: teal (#2a9d8f) for vegetarians, coral (#e76f51) for meat-eaters
- Fonts, spine formatting, grid style
- PDF export settings for journal submission

**`plot_config.yaml`** — configuration file specifying input file paths for the plotting scripts

**Other plotting scripts:**
| Script | Purpose |
|--------|---------|
| `utility_probability_plot.py` | Visualizes the switching probability surface (how likely is a switch given different parameter values?) |
| `agency_predictor_analysis.py` | Which agent properties predict large cascade influence? |
| `compare_social_functions.py` | Compares different ways of computing social influence |
| `demo_patch_network.py` | Demonstrates the PATCH network topology |
| `explore_derivatives.py` | Analyzes inflection behavior in adoption curves |
| `test_network_plot.py`, `test_layout_algorithms.py` | Prototyping network visualizations |

---

## 6. `viz/` — Schematic Diagram

Contains a LaTeX file (`cascade_attribution_schematic.tex`) and its compiled PDF output. This is the conceptual diagram showing how cascade attribution works — i.e., how CO₂ reduction credit propagates up an influence chain.

---

## Key Concepts Explained

### What is "complex contagion"?
Simple contagion (like a cold) spreads with a single exposure. Complex contagion requires *multiple reinforcing exposures* before a person changes. The `gamma` parameter (default 0.3) controls diminishing returns — seeing the same friend eat veg 10 times doesn't count 10 times; each additional exposure has less impact. This produces the characteristic S-curve adoption pattern.

### What is "cascade attribution"?
When agent A switches to veg because of agent B, and later agent C switches because of A, then B gets partial credit for C's switch too. The credit decays with depth (controlled by `decay=0.7`) and with how briefly/long the influencer stayed vegetarian (dwell-time weighting). The result: a power-law distribution where a small number of "super-spreaders" account for a huge share of total reductions.

### What are theta, rho, alpha?
- **theta** (θ): *How much do you like vegetarian food?* Scale: -1 (strongly prefers meat) to +1 (strongly prefers veg). Comes from a survey about food attitudes.
- **rho** (ρ): *How ready are you to change?* Behavioral intention. 0 = no intention, 1 = strong intention. Acts as a "default preference" before social influence.
- **alpha** (α): *How much is your diet part of your identity?* Self-identity weight. High alpha = diet is central to who you are = harder to change but if you do change, you influence others more strongly.

### What is "homophily"?
The tendency to connect with similar others. In this model, people are more likely to be friends with others of the same age, gender, income, education, and dietary preference. This creates echo chambers: vegetarians tend to cluster together, which can both accelerate change within the cluster but limit cross-group spread.

### What is the "effective threshold"?
`effective_threshold = rho - alpha × dissonance`

For a meat-eater to be stable (not immediately flip to veg at t=0), this value needs to be above 0.20. The data preparation pipeline checks this.

---

## Complete Parameter Reference

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `N` | 2000 | Number of agents in the simulation |
| `steps` | 150,000 | How many interaction steps to simulate |
| `beta` | 13 | Decision sharpness (higher = more deterministic switching) |
| `alpha` | 0.35 | Self-identity weight (compressed to [0.05, 0.80]) |
| `rho` | 0.45 | Behavioral intention threshold |
| `theta` | agent-specific | Intrinsic vegetarian preference |
| `gamma` | 0.3 | Diminishing returns on repeated contacts [0=max diminishing, 1=no diminishing] |
| `tau` | 0.035 | External pro-veg field (media, policy pressure) |
| `mu` | 0.2 | Status-quo bias (inertia against switching) |
| `M` | 9 | Memory buffer length (how many past contacts to remember) |
| `tc` | 0.7 | Triadic closure probability in network rewiring |
| `decay` | 0.7 | Depth decay for cascade credit attribution |
| `veg_f` | 0.06 | Initial fraction of vegetarians (6%, matching Dutch surveys) |
| `topology` | `homophilic_emp` | Network type |
| `agent_ini` | `sample-max` | How agents are initialized from data |

**Network topology options:**
- `homophilic_emp` — empirically-grounded homophilic network (default, most realistic)
- `BA` — Barabási-Albert scale-free network
- `complete` — everyone connected to everyone
- `WS` — Watts-Strogatz small-world network
- `CSF` — configuration scale-free
- `PATCH` — modular/patch network
- `prebuilt` — load from file

**Agent initialization modes:**
- `twin` — pair each model agent with a real survey respondent (best empirical grounding)
- `sample-max` — use 385 complete-case agents (all empirical, no imputation)
- `synthetic` — generate agents from PMF tables only
- `parameterized` — use provided fixed parameter values

---

## Typical Workflow

### First-time setup (run once)

```bash
# 1. Create the conda environment
conda env create -f environment_foodfriends.yml
conda activate foodfriends   # (name may vary, check the yml file)

# 2. Prepare the data
cd auxillary
python create_hierarchical_agents.py    # creates data/hierarchical_agents.csv
python create_pmf_tables.py             # creates data/demographic_pmfs.pkl
python validate_theta_stratification.py # validates the imputation approach
python analyze_sample_size.py           # confirms N=2000 recommendation
```

### Running the model

```bash
cd model_src
# Standard run: 50 simulations with N=2000, using twin initialization
python model_runner_mp.py --analysis trajectory --agent_ini twin --runs 50

# Outputs saved to model_output/trajectory_analysis_twin_YYYYMMDD.pkl
```

### Analyzing results

```bash
cd analysis
python results_analysis.py
```

### Making figures

```bash
cd plotting
python publication_plots_main.py
```

---

## Key Findings (What the Model Shows)

1. **Tipping point exists**: Once ~10% of the population is vegetarian, adoption accelerates sharply.
2. **Saturation**: The model naturally levels off around 30–40% vegetarian fraction.
3. **Inequality in influence**: The top 10% of agents account for 60–75% of total CO₂ reductions. The distribution follows a power law.
4. **Network structure matters**: Homophilic networks produce slower initial spread but more stable, cohesive vegetarian clusters.
5. **Self-identity amplifies cascades**: High-alpha agents are harder to convert but are more influential once converted — they act as stable "anchors" in vegetarian clusters.
6. **N=2000 is the sweet spot**: Large enough for statistical reliability (CV=2.2%), small enough that 65% of agents are empirically grounded rather than imputed.

---

## File Status Summary

| Location | Status | Notes |
|----------|--------|-------|
| `model_src/model_main.py` | Current | Main model |
| `model_src/model_runner_mp.py` | Current | Main runner |
| `model_src/extended_model_runner.py` | Current | Supplementary analyses |
| `model_src/model_runn.py` | Variant | Earlier runner variant |
| `model_src/model_main_backup_*.py` | Archive | Snapshots before major changes |
| `model_src/model_main_cumulative_test.py` | Experimental | Cumulative attribution test |
| `model_src/old/` | Archive | Earlier model variants |
| `model_src/testing/` | Dev tests | Useful for understanding model behavior |
| `auxillary/homophily_network_v2_original.py` | Archive | Original before v2 refactor |
| `auxillary/old/` | Archive | Archived development scripts |
| `plotting/publication_plots_backup.py` | Archive | Backup before last plotting refactor |
| `plotting/old/` | Archive | Earlier figure scripts |
| `data/data_analysis/` | Current | Parameter distribution analysis scripts and outputs |
| `data/data_construction_paper/` | Archive | Raw Stata files and do-file for upstream data construction |

---

*Last updated: April 2026.*
