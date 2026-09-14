# Foodfriends

Agent-based model of vegetarian diet spreading through a social network,
parameterised from a survey of 5,602 Dutch participants (LISS panel). It reports
two outcomes: adoption of the vegetarian diet, and the amplification credited to
each converter through the chain of conversions that follows them.

Started at the CSH Winterschool.

## Layout

```
data/          Parameter tables (LISS inputs not redistributed, see LICENSE-DATA)
model_src/     Simulation engine, runners, sensitivity campaign; testing/ = scaling tests
auxillary/     Survey preparation, sampling, network construction
analysis/      Ledgers, nulls, window fits, statistics, source-data export
plotting/      Paper figures
viz/           LaTeX schematic of the attribution diagram
model_output/  Results (.pkl)          visualisations_output/  Figures
source_data/   Per-panel CSVs behind the main figures
```

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements_foodfriends.txt --no-deps
```

`--no-deps` because netin's pinned bounds are stale. With conda, create the
environment first, then install from the same requirements file with pip:
conda's solver rejects `pandas==3.0.5`, which older pandas cannot substitute
because it cannot read the result pickles.

```bash
conda env create -f environment_foodfriends.yml
conda activate foodfriends
pip install -r requirements_foodfriends.txt --no-deps
```

## Workflow

```bash
cd auxillary && python create_hierarchical_agents.py   # one-time: merge survey files
cd auxillary && python create_pmf_tables.py            # one-time: imputation PMFs
cd model_src && python model_runn.py                   # ensemble run
cd analysis && python results_analysis.py              # statistics
cd plotting && python publication_plots_main.py        # figures
```

Without LISS access, run `auxillary/create_synthetic_agents.py` in place of the
two preparation steps and point `survey_file` at `data/synthetic_agents.csv`. It
samples from aggregate cell counts, so it reproduces the qualitative results but
not the paper's numbers.

## Model (`model_src/model_main.py`)

Each step an agent samples a neighbour, appends that neighbour's diet to a memory
buffer of length `M`, and switches diet with Boltzmann probability over

```
H(s) = (1 - w)(s - h_ind)^2 + w(s - h_soc)^2
```

`h_ind` is the individual field, built from intrinsic preference `theta` and
intention `rho` discounted by `kappa`, and gated so that `theta` activates only
above a threshold share of opposing contacts. `h_soc` is the diet share in the
memory buffer, with exponent `gamma` damping repeated contacts from one source.
`w = 1 - alpha` is the agent's social weight.

`Model` builds the network, loads agents, runs the dynamics, and logs every
conversion event with its sampled parent, which `analysis/attribution_ledger.py`
replays into any credit convention.

### Runners

- `model_runn.py` — production. `DEFAULT_PARAMS` here is the source of truth.
  Saves snapshots, `steady_state_t` and the event log.
- `model_runner_mp.py` — parallel sweeps. Drops snapshots in `sample-max` mode
  and silently overwrites a same-day `.pkl`; not for anything needing snapshots.
- `sensitivity_campaign.py` — one-at-a-time sweep over the parameter table.
- `extended_model_runner.py` — emissions and vegetarian-fraction studies.

## Parameters (`model_runn.DEFAULT_PARAMS`)

| Param | Default | Meaning |
|---|---|---|
| `N` | 650 (2000 in the paper) | Population size |
| `steps` | 400,000 | Interaction steps; set from the fitted t_end |
| `beta` | 13 | Inverse temperature |
| `alpha` | survey/imputed | Self-reliance, compressed to [0.05, 0.80] |
| `rho` | survey/imputed | Behavioural intention |
| `theta` | survey | Intrinsic preference, -1 meat to +1 veg |
| `kappa` | 0.55 | Intention-behaviour discount on `rho` |
| `theta_gate_c`, `theta_gate_k` | 0.35, 35 | Gate threshold and steepness |
| `gamma` | 0.3 | Diminishing returns on repeated contacts |
| `M` | 9 | Memory buffer length |
| `decay` | 0.7 | Depth decay on cascade credit |
| `immune_n` | 0.10 | Fraction of agents that never switch |
| `topology` | `homophilic_emp` | Also `BA`, `complete`, `WS`, `PATCH`, `prebuilt` |
| `agent_ini` | `sample-max` | Also `twin`, `synthetic`, `parameterized` |

Agents are drawn from `data/hierarchical_agents.csv`; `theta` and diet are
measured for all 5,602 respondents, `alpha` and `rho` for 1,298, the rest imputed
from conditional PMFs that preserve the empirical theta-rho correlation
(r = +0.34). Below N = 5602 the sample is stratified on gender, age, income and
education. See `auxillary/README.md`.

## Licensing and citation

Code is MIT (`LICENSE`). Data we derived are CC BY 4.0; the underlying LISS
microdata are not ours to license and are not redistributed here, and must be
obtained from the LISS Data Archive (`LICENSE-DATA`). Manuscript text is rights
reserved. Machine-readable metadata is in `CITATION.cff`: cite the archived
Zenodo release for the code and the paper for the model.
