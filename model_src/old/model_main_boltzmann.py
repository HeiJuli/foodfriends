# -*- coding: utf-8 -*-
"""
Minimal Boltzmann dissonance model for dietary behavior change.

Based on Galesic et al. (2021) J. R. Soc. Interface, 18, 20200857.
Stripped to core: individual dissonance + social dissonance + external field.

Hamiltonian per agent i for diet state s (s_num: meat=0, veg=1):
  H_i(s) = (1 - w_i) * (s_num - h_ind)^2        [individual dissonance]
         + w_i * (s_num - h_soc)^2                [social dissonance]
         - tau * s_num                             [external field, Galesic & Stein 2019]
         + mu * (1-f_opp) * (s_num - s_cur)^2     [status-quo bias, erodes w/ exposure]

  h_ind = (1-gate) * rho + gate * theta_01        [belief vector: rho -> theta]
  theta_01 = (theta + 1) / 2                      [rescale theta from [-1,1] to [0,1]]
  gate(p_opp) = sigmoid(k * (p_opp - c))          [social exposure gate]
  w_i = 1 - alpha_i    [per-agent social weight, from survey]
  h_soc = frac veg in memory buffer
  beta_inv = inverse temperature (global, single tuning parameter)

Alpha compression: raw Likert [0,1] -> [alpha_min, alpha_max] to eliminate
degenerate extremes (pure individualists / pure conformists).

P(switch) = exp(-beta_inv * H_switch) / Z

@author: everall
"""

import networkx as nx
import numpy as np
import random
import scipy.stats as st
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from netin import PATCH, PAH
from netin import viz
import sys
import os
sys.path.append('..')
from auxillary.homophily_network_v2 import generate_homophily_network_v2
from auxillary.sampling_utils import stratified_sample_agents
from auxillary import network_stats

# %% Preliminary settings
# CO2 measures are in kg/year, source: https://pubmed.ncbi.nlm.nih.gov/25834298/
params = {"veg_CO2": 1390,
          "vegan_CO2": 1054,
          "meat_CO2": 2054,
          "N": 650,
          "erdos_p": 3,
          "steps": 110000,
          "k": 8,
          "immune_n": 0,
          "M": 9,
          "veg_f": 0,
          "meat_f": 0.95,
          "p_rewire": 0.01,
          "rewire_h": 0.1,
          "tc": 0.7,
          'topology': "homophilic_emp",
          # --- Boltzmann parameters (Galesic et al. 2021) ---
          # beta: inverse temperature / attention (single global tuning knob)
          #   low (~5)  -> noisy/stochastic     mid (~15-25) -> gradual
          #   high (~50+) -> sharp/deterministic
          #   Galesic et al.: 23 for GM food, 10-50 range across scenarios
          "beta": 27,
          # --- Agent initialization ---
          "alpha": 0.36,
          "rho": 0.45,
          "theta": 0.44,
          "agent_ini": "sample-max",
          "survey_file": "../data/hierarchical_agents.csv",
          "adjust_veg_fraction": False,
          "target_veg_fraction": 0.06,
          "tau": 0.035,
          # --- Theta gating (social exposure threshold) ---
          # theta only activates once p_opp in memory exceeds c
          #   c=0.3: ~3/9 opposite-diet memories needed to "wake up" theta
          #   k=10:  sigmoid steepness (higher = sharper gate)
          "theta_gate_c": 0.35,
          "theta_gate_k": 35,
          # --- Alpha compression (Likert -> bounded weight) ---
          "alpha_min": 0.15,
          "alpha_max": 0.85,
          # --- Status-quo bias (Samuelson & Zeckhauser 1988) ---
          # Behavioral inertia that erodes with exposure to opposite-diet neighbors
          "mu": 0.2,
          }


# %% Auxillary/Helpers

def hamiltonian(s_num, theta, w, memory, M, rho, current_diet,
                theta_gate_c=0.3, theta_gate_k=10, tau=0.0, mu=0.1):
    """Hamiltonian for diet state s (Galesic et al. 2021 + gated belief vector).

    H(s) = (1-w)(s - h_ind)^2 + w(s - h_soc)^2 - tau*s + mu*(1-f_opp)*(s - s_cur)^2

    h_ind = (1-gate)*rho + gate*theta_01
    mu: status-quo bias (Samuelson & Zeckhauser 1988). Behavioral inertia that
    erodes with exposure to opposite-diet neighbors (f_opp).

    s_num: 0=meat, 1=veg.  h_soc: fraction veg in memory.
    """
    mem = memory[-M:]
    h_soc = sum(1 for d in mem if d == "veg") / len(mem) if mem else 0.0
    p_opp = sum(1 for d in mem if d != current_diet) / len(mem) if mem else 0.0
    gate = 1 / (1 + math.exp(-theta_gate_k * (p_opp - theta_gate_c)))
    s = s_num
    s_cur = 1.0 if current_diet == "veg" else 0.0

    theta_01 = (theta + 1.0) / 2.0  # [-1,1] -> [0,1]
    h_ind = (1 - gate) * rho + gate * theta_01
    inertia = mu * (1 - p_opp) * (s - s_cur)**2
    return (1 - w) * (s - h_ind)**2 + w * (s - h_soc)**2 - tau * s + inertia



def boltzmann_prob(H_switch, H_stay, beta):
    """P(switch) = exp(-beta * H_switch) / Z.  Log-sum-exp for stability."""
    x, y = -beta * H_switch, -beta * H_stay
    m = max(x, y)
    return math.exp(x - m) / (math.exp(x - m) + math.exp(y - m))


def sample_from_pmf(demo_key, pmf_tables, param, theta=None):
    """Sample parameter from theta-stratified PMF."""
    if not pmf_tables:
        return 0.5

    metadata = pmf_tables.get('_metadata', {})
    stratified_params = metadata.get('stratified_params', ['alpha', 'rho'])

    if param in stratified_params and theta is not None:
        theta_bins = metadata.get('theta_bins', [-1.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        theta_labels = metadata.get('theta_labels', [
            '(-1.0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1.0]'])
        theta_bin = None
        for i, (low, high) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):
            if low <= theta < high:
                theta_bin = theta_labels[i]
                break
        if theta_bin is None and theta >= theta_bins[-2]:
            theta_bin = theta_labels[-1]
        lookup_key = demo_key + (theta_bin,) if theta_bin else demo_key
    else:
        lookup_key = demo_key

    if lookup_key in pmf_tables[param]:
        pmf = pmf_tables[param][lookup_key]
        vals, probs = pmf['values'], pmf['probabilities']
        nz = [(v, p) for v, p in zip(vals, probs) if p > 0]
        if nz:
            v, p = zip(*nz)
            return np.random.choice(v, p=np.array(p)/sum(p))

    if param in stratified_params and len(lookup_key) > 4:
        if demo_key in pmf_tables[param]:
            pmf = pmf_tables[param][demo_key]
            vals, probs = pmf['values'], pmf['probabilities']
            nz = [(v, p) for v, p in zip(vals, probs) if p > 0]
            if nz:
                v, p = zip(*nz)
                return np.random.choice(v, p=np.array(p)/sum(p))

    all_vals = [v for cell in pmf_tables[param].values() for v in cell['values']]
    return np.random.choice(all_vals) if all_vals else 0.5


def load_sample_max_agents(filepath="../data/hierarchical_agents.csv"):
    """Load 385 demographically representative complete-case agents."""
    df = pd.read_csv(filepath)
    complete = df[df['has_alpha'] & df['has_rho']].copy()
    complete = complete.sort_values('nomem_encr').reset_index(drop=True)

    age_targets = {
        '18-29': 56, '30-39': 54, '40-49': 56,
        '50-59': 68, '60-69': 80, '70+': 71
    }

    sampled = []
    for age_group, n_target in age_targets.items():
        group = complete[complete['age_group'] == age_group].reset_index(drop=True)
        if len(group) < n_target:
            print(f"WARNING: Only {len(group)} agents in {age_group}, need {n_target}")
            sampled.append(group)
        else:
            sampled.append(group.sample(n=n_target, replace=False, random_state=42))

    result = pd.concat(sampled, ignore_index=True)
    print(f"Sample-max mode: {len(result)} agents with perfect age stratification")
    return result


# %% Agent

class Agent():

    def __init__(self, i, params, **kwargs):
        self.i = i
        self.params = params
        self.set_params(**kwargs)
        self.C = self.diet_emissions(self.diet)
        self.memory = []
        self.survey_id = kwargs.get('survey_id', i)
        self.reduction_out = 0
        self.diet_duration = 0
        self.diet_history = []
        self.last_change_time = 0
        self.immune = False

        # Cascade attribution tracking (Guilbeault & Centola 2021)
        self.influence_parent = None
        self.influenced_agents = []
        self.change_time = None

    def set_params(self, **kwargs):
        if self.params["agent_ini"] not in ["twin", "sample-max"]:
            self.diet = self.choose_diet()
            self.rho = self.choose_alpha_beta(self.params["rho"])
            self.theta = st.truncnorm.rvs(-1, 1, loc=self.params["theta"])
            self.alpha = self.choose_alpha_beta(self.params["alpha"])
            self.beta = 1 - (self.alpha)
            self.w_i = self.beta
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)

            if hasattr(self, 'alpha') and not hasattr(self, 'beta'):
                self.beta = 1 - self.alpha

            is_complete_case = (hasattr(self, 'has_alpha') and self.has_alpha and
                               hasattr(self, 'has_rho') and self.has_rho)

            if is_complete_case:
                self.beta = 1 - self.alpha
            elif hasattr(self, 'demographics') and hasattr(self, 'pmf_tables') and self.pmf_tables:
                if not (hasattr(self, 'has_alpha') and self.has_alpha and hasattr(self, 'alpha')):
                    self.alpha = sample_from_pmf(
                        self.demographics, self.pmf_tables, 'alpha', theta=self.theta)
                if not (hasattr(self, 'has_rho') and self.has_rho and hasattr(self, 'rho')):
                    self.rho = sample_from_pmf(
                        self.demographics, self.pmf_tables, 'rho', theta=self.theta)
                self.beta = 1 - self.alpha
            else:
                self.rho = st.truncnorm.rvs(-1, 1, loc=0.48, scale=0.31)
                self.alpha = st.truncweibull_min.rvs(248.69, 0, 1, loc=-47.38, scale=48.2)
                self.beta = 1 - self.alpha

            self.w_i = self.beta

    def choose_diet(self):
        return np.random.choice(
            ["veg", "meat"], p=[self.params["veg_f"], self.params["meat_f"]])

    def initialize_memory_from_neighbours(self, G, agents):
        """Seed memory deterministically to match current neighbor distribution."""
        neigh_ids = list(G.neighbors(self.i))
        if not neigh_ids:
            self.memory = [self.diet] * self.params["M"]
            return
        neigh_diets = [agents[j].diet for j in neigh_ids]
        n_veg = sum(d == "veg" for d in neigh_diets)
        veg_in_mem = round(self.params["M"] * n_veg / len(neigh_diets))
        self.memory = ["veg"] * veg_in_mem + ["meat"] * (self.params["M"] - veg_in_mem)
        np.random.shuffle(self.memory)

    def diet_emissions(self, diet):
        veg, meat = (st.norm.rvs(loc=x, scale=0.1*x)
                     for x in (self.params["veg_CO2"], self.params["meat_CO2"]))
        return {"veg": veg, "meat": meat}[diet]

    def choose_alpha_beta(self, mean):
        return st.truncnorm.rvs(0, 1, loc=mean, scale=mean*0.2)

    # --- Boltzmann switching ---

    def prob_calc(self, other_agent, G, agents):
        """P(switch) via Boltzmann comparison of H_stay vs H_switch.

        Theta is gated by social exposure inside the Hamiltonian (sigmoid on p_opp).
        No external sqrt(p_opp) gate -- avoids double-counting social influence.
        """
        s_stay = 1.0 if self.diet == "veg" else 0.0
        s_switch = 1.0 - s_stay
        M = self.params["M"]
        gate_c = self.params.get("theta_gate_c", 0.3)
        gate_k = self.params.get("theta_gate_k", 10)
        tau = self.params.get("tau", 0.0)
        mu = self.params.get("mu", 0.1)
        H_stay = hamiltonian(s_stay, self.theta, self.w_i, self.memory, M, self.rho, self.diet, gate_c, gate_k, tau, mu)
        H_switch = hamiltonian(s_switch, self.theta, self.w_i, self.memory, M, self.rho, self.diet, gate_c, gate_k, tau, mu)
        return boltzmann_prob(H_switch, H_stay, self.params["beta"])

    # --- Cascade attribution ---

    def reduction_tracker(self, old_c, influencer, agents_list,
                          cascade_depth=1, decay=0.7, max_depth=5, visited=None):
        if visited is None:
            visited = set()
        if cascade_depth > max_depth or influencer.i in visited:
            return
        visited.add(influencer.i)
        delta = old_c - self.C
        if delta > 0:
            influencer.reduction_out += delta * (decay ** (cascade_depth - 1))
            if influencer.influence_parent is not None:
                parent = agents_list[influencer.influence_parent]
                self.reduction_tracker(
                    old_c, parent, agents_list, cascade_depth + 1, decay, max_depth, visited)

    # --- Step ---

    def step(self, G, agents, t):
        """Step agent forward one timestep via pairwise interaction."""
        neighbours = [agents[n] for n in G.neighbors(self.i)]
        if not neighbours:
            return

        other_agent = random.choice(neighbours)
        self.memory.append(other_agent.diet)

        prob_switch = self.prob_calc(other_agent, G, agents)

        if not self.immune and self.flip(prob_switch):
            old_C, old_diet = self.C, self.diet
            self.diet = "meat" if self.diet == "veg" else "veg"
            self.C = self.diet_emissions(self.diet)

            if old_diet == "meat" and self.diet == "veg":
                self.influence_parent = other_agent.i
                self.change_time = t
                other_agent.influenced_agents.append(self.i)
                self.reduction_tracker(old_C, other_agent, agents, cascade_depth=1)
        else:
            self.C = np.random.normal(self.C, 0.1 * self.C)

    def flip(self, p):
        return np.random.random() < p


# %% Model

class Model():
    def __init__(self, params, pmf_tables=None):
        self.pmf_tables = pmf_tables
        self.params = params
        self.snapshots = {}
        self.snapshot_times = [params["steps"] // 3, 2 * params["steps"] // 3]
        self._generate_network()
        self.system_C = []
        self.fraction_veg = []

    def _generate_network(self):
        topo = self.params['topology']
        N = self.params["N"]
        if topo == "complete":
            self.G1 = nx.complete_graph(N)
        elif topo == "BA":
            self.G1 = nx.erdos_renyi_graph(N, self.params["erdos_p"])
        elif topo == "CSF":
            self.G1 = nx.powerlaw_cluster_graph(N, 6, self.params["tc"])
        elif topo == "WS":
            self.G1 = nx.watts_strogatz_graph(N, 6, self.params["tc"])
        elif topo == "PATCH":
            self.G1 = PATCH(N, self.params["k"], float(
                self.params["veg_f"]), h_MM=0.6, tc=self.params["tc"], h_mm=0.7)
            self.G1.generate()
        elif topo == "homophilic_emp":
            self.G1 = nx.empty_graph(N)

    def record_fraction(self):
        self.fraction_veg.append(
            sum(d == "veg" for d in self.get_attributes("diet")) / self.params["N"])

    def harmonise_netIn(self):
        for i, agent in enumerate(self.agents):
            self.G1.nodes[i]["m"] = 1 if agent.diet == "veg" else 0

    def agent_ini(self):
        if self.params["agent_ini"] in ["twin", "sample-max"]:
            self._init_survey_agents()
        else:
            self.agents = [Agent(node, self.params) for node in self.G1.nodes()]
            for agent in self.agents:
                agent.initialize_memory_from_neighbours(self.G1, self.agents)

        print(f"INFO: beta (inv. temperature) = {self.params['beta']}")

        n_immune = int(self.params["immune_n"] * len(self.agents))
        if n_immune > 0:
            for i in np.random.choice(len(self.agents), n_immune, replace=False):
                self.agents[i].immune = True

    def _init_survey_agents(self):
        """Initialize agents from survey data (twin/sample-max mode)."""
        if self.params["agent_ini"] == "sample-max":
            self.survey_data = load_sample_max_agents(self.params["survey_file"])
            if self.params["N"] != len(self.survey_data):
                old_N = self.params["N"]
                self.params["N"] = len(self.survey_data)
                print(f"INFO: Overriding N={old_N} to {len(self.survey_data)} for sample-max mode")
                if self.params['topology'] != "homophilic_emp":
                    self._generate_network()
        else:
            self.survey_data = pd.read_csv(self.params["survey_file"])

        n_survey = len(self.survey_data)
        N = self.params["N"]
        if n_survey > N:
            print(f"INFO: Stratified sampling {n_survey} -> N={N}")
            self.survey_data = stratified_sample_agents(
                self.survey_data, n_target=N,
                strata_cols=['gender', 'age_group', 'incquart', 'educlevel'],
                random_state=42, verbose=True
            ).reset_index(drop=True)
        elif n_survey < N:
            raise ValueError(f"Cannot create {N} agents from {n_survey} survey participants.")
        else:
            print(f"INFO: Using all {n_survey} survey participants")

        self.agents = []
        demo_cols = ['gender', 'age_group', 'incquart', 'educlevel']
        for agent_id, (_, row) in enumerate(self.survey_data.iterrows()):
            demo_key = (tuple(row[c] for c in demo_cols if c in row)
                        if self.pmf_tables else None)
            kwargs = {
                'theta': row["theta"], 'diet': row["diet"],
                'demographics': demo_key, 'pmf_tables': self.pmf_tables,
                'survey_id': row["nomem_encr"],
                'has_rho': bool(row.get("has_rho", False)),
                'has_alpha': bool(row.get("has_alpha", False)),
            }
            if kwargs['has_rho']:
                kwargs['rho'] = row["rho"]
            if kwargs['has_alpha']:
                kwargs['alpha'] = row["alpha"]
            self.agents.append(Agent(i=agent_id, params=self.params, **kwargs))

        # Alpha compression: raw Likert [0,1] -> [alpha_min, alpha_max]
        a_min = self.params.get("alpha_min")
        a_max = self.params.get("alpha_max")
        if a_min is not None and a_max is not None:
            raw_alphas = [a.alpha for a in self.agents]
            for a in self.agents:
                a.alpha = a_min + (a_max - a_min) * a.alpha
                a.beta = 1 - a.alpha
                a.w_i = a.beta
            print(f"INFO: Alpha compressed [{a_min},{a_max}]: "
                  f"mean {np.mean(raw_alphas):.3f} -> {np.mean([a.alpha for a in self.agents]):.3f}")

        self.adjust_veg_fraction_to_target()

        if self.params['topology'] == "homophilic_emp":
            print("INFO: Generating empirical homophily network")
            self.G1, self.sim_matrix = generate_homophily_network_v2(
                N=self.params["N"], avg_degree=8,
                agents_df=self.survey_data,
                attribute_weights=np.array([0.20, 0.35, 0.18, 0.32, 0.05]),
                seed=self.params.get("seed", 42),
                tc=self.params.get("tc", 0.7)
            )
            print(f"INFO: Network: {self.G1.number_of_edges()} edges, "
                  f"mean degree {np.mean([d for _, d in self.G1.degree()]):.2f}")

        for agent in self.agents:
            agent.initialize_memory_from_neighbours(self.G1, self.agents)

    def adjust_veg_fraction_to_target(self):
        if not self.params.get("adjust_veg_fraction", False):
            return
        target = self.params["target_veg_fraction"]
        current_veg = sum(1 for a in self.agents if a.diet == "veg")
        current_frac = current_veg / len(self.agents)
        if current_frac >= target:
            print(f"INFO: Veg fraction {current_frac:.3f} already >= target {target:.3f}")
            return
        needed = int(target * len(self.agents)) - current_veg
        print(f"INFO: Flipping {needed} meat-eaters to veg ({current_frac:.3f} -> {target:.3f})")
        meat_ranked = sorted(
            ((a, a.theta + a.rho) for a in self.agents if a.diet == "meat"),
            key=lambda x: x[1], reverse=True)
        for agent, _ in meat_ranked[:needed]:
            agent.diet = "veg"
            agent.C = agent.diet_emissions("veg")
        final = sum(1 for a in self.agents if a.diet == "veg") / len(self.agents)
        print(f"INFO: Final veg fraction: {final:.3f}")

    def get_attribute(self, attribute):
        return sum(getattr(a, attribute) for a in self.agents)

    def get_attributes(self, attribute):
        return [getattr(a, attribute) for a in self.agents]

    def plot_params(self):
        param_names = ['alpha', 'beta', 'rho', 'theta']
        vals = [self.get_attributes(p) for p in param_names]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(4):
            axes[i].hist(vals[i], bins=30, alpha=0.7)
            axes[i].set_title(f'{param_names[i]} (u={np.mean(vals[i]):.2f})')
        os.makedirs("../visualisations_output", exist_ok=True)
        plt.savefig(f"../visualisations_output/distro_plots_boltzmann.jpg")
        print(f"Averages: a={np.mean(vals[0]):.2f} b={np.mean(vals[1]):.2f} "
              f"r={np.mean(vals[2]):.2f} t={np.mean(vals[3]):.2f}")
        print(f"Diet: {sum(d=='veg' for d in self.get_attributes('diet'))/len(self.agents):.2f} veg")

    def record_snapshot(self, t):
        try:
            graph_copy = self.G1.copy()
        except TypeError:
            graph_copy = nx.Graph(self.G1)
        self.snapshots[t] = {
            'diets': self.get_attributes("diet"),
            'reductions': self.get_attributes("reduction_out"),
            'graph': graph_copy,
            'veg_fraction': self.fraction_veg[-1]
        }

    def cascade_statistics(self):
        stats = []
        for agent in self.agents:
            direct = len(agent.influenced_agents)
            visited = set()
            total = self._count_indirect_influence(agent.i, 0, visited=visited)
            stats.append({
                'agent_id': agent.i, 'direct_influence': direct,
                'total_cascade': total, 'attributed_reduction': agent.reduction_out,
                'multiplier': total / direct if direct > 0 else 0
            })
        return pd.DataFrame(stats)

    def _count_indirect_influence(self, agent_id, depth, max_depth=5, visited=None):
        if visited is None:
            visited = set()
        if depth > max_depth or agent_id in visited:
            return 0
        visited.add(agent_id)
        count = len(self.agents[agent_id].influenced_agents)
        for child_id in self.agents[agent_id].influenced_agents:
            if child_id not in visited:
                count += self._count_indirect_influence(child_id, depth + 1, max_depth, visited)
        return count

    def flip(self, p):
        return np.random.random() < p

    def rewire(self, i):
        """Rewire: TC with prob tc, else homophily * (degree + eps)."""
        if not self.flip(self.params["p_rewire"]):
            return
        neighbors = set(self.G1.neighbors(i.i))
        if not neighbors:
            return

        fof = {}
        for nb in neighbors:
            for nn in self.G1.neighbors(nb):
                if nn != i.i and nn not in neighbors:
                    fof[nn] = fof.get(nn, 0) + 1

        EPSILON = 1e-5
        tc = self.params.get("tc", 0.7)
        j = None

        if fof and random.random() < tc:
            candidates = np.array(list(fof.keys()))
            weights = np.array([fof[c] for c in candidates], dtype=float)
            j = int(np.random.choice(candidates, p=weights / weights.sum()))
        elif hasattr(self, 'sim_matrix'):
            non_nb = np.array([n for n in self.G1.nodes()
                              if n != i.i and n not in neighbors])
            if len(non_nb) > 0:
                weights = self.sim_matrix[i.i, non_nb] * np.array(
                    [self.G1.degree(n) + EPSILON for n in non_nb])
                s = weights.sum()
                j = int(np.random.choice(non_nb, p=weights / s)
                        ) if s > 0 else int(np.random.choice(non_nb))
        else:
            non_nb = [n for n in self.G1.nodes() if n != i.i and n not in neighbors]
            if non_nb:
                j = random.choice(non_nb)

        if j is None:
            return

        nb_list = list(neighbors)
        tri_counts = [
            len(neighbors & set(self.G1.neighbors(nb)) - {i.i}) for nb in nb_list]
        min_tri = min(tri_counts)
        remove_target = random.choice(
            [nb for nb, tc_ in zip(nb_list, tri_counts) if tc_ == min_tri])

        self.G1.add_edge(i.i, j)
        self.G1.remove_edge(i.i, remove_target)

    def current_veg_fraction(self):
        return sum(1 for a in self.agents if a.diet == "veg") / len(self.agents)

    def run(self):
        self.agent_ini()
        self.harmonise_netIn()
        self.record_fraction()
        self.record_snapshot(0)

        for t in range(self.params["steps"]):
            i = np.random.choice(len(self.agents))

            if self.flip(0.2):
                self.agents[i].step(self.G1, self.agents, t)
                self.rewire(self.agents[i])

            self.system_C.append(self.get_attribute("C") / self.params["N"])
            self.record_fraction()

            if t in self.snapshot_times:
                self.record_snapshot(t)

            self.harmonise_netIn()

        self.record_snapshot('final')


# %%
if __name__ == '__main__':

    n_trajectories = 3

    all_trajectories = []

    for traj in range(n_trajectories):
        print(f"\n--- Trajectory {traj+1}/{n_trajectories} ---")
        test_model = Model(params)
        test_model.run()
        all_trajectories.append(test_model.fraction_veg)

    plt.figure(figsize=(8, 6))
    for i, trajec in enumerate(all_trajectories):
        plt.plot(trajec, alpha=0.7, label=f'Trajectory {i+1}')

    plt.ylabel("Vegetarian Fraction")
    plt.xlabel("t (steps)")
    plt.title("Minimal Boltzmann (Galesic et al. 2021)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("../visualisations_output", exist_ok=True)
    plt.savefig("../visualisations_output/boltzmann_minimal_test.png", dpi=150)
    print("Saved: ../visualisations_output/boltzmann_minimal_test.png")

    print("\nDone.")
