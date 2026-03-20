# -*- coding: utf-8 -*-
"""
Boltzmann dissonance model with complex contagion reinforcement.

Extension of model_main.py (Galesic et al. 2021). Adds diminishing marginal
returns to repeated contacts in the social signal h_soc:

  For each source with n contacts in memory, effective weight = n^gamma.
  h_soc = sum(n_i^gamma for veg sources) / sum(n_j^gamma for all sources)

Where gamma in (0,1) controls diminishing returns: gamma=1 means all contacts
equal (standard model), gamma->0 means only unique sources matter.
E.g. gamma=0.5: 3 contacts from same person -> sqrt(3)=1.73 effective.

Memory stores (diet, source_id) tuples instead of plain diet strings.

@author: everall
"""

import networkx as nx
import numpy as np
import random
import scipy.stats as st
import math
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from netin import PATCH
import sys
import os
sys.path.append('..')
from auxillary.homophily_network_v2 import generate_homophily_network_v2
from auxillary.sampling_utils import stratified_sample_agents
from auxillary import network_stats

# %% Parameters
params = {
    "veg_CO2": 1390, "meat_CO2": 2054,
    "N": 650,
    "erdos_p": 3,
    "steps": 35000,
    "k": 8,
    "immune_n": 0.10,
    "M": 9,
    "veg_f": 0,
    "meat_f": 0.95,
    "p_rewire": 0.01,
    "tc": 0.7,
    "topology": "homophilic_emp",
    "beta": 13,
    "alpha": 0.36,
    "rho": 0.45,
    "theta": 0.44,
    "agent_ini": "sample-max",
    "survey_file": "../data/hierarchical_agents.csv",
    "adjust_veg_fraction": True,
    "target_veg_fraction": 0.06,
    "tau": 0.015,
    "theta_gate_c": 0.35,
    "theta_gate_k": 25,
    "alpha_min": 0.05,
    "alpha_max": 0.80,
    "mu": 0.3,
    "gamma": 0.3,  # diminishing returns exponent: n contacts from same source -> n^gamma effective
    "tau_persistence": None,  # computed as M*2*N (memory renewal window); Takaguchi et al. 2012 / Newman 2002 infectious period
    "decay": 0.7,  # geometric decay per cascade depth for emission credit attribution
}


# %% Helpers

def _status_quo_bias(s, s_cur, p_opp, mu):
    """Inertia term (Samuelson & Zeckhauser 1988): erodes with opposite-diet exposure.
    Ignore this for now"""

def hamiltonian(s_num, theta, w, memory, M, rho, current_diet,
                theta_gate_c=0.3, theta_gate_k=10, tau=0.0, mu=0.1, gamma=0.5):
    """H(s) = (1-w)(s-h_ind)^2 + w(s-h_soc)^2 - tau*s

    h_ind = (1-gate)*rho + gate*theta_01  where gate = sigmoid(k*(p_opp - c))
    h_soc uses diminishing marginal returns per source: n contacts -> n^gamma
      effective weight. gamma=1 => all contacts equal, gamma->0 => only unique
      sources matter.

    Memory entries are (diet, source_id) tuples.
    """
    mem = memory[-M:]
    if not mem:
        return (1 - w) * (s_num - rho)**2

    # Effective counts with diminishing returns per source
    veg_src = Counter(src for d, src in mem if d == "veg")
    all_src = Counter(src for _, src in mem)
    eff_veg = sum(n ** gamma for n in veg_src.values())
    eff_total = sum(n ** gamma for n in all_src.values())

    h_soc = eff_veg / eff_total if eff_total > 0 else 0.0
    p_opp = sum(1 for d, _ in mem if d != current_diet) / len(mem)
    gate = 1 / (1 + math.exp(-theta_gate_k * (p_opp - theta_gate_c)))

    s, s_cur = s_num, 1.0 if current_diet == "veg" else 0.0
    h_ind = (1 - gate) * rho + gate * (theta + 1.0) / 2.0
    return (1 - w) * (s - h_ind)**2 + w * (s - h_soc)**2


def boltzmann_prob(H_switch, H_stay, beta):
    """P(switch) = exp(-beta*H_switch) / Z. Log-sum-exp for stability."""
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
        theta_labels = metadata.get('theta_labels',
            ['(-1.0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1.0]'])
        theta_bin = next((theta_labels[i] for i, (lo, hi) in
                          enumerate(zip(theta_bins[:-1], theta_bins[1:])) if lo <= theta < hi), None)
        if theta_bin is None and theta >= theta_bins[-2]:
            theta_bin = theta_labels[-1]
        lookup_key = demo_key + (theta_bin,) if theta_bin else demo_key
    else:
        lookup_key = demo_key

    def _draw(pmf):
        vals, probs = pmf['values'], pmf['probabilities']
        nz = [(v, p) for v, p in zip(vals, probs) if p > 0]
        if nz:
            v, p = zip(*nz)
            return np.random.choice(v, p=np.array(p) / sum(p))
        return None

    if lookup_key in pmf_tables[param]:
        result = _draw(pmf_tables[param][lookup_key])
        if result is not None:
            return result
    if param in stratified_params and len(lookup_key) > 4 and demo_key in pmf_tables[param]:
        result = _draw(pmf_tables[param][demo_key])
        if result is not None:
            return result
    all_vals = [v for cell in pmf_tables[param].values() for v in cell['values']]
    return np.random.choice(all_vals) if all_vals else 0.5


def load_sample_max_agents(filepath="../data/hierarchical_agents.csv"):
    """Load demographically stratified complete-case agents."""
    df = pd.read_csv(filepath)
    complete = df[df['has_alpha'] & df['has_rho']].copy().sort_values('nomem_encr').reset_index(drop=True)
    age_targets = {'18-29': 56, '30-39': 54, '40-49': 56, '50-59': 68, '60-69': 80, '70+': 71}
    sampled = []
    for age_group, n_target in age_targets.items():
        group = complete[complete['age_group'] == age_group].reset_index(drop=True)
        if len(group) < n_target:
            print(f"WARNING: Only {len(group)} agents in {age_group}, need {n_target}")
            sampled.append(group)
        else:
            sampled.append(group.sample(n=n_target, replace=False, random_state=42))
    result = pd.concat(sampled, ignore_index=True)
    print(f"Sample-max: {len(result)} agents with perfect age stratification")
    return result


# %% Agent

class Agent():

    def __init__(self, i, params, **kwargs):
        self.i = i
        self.params = params
        self.set_params(**kwargs)
        self.C_base = {"veg": self.params["veg_CO2"], "meat": self.params["meat_CO2"]}
        self.C = self.diet_emissions(self.diet)
        self.memory = []  # list of (diet, source_id) tuples
        self.survey_id = kwargs.get('survey_id', i)
        self.reduction_out = 0  # cumulative cascade credit (monotonically non-decreasing; Zhou et al. 2014)
        self.diet_duration = 0
        self.diet_history = []
        self.last_change_time = 0
        self.immune = False
        self.influence_parent = None
        self.influenced_agents = set()
        self.change_time = None

    def set_params(self, **kwargs):
        if self.params["agent_ini"] not in ["twin", "sample-max"]:
            self.diet = self.choose_diet()
            self.rho = self.choose_alpha_beta(self.params["rho"])
            self.theta = st.truncnorm.rvs(-1, 1, loc=self.params["theta"])
            self.alpha = self.choose_alpha_beta(self.params["alpha"])
            self.beta = 1 - self.alpha
            self.w_i = self.beta
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)
            if hasattr(self, 'alpha') and not hasattr(self, 'beta'):
                self.beta = 1 - self.alpha

            is_complete = (getattr(self, 'has_alpha', False) and getattr(self, 'has_rho', False))
            if is_complete:
                self.beta = 1 - self.alpha
            elif hasattr(self, 'demographics') and getattr(self, 'pmf_tables', None):
                if not getattr(self, 'has_alpha', False):
                    self.alpha = sample_from_pmf(self.demographics, self.pmf_tables, 'alpha', theta=self.theta)
                if not getattr(self, 'has_rho', False):
                    self.rho = sample_from_pmf(self.demographics, self.pmf_tables, 'rho', theta=self.theta)
                self.beta = 1 - self.alpha
            else:
                self.rho = st.truncnorm.rvs(-1, 1, loc=0.48, scale=0.31)
                self.alpha = st.truncweibull_min.rvs(248.69, 0, 1, loc=-47.38, scale=48.2)
                self.beta = 1 - self.alpha
            self.w_i = self.beta

    def choose_diet(self):
        return np.random.choice(["veg", "meat"], p=[self.params["veg_f"], self.params["meat_f"]])

    def initialize_memory_from_neighbours(self, G, agents):
        """Seed memory as (diet, source_id) tuples matching neighbor distribution."""
        neigh_ids = list(G.neighbors(self.i))
        if not neigh_ids:
            self.memory = [(self.diet, self.i)] * self.params["M"]
            return
        neigh_diets = [(agents[j].diet, j) for j in neigh_ids]
        n_veg = sum(1 for d, _ in neigh_diets if d == "veg")
        veg_in_mem = round(self.params["M"] * n_veg / len(neigh_diets))
        # Sample source IDs from actual neighbors to preserve diversity info
        veg_nbrs = [j for j in neigh_ids if agents[j].diet == "veg"]
        meat_nbrs = [j for j in neigh_ids if agents[j].diet == "meat"]
        mem = []
        for _ in range(veg_in_mem):
            src = random.choice(veg_nbrs) if veg_nbrs else self.i
            mem.append(("veg", src))
        for _ in range(self.params["M"] - veg_in_mem):
            src = random.choice(meat_nbrs) if meat_nbrs else self.i
            mem.append(("meat", src))
        random.shuffle(mem)
        self.memory = mem

    def diet_emissions(self, diet):
        base = self.params["veg_CO2"] if diet == "veg" else self.params["meat_CO2"]
        # Log-normal with CV~0.20 (Temme 2015; Vellinga 2019 NL dietary data)
        sigma = 0.20
        return st.lognorm.rvs(s=sigma, scale=base)

    def choose_alpha_beta(self, mean):
        return st.truncnorm.rvs(0, 1, loc=mean, scale=mean*0.2)

    def prob_calc(self, other_agent):
        """P(switch) via Boltzmann comparison of H_stay vs H_switch."""
        s_stay = 1.0 if self.diet == "veg" else 0.0
        gate_c, gate_k = self.params.get("theta_gate_c", 0.3), self.params.get("theta_gate_k", 10)
        tau, mu = self.params.get("tau", 0.0), self.params.get("mu", 0.1)
        M, gamma = self.params["M"], self.params.get("gamma", 0.5)
        H_stay   = hamiltonian(s_stay,       self.theta, self.w_i, self.memory, M, self.rho, self.diet, gate_c, gate_k, tau, mu, gamma)
        H_switch = hamiltonian(1.0 - s_stay, self.theta, self.w_i, self.memory, M, self.rho, self.diet, gate_c, gate_k, tau, mu, gamma)
        return boltzmann_prob(H_switch, H_stay, self.params["beta"])

    def _cascade_attribute(self, delta, influencer, agents_list, t,
                           cascade_depth=1, decay=None, visited=None):
        """Propagate emission credit up the influence chain.
        Per-ancestor dwell-time weight (Takaguchi et al. 2012) + geometric depth decay.
        Cumulative-only: back-switching is causally independent (Karimi & Holme 2013;
        Zhou et al. 2014; Liu et al. 2017). No hard depth cap -- decay^d attenuates
        naturally; visited prevents cycles."""
        if decay is None:
            decay = self.params.get("decay", 0.7)
        if visited is None:
            visited = set()
        if influencer.i in visited:
            return
        visited.add(influencer.i)
        tau_p = self.params["tau_persistence"]
        dur = (t - influencer.change_time) if influencer.change_time is not None else t
        w = 1.0 - np.exp(-dur / tau_p)
        influencer.reduction_out += delta * w * (decay ** (cascade_depth - 1))
        if influencer.influence_parent is not None:
            self._cascade_attribute(delta, agents_list[influencer.influence_parent],
                                    agents_list, t, cascade_depth + 1, decay, visited)

    def step(self, G, agents, t):
        """Step agent forward one timestep via pairwise interaction."""
        neighbours = [agents[n] for n in G.neighbors(self.i)]
        if not neighbours:
            return
        other_agent = random.choice(neighbours)
        self.memory.append((other_agent.diet, other_agent.i))

        if not self.immune and self.flip(self.prob_calc(other_agent)):
            old_diet = self.diet
            self.diet = "meat" if self.diet == "veg" else "veg"
            self.C = self.diet_emissions(self.diet)
            # Use base emissions for stable attribution delta
            delta = self.C_base["meat"] - self.C_base["veg"]

            if old_diet == "meat" and self.diet == "veg":
                # meat -> veg: credit the influence chain
                self.influence_parent = other_agent.i
                self.change_time = t
                other_agent.influenced_agents.add(self.i)
                self._cascade_attribute(delta, other_agent, agents, t)

            elif old_diet == "veg" and self.diet == "meat":
                # veg -> meat: detach from tree only -- no debits (Holme & Saramäki 2012; Zhou et al. 2014)
                if self.influence_parent is not None:
                    agents[self.influence_parent].influenced_agents.discard(self.i)
                    self.influence_parent = None
                    self.change_time = None
        else:
            self.C = st.lognorm.rvs(s=0.20, scale=self.C_base[self.diet])

    def flip(self, p):
        return np.random.random() < p


# %% Model

class Model():
    def __init__(self, params, pmf_tables=None):
        self.pmf_tables = pmf_tables
        self.params = params
        # Resolve tau_persistence: memory renewal window = M * 2 * N (Newman 2002; Takaguchi et al. 2012)
        if self.params.get("tau_persistence") is None:
            self.params["tau_persistence"] = self.params["M"] * 2 * self.params["N"]
        self.snapshots = {}
        self.snapshot_times = [params["steps"] * i // 4 for i in range(1, 4)]
        dense_start = params.get("snapshot_dense_start", 0)
        if dense_start and params["steps"] > dense_start:
            self.snapshot_times += list(range(dense_start, params["steps"] + 1, 2000))
        self._generate_network()
        self.system_C = []
        self.fraction_veg = []
        self.steady_state_t = None

    def _generate_network(self):
        topo, N = self.params['topology'], self.params["N"]
        if topo == "complete":
            self.G1 = nx.complete_graph(N)
        elif topo == "BA":
            self.G1 = nx.erdos_renyi_graph(N, self.params["erdos_p"])
        elif topo == "CSF":
            self.G1 = nx.powerlaw_cluster_graph(N, 6, self.params["tc"])
        elif topo == "WS":
            self.G1 = nx.watts_strogatz_graph(N, 6, self.params["tc"])
        elif topo == "PATCH":
            self.G1 = PATCH(N, self.params["k"], float(self.params["veg_f"]),
                            h_MM=0.6, tc=self.params["tc"], h_mm=0.7)
            self.G1.generate()
        elif topo == "homophilic_emp":
            self.G1 = nx.empty_graph(N)

    def record_fraction(self):
        self.fraction_veg.append(
            sum(d == "veg" for d in self.get_attributes("diet")) / self.params["N"])

    def _check_steady_state(self, t, window=5000, threshold=1e-4, min_t=10000):
        """Detect convergence: std of fraction_veg over window < threshold.
        Records a 'steady' snapshot on first detection."""
        if t < min_t or self.steady_state_t is not None or len(self.fraction_veg) < window:
            return
        recent = self.fraction_veg[-window:]
        if np.std(recent) < threshold:
            self.steady_state_t = t
            self.record_snapshot('steady')
            print(f"INFO: Steady state detected at t={t} (std={np.std(recent):.2e} over {window} steps)")

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

        print(f"INFO: beta={self.params['beta']}, gamma={self.params.get('gamma', 0.5)}")
        n_immune = int(self.params["immune_n"] * len(self.agents))
        if n_immune > 0:
            scores = np.array([(a.theta + a.rho) / 2 for a in self.agents])
            n_lo, n_hi = n_immune // 2, n_immune - n_immune // 2
            idx = np.concatenate([np.argsort(scores)[:n_lo], np.argsort(scores)[-n_hi:]])
            for i in idx:
                self.agents[i].immune = True
            print(f"INFO: {n_immune} immune agents ({n_lo} low-tail, {n_hi} high-tail)")

    def _init_survey_agents(self):
        """Initialize agents from survey data (twin/sample-max mode)."""
        if self.params["agent_ini"] == "sample-max":
            self.survey_data = load_sample_max_agents(self.params["survey_file"])
            if self.params["N"] != len(self.survey_data):
                old_N = self.params["N"]
                self.params["N"] = len(self.survey_data)
                print(f"INFO: Overriding N={old_N} -> {self.params['N']} for sample-max mode")
                if self.params['topology'] != "homophilic_emp":
                    self._generate_network()
        else:
            self.survey_data = pd.read_csv(self.params["survey_file"])

        n_survey, N = len(self.survey_data), self.params["N"]
        if n_survey > N:
            print(f"INFO: Stratified sampling {n_survey} -> N={N}")
            self.survey_data = stratified_sample_agents(
                self.survey_data, n_target=N,
                strata_cols=['gender', 'age_group', 'incquart', 'educlevel'],
                random_state=42, verbose=True
            ).reset_index(drop=True)
        elif n_survey < N:
            print(f"INFO: Stratified upsampling {n_survey} -> N={N} (with replacement)")
            self.survey_data = stratified_sample_agents(
                self.survey_data, n_target=N,
                strata_cols=['gender', 'age_group', 'incquart', 'educlevel'],
                random_state=42, verbose=False
            ).reset_index(drop=True)
        else:
            print(f"INFO: Using all {n_survey} survey participants")

        demo_cols = ['gender', 'age_group', 'incquart', 'educlevel']
        self.agents = []
        for agent_id, (_, row) in enumerate(self.survey_data.iterrows()):
            demo_key = tuple(row[c] for c in demo_cols if c in row) if self.pmf_tables else None
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

        a_min, a_max = self.params.get("alpha_min"), self.params.get("alpha_max")
        if a_min is not None and a_max is not None:
            raw_mean = np.mean([a.alpha for a in self.agents])
            for a in self.agents:
                a.alpha = a_min + (a_max - a_min) * a.alpha
                a.beta = 1 - a.alpha
                a.w_i = a.beta
            print(f"INFO: Alpha compressed [{a_min},{a_max}]: mean {raw_mean:.3f} -> {np.mean([a.alpha for a in self.agents]):.3f}")

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
        target = self.params.get("target_veg_fraction", 0.06)
        current_veg = sum(1 for a in self.agents if a.diet == "veg")
        candidates = sorted(
            (a for a in self.agents if a.diet == "meat" and not a.immune),
            key=lambda a: (a.rho, a.alpha), reverse=True)
        needed = max(0, int(target * len(self.agents)) - current_veg)
        flipped = 0
        for a in candidates[:needed]:
            a.diet = "veg"
            a.C = a.diet_emissions("veg")
            flipped += 1
        frac = sum(1 for a in self.agents if a.diet == "veg") / len(self.agents)
        print(f"INFO: Flipped {flipped} meat-eaters -> veg (target: {target:.3f}, actual: {frac:.3f})")

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
        plt.savefig("../visualisations_output/distro_plots_boltzmann_complex.jpg")
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
            'change_times': self.get_attributes("change_time"),
            'alphas': self.get_attributes("alpha"),
            'rhos': self.get_attributes("rho"),
            'graph': graph_copy,
            'veg_fraction': self.fraction_veg[-1],
            'influence_parents': [a.influence_parent for a in self.agents],
            'direct_conversions': [len(a.influenced_agents) for a in self.agents],
        }

    def cascade_statistics(self):
        stats = []
        for agent in self.agents:
            direct = len(agent.influenced_agents)
            total = self._count_indirect_influence(agent.i, 0, visited=set())
            stats.append({
                'agent_id': agent.i, 'direct_influence': direct,
                'total_cascade': total, 'attributed_reduction': agent.reduction_out,
                'multiplier': total / direct if direct > 0 else 0
            })
        return pd.DataFrame(stats)

    def _count_indirect_influence(self, agent_id, depth, visited=None):
        """Count cascade tree size. No hard depth cap -- bounded by network size."""
        if visited is None:
            visited = set()
        if agent_id in visited:
            return 0
        visited.add(agent_id)
        count = len(self.agents[agent_id].influenced_agents)
        for child_id in self.agents[agent_id].influenced_agents:
            if child_id not in visited:
                count += self._count_indirect_influence(child_id, depth + 1, visited)
        return count

    def theoretical_max_reduction(self, decay=None):
        """Analytical upper bound on reduction_out for each agent.
        Assumes: agent converts ALL meat-eating neighbors, each of those
        converts all of theirs, etc. -- perfect tree, no back-switches,
        no overlap. BFS on actual network topology respects real degree
        distribution and finite N.
        Returns DataFrame with agent_id, degree, max_reduction, observed."""
        if decay is None:
            decay = self.params.get("decay", 0.7)
        delta = self.params["meat_CO2"] - self.params["veg_CO2"]
        results = []
        for agent in self.agents:
            visited, frontier = {agent.i}, {agent.i}
            max_red, depth = 0.0, 0
            while frontier:
                next_frontier = {nb for node in frontier
                                 for nb in self.G1.neighbors(node) if nb not in visited}
                if not next_frontier:
                    break
                depth += 1
                visited.update(next_frontier)
                max_red += len(next_frontier) * delta * (decay ** (depth - 1))
                frontier = next_frontier
            results.append({
                'agent_id': agent.i, 'degree': self.G1.degree(agent.i),
                'max_cascade_depth': depth, 'max_reachable': len(visited) - 1,
                'theoretical_max_kg': max_red, 'observed_kg': agent.reduction_out,
                'pct_of_max': (agent.reduction_out / max_red * 100) if max_red > 0 else 0
            })
        inf_bound = delta / (1 - decay)
        avg_deg = np.mean([self.G1.degree(n) for n in self.G1.nodes()])
        print(f"-- Theoretical max reduction sanity check --")
        print(f"   delta={delta} kg, decay={decay}, inf-chain bound={inf_bound:.0f} kg")
        print(f"   avg_degree={avg_deg:.1f}, N={self.params['N']}")
        df = pd.DataFrame(results)
        top = df.nlargest(5, 'theoretical_max_kg')
        print(f"   Top 5 max (kg):     {top['theoretical_max_kg'].values.astype(int)}")
        print(f"   Top 5 observed (kg): {top['observed_kg'].values.astype(int)}")
        print(f"   Top 5 % of max:     {[f'{x:.1f}%' for x in top['pct_of_max'].values]}")
        return df

    def flip(self, p):
        return np.random.random() < p

    def rewire(self, i):
        """Rewire: triadic closure with prob tc, else homophily * (degree + eps)."""
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

        EPSILON, tc, j = 1e-5, self.params.get("tc", 0.7), None

        if fof and random.random() < tc:
            candidates = np.array(list(fof.keys()))
            weights = np.array([fof[c] for c in candidates], dtype=float)
            j = int(np.random.choice(candidates, p=weights / weights.sum()))
        elif hasattr(self, 'sim_matrix'):
            non_nb = np.array([n for n in self.G1.nodes() if n != i.i and n not in neighbors])
            if len(non_nb) > 0:
                weights = self.sim_matrix[i.i, non_nb] * np.array(
                    [self.G1.degree(n) + EPSILON for n in non_nb])
                s = weights.sum()
                j = int(np.random.choice(non_nb, p=weights / s)) if s > 0 else int(np.random.choice(non_nb))
        else:
            non_nb = [n for n in self.G1.nodes() if n != i.i and n not in neighbors]
            if non_nb:
                j = random.choice(non_nb)

        if j is None:
            return

        nb_list = list(neighbors)
        tri_counts = [len(neighbors & set(self.G1.neighbors(nb)) - {i.i}) for nb in nb_list]
        min_tri = min(tri_counts)
        remove_target = random.choice([nb for nb, tc_ in zip(nb_list, tri_counts) if tc_ == min_tri])

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
            if self.flip(0.50):
                self.agents[i].step(self.G1, self.agents, t)
                self.rewire(self.agents[i])

            self.system_C.append(self.get_attribute("C") / self.params["N"])
            self.record_fraction()

            if t in self.snapshot_times:
                self.record_snapshot(t)
            if t % 1000 == 0:
                self._check_steady_state(t)

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
    plt.title(f"Boltzmann + diminishing returns (gamma={params['gamma']})")
    plt.legend()
    plt.tight_layout()
    os.makedirs("../visualisations_output", exist_ok=True)
    plt.savefig("../visualisations_output/boltzmann_complex_test.png", dpi=150)
    print("Saved: ../visualisations_output/boltzmann_complex_test.png\nDone.")
