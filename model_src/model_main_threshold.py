# -*- coding: utf-8 -*-
"""
BASIC THRESHOLD MODEL VERSION

Simple threshold-based diet switching:
- Agents switch when proportion of opposite diet in memory exceeds personal threshold (rho)
- No utility calculations, no dissonance adjustments, no beta weighting
- Clean slate to test pure threshold dynamics

Created from model_main_single.py
@author: everall
"""

import networkx as nx
import numpy as np
import random
import scipy.stats as st
import math
import seaborn as sns
import matplotlib
#matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd
from netin import PATCH, PAH
from netin import viz
import sys
import os
sys.path.append('..')
from auxillary import network_stats
from auxillary.sampling_utils import stratified_sample_agents, load_sample_max_agents
from auxillary.homophily_network_v2 import generate_homophily_network_v2

# Fix for Windows numpy randint issue with netin
# Only patch once to avoid recursion on module reload
# if not hasattr(np.random.randint, '_is_patched'):
#     _original_randint = np.random.randint

#     def patched_randint(low, high=None, size=None, dtype=int):
#         if high is not None and high >= 2**31:
#             # Use a smaller bound that works on Windows
#             high = 2**31 - 1
#         return _original_randint(low, high, size=size, dtype=dtype)

#     patched_randint._is_patched = True
#     np.random.randint = patched_randint


# %% Preliminary settings
#random.seed(30)
#np.random.seed(30)
#currently agents being incentives to go to other diet
# # CO2 measures are in kg/year, source: https://pubmed.ncbi.nlm.nih.gov/25834298/
params = {"veg_CO2": 1390,
          "vegan_CO2": 1054,
          "meat_CO2": 2054,
          "N": 385,
          "erdos_p": 3,
          "steps": 50000,
          "k": 8, #initial edges per node for graph generation
          "w_i": 5, #weight of the replicator function
          "sigmoid_k": 12, #sigmoid steepness for dissonance scaling
          "immune_n": 0.10,
          "M": 8, # memory length use 7 or 9 maybe.
          "veg_f":0.1, #vegetarian fraction
          "meat_f": 0.9,  #meat eater fraction
          "p_rewire": 0.1, #probability of rewire step
          "rewire_h": 0.1, # slightly preference for same diet
          "tc": 0.7, #probability of triadic closure (tc~0.7 gives clustering C~0.3)
          'topology': "homophilic_emp", #can either be barabasi albert with "BA", or fully connected with "complete"
          "alpha": 0.68, #self dissonance
          "rho": 0.45, #behavioural intentions
          "theta": 0.58, #intrinsic preference (- is for meat, + for vego)
          "agent_ini": 'sample-max', #'synthetic', #choose between "twin" "parameterized" or "synthetic"
          "survey_file": "../data/hierarchical_agents.csv",
          "adjust_veg_fraction": False, #artificially increase veg fraction to match NL demographics
          "target_veg_fraction": 0.06 #target vegetarian fraction (6% for Netherlands)
          }

#%% Auxillary/Helpers

def sample_from_pmf(demo_key, pmf_tables, param, theta=None):
    """Sample parameter from theta-stratified PMF

    Args:
        demo_key: Tuple of (gender, age_group, incquart, educlevel)
        pmf_tables: Dict of PMF tables
        param: Parameter name ('alpha', 'rho', or 'theta')
        theta: Agent's theta value (required for alpha/rho, ignored for theta)
    """
    if not pmf_tables:
        return 0.5

    # Get metadata
    metadata = pmf_tables.get('_metadata', {})
    stratified_params = metadata.get('stratified_params', ['alpha', 'rho'])

    # Build lookup key
    if param in stratified_params and theta is not None:
        # Bin theta value
        theta_bins = metadata.get('theta_bins', [-1.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        theta_labels = metadata.get('theta_labels', ['(-1.0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1.0]'])

        # Find theta bin
        theta_bin = None
        for i, (low, high) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):
            if i == 0:  # First bin includes lower bound
                if low <= theta < high:
                    theta_bin = theta_labels[i]
                    break
            else:
                if low <= theta < high:
                    theta_bin = theta_labels[i]
                    break

        # Handle edge case: theta exactly equals upper bound
        if theta_bin is None and theta >= theta_bins[-2]:
            theta_bin = theta_labels[-1]

        if theta_bin is None:
            # Theta out of range, use fallback
            lookup_key = demo_key
        else:
            # Create full lookup key with theta bin
            lookup_key = demo_key + (theta_bin,)
    else:
        # Theta or unstratified parameter
        lookup_key = demo_key

    # Try to find exact match
    if lookup_key in pmf_tables[param]:
        pmf = pmf_tables[param][lookup_key]
        vals, probs = pmf['values'], pmf['probabilities']
        nz = [(v,p) for v,p in zip(vals, probs) if p > 0]
        if nz:
            v, p = zip(*nz)
            return np.random.choice(v, p=np.array(p)/sum(p))

    # Fallback 1: Try without theta bin (demographics only) for stratified params
    if param in stratified_params and len(lookup_key) > 4:
        demo_only_key = demo_key
        if demo_only_key in pmf_tables[param]:
            pmf = pmf_tables[param][demo_only_key]
            vals, probs = pmf['values'], pmf['probabilities']
            nz = [(v,p) for v,p in zip(vals, probs) if p > 0]
            if nz:
                v, p = zip(*nz)
                return np.random.choice(v, p=np.array(p)/sum(p))

    # Fallback 2: Sample from all values
    all_vals = []
    for cell in pmf_tables[param].values():
        all_vals.extend(cell['values'])
    return np.random.choice(all_vals) if all_vals else 0.5

def load_sample_max_agents(filepath="../data/hierarchical_agents.csv"):
    """Load 385 demographically representative complete-case agents"""
    df = pd.read_csv(filepath)
    complete = df[df['has_alpha'] & df['has_rho']].copy()

    # Sort by nomem_encr to ensure stable row ordering across runs
    complete = complete.sort_values('nomem_encr').reset_index(drop=True)

    # Target stratification for n=385 (perfect age representation)
    age_targets = {
        '18-29': 56,  # All available (bottleneck)
        '30-39': 54,
        '40-49': 56,
        '50-59': 68,
        '60-69': 80,
        '70+': 71
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

        self.i = i  # Set ID first for debugging
        self.params = params
        # types can be vegetarian or meat eater
        self.set_params(**kwargs)
        self.C = self.diet_emissions(self.diet)
        self.memory = []
        self.survey_id = kwargs.get('survey_id', i)  # Original survey ID if available
        self.reduction_out = 0
        self.diet_duration = 0  # Track how long agent maintains current diet
        self.diet_history = []  # Track diet changes
        self.last_change_time = 0  # Track when diet last changed
        self.immune = False

        # Cascade attribution tracking (Guilbeault & Centola 2021)
        self.influence_parent = None  # Direct influencer of last diet change
        self.influenced_agents = []   # Agents this agent directly influenced
        self.change_time = None       # Timestep when diet last changed
    
    
    def set_params(self, **kwargs):

        #TODO: need to change these "synthetic" distibutions to have the right form
        # e.g theta is not normally distributed, but has a left-skewed distribution
        # thinking of not doing this
        if self.params["agent_ini"] not in ["twin", "sample-max"]:
            self.diet = self.choose_diet()
            self.rho = self.choose_alpha_beta(self.params["rho"])
            self.theta = st.truncnorm.rvs(-1, 1, loc=self.params["theta"])
            self.alpha = self.choose_alpha_beta(self.params["alpha"])
            self.beta = 1-self.alpha
        else:
            # Twin/sample-max mode: set theta/diet from survey, sample alpha/rho from PMF if needed
            for key, value in kwargs.items():
                setattr(self, key, value)

            if hasattr(self, 'alpha') and not hasattr(self, 'beta'):
                self.beta = 1 - self.alpha

            # Check if this is a complete case (has all survey parameters)
            is_complete_case = (hasattr(self, 'has_alpha') and self.has_alpha and
                               hasattr(self, 'has_rho') and self.has_rho)

            # Use hierarchical matching first, then PMF for gaps, finally synthetic fallback
            if is_complete_case:
                # Complete case: use survey values directly, no PMF needed
                self.beta = 1 - self.alpha
            elif hasattr(self, 'demographics') and hasattr(self, 'pmf_tables') and self.pmf_tables:
                # Partial case with PMF tables: use survey values where available, sample from PMF for gaps
                if not (hasattr(self, 'has_alpha') and self.has_alpha and hasattr(self, 'alpha')):
                    self.alpha = sample_from_pmf(self.demographics, self.pmf_tables, 'alpha', theta=self.theta)

                if not (hasattr(self, 'has_rho') and self.has_rho and hasattr(self, 'rho')):
                    self.rho = sample_from_pmf(self.demographics, self.pmf_tables, 'rho', theta=self.theta)

                self.beta = 1 - self.alpha
            else:
                # No survey data and no PMF tables: use synthetic fallback
                self.rho = st.truncnorm.rvs(-1, 1, loc=0.48, scale=0.31)
                self.alpha = st.truncweibull_min.rvs(248.69, 0, 1, loc=-47.38, scale=48.2)
                self.beta = 1 - self.alpha
                
        
    def choose_diet(self):
        
        choices = ["veg",  "meat"] #"vegan",
        #TODO: implement determenistic way of initalising agent diets if required
        #currently this should work for networks N >> 1 
        return np.random.choice(choices, p=[self.params["veg_f"], self.params["meat_f"]])
    
    
    def initialize_memory_from_neighbours(self, G, agents):

        """Seed memory deterministically to match current neighbor distribution."""

        neigh_ids = list(G.neighbors(self.i))
        if len(neigh_ids)==0:
            # Fallback: if isolated, just seed with own diet
            self.memory = [self.diet]* self.params["M"]
            return

        neigh_diets = [agents[j].diet for j in neigh_ids]
        n_veg = sum(d == "veg" for d in neigh_diets)
        n_meat = len(neigh_diets) - n_veg

        # Create memory deterministically proportional to neighbor diets
        veg_in_mem = round(self.params["M"] * n_veg / len(neigh_diets))
        meat_in_mem = self.params["M"] - veg_in_mem

        self.memory = ["veg"] * veg_in_mem + ["meat"] * meat_in_mem
        np.random.shuffle(self.memory)  # Shuffle order but keep counts exact
        
    def initialize_memory(self):
        """Initialize agent memory with population-based sampling to represent realistic social context"""
        choices = ["veg", "meat"]
        probabilities = [self.params["veg_f"], self.params["meat_f"]]
        
        for _ in range(self.params["M"]):
            sampled_diet = np.random.choice(choices, p=probabilities)
            self.memory.append(sampled_diet)
        
            
    def diet_emissions(self, diet):
        veg, meat = list(map(lambda x: st.norm.rvs(loc=x, scale=0.1*x),
                                list(map(self.params.get, ["veg_CO2", "meat_CO2"]))))
        lookup = {"veg": veg, "meat": meat}

        return lookup[diet]

    
    
    def choose_alpha_beta(self, mean):
        a, b = 0, 1
        val = st.truncnorm.rvs(a, b, loc=mean, scale=mean*0.2)
        return val
        
        
    def prob_calc(self, other_agent):
        """
        THRESHOLD MODEL WITH DISSONANCE: Additive social + internal pressure

        Mechanism:
        - Base threshold from rho (behavioral intentions)
        - Social pressure: beta-weighted proportion (conformity/social influence)
        - Internal pressure: alpha-weighted dissonance (independence/internal conflict)
        - Combined signal compared to threshold via sigmoid
        """
        # Get proportion of opposite diet in recent memory
        opposite_diet = "meat" if self.diet == "veg" else "veg"
        mem = self.memory[-self.params["M"]:]

        if len(mem) == 0:
            return 0.0

        proportion = sum(d == opposite_diet for d in mem) / len(mem)

        # Beta-weighted social pressure (pull from neighbors)
        social_pressure = self.beta * proportion

        # Alpha-weighted internal pressure (push from dissonance)
        dissonance = self.dissonance_new()
        internal_pressure = self.alpha * dissonance

        # Combined signal vs threshold
        total_signal = social_pressure + internal_pressure
        threshold = self.rho

        # Sigmoid: high probability when total signal exceeds threshold
        prob_switch = 1 / (1 + math.exp(-self.params["w_i"] * (total_signal - threshold)))

        return prob_switch

    def dissonance_new(self):
        """
        Cognitive dissonance for threshold adjustment

        Returns dissonance magnitude when preference (theta) misaligns with diet,
        weighted by exposure-dependent sigmoid activation.

        Theta scale: 0 = meat preference, 1 = veg preference (high theta = high veg preference)

        For meat-eaters:
        - Dissonance activates when theta > 0.5 (prefer veg but eating meat)
        - Magnitude = theta (distance from meat end)
        - Scales with veg exposure in memory via sigmoid (inflection at 0.25)

        For veg-eaters:
        - Dissonance activates when theta < 0.5 (prefer meat but eating veg)
        - Magnitude = 1 - theta (distance from veg end)
        - Scales with meat exposure in memory via sigmoid (inflection at 0.25)
        """
        # Get recent memory
        mem = self.memory[-self.params["M"]:]
        if len(mem) == 0:
            return 0.0

        # Calculate diet fractions in memory
        veg_fraction = sum(d == "veg" for d in mem) / len(mem)
        meat_fraction = 1 - veg_fraction

        # Sigmoid weighting with inflection at 0.25 for both directions
        veg_sigmoid = 1 / (1 + math.exp(-self.params["sigmoid_k"] * (veg_fraction - 0.25)))
        meat_sigmoid = 1 / (1 + math.exp(-self.params["sigmoid_k"] * (meat_fraction - 0.25)))

        # Meat-eaters: dissonance when prefer veg but eating meat
        if self.diet == "meat":
            theta_misaligned = self.theta > 0.5  # Fixed: was <
            if theta_misaligned:
                # Dissonance = distance from meat end (0), weighted by veg exposure
                return veg_sigmoid * self.theta

        # Veg-eaters: dissonance when prefer meat but eating veg
        else:
            theta_misaligned = self.theta < 0.5  # Fixed: was >
            if theta_misaligned:
                # Dissonance = distance from veg end (1), weighted by meat exposure
                return meat_sigmoid * (1 - self.theta)

        return 0.0
    
      

        
    def select_node(self, i, G, i_x=None):
        neighbours = set(G.neighbors(i))
        if i_x is not None:
            neighbours.discard(i_x)

        # Add additional debugging to check the correctness of the neighbours list
        assert i not in neighbours, f"Agent {i} found in its own neighbours list: {neighbours}"

        neighbour_node = random.choice(list(neighbours))
        assert neighbour_node != i, f"node: {i} and neighbour: {neighbour_node} same"

        return neighbour_node
    
    def reduction_tracker(self, old_c, influencer, agents_list, cascade_depth=1, decay=0.7, max_depth=5, visited=None):
        """
        Tracks direct + cascading emission reductions with generational decay

        Implements cascade attribution model combining:
        - Guilbeault & Centola (2021): Complex contagion influence chains
        - Banerjee et al. (2013): Recursive cascade tracking
        - Pentland (2014): Generational decay parameter

        Args:
            old_c: previous consumption level
            influencer: agent who directly influenced the change
            agents_list: reference to all agents for recursive attribution
            cascade_depth: generation distance (1=direct, 2=secondary, etc.)
            decay: discount factor per generation (default 0.7, literature range 0.5-0.8)
            max_depth: maximum cascade depth to prevent infinite recursion (default 5)
            visited: set of agent IDs already visited to prevent cycles
        """
        # Initialize visited set on first call
        if visited is None:
            visited = set()

        # Stop if max depth reached or agent already visited (circular reference)
        if cascade_depth > max_depth or influencer.i in visited:
            return

        # Mark this agent as visited
        visited.add(influencer.i)

        delta = old_c - self.C
        if delta > 0:
            # Attribute to direct influencer with generational decay
            attribution = delta * (decay ** (cascade_depth - 1))
            influencer.reduction_out += attribution

            # Recursive attribution through influence chain
            if influencer.influence_parent is not None:
                parent = agents_list[influencer.influence_parent]
                self.reduction_tracker(old_c, parent, agents_list, cascade_depth + 1, decay, max_depth, visited)
        
    
    def get_neighbour_attributes(self, attribute):
        """
       gets a list of neighbour attributes
    
       Args:
           attribute (str): the desired agent attribute, e.g C, or diet
           neighbours (list): neighbour indexes of i
    
       Returns:
           list: contains all attributes from N agents 
        """
        
        attribute = str(attribute)
        
        # get all agent attibutes from graph single
        attribute_l = [getattr(neighbour, attribute)
                       for neighbour in self.neighbours]
        return attribute_l
    
    
        
    
    def step(self, G, agents, t):
        """
       Steps agent i forward one t

       Args:
           G (dic): an nx graph object
           agents (list): list of agents in G
           t (int): current timestep
       Returns:

        """

        # Select random neighbor
        self.neighbours = [agents[neighbour] for neighbour in G.neighbors(self.i)]

        if not self.neighbours:  # Skip if isolated node
            return

        other_agent = random.choice(self.neighbours)
        self.memory.append(other_agent.diet)

        # Calculate probability of switching based on pairwise comparison
        prob_switch = self.prob_calc(other_agent)

        if not self.immune and self.flip(prob_switch):
            old_C, old_diet = self.C, self.diet
            self.diet = "meat" if self.diet == "veg" else "veg"

            # Recalculate emissions for new diet
            self.C = self.diet_emissions(self.diet)

            # If emissions reduced, track influence and cascade attribution
            if old_diet == "meat" and self.diet == "veg":
                # Record influence relationship (Banerjee et al. 2013)
                self.influence_parent = other_agent.i
                self.change_time = t
                other_agent.influenced_agents.append(self.i)

                # Cascade attribution with decay (Guilbeault & Centola 2021)
                self.reduction_tracker(old_C, other_agent, agents, cascade_depth=1)

        else:
            # Add same noise scale to current consumption without switching
            self.C = np.random.normal(self.C, 0.1 * self.C)
        
      
        
    def flip(self, p):
        return np.random.random() < p

#%% Model 
class Model():
    def __init__(self, params, pmf_tables = None):


        self.pmf_tables = pmf_tables
        self.params = params
        self.snapshots = {}  # Store network snapshots
        self.snapshot_times = [params["steps"] // 3, 2 * params["steps"] // 3]

        self._generate_network()

        self.system_C = []
        self.fraction_veg = []

    def _generate_network(self):
        """Generate network topology based on params"""
        if self.params['topology'] == "complete":
            self.G1 = nx.complete_graph(self.params["N"])
        elif self.params['topology'] == "BA":
            self.G1 = nx.erdos_renyi_graph(self.params["N"], self.params["erdos_p"])
        elif self.params['topology'] == "CSF":
            self.G1 = nx.powerlaw_cluster_graph(self.params["N"], 6, self.params["tc"])
        elif self.params['topology'] == "WS":
            self.G1 = nx.watts_strogatz_graph(self.params["N"], 6, self.params["tc"])
        elif self.params['topology'] == "PATCH":
            self.G1 = PATCH(self.params["N"], self.params["k"], float(self.params["veg_f"]), h_MM=0.6, tc=self.params["tc"], h_mm=0.7)
            self.G1.generate()
        elif self.params['topology'] == "homophilic_emp":
            # Placeholder - will be generated in agent_ini() after loading survey data
            self.G1 = nx.empty_graph(self.params["N"])  
    
    def record_fraction(self):
        fraction_veg = sum(i == "veg" for i in self.get_attributes("diet"))/self.params["N"]
        self.fraction_veg.append(fraction_veg)

    def harmonise_netIn(self):
        """
        Assigns agents to the PATCH minority or majority class based on diet. 
        Initial agent diet fraction which is in the minority will be assigned such.
        
        """
        # if m is one, node is minority, else majority
        for i in range(len(self.agents)):
            self.G1.nodes[i]["m"] = 1 if self.agents[i].diet in "veg" else 0
            
        return
            
    

    def agent_ini(self):


        if self.params["agent_ini"] in ["twin", "sample-max"]:
            if self.params["agent_ini"] == "sample-max":
                # Load pre-stratified 385 complete cases
                self.survey_data = load_sample_max_agents(self.params["survey_file"])
                # Override N to match sample-max size and regenerate network
                if self.params["N"] != len(self.survey_data):
                    old_N = self.params["N"]
                    self.params["N"] = len(self.survey_data)
                    print(f"INFO: Overriding N={old_N} to {len(self.survey_data)} for sample-max mode")
                    print(f"INFO: Regenerating network with correct size")
                    self._generate_network()
            else:
                self.survey_data = pd.read_csv(self.params["survey_file"])
            
            # Handle case where N is smaller than survey data
            if len(self.survey_data) > self.params["N"]:
                print(f"INFO: Survey data has {len(self.survey_data)} participants but N={self.params['N']}")
                print(f"INFO: Using stratified sampling to preserve demographic distributions")
                self.survey_data = stratified_sample_agents(
                    self.survey_data,
                    n_target=self.params["N"],
                    strata_cols=['gender', 'age_group', 'incquart', 'educlevel'],
                    random_state=42,
                    verbose=True
                ).reset_index(drop=True)
            elif len(self.survey_data) < self.params["N"]:
                raise ValueError(f"Cannot create {self.params['N']} agents from only {len(self.survey_data)} survey participants. "
                               f"Either reduce N or provide more survey data.")
            else:
                print(f"INFO: Using all {len(self.survey_data)} survey participants for twin mode")
            
            self.agents = []
            for agent_id, (index, row) in enumerate(self.survey_data.iterrows()):
                # Create demographic key if PMF tables available
                demo_key = None
                if self.pmf_tables:
                    demo_cols = ['gender', 'age_group', 'incquart', 'educlevel']
                    demo_key = tuple(row[col] for col in demo_cols if col in row)

                # Prepare agent kwargs with hierarchical matching data
                agent_kwargs = {
                    'theta': row["theta"],
                    'diet': row["diet"],
                    'demographics': demo_key,
                    'pmf_tables': self.pmf_tables,
                    'survey_id': row["nomem_encr"]
                }

                # Add hierarchical matching flags and values if present
                if "has_rho" in row and row["has_rho"]:
                    agent_kwargs['has_rho'] = True
                    agent_kwargs['rho'] = row["rho"]
                else:
                    agent_kwargs['has_rho'] = False

                if "has_alpha" in row and row["has_alpha"]:
                    agent_kwargs['has_alpha'] = True
                    agent_kwargs['alpha'] = row["alpha"]
                else:
                    agent_kwargs['has_alpha'] = False

                agent = Agent(
                    i=agent_id,  # Use sequential ID to match network nodes
                    params=self.params,
                    **agent_kwargs
                )

                self.agents.append(agent)

            # Adjust vegetarian fraction to match Netherlands demographics
            self.adjust_veg_fraction_to_target()

            # Generate homophilic network if requested
            if self.params['topology'] == "homophilic_emp":
                print(f"INFO: Generating empirical homophily network (Option A: moderate homophily)")
                # Option A weights: [gender, age, income, edu, theta] = [0.20, 0.35, 0.18, 0.32, 0.05]
                self.G1 = generate_homophily_network_v2(
                    N=self.params["N"],
                    avg_degree=8,
                    agents_df=self.survey_data,
                    attribute_weights=np.array([0.20, 0.35, 0.18, 0.32, 0.05]),
                    seed=self.params.get("seed", 42),
                    tc=self.params.get("tc", 0.7)
                )
                print(f"INFO: Generated homophily network with {self.G1.number_of_edges()} edges, mean degree {np.mean([d for n,d in self.G1.degree()]):.2f}")

            # Initialize memory after all agents created
            for agent in self.agents:
                agent.initialize_memory_from_neighbours(self.G1, self.agents)
        else:
            self.agents = [Agent(node, self.params) for node in self.G1.nodes()]
            # Initialize memory for parameterized and synthetic modes
            for agent in self.agents:
                agent.initialize_memory_from_neighbours(self.G1, self.agents)
                
        
        n_immune = int(self.params["immune_n"] * len(self.agents))
        immune_idx = np.random.choice(len(self.agents), n_immune, replace=False)

        for i in immune_idx:
            self.agents[i].immune = True

    def adjust_veg_fraction_to_target(self):
        """
        Adjust vegetarian fraction to match Netherlands demographics.
        Flips meat-eaters with highest theta+rho to vegetarian.
        Vegans/vegetarians from survey data are kept as-is.
        """
        if not self.params.get("adjust_veg_fraction", False):
            return

        target_fraction = self.params["target_veg_fraction"]
        current_veg = sum(1 for a in self.agents if a.diet == "veg")
        current_fraction = current_veg / len(self.agents)

        if current_fraction >= target_fraction:
            print(f"INFO: Current veg fraction {current_fraction:.3f} already >= target {target_fraction:.3f}")
            return

        target_count = int(target_fraction * len(self.agents))
        needed_flips = target_count - current_veg

        print(f"INFO: Adjusting veg fraction from {current_fraction:.3f} to {target_fraction:.3f}")
        print(f"INFO: Flipping {needed_flips} high-theta/rho meat-eaters to vegetarian")

        # Find meat-eaters sorted by theta + rho (highest vegetarian potential)
        meat_eaters = [(a, a.theta + a.rho) for a in self.agents if a.diet == "meat"]
        meat_eaters.sort(key=lambda x: x[1], reverse=True)

        # Flip top N meat-eaters to vegetarian
        for i in range(min(needed_flips, len(meat_eaters))):
            agent, score = meat_eaters[i]
            agent.diet = "veg"
            agent.C = agent.diet_emissions("veg")

        final_veg = sum(1 for a in self.agents if a.diet == "veg")
        final_fraction = final_veg / len(self.agents)
        print(f"INFO: Final veg fraction: {final_fraction:.3f} ({final_veg}/{len(self.agents)} agents)")

    def get_attribute(self, attribute):
        """
        sums a given attribute over N agents 
    
       Args:
           attribute (str): the desired agent attribute, e.g C, or diet
           
    
       Returns:
           int: the sum of the given attribute over all agents N 
        """
        
        attribute = str(attribute)
        # get all agent objects from graph
        attribute_t = 0
        #getattr(self.agents[i], attribute)
        [attribute_t := attribute_t +
            getattr(self.agents[i], attribute) for i in range(len(self.agents))]
        return attribute_t

    def get_attributes(self, attribute):
        """
       gets a list of individual agents attributes based on input 
    
       Args:
           attribute (str): the desired agent attribute, e.g C, or diet
           
    
       Returns:
           list: contains all attributes from N agents 
        """
        
        attribute = str(attribute)
        # get all agent attibutes from graph single
        attribute_l = [getattr(self.agents[i], attribute)
                       for i in range(len(self.agents))]
        return attribute_l
    
    def plot_params(self):
        params = ['alpha', 'beta', 'rho', 'theta']
        vals = [self.get_attributes(p) for p in params]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(4):
            axes[i].hist(vals[i], bins=30, alpha=0.7)
            axes[i].set_title(f'{params[i]} (μ={np.mean(vals[i]):.2f})')
        #plt.tight_layout()
        
        os.makedirs("../visualisations_output", exist_ok=True)
        suffix = f"_{self.params['agent_ini']}"
        plt.savefig(f"../visualisations_output/distro_plots{suffix}.jpg")
        print(f"Averages: α={np.mean(vals[0]):.2f} β={np.mean(vals[1]):.2f} ρ={np.mean(vals[2]):.2f} θ={np.mean(vals[3]):.2f}")
        print(f"Diet: {sum(d=='veg' for d in self.get_attributes('diet'))/len(self.agents):.2f} veg")
        
    def record_snapshot(self, t):
        """Record network state and agent attributes at time t"""
        # Handle PATCH graphs which may not copy properly in some netin versions
        try:
            graph_copy = self.G1.copy()
        except TypeError:
            # Fallback: convert to standard nx.Graph
            graph_copy = nx.Graph(self.G1)

        self.snapshots[t] = {
            'diets': self.get_attributes("diet"),
            'reductions': self.get_attributes("reduction_out"),
            'graph': graph_copy,
            'veg_fraction': self.fraction_veg[-1]
        }
    
    def cascade_statistics(self):
        """
        Calculate cascade metrics for each agent

        Implements metrics from:
        - Banerjee et al. (2013): Diffusion centrality and cascade sizes
        - Guilbeault & Centola (2021): Complex path influence measurement

        Returns:
            pd.DataFrame with columns:
                - agent_id: agent index
                - direct_influence: first-generation adopters influenced
                - total_cascade: full cascade size (all generations)
                - attributed_reduction: total CO2 reduction with decay
                - multiplier: network amplification (total/direct)
        """
        stats = []
        for agent in self.agents:
            cascade_size = len(agent.influenced_agents)

            # Count indirect influence via DFS through influence graph
            visited = set()
            indirect = self._count_indirect_influence(agent.i, depth=0, visited=visited)

            stats.append({
                'agent_id': agent.i,
                'direct_influence': cascade_size,
                'total_cascade': indirect,
                'attributed_reduction': agent.reduction_out,
                'multiplier': indirect / cascade_size if cascade_size > 0 else 0
            })
        return pd.DataFrame(stats)

    def _count_indirect_influence(self, agent_id, depth, max_depth=5, visited=None):
        """
        Recursively count cascade size with depth limit and cycle protection

        Args:
            agent_id: starting agent
            depth: current recursion depth
            max_depth: maximum cascade depth to track (prevents infinite recursion)
            visited: set of already counted agent IDs (prevents double-counting and cycles)

        Returns:
            int: total number of unique agents in cascade
        """
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
        
        # Think about characteristic rewire timescale 
        if self.flip(self.params["p_rewire"]):
            
            non_neighbors = [k for k in nx.non_neighbors(self.G1, i.i)]
            if not non_neighbors:
                return
            
            
            j = random.choice(non_neighbors)
            
            if self.agents[j].diet != i.diet:
                if self.flip(0.10):
                    return
            else:
                # Get the neighbours before rewiirng so j is excluded
                remove_neighbours = list(self.G1.neighbors(i.i))

                if not remove_neighbours:
                    return

                self.G1.add_edge(i.i, j)

                self.G1.remove_edge(i.i, random.choice(remove_neighbours))

    def current_veg_fraction(self):
        """Calculate current fraction of vegetarians in population"""
        return sum(1 for agent in self.agents if agent.diet == "veg") / len(self.agents)

    def run(self):
        
        #print(f"Starting model with agent initation mode: {self.params['agent_ini']}")
        self.agent_ini()
        self.plot_params()
        self.harmonise_netIn()
        self.record_fraction()
        self.record_snapshot(0)
        
        time_array = list(range(self.params["steps"]))
        
        for t in time_array:

            # Select random agent
            i = np.random.choice(range(len(self.agents)))
            
            # Update based on pairwise interaction
            self.agents[i].step(self.G1, self.agents, t)
            self.rewire(self.agents[i],)
            
            
            # Record system state
            self.system_C.append(self.get_attribute("C")/self.params["N"])
            self.record_fraction()
            
            # Record snapshot if required
            if t in self.snapshot_times:
                self.record_snapshot(t)
                
            self.harmonise_netIn()
        # Final snapshot
        self.record_snapshot('final')
    
    

# %%
if __name__ == '__main__': 
	
	n_trajectories = 5
	
	#params.update({'topology': 'PATCH'})
	
	all_trajectories = []
	
	for traj in range(n_trajectories):
    
            print(f"Running trajectory {traj+1}/{n_trajectories}")
            
            test_model = Model(params)
            test_model.run()
            all_trajectories.append(test_model.fraction_veg)
              	
	# Plot all trajectories
	plt.figure(figsize=(8, 6))
	for i, trajec in enumerate(all_trajectories):
		plt.plot(trajec, alpha=0.7, label=f'Trajectory {i+1}')

	plt.ylabel("Vegetarian Fraction")
	plt.xlabel("t (steps)")
	plt.legend()
	plt.show()

	#plt.savefig('../visualisations_output/trajectories.png', dpi=300, bbox_inches='tight')
	#print("Plot saved to visualisations_output/trajectories.png")
	# end_state_A = test_model.get_attributes("reduction_out")
	# end_state_frac = test_model.get_attributes("threshold")
	#viz.handlers.plot_graph(test_model.G1, edge_width = 0.2, edge_arrows =False)
	#network_stats.infer_homophily_values(test_model.G1, test_model.fraction_veg[-1])
