# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:36:06 2024

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
          "N": 300,
          "erdos_p": 3,
          "steps": 15000,
          "k": 8, #initial edges per node for graph generation
          "w_i": 5, #weight of the replicator function
          "immune_n": 0.10,
          "M": 7, # memory length
          "veg_f":0.1, #vegetarian fraction
          "meat_f": 0.9,  #meat eater fraction
          "p_rewire": 0.1, #probability of rewire step
          "rewire_h": 0.1, # slightly preference for same diet
          "tc": 0.2, #probability of triadic closure for CSF, PATCH network gens
          'topology': "PATCH", #can either be barabasi albert with "BA", or fully connected with "complete"
          "alpha": 0.68, #self dissonance
          "rho": 0.45, #behavioural intentions
          "theta": 0.58, #intrinsic preference (- is for meat, + for vego)
          "agent_ini": 'twin', #'synthetic', #choose between "twin" "parameterized" or "synthetic" 
          "survey_file": "../data/hierarchical_agents.csv"
          }

#%% Auxillary/Helpers

def sample_from_pmf(demo_key, pmf_tables, param):
    """Sample single parameter from PMF"""
    if pmf_tables and demo_key in pmf_tables[param]:
        pmf = pmf_tables[param][demo_key]
        vals, probs = pmf['values'], pmf['probabilities']
        nz = [(v,p) for v,p in zip(vals, probs) if p > 0]
        if nz:
            v, p = zip(*nz)
            return np.random.choice(v, p=np.array(p)/sum(p))
    
    # Fallback: sample from all values or default
    if pmf_tables:
        all_vals = []
        for cell in pmf_tables[param].values():
            all_vals.extend(cell['values'])
        return np.random.choice(all_vals) if all_vals else 0.5
    return 0.5




# %% Agent

class Agent():
    
    def __init__(self, i, params, **kwargs):
        
        self.params = params
        # types can be vegetarian or meat eater
        self.set_params(**kwargs)
        self.C = self.diet_emissions(self.diet)
        self.memory = []
        self.i = i
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
        if self.params["agent_ini"] != "twin":
            self.diet = self.choose_diet()
            self.rho = self.choose_alpha_beta(self.params["rho"])
            self.theta = st.truncnorm.rvs(-1, 1, loc=self.params["theta"])
            self.alpha = self.choose_alpha_beta(self.params["alpha"])
            self.beta = 1-self.alpha
        else:
            # Twin mode: set theta/diet from survey, sample alpha/rho from PMF
            for key, value in kwargs.items():
                setattr(self, key, value)
                # Ensure alpha-beta complementarity for hierarchical CSV
                
            if hasattr(self, 'alpha') and not hasattr(self, 'beta'):
                self.beta = 1 - self.alpha
                
            # Use hierarchical matching first, then PMF for gaps, finally synthetic fallback
            if hasattr(self, 'demographics') and hasattr(self, 'pmf_tables') and self.pmf_tables:
                # Check if we have direct alpha match from survey
                if hasattr(self, 'has_alpha') and self.has_alpha and hasattr(self, 'alpha'):
                    # Use direct alpha from hierarchical matching
                    pass  # self.alpha already set from kwargs
                else:
                    # Sample alpha from PMF using demographics
                    self.alpha = sample_from_pmf(self.demographics, self.pmf_tables, 'alpha')
                
                # Check if we have direct rho match from survey  
                if hasattr(self, 'has_rho') and self.has_rho and hasattr(self, 'rho'):
                    # Use direct rho from hierarchical matching
                    pass  # self.rho already set from kwargs
                else:
                    # Sample rho from PMF using demographics
                    self.rho = sample_from_pmf(self.demographics, self.pmf_tables, 'rho')
                
                self.beta = 1 - self.alpha  # Recalculate beta after new alpha
            else:
                # Existing synthetic fallback
                self.rho = st.truncnorm.rvs(-1, 1, loc=0.48, scale=0.31)
                self.alpha = st.truncweibull_min.rvs(248.69, 0, 1, loc=-47.38, scale=48.2)
                self.beta = 1 - self.alpha  # Recalculate beta after new alpha
                
        
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
        Calculates probability of diet change based on pairwise comparison
        
        Args:
            other_agent: the agent being compared with
        """
        u_i = self.calc_utility(other_agent, mode="same")
        
        #utility "shadow", i.e alternative choice
        u_s = self.calc_utility(other_agent, mode="diff")
        

        
        prob_switch = 1/(1+math.exp(-1.7*(u_s-u_i)))
        

        #scale by readiness to switch - only applies to meat-eaters (belief-action gap)
        if self.diet == 'meat':
            return prob_switch #self.rho

        else:
            return prob_switch 

    def dissonance_new(self, case, mode):
        """
        Cognitive dissonance based on alignment between preference (theta) and behavior.
        Dissonance magnitude scales with social exposure to alternative diet.

        theta > 0: prefers veg
        theta < 0: prefers meat
        rho: behavioral intention (belief-action gap for meat-eaters)
        """
        if mode == "same":
            diet = self.diet
        else:
            diet = "meat" if self.diet == "veg" else "veg"

        prefers_veg = (self.theta > 0)
        diet_is_veg = (diet == "veg")
        aligned = (prefers_veg == diet_is_veg)

        if aligned or len(self.memory) == 0:
            return 0.0

        # Dissonance activates only with social exposure to alternative
        veg_exposure = sum(1 for d in self.memory if d == "veg") / len(self.memory)
        exposure_factor = 1 / (1 + np.exp(-20 * (veg_exposure)))
        gap = abs(self.theta - self.rho)

        return -gap * exposure_factor
    
      

        
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
    
    
 
    def calc_utility(self, other_agent, mode):
        """
        Calculates utility for pairwise interaction
        
        Args:
            other_agent: the agent being compared with
            mode: whether calculating utility for same or different diet
        """
        if mode == "same":
            diet = self.diet
        else:
            diet = "meat" if self.diet == "veg" else "veg"
        
        
        # Calculate ratio based on single comparison
        if len(self.memory) == 0:
            print("memory empty!")
            return 0.0  # Return neutral utility for empty memory
           
        
        mem_same = sum(1 for x in self.memory[-self.params["M"]:] if x == diet)

        ratio = mem_same/len(self.memory[-self.params["M"]:])  # Remove unnecessary list wrapping


        util = self.beta*(3*ratio-1.5) + self.alpha*self.dissonance_new("simple", mode)

        return util
    
        
    
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
        self.snapshot_times = [int(params["steps"] * r) for r in [0.33, 0.66]]
            
        if params['topology'] == "complete":
            
            self.G1 = nx.complete_graph(params["N"])
        elif params['topology'] == "BA":  
            self.G1 = nx.erdos_renyi_graph(
                self.params["N"], self.params["erdos_p"])
        
        elif params['topology'] == "CSF":  
             self.G1 = nx.powerlaw_cluster_graph(params["N"], 6, params["tc"])
             
        elif params['topology'] == "WS":  
             self.G1 = nx.watts_strogatz_graph(params["N"], 6, params["tc"])
        
        #MM stands for majority class, mm for minority, tc is triadic closure
        elif params['topology'] == "PATCH":
            self.G1 = PATCH(params["N"], params["k"], float(params["veg_f"]), h_MM=0.6, tc=params["tc"], h_mm=0.7)
            self.G1.generate()
            
        self.system_C = []
        self.fraction_veg = []  
    
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
        
        
        if self.params["agent_ini"] == "twin":
            self.survey_data = pd.read_csv(self.params["survey_file"])
            
            # Handle case where N is smaller than survey data
            if len(self.survey_data) > self.params["N"]:
                print(f"WARNING: Survey data has {len(self.survey_data)} participants but N={self.params['N']}")
                print(f"WARNING: Randomly sampling {self.params['N']} participants from survey data")
                self.survey_data = self.survey_data.sample(n=self.params["N"], random_state=42).reset_index(drop=True)
                print(f"WARNING: Using {len(self.survey_data)} sampled participants for twin mode")
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
	
	n_trajectories = 2
	
	params.update({'topology': 'PATCH'})
	
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
	print("Plot saved to visualisations_output/trajectories.png")
	# end_state_A = test_model.get_attributes("reduction_out")
	# end_state_frac = test_model.get_attributes("threshold")
	#viz.handlers.plot_graph(test_model.G1, edge_width = 0.2, edge_arrows =False)
	#network_stats.infer_homophily_values(test_model.G1, test_model.fraction_veg[-1])
