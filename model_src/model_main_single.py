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
import matplotlib.pyplot as plt
import pandas as pd
from netin import PATCH, PAH
from netin import viz
import sys
sys.path.append('..')
from auxillary import network_stats


# %% Preliminary settings
#random.seed(30)
#np.random.seed(30)
#currently agents being incentives to go to other diet
# # CO2 measures are in kg/year, source: https://pubmed.ncbi.nlm.nih.gov/25834298/
params = {"veg_CO2": 1390,
          "vegan_CO2": 1054,
          "meat_CO2": 2054,
          "N": 699,
          "erdos_p": 3,
          "steps": 25000,
          "k": 8, #initial edges per node for graph generation
          "w_i": 5, #weight of the replicator function
          "immune_n": 0.10,
          "M": 10, # memory length
          "veg_f":0.5, #vegetarian fraction
          "meat_f": 0.5,  #meat eater fraciton
          "p_rewire": 0.1, #probability of rewire step
          "rewire_h": 0.1, # slightly preference for same diet
          "tc": 0.3, #probability of triadic closure for CSF, PATCH network gens
          'topology': "PATCH", #can either be barabasi albert with "BA", or fully connected with "complete"
          "alpha": 0.35, #self dissonance
          "beta": 0.65, #social dissonance
          "rho": 0.1, #behavioural intentions
          "theta": 0, #intrinsic preference (- is for meat, + for vego)
          "agent_ini": "synthetic", #choose between "twin" "parameterized" or "synthetic" 
          "survey_file": "../data/final_data_parameters.csv"
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
        self.global_norm = 0.5
        self.reduction_out = 0
        self.diet_duration = 0  # Track how long agent maintains current diet
        self.diet_history = []  # Track diet changes
        self.last_change_time = 0  # Track when diet last changed
        self.immune = False
    
    
    def set_params(self, **kwargs):
        
        #TODO: need to change these "synthetic" distibutions to have the right form 
        # e.g theta is not normally distributed, but has a left-skewed distribution
        # thinking of not doing this
        if self.params["agent_ini"] != "twin":
            self.diet = self.choose_diet()
            self.rho = self.choose_alpha_beta(self.params["rho"])
            self.theta = st.truncnorm.rvs(-1, 1, loc=self.params["theta"])
            self.alpha = self.choose_alpha_beta(self.params["alpha"])
            self.beta = self.choose_alpha_beta(self.params["beta"])
        else:
            # Twin mode: set theta/diet from survey, sample alpha/rho from PMF
            for key, value in kwargs.items():
                setattr(self, key, value)
                # Ensure alpha-beta complementarity for hierarchical CSV
                
            if hasattr(self, 'alpha') and not hasattr(self, 'beta'):
                self.beta = 1 - self.alpha
                
            # Use PMF if available, else synthetic fallback
            if hasattr(self, 'demographics') and hasattr(self, 'pmf_tables') and self.pmf_tables:
                self.alpha = sample_from_pmf(self.demographics, self.pmf_tables, 'alpha')
                self.rho = sample_from_pmf(self.demographics, self.pmf_tables, 'rho')
            else:
                # Existing synthetic fallback
                self.rho = st.truncnorm.rvs(-1, 1, loc=0.48, scale=0.31)
                self.alpha = st.truncweibull_min(248.69, 0, 1, loc=-47.38, scale=48.2)
                
        
    def choose_diet(self):
        
        choices = ["veg",  "meat"] #"vegan",
        #TODO: implement determenistic way of initalising agent diets if required
        #currently this should work for networks N >> 1 
        return np.random.choice(choices, p=[self.params["veg_f"], self.params["meat_f"]])

    
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
        u_s = self.calc_utility(other_agent, mode="diff")
        
        prob_switch = 1/(1+math.exp(-5*(u_s-u_i)))
        
        
        #scale by readiness to switch
        return prob_switch * self.rho


    def dissonance_new(self, case, mode):
        
        if mode == "same":
            diet = self.diet
       
        else:
            diet = "meat" if self.diet == "veg" else "veg"
        
        if diet == "veg":
            return self.theta
        else:
            return -1*self.theta
    
    #uses the sigmoid function to calculate dissonance
   #     elif case == "sigmoid":
   #         current_diet = 1 if self.diet == "veg" else -1
   #         # The devision of 0.4621171572600098 is to normalize the sigmoid function in the interval of[-1,1].
   #         return (2/(1+math.exp(-1*(self.theta*current_diet)))-1)/0.46


        
    def select_node(self, i, G, i_x=None):
        neighbours = set(G.neighbors(i))
        if i_x is not None:
            neighbours.discard(i_x)

        # Add additional debugging to check the correctness of the neighbours list
        assert i not in neighbours, f"Agent {i} found in its own neighbours list: {neighbours}"

        neighbour_node = random.choice(list(neighbours))
        assert neighbour_node != i, f"node: {i} and neighbour: {neighbour_node} same"

        return neighbour_node
    
    def reduction_tracker(self, old_c, influencer):
        """
        Tracks emission reductions attributed to single influencing agent
        
        Args:
            old_c: previous consumption level
            influencer: agent who influenced the change
        """
        delta = old_c - self.C
        if delta > 0: 
            current = getattr(influencer, "reduction_out")
            setattr(influencer, "reduction_out", current + delta)
        
    
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
    
    
    #get ratio of meat eaters for a given agent
    #if mode = same, 
    #working

    #working
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
            return
        mem_same = sum(1 for x in self.memory[-self.params["M"]:] if x == diet)
        
        ratio =  [mem_same/len(self.memory[-self.params["M"]:])][0]

        
        util = self.beta*(2*ratio-1) + self.alpha*self.dissonance_new("simple", mode)
        
        return util
    
        
    
    def step(self, G, agents):
        """
       Steps agent i forward one t
    
       Args:
           G (dic): an nx graph object
           agents (list): list of agents in G
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
            
            # Update consumption based on influencer
            if other_agent.diet == self.diet:
                self.C = other_agent.C
            else:
                self.C = self.diet_emissions(self.diet)
            
            
            # If emissions reduced, attribute to influencing agent
            if old_diet == "meat" and self.diet == "veg":
                self.reduction_tracker(old_C, other_agent)
        
        else: 
            self.C = self.diet_emissions(self.diet)
        
      
        
        
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
            self.G1 = PATCH(params["N"], params["k"], float(params["veg_f"]), h_MM=0.6, tc=params["tc"], h_mm=0.6)
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
            assert len(self.survey_data) == self.params["N"], "number of nodes does not match number of survey participants"
            
            self.agents = []
            for index, row in self.survey_data.iterrows():
                # Create demographic key if PMF tables available
                demo_key = None
                if self.pmf_tables:
                    demo_cols = ['gender', 'age_cat', 'netinc_cat', 'educcat']
                    demo_key = tuple(row[col] for col in demo_cols if col in row)
                
                agent = Agent(
                    i=row["nomem_encr"],
                    params=self.params,
                    theta=row["theta"],
                    diet=row["diet"],
                    demographics=demo_key,
                    pmf_tables=self.pmf_tables
                )
                self.agents.append(agent)
        else:
            self.agents = [Agent(node, self.params) for node in self.G1.nodes()]
                
        
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
    
    def record_snapshot(self, t):
        """Record network state and agent attributes at time t"""
        self.snapshots[t] = {
            'diets': self.get_attributes("diet"),
            'reductions': self.get_attributes("reduction_out"),
            'graph': self.G1.copy(),
            'veg_fraction': self.fraction_veg[-1]
        }
    
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
                
        
    def run(self):
        
        #print(f"Starting model with agent initation mode: {self.params['agent_ini']}")
        self.agent_ini()
        self.harmonise_netIn()
        self.record_fraction()
        self.record_snapshot(0)
        
        time_array = list(range(self.params["steps"]))
        
        for t in time_array:
                
            # Select random agent
            i = np.random.choice(range(len(self.agents)))
            
            # Update based on pairwise interaction
            self.agents[i].step(self.G1, self.agents)
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
	
	params.update({'topology': 'PATCH'})
	
	test_model = Model(params)
	
	test_model.run()
	#nx.draw(test_model.G1, node_size = 25, width = 0.5)
    
	trajec = test_model.fraction_veg
	plt.plot(trajec)
	plt.ylabel("Vegetarian Fraction")
	plt.xlabel("t (steps)")
	plt.show()
    # end_state_A = test_model.get_attributes("reduction_out")
    # end_state_frac = test_model.get_attributes("threshold")
#viz.handlers.plot_graph(test_model.G1, edge_width = 0.2, edge_arrows =False)
    #network_stats.infer_homophily_values(test_model.G1, test_model.fraction_veg[-1])
