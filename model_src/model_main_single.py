# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:36:06 2024

@author: everall
"""

import networkx as nx
import numpy as np
import random
from scipy.stats import norm
from scipy.stats import truncnorm
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
          "steps":50000,
          "w_i": 5, #weight of the replicator function
          "immune_n": 0.1,
          "M": 10, # memory length
          "veg_f":0.3, #vegetarian fraction
          "meat_f": 0.7,  #meat eater fraciton
          "n": 5,
          "v": 10,
          'topology': "complete", #can either be barabasi albert with "BA", or fully connected with "complete"
          "alpha": 0.35, #self dissonance
          "beta": 0.65, #social dissonance
          "agent_ini": "twin", #choose between "twin", "empirical", "synthetic" 
          "survey_file": "../data/final_data_parameters.csv"
          }

# %% Agent

class Agent():
    
    
    def __init__(self, i, params, **kwargs):
        
        self.params = params
        # types can be vegetarian or meat eater
        self.set_params(self.params, **kwargs)
        self.C = self.diet_emissions(self.diet, self.params)
        self.memory = []
        self.i = i
        self.global_norm = 0.5
        self.reduction_out = 0
        self.diet_duration = 0  # Track how long agent maintains current diet
        self.diet_history = []  # Track diet changes
        self.last_change_time = 0  # Track when diet last changed
    
    
    def set_params(self, params, **kwargs):
        
        if params["agent_ini"] != "twin":
            self.diet = self.choose_diet()
            self.theta = truncnorm.rvs(-1, 1)
            self.alpha = self.choose_alpha_beta(params["alpha"])
            self.beta = self.choose_alpha_beta(params["beta"])
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)
                
            
        
    def choose_diet(self, params):
        
        choices = ["veg",  "meat"] #"vegan",
        #TODO: implement determenistic way of initalising agent diets if required
        #currently this should work for networks N >> 1 
        return np.random.choice(choices, p=[params["veg_f"], params["meat_f"]])

    
    def diet_emissions(self, diet, params):

        veg, meat = list(map(lambda x: norm.rvs(loc=x, scale=0.1*x),
                                list(map(params.get, ["veg_CO2", "meat_CO2"]))))
        lookup = {"veg": veg, "meat": meat}

        return lookup[diet]

    
    
    def choose_alpha_beta(self, mean, params):
        
    
        a, b = 0, 1
        val = truncnorm.rvs(a, b, loc=mean, scale=mean*0.2)
        
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
        
        return prob_switch


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
        mem_same = sum(1 for x in self.memory[-params["M"]:] if x == diet)
        
        ratio =  [mem_same/len(self.memory[-params["M"]:])][0]

        
        util = self.beta*(2*ratio-1) + self.alpha*self.dissonance_new("simple", mode)
        
        return util
    
    
    def step(self, G, agents, params):
        """
       Steps agent i forward one t
    
       Args:
           G (dic): an nx graph object
           agents (list): list of agents in G
           params (dic): model parameters
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
        
        if self.flip(prob_switch):
            old_C = self.C
            self.diet = "meat" if self.diet == "veg" else "veg"
            
            # Update consumption based on influencer
            self.C = other_agent.C if other_agent.diet == self.diet else \
                self.diet_emissions(self.diet, params)
                
            # If emissions reduced, attribute to influencing agent
            if self.diet == "veg":
                self.reduction_tracker(old_C, other_agent)
        
        self.C = self.diet_emissions(self.diet, self.params)
        
      
        
        
    def flip(self, p):
        return np.random.random() < p

#%% Model 
class Model():
    def __init__(self, params):
        

        self.params = params
  
            
        if params['topology'] == "complete":
            
            self.G1 = nx.complete_graph(params["N"])
        elif params['topology'] == "BA":  
            self.G1 = nx.erdos_renyi_graph(
                self.params["N"], self.params["erdos_p"])
        
        elif params['topology'] == "CSF":  
             self.G1 = nx.powerlaw_cluster_graph(params["N"], 6, 0.4)
        
        self.system_C = []
        self.fraction_veg = []  
    
    def record_fraction(self):
        fraction_veg = sum(i == "veg" for i in self.get_attributes("diet"))/self.params["N"]
        self.fraction_veg.append(fraction_veg)

    


    def agent_ini(self, params):
        
        
        if params["agent_ini"] == "twin":
            
            self.survey_data = pd.read_csv(params["survey_file"])
            assert len(self.survey_data) == params["N"], "number of nodes does not match number of survey participants"
            
            
            self.agents=[]
            for index, row in self.survey_data.iterrows():
                agent = Agent(
                    i=row["nomem_encr"],
                    params=params, 
                    alpha=row["alpha"],
                    beta=row["beta"],
                    theta=row["theta"],
                    diet =row["diet"]
                )
                self.agents.append(agent)
            print(f"Created {len(self.agents)} agents for {self.G1.number_of_nodes()} nodes")
        
        # elif params["agent_ini"] == "empirical":
        #     # Ensure agents are created for each node specifically
        #     self.agents = [Agent(node, params) for node in self.G1.nodes()]
            
        else:
            # Ensure agents are created for each node specifically
            self.agents = [Agent(node, params) for node in self.G1.nodes()]
            


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
    

    def run(self):
        
        print(f"Starting model with agent initation mode: {params['agent_ini']}")
        self.agent_ini(self.params)
        self.record_fraction()
        
        time_array = list(range(self.params["steps"]))
        for t in time_array:
            # Select random agent
            i = np.random.choice(range(len(self.agents)))
            
            # Update based on pairwise interaction
            self.agents[i].step(self.G1, self.agents, self.params)
            
            # Record system state
            self.system_C.append(self.get_attribute("C")/self.params["N"])
            self.record_fraction()
    
    


# %%
if  __name__ ==  '__main__': 
    
	test_model = Model(params)

	test_model.run()
	trajec = test_model.fraction_veg

	plt.plot(trajec)
	plt.ylabel("Vegetarian Fraction")
	plt.xlabel("t (steps)")
	plt.show()
# end_state_A = test_model.get_attributes("reduction_out")
# end_state_frac = test_model.get_attributes("threshold")


