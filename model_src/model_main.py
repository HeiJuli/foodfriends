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

# %% Preliminary settings
#random.seed(30)
#np.random.seed(30)
#currently agents being incentives to go to other diet
# # CO2 measures are in kg/year, source: https://pubmed.ncbi.nlm.nih.gov/25834298/
params = {"veg_CO2": 1390,
          "vegan_CO2": 1054,
          "meat_CO2": 2054,
          "N": 100,
          "erdos_p": 3,
          "steps":2000,
          "w_i": 5, #weight of the replicator function
          "immune_n": 0.1,
          "M": 4,
          "veg_f":0.6, #vegetarian fraction
          "meat_f": 0.4,  #meat eater fraciton
          "n": 5,
          "v": 10,
          'topology': "BA", #can either be barabasi albert with "BA", or fully connected with "complete"
          "alpha": 0.8,
          "beta": 0.2
          
          }

# %% Agent

class Agent():

    def __init__(self, i, params):

        # types can be vegetarian or meat eater
        self.params = params
        self.diet = self.choose_diet(self.params)
        self.C = self.diet_emissions(self.diet, self.params)
        self.memory = []
        self.i = i
        self.individual_norm = truncnorm.rvs(-1, 1)
        self.global_norm = 0.5
        self.reduction_out = 0
        # implement other distributions (pareto)
        self.alpha = self.params["alpha"]
        self.beta = self.params["beta"]
        self.diet_duration = 0  # Track how long agent maintains current diet
        self.diet_history = []  # Track diet changes
        self.last_change_time = 0  # Track when diet last changed
        
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

    

    def prob_calc(self):
        """
        Calculates the probability of a dietry change in either direction
        Args:
            params (dic): a dictionary of model parameters
            sign (bool): indicates positive or negative interaction
    
        Returns:
            float: the probability of change
        """
    
        #actually utility, if you don;t change, mode means whether calcing same diet or opposite 
        u_i = self.calc_utility(mode = "same")
        
        #utlity shadow - alternative utlity - if agent were to change
        u_s = self.calc_utility(mode = "diff")
        
    
        prob_switch = 1/(1+math.exp(-5*(u_s-u_i)))
        #print(f"u_s: {u_s}, u_i: {u_i}, Switching p: {prob_switch}")
     
        return prob_switch
    


    def dissonance_new(self, case, mode):
        
        if mode == "same":
            diet = self.diet
       
        else:
            diet = "meat" if self.diet == "veg" else "veg"
        
        if diet == "veg":
            return self.individual_norm
        else:
            return -1*self.individual_norm
    
    #uses the sigmoid function to calculate dissonance
   #     elif case == "sigmoid":
   #         current_diet = 1 if self.diet == "veg" else -1
   #         # The devision of 0.4621171572600098 is to normalize the sigmoid function in the interval of[-1,1].
   #         return (2/(1+math.exp(-1*(self.individual_norm*current_diet)))-1)/0.46


        
    def select_node(self, i, G, i_x=None):
        neighbours = set(G.neighbors(i))
        if i_x is not None:
            neighbours.discard(i_x)

        # Add additional debugging to check the correctness of the neighbours list
        assert i not in neighbours, f"Agent {i} found in its own neighbours list: {neighbours}"

        neighbour_node = random.choice(list(neighbours))
        assert neighbour_node != i, f"node: {i} and neighbour: {neighbour_node} same"

        return neighbour_node
    
    # def reduction_tracker(self, old_c, similar_neighbours):
    #     """
    #    Takes the reduction of consumption emissions as a result of an 
    #    interaction with agent j, and adds it to agent i's total reduction caused
    
    #    Args:
    #        agents: agent objects
    
    #    Returns:
    #        int: The product of a and b.
    #     """
        
        
    #     delta = old_c - self.C
        
    #     if delta <= 0:  # Only track actual reductions
    #         return
        
    #     #TODO: this is uneccesarily intensive, optimise
    #     for i in self.neighbours:
    #         # get current reduction amount
    #         current = getattr(i, "reduction_out")
    #         #set the new reduction amount
    #         setattr(i, "reduction_out", current + delta/len(self.neighbours))
    
    def reduction_tracker(self, old_c, similar_neighbours, G):
        """
        Tracks emission reductions and attributes them to influential neighbors
        based on their relative contribution to the agent's decision
        
        Args:
            old_c: previous consumption level
            similar_neighbours: list of neighbors with same diet
        """
        delta = old_c - self.C
        
        if delta <= 0:  # Only track actual reductions
            return
            
        # Calculate influence weights based on each neighbor's characteristics
        weights = []
        for neighbor in similar_neighbours:
            # Weight based on:
            # 1. Neighbor's time with current diet (stability)
            # 2. Neighbor's network centrality 
            # 3. Neighbor's influence history
            w = 1.0  # Base weight
            
            # Add weight if neighbor maintained diet for longer
            if hasattr(neighbor, 'diet_duration'):
                w *= (1 + 0.1 * neighbor.diet_duration)
                
            # Add centrality-based weight (degree centrality as proxy)
            w *= (1 + 0.1 * sum(G.neighbors(neighbor.i)))
            
            weights.append(w)
    
        # Normalize weights
        if weights:
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights] if total_weight > 0 else [1.0/len(weights)] * len(weights)
            
            # Attribute reductions proportionally
            for neighbor, weight in zip(similar_neighbours, weights):
                current = getattr(neighbor, "reduction_out")
                setattr(neighbor, "reduction_out", current + delta * weight)
        
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
    def get_ratio(self, mode = "same"):
        """
       gets the ratio of agents with a certain diet over k neighbours of the agent
       object. This ratio is based on neighbours with the same diet of the agent with
       mode = same, or the opposite diet if anything else.
    
       Args:
           mode (str): the mode of counting
    
       Returns:
           float: fraction of specified diet over the total neighbours k
        """
        
        if mode == "same":
            diet = self.diet
        else:
            diet = "meat" if self.diet == "veg" else "veg"
            
        neighbour_diets = self.get_neighbour_attributes("diet")
        
        count=0
        for i in neighbour_diets:
            if i == diet:
                count += 1 
        ratio_diet = count/len(neighbour_diets)
        
      
        
        return ratio_diet 

    #working
    def calc_utility(self, mode):
       
    
        util = self.beta*(1-2*self.get_ratio(mode)) + self.alpha*self.dissonance_new("simple", mode)#- self.beta*self.global_norm)
        
        
       
        return util 
    
    def step(self, G, agents, params, t):
        """
       Steps agent i forward one t
    
       Args:
           G (dic): an nx graph object
           agents (list): list of agents in G
           params (dic): model parameters
       Returns:
           
        """
        
        # need to implent this recursively to avoid high-degree node bias
        self.neighbours = [agents[neighbour] for neighbour in G.neighbors(self.i)]
        
        
        prob_switch = self.prob_calc()
      
        if self.flip(prob_switch):
            old_C = self.C
            self.diet = "meat" if self.diet == "veg" else "veg"
            
            
            self.diet_duration = 0
            self.last_change_time = t  # Assuming you pass current time t
            self.diet_history.append((t, self.diet))
            #getting neighbours with similar diet
            similar_neighbours = [i for i in self.neighbours if i.diet == self.diet]
            
            #getting list
            neighbours_C = [neighbour.C for neighbour in self.neighbours]
            
            #makes C (emissions) the average of neighbours with the same diet
            self.C = np.mean(neighbours_C) if len(neighbours_C) >= 1 else \
                self.diet_emissions(self.diet, params)
            
          
            # if dietry emissions are reduced, attribute this to veg network neighbours
            if self.diet == "veg":
                self.reduction_tracker(old_C, similar_neighbours, G)
            
        else:
            self.diet_duration += 1
            
            
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
        
        # elif params['topology'] == "FB":  
        #     self.G1 = 
        
        self.system_C = []
        self.fraction_veg = []  
    
    def record_fraction(self):
        fraction_veg = sum(i == "veg" for i in self.get_attributes("diet"))/self.params["N"]
        self.fraction_veg.append(fraction_veg)

    


    def agent_ini(self, params):
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
        #initiate agents
        self.agent_ini(self.params)
        #self.map_agents() 
        self.record_fraction()
        time_array = list(range(self.params["steps"]))
        for t in time_array:
            #selecting an index at random
            i = np.random.choice(range(len(self.agents)))
            #for i in self.agents:
            self.agents[i].step(self.G1, self.agents, self.params, t)
            self.system_C.append(self.get_attribute("C")/self.params["N"])
            self.record_fraction()
            #print(self.G1.nodes[1]["agent"].C, self.agents[1].C)
    
    


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


