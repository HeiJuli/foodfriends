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

# %% Preliminary settings
random.seed(30)
np.random.seed(30)

# # CO2 measures are in kg/year, source: https://pubmed.ncbi.nlm.nih.gov/25834298/

params = {"veg_CO2": 1390,
          "vegan_CO2": 1054,
          "meat_CO2": 2054,
          "N": 100,
          "erdos_p": 3,
          "steps": 500,
          "w_i": 5,
          "immune_n": 0.25,
          "M": 4,
          "veg_f":0.14,
          "meat_f": 0.86,  
          "n": 5,
          "v": 10
          }

# %% Agent

class Agent():

    def __init__(self, i, params):

        # types can be vegan, vegetarian or meat eater
        self.params = params
        self.diet = self.choose_diet(self.params)
        self.C = self.diet_emissions(self.diet, self.params)
        self.memory = []
        self.i = i
        self.individual_norm = truncnorm.rvs(0, 1)
        self.global_norm = truncnorm.rvs(0, 1)
        self.reduction_out = 0
        # implement other distributions (pareto)
        self.alpha = truncnorm.rvs(0, 1)
        self.beta = truncnorm.rvs(0, 1)

    def choose_diet(self, params):
        
        choices = ["veg",  "meat"] #"vegan",
        return np.random.choice(choices, p=[params["veg_f"], params["meat_f"]])

    # need to add probabilstic selection
    def diet_emissions(self, diet, params):

        veg, meat = list(map(lambda x: norm.rvs(loc=x, scale=0.1*x),
                                list(map(params.get, ["veg_CO2", "meat_CO2"]))))
        lookup = {"veg": veg, "meat": meat}

        return lookup[diet]

    #def interact(self, neighbour):

    
        
    def prob_calc(self):
        """
        Calculates the probability of a dietry change in either direction
        Args:
            params (dic): a dictionary of model parameters
            sign (bool): indicates positive or negative interaction
    
        Returns:
            float: the probability of change
        """
        
        alternative_diet = "meat" if self.diet == "veg" else "meat"
        u_i = self.calc_utility(self.diet)
        u_s = self.calc_utility(alternative_diet)
        
        prob_switch = 1/(1+math.exp(u_i-u_s))
            
        return prob_switch
        
        
    def dissonance_calc(self, signs):
        
        return 0.1
        
        
    def select_node(self, i, G, i_x=None):
        neighbours = set(G.neighbors(i))
        if i_x is not None:
            neighbours.discard(i_x)

        # Add additional debugging to check the correctness of the neighbours list
        assert i not in neighbours, f"Agent {i} found in its own neighbours list: {neighbours}"

        neighbour_node = random.choice(list(neighbours))
        assert neighbour_node != i, f"node: {i} and neighbour: {neighbour_node} same"

        return neighbour_node
    
    def reduction_tracker(self, C_j, agents):
        """
       Takes the reduction of consumption emissions as a result of an 
       interaction with agent j, and adds it to agent i's total reduction caused
    
       Args:
           C_j: An agent object
    
       Returns:
           int: The product of a and b.
        """
        
        neighbour = agents[C_j]
        #print(self.C,  neighbour.C)
        delta = self.C - neighbour.C
      
        neighbour.reduction_out += delta
        
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
    def get_ratio(self):
        
        neighbour_diets = self.get_neighbour_attributes("diet")
    
        count=0
        for i in neighbour_diets:
            if i == self.diet:
                count += 1 
        ratio_similar = count/len(neighbour_diets)
        
        ratio_dissimilar = 1 - ratio_similar
        
        return ratio_dissimilar, ratio_similar 


    def calc_utility(self,ut_diet):
        if ut_diet == "veg":
            sign = 1
        elif ut_diet == "meat":    
            sign = -1
        else: 
            return ValueError("This " + self.diet +" diet is not defined!")
        
        return sign * self.individual_norm + self.alpha * (1-2*self.get_ratio()) + self.beta * self.global_norm
    
        
    def step(self, G, agents, params):
        """
       Steps agent i forward one t
    
       Args:
           G (dic): a dictionary representing a graph G
           agents (list): list of agents in G
           params (dic): model parameters
       Returns:
           
        """

        # need to implent this recursively to avoid high-degree node bias
        self.neighbours = [agents[neighbour] for neighbour in G.neighbors(self.i)]

        prob_switch = self.prob_calc()
        if self.flip(prob_switch):
            self.diet = "meat" if self.diet == "veg" else "meat"
        
        #neighbour_node = self.select_node(first_n, G, i_x = self.i)
        
        
    def flip(self, p):
        return np.random.random() < p

#%% Model 
class Model():
    def __init__(self, params):

        self.params = params
        self.G1 = nx.erdos_renyi_graph(
            self.params["N"], self.params["erdos_p"])
        self.system_C = []

    #initates agents
    def agent_ini(self, params):
        self.agents = [Agent(i, params) for i in range(len(self.G1.nodes))]
        
        
        #this calculates the immune node f
        n = 0
        
        for i in range(len(self.G1.nodes)):
            self.agents[np.random.choice(
                range(len(self.G1.nodes)))].threshold = 1
            n += 1
            frac = n/len(self.agents)
            if frac > self.params["immune_n"]:
                break
        
        return
    
    
    
    def get_attribute(self, attribute, ):
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
        self.agent_ini(self.params)
        
        time_array = list(range(self.params["steps"]))
        for t in time_array:
        
            for i in self.agents:
                i.step(self.G1, self.agents, self.params)
            self.system_C.append(self.get_attribute("C")/self.params["N"])
            
    
    


# %%
test_model = Model(params)
test_model.run()
# trajec = test_model.system_C
# end_state_A = test_model.get_attributes("reduction_out")
# end_state_frac = test_model.get_attributes("threshold")


