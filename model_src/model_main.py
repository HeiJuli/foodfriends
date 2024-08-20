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
random.seed(30)
np.random.seed(30)
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
          'topology': "BA" #can either be barabasi albert with "BA", or fully connected with "complete"
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
        self.alpha = 0.5
        self.beta = 0.5#0.3
        
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
    
        #actually utility, mode means whether calcing sam diet or opposit 
        u_i = self.calc_utility(mode = "same")
        
        #utlity shadow - alternative utlity
        u_s = self.calc_utility(mode = "diff")
        
    
        prob_switch = 1/(1+math.exp(-3*(u_s-u_i)))
        #print(f"u_s: {u_s}, u_i: {u_i}, Switching p: {prob_switch}")
     
        return prob_switch
    
    def dissonance(self, case, mode):
        
        
        
        if mode == "same":
            diet = self.diet
        #if mode diff
        else:
            diet = "meat" if self.diet == "veg" else "veg"
            
        #uses the discrete dissonance function
        
        if case == "simple":
            
            if diet == "veg":
                #norm from 0-1 means you prefer veg inherently 
                if self.individual_norm >= 0:
                    sign = 1
                else:
                    sign = -1
                    
            elif diet == "meat":    
                
                if self.individual_norm >= 0:
                    sign = -1
                else:
                    sign = 1
            else: 
                return ValueError("This " + self.diet +" diet is not defined!")
            
            return sign * self.individual_norm
        
        #uses the sigmoid function to calculate dissonance
        elif case == "sigmoid":
            current_diet = 1 if self.diet == "veg" else -1
            # The devision of 0.4621171572600098 is to normalize the sigmoid function in the interval of[-1,1].
            return (2/(1+math.exp(-1*(self.individual_norm*current_diet)))-1)/0.46

        else:
            return ValueError("You can only select form either 'simple' or 'sigmoid'. ")



        
        
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
       
        #TODO: reformulate this to use similar agents  
        util = self.alpha*(2*self.get_ratio(mode)-1) #- self.beta*self.global_norm)
        
        
        #util_first_order = self.dissonance("simple", mode) + self.get_ratio(mode)
        #print(util)
        # util = self.alpha*self.get_ratio(mode)
       
        return util 
    
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
            self.diet = "meat" if self.diet == "veg" else "veg"
            
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
        
        self.system_C = []
        self.fraction_veg = []  
    
    def record_fraction(self):
        fraction_veg = sum(i == "veg" for i in self.get_attributes("diet"))/self.params["N"]
        self.fraction_veg.append(fraction_veg)

    


    def agent_ini(self, params):
        # Ensure agents are created for each node specifically
        self.agents = [Agent(node, params) for node in self.G1.nodes()]
        


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
        #initiate agents
        self.agent_ini(self.params)
        #self.map_agents() 
        self.record_fraction()
        time_array = list(range(self.params["steps"]))
        for t in time_array:
            #selecting an index at random
            i = np.random.choice(range(len(self.agents)))
            #for i in self.agents:
            self.agents[i].step(self.G1, self.agents, self.params)
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


