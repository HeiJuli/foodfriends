# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:03:54 2023

@author: everall
"""
#%% Imports
import numpy as np
import seaborn as sns
import pandas as pd
import model_main
import matplotlib.pyplot as plt
import time

#%% Setting parameters

params = {"veg_CO2": 1390,
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

#%% Main functions

def run_model(params=params):

    test_model = model_main.Model(params)
    test_model.run()
    # end_state_A = test_model.get_attributes("C")
    # end_state_frac = test_model.get_attributes("threshold")
    #print(params)
    return test_model

#expand this to include other parameters/be generic
def parameter_sweep(params, param_to_vary, num_iterations, param_min, param_max, num_samples):
    """
    Runs multiple model simulations of the ABM while varying a given parameter.
    
    Args:
        params (dict): The dictionary of model parameters.
        param_to_vary (str): The name of the parameter to be varied.
        num_iterations (int): The number of iterations to run.
        param_min (float): The minimum value of the parameter to be tested.
        param_max (float): The maximum value of the parameter to be tested.
        num_samples (int): The number of samples to take from the parameter space.
        
    Returns:
        list: A list of lists, where each inner list corresponds to one model output.
    """
    param_values = np.linspace(param_min, param_max, num_samples)
    
    runs = []
    for _ in range(num_iterations):
        print(f"Started iteration: {_ + 1}/{num_iterations}")
        for p in param_values:
            # Ensure that the fractions always sum to 1
            if param_to_vary == "veg_f":
                params["meat_f"] = 1 - p
            elif param_to_vary == "meat_f":
                params["veg_f"] = 1 - p
            
            params.update({param_to_vary: p})
            test_model = run_model(params)
            trajec_end = test_model.system_C[-1:]
            runs.append([trajec_end, p])
    
    return runs

def timer(func, *args):
    start = time.time()
    outputs = func(*args)
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime complete: {mins:5.0f} mins {sec}s\n')
    
    return outputs
    

#%% Running sensitivity analysis
#run_model()
#params.update({"n":5,"v": 10})
model_out = timer(parameter_sweep, params, "veg_f", params["n"], 0.0, 1.0, params["v"])


#%% Processesing data frames

####consumption
df_C = pd.DataFrame(model_out)
df_C = df_C.explode(0).rename(columns={0:"C_t_max", 1: "paramater_value"} )
df_C.to_csv(f"../model_output/output_N_{params['N']}_n_{params['n']}_v_{params['v']}_.csv")
#df_C = pd.read_csv(f"../model_output/output_N_{params['N']}_n_{params['n']}_v_{params['v']}_.csv")

####agents reduction
# df_reduc = pd.DataFrame(model_out[1])
# df_reduc = df_reduc.explode(0).rename(columns={0:"Reduced", 1: "paramater_value"} )


#%% Rough/Demo plots
sp = sns.scatterplot(data = df_C, x = "paramater_value", y = "C_t_max" )  
plt.xlabel('% vegans & vegetarians')
plt.ylabel('Final average dietry consumption [kg/c02/year]')  
plt.savefig("../visualisations_output/Example_consumption.png", dpi = 600)

# reduc_plot = sns.histplot(df_reduc["Reduced"])
# plt.xlabel('Final reduced average dietry consumption [kg/c02/year]') 
# plt.savefig("../visualisations_output/Example_reduc_distrtibutions.png", dpi = 600)

# reduc_cdf = sns.ecdfplot(df_reduc["Reduced"])
# plt.ylabel('Final average dietry consumption [kg/c02/year]') 
# plt.xlabel('Reduced dietry consumption [kg/c02/year]')
# plt.savefig("../visualisations_output/Example_reduc_distrtibutions_ecdf.png", dpi = 600)





