# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:03:54 2023

@author: everall
"""
#%% Imports
import numpy as np
import seaborn as sns
import pandas as pd
#import model_main
import model_main_single as model_main
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from datetime import date
import time

#%% Setting parameters



params = {"veg_CO2": 1390,
          "meat_CO2": 2054,
          "N": 300,
          "erdos_p": 3,
          "steps":5000,
          "w_i": 5, #weight of the replicator function
          "immune_n": 0.1,
          "M": 4,
          "veg_f":0.6, #vegetarian fraction
          "meat_f": 0.4,  #meat eater fraciton
          "n": 5,
          "v": 10,
          'topology': "BA", #can either be barabasi albert with "BA", or fully connected with "complete"
          "alpha": 0.5,
          "beta": 0.5
          }

#%% Main functions

def run_model(params=params):

    test_model = model_main.Model(params)
    test_model.run()
    # end_state_A = test_model.get_attributes("C")
    # end_state_frac = test_model.get_attributes("threshold")
    #print(params)
    return test_model



def basic_run():

    pass


def parameter_sweep(params, param_ranges, num_iterations):
    """
    Runs multiple model simulations of the ABM while varying multiple parameters.
    
    Args:
        params (dict): The base dictionary of model parameters
        param_ranges (dict): Dictionary where keys are parameter names and values are lists of values to test
        num_iterations (int): Number of iterations per parameter combination
        
    Returns:
        pd.DataFrame: DataFrame containing results of all runs
    """
    results = []
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    total_combinations = len(param_combinations)
    
    for i, combination in enumerate(param_combinations):
        print(f"Processing combination {i+1}/{total_combinations}")
        
        # Update params with current combination
        current_params = params.copy()
        for name, value in zip(param_names, combination):
            current_params[name] = value
            if name == "veg_f":
                current_params["meat_f"] = 1 - value
            elif name == "meat_f":
                current_params["veg_f"] = 1 - value
        
        for iteration in range(num_iterations):
            test_model = run_model(current_params)
            
            # Store detailed results for each run
            run_results = {
                'run_id': f"{i}_{iteration}",
                'final_system_C': test_model.system_C[-1],
                'mean_system_C': np.mean(test_model.system_C),
                'std_system_C': np.std(test_model.system_C),
                'final_fraction_veg': test_model.fraction_veg[-1],
                'mean_fraction_veg': np.mean(test_model.fraction_veg),
                'std_fraction_veg': np.std(test_model.fraction_veg),
                'individual_reductions': test_model.get_attributes("reduction_out"),
                'topology': current_params['topology'],
                'system_C_trajectory': test_model.system_C,
                'fraction_veg_trajectory': test_model.fraction_veg
            }
            
            # Add all parameter values to results
            for name, value in current_params.items():
                run_results[name] = value
                
            results.append(run_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    today = date.today()
    date_str = today.strftime("%b_%d_%Y")
    filename = f'parameter_sweep_{date_str}_N_{params["N"]}_{"_".join(param_names)}'
    
    # Save full results as pickle for complete data preservation
    results_df.to_pickle(f'../model_output/{filename}.pkl')
    
    # Save a CSV with flattened data for basic analysis
    # Convert complex columns to string representation for CSV
    # csv_df = results_df.copy()
    # csv_df['individual_reductions'] = csv_df['individual_reductions'].apply(str)
    # csv_df['system_C_trajectory'] = csv_df['system_C_trajectory'].apply(str)
    # csv_df['fraction_veg_trajectory'] = csv_df['fraction_veg_trajectory'].apply(str)
    # csv_df.to_csv(f'../model_output/{filename}.csv', index=False)
    
    return results_df

def timer(func, *args):
    start = time.time()
    outputs = func(*args)
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime complete: {mins:5.0f} mins {sec}s\n')
    
    return outputs

#%% Default runs with trajectories

#%% Running sensitivity analysis


param_sweeps = ["alpha", "beta"]

# # Example usage:
# param_ranges_t = {
#     "veg_f": np.linspace(0.0, 1.0, 10),
# }

param_ranges = {i:np.linspace(0.0, 1.0, 5) for i in param_sweeps}

num_iterations = 3
results_df = timer(parameter_sweep, params, param_ranges, num_iterations)







