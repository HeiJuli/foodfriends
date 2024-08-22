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
import itertools
import pandas as pd
from datetime import date
import time

#%% Setting parameters

params = {"veg_CO2": 1390,
          "meat_CO2": 2054,
          "N": 200,
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
        params (dict): The base dictionary of model parameters.
        param_ranges (dict): A dictionary where keys are parameter names and values are lists of values to test.
        num_iterations (int): The number of iterations to run for each parameter combination.
        
    Returns:
        pd.DataFrame: A DataFrame containing the results of all runs.
    """
    # Generate all combinations of parameter values
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    
    runs = []
    total_combinations = len(param_combinations)
    
    for i, combination in enumerate(param_combinations):
        print(f"Processing combination {i+1}/{total_combinations}")
        
        # Update params with the current combination
        current_params = params.copy()
        for name, value in zip(param_names, combination):
            current_params[name] = value
            
            # Ensure that the fractions always sum to 1 if we're changing veg_f or meat_f
            if name == "veg_f":
                current_params["meat_f"] = 1 - value
            elif name == "meat_f":
                current_params["veg_f"] = 1 - value
        
        for _ in range(num_iterations):
            test_model = run_model(current_params)
            
            # Collect results
            result = [
                test_model.system_C[-1],  # Final system C
                test_model.fraction_veg[-1],  # Final fraction of vegetarians
                test_model.get_attributes("reduction_out")  # Individual agent emissions reduction
            ]
            result.extend(combination)  # Add the parameter values
            runs.append(result)
    
    # Create DataFrame
    columns = ['final_system_C', 'final_fraction_veg', 'individual_reduction'] + param_names
    df = pd.DataFrame(runs, columns=columns)
    
    # Save to CSV
    today = date.today()
    date_str = today.strftime("%b_%d_%Y")
    fname = f'../model_output/parameter_sweep_{date_str}_N_{params["N"]}_{"_".join(param_names)}.csv'
    df.to_csv(fname, index=False)
    
    return df


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

# Example usage:
param_ranges = {
    "veg_f": np.linspace(0.0, 1.0, 10),
}


num_iterations = 3
results_df = timer(parameter_sweep, params, param_ranges, num_iterations)


#%% Processesing data frames and Rough/Demo plots
#TODO: these sections will be put into a differnt script eventually

# Plot for final average dietary consumption vs. veg_f
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x="veg_f", y="final_system_C")
plt.xlabel('% vegans & vegetarians')
plt.ylabel('Final average dietary consumption [kg/CO2/year]')
plt.title('Final Consumption vs. Vegetarian Fraction')
plt.savefig("../visualisations_output/Example_consumption.png", dpi=300)
plt.show()

# Histogram of individual reductions
plt.figure(figsize=(10, 6))
sns.histplot(results_df["individual_reduction"].explode())
plt.xlabel('Final reduced average dietary consumption [kg/CO2/year]')
plt.title('Distribution of Individual Reductions')
plt.savefig("../visualisations_output/Example_reduc_distributions.png", dpi=300)
plt.show()

# IECDF
plt.figure(figsize=(10, 6))
sns.ecdfplot(results_df["individual_reduction"].explode())
plt.ylabel('Cumulative Probability')
plt.xlabel('Reduced dietary consumption [kg/CO2/year]')
plt.title('ECDF of Individual Reductions')
plt.savefig("../visualisations_output/Example_reduc_distributions_ecdf.png", dpi=600)
plt.show()

#%%% Other plotting ideas


# Heatmap for n and v (replace values)
if "n" in results_df.columns and "v" in results_df.columns:
    plt.figure(figsize=(12, 8))
    pivot_df = results_df.pivot(index="n", columns="v", values="final_system_C")
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title('Final Consumption for different n and v values')
    plt.xlabel('v')
    plt.ylabel('n')
    plt.savefig("../visualisations_output/heatmap_n_v.png", dpi=300)
    plt.show()

# Boxplot of final consumption for different parameter values
if "n" in results_df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="n", y="final_system_C", data=results_df)
    plt.xlabel('n')
    plt.ylabel('Final average dietary consumption [kg/CO2/year]')
    plt.title('Final Consumption Distribution for Different n Values')
    plt.savefig("../visualisations_output/boxplot_n_consumption.png", dpi=300)
    plt.show()

# Scatter plot with different colors for different parameter
if "v" in results_df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x="veg_f", y="final_system_C", hue="v", palette="viridis")
    plt.xlabel('% vegans & vegetarians')
    plt.ylabel('Final average dietary consumption [kg/CO2/year]')
    plt.title('Final Consumption vs. Vegetarian Fraction for Different v Values')
    plt.savefig("../visualisations_output/scatter_veg_f_consumption_v.png", dpi=300)
    plt.show()




