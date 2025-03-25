# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:25:00 2025

@author: emma.thill
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlined model running script for dietary contagion model
Focused only on specific analyses for publication plots
"""
import os
import numpy as np
import pandas as pd
import time
from datetime import date

# Import model modules
import model_main_single as model_main
from extended_survey_runs import (
    run_emissions_vs_vegetarian_fraction,
    run_parameter_sensitivity,
    analyze_cluster_formation
)

# Default model parameters
DEFAULT_PARAMS = {
    "veg_CO2": 1390,
    "meat_CO2": 2054,
    "N": 699,
    "erdos_p": 3,
    "steps": 5000,
    "w_i": 5,
    "immune_n": 0.1,
    "M": 5,
    "veg_f": 0.3,
    "meat_f": 0.7,
    "n": 5,
    "v": 10,
    'topology': "CSF",
    "alpha": 0.4,
    "beta": 0.6
}

def load_survey_data(filepath,variables_to_include):
    """
    Load survey file and filter only the needed variables.
    
    Args:
        filepath (str): Path to the survey CSV file.
        variables_to_include (lists): List of variables to include
        
    Returns:
        pd.DataFrame: Filtered survey data with only necessary columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Survey data file not found: {filepath}")
        
    survey_data = pd.read_csv(filepath)
    
    print(f"Loaded survey data with {survey_data.shape[0]} respondents and {survey_data.shape[1]} attributes")
    
    #Select only required columns
    filtered_data=survey_data[variables_to_include]
    
    return filtered_data
    

def ensure_output_dir():
    """Ensure model_output directory exists"""
    if not os.path.exists('../model_output'):
        os.makedirs('../model_output')

def timer(func, *args, **kwargs):
    """Time a function execution"""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    mins = int(elapsed / 60)
    secs = elapsed % 60
    print(f"Runtime: {mins} mins {secs:.1f}s")
    return result

def run_basic_model(params=None, survey_filepath=None):
    """Run a single model simulation with given parameters"""
    params = params or DEFAULT_PARAMS.copy()
    
    survey_data = None
    if survey_filepath:
        survey_data = load_survey_data(survey_filepath, ["nomem_encr","alpha", "beta", "theta", "diet"])
        # Ensure we have exactly `N` respondents for `N` agents
        if len(survey_data) < params["N"]:
           raise ValueError(f"Survey data has only {len(survey_data)} respondents, but model expects {params['N']} agents.")

    model = model_main.Model(params, survey_data=survey_data)
    model.run()
    return model

def run_emissions_analysis(params=None, num_runs=3, veg_fractions=None, survey_filepath=None):
    """
    Run CO2 emissions vs vegetarian fraction analysis
    
    Args:
        params (dict): Model parameters
        num_runs (int): Number of runs per vegetarian fraction
        veg_fractions (array): Array of vegetarian fractions to test
        
    Returns:
        pd.DataFrame: Results with veg_fraction and final_CO2 columns
    """
    params = DEFAULT_PARAMS.copy() if params is None else params
    if veg_fractions is None:
        veg_fractions = np.linspace(0, 1, 5)
    
    if survey_filepath:
        survey_params = load_survey_data(survey_filepath, ["nomem_encr","alpha", "beta", "theta"])
        params.update(survey_params)
    
    print(f"Running emissions analysis with {len(veg_fractions)} vegetarian fractions...")
    
    results_df = run_emissions_vs_vegetarian_fraction(
        params, 
        num_runs=num_runs, 
        veg_fractions=veg_fractions
    )
    
    # Save results
    ensure_output_dir()
    date_str = date.today().strftime("%Y%m%d")
    filename = f'emissions_{date_str}.pkl'
    results_df.to_pickle(f'../model_output/{filename}')
    print(f"Results saved to ../model_output/{filename}")
    
    return results_df

def run_tipping_point_analysis(params=None, alpha_range=None, beta_range=None, veg_fractions=None, survey_filepath=None):
    """
    Run parameter sensitivity analysis to find tipping points
    
    Args:
        params (dict): Model parameters
        alpha_range (array): Array of alpha values
        beta_range (array): Array of beta values
        veg_fractions (array): Array of initial vegetarian fractions
        
    Returns:
        pd.DataFrame: Combined results for all vegetarian fractions
    """
    params = DEFAULT_PARAMS.copy() if params is None else params
    if alpha_range is None:
        alpha_range = np.linspace(0.1, 0.9, 10)
    if beta_range is None:
        beta_range = np.linspace(0.1, 0.9, 10)
    if veg_fractions is None:
        veg_fractions = [0.2]
    
    if survey_filepath:
        survey_params = load_survey_data(survey_filepath, ["nomem_encr","diet","theta"])
        params.update(survey_params)
    
    print(f"Running tipping point analysis with {len(alpha_range)}x{len(beta_range)} parameter combinations...")
    
    all_results = []
    
    for veg_f in veg_fractions:
        print(f"Testing initial vegetarian fraction: {veg_f}")
        
        results = run_parameter_sensitivity(
            params,
            alpha_range=alpha_range,
            beta_range=beta_range,
            fixed_veg_f=veg_f
        )
        
        results['initial_veg_f'] = veg_f
        all_results.append(results)
    
    combined_df = pd.concat(all_results)
    
    # Save results
    ensure_output_dir()
    date_str = date.today().strftime("%Y%m%d")
    filename = f'tipping_all_{date_str}.pkl'
    combined_df.to_pickle(f'../model_output/{filename}')
    print(f"Results saved to ../model_output/{filename}")
    
    return combined_df

def run_veg_growth_analysis(params=None, veg_fractions=None, max_veg_fraction=1, survey_filepath=None):
    """
    Run simulations to analyze growth in vegetarian population
    
    Args:
        params (dict): Model parameters
        veg_fractions (array): Array of vegetarian fractions to test
        max_veg_fraction (float): Maximum vegetarian fraction to analyze
        
    Returns:
        pd.DataFrame: Results with initial and final vegetarian fractions
    """
    params = DEFAULT_PARAMS.copy() if params is None else params
    if veg_fractions is None:
        veg_fractions = np.linspace(0, max_veg_fraction, 10)
    if survey_filepath:
        survey_params = load_survey_data(survey_filepath, ["nomem_encr","alpha", "beta", "theta"])
        params.update(survey_params)
    
    # Filter fractions to respect max_veg_fraction
    veg_fractions = veg_fractions[veg_fractions <= max_veg_fraction]
    
    print(f"Running vegetarian growth analysis with {len(veg_fractions)} initial fractions...")
    
    results = []
    
    for veg_f in veg_fractions:
        print(f"Testing initial veg fraction: {veg_f:.2f}")
        
        # Run model with this vegetarian fraction
        test_params = params.copy()
        test_params["veg_f"] = veg_f
        test_params["meat_f"] = 1 - veg_f
        
        model = run_basic_model(test_params)
        
        # Store initial and final vegetarian fractions
        results.append({
            'initial_veg_fraction': veg_f,
            'final_veg_fraction': model.fraction_veg[-1]
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    ensure_output_dir()
    date_str = date.today().strftime("%Y%m%d")
    filename = f'veg_growth_{date_str}.pkl'
    results_df.to_pickle(f'../model_output/{filename}')
    print(f"Results saved to ../model_output/{filename}")
    
    return results_df


def run_parameter_sweep(params=None, alpha_range=None, beta_range=None, runs_per_combo=3, survey_filepath=None):
    """
    Run parameter sweep focusing on individual reductions attribution
    
    Args:
        params (dict): Model parameters
        alpha_range (array): Array of alpha values
        beta_range (array): Array of beta values
        runs_per_combo (int): Number of runs per parameter combination
        
    Returns:
        pd.DataFrame: Results with individual_reductions column
    """
    params = DEFAULT_PARAMS.copy() if params is None else params
    if alpha_range is None:
        alpha_range = np.linspace(0.1, 0.9, 5)
    if beta_range is None:
        beta_range = np.linspace(0.1, 0.9, 5)
    
    if survey_filepath:
        survey_params = load_survey_data(survey_filepath, ["nomem_encr","diet","theta"])
        params.update(survey_params)
    
    print(f"Running parameter sweep with {len(alpha_range)}x{len(beta_range)} combinations...")
    
    results = []
    total_runs = len(alpha_range) * len(beta_range) * runs_per_combo
    run_count = 0
    
    for alpha in alpha_range:
        for beta in beta_range:
            run_params = params.copy()
            run_params["alpha"] = alpha
            run_params["beta"] = beta
            
            for run in range(runs_per_combo):
                run_count += 1
                print(f"Run {run_count}/{total_runs}: α={alpha:.2f}, β={beta:.2f}, run {run+1}/{runs_per_combo}")
                
                model = run_basic_model(run_params)
                
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'run': run,
                    'final_veg_fraction': model.fraction_veg[-1],
                    'individual_reductions': model.get_attributes("reduction_out")
                })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    ensure_output_dir()
    date_str = date.today().strftime("%Y%m%d")
    filename = f'parameter_sweep_{date_str}.pkl'
    results_df.to_pickle(f'../model_output/{filename}')
    print(f"Results saved to ../model_output/{filename}")
    
    return results_df

def run_3d_parameter_analysis(params=None, alpha_range=None, beta_range=None, veg_fractions=None, runs_per_combo=3, survey_filepath=None):
    """
    Run a 3D parameter analysis varying alpha, beta, and initial vegetarian fraction
    
    Args:
        params (dict): Base model parameters
        alpha_range (array): Array of alpha values to test
        beta_range (array): Array of beta values to test
        veg_fractions (array): Array of initial vegetarian fractions to test
        runs_per_combo (int): Number of runs per parameter combination
        
    Returns:
        pd.DataFrame: Results with parameter combinations and outcomes
    """
    params = DEFAULT_PARAMS.copy() if params is None else params
    if alpha_range is None:
        alpha_range = np.linspace(0.1, 0.9, 5)
    if beta_range is None:
        beta_range = np.linspace(0.1, 0.9, 5)
    if veg_fractions is None:
        veg_fractions = np.linspace(0.1, 0.9, 5)
    
    if survey_filepath:
        survey_params = load_survey_data(survey_filepath, ["nomem_encr","theta"])
        params.update(survey_params)
    
    print(f"Running 3D parameter analysis with {len(alpha_range)}x{len(beta_range)}x{len(veg_fractions)} combinations...")
    
    results = []
    total_runs = len(alpha_range) * len(beta_range) * len(veg_fractions) * runs_per_combo
    run_count = 0
    
    for alpha in alpha_range:
        for beta in beta_range:
            for veg_f in veg_fractions:
                # Update parameters
                test_params = params.copy()
                test_params["alpha"] = alpha
                test_params["beta"] = beta
                test_params["veg_f"] = veg_f
                test_params["meat_f"] = 1.0 - veg_f
                
                for run in range(runs_per_combo):
                    run_count += 1
                    print(f"Run {run_count}/{total_runs}: α={alpha:.2f}, β={beta:.2f}, veg_f={veg_f:.2f}, run {run+1}/{runs_per_combo}")
                    
                    # Run model
                    model = model_main.Model(test_params)
                    model.run()
                    
                    # Record results
                    results.append({
                        'alpha': alpha,
                        'beta': beta,
                        'initial_veg_f': veg_f,
                        'final_veg_f': model.fraction_veg[-1],
                        'change': model.fraction_veg[-1] - veg_f,
                        'tipped': model.fraction_veg[-1] > (veg_f * 1.2),  # 20% increase threshold
                        'final_CO2': model.system_C[-1]
                    })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    ensure_output_dir()
    date_str = date.today().strftime("%Y%m%d")
    filename = f'3d_parameter_analysis_{date_str}.pkl'
    results_df.to_pickle(f'../model_output/{filename}')
    print(f"Results saved to ../model_output/{filename}")
    
    return results_df

def run_trajectory_analysis(params=None, alpha_values=None, beta_values=None, runs_per_combo=3, survey_filepath=None):
    """
    Run simulations and save full trajectories for different parameter combinations
    
    Args:
        params (dict): Base model parameters
        alpha_values (list): Alpha values to test
        beta_values (list): Beta values to test
        runs_per_combo (int): Number of runs per parameter combination
        
    Returns:
        pd.DataFrame: Results with trajectory data
    """
    params = DEFAULT_PARAMS.copy() if params is None else params
    if alpha_values is None:
        alpha_values = [0.25, 0.5, 0.75]
    if beta_values is None:
        beta_values = [0.25, 0.5, 0.75]
    
    if survey_filepath:
        survey_params = load_survey_data(survey_filepath, ["diet","theta"])
        params.update(survey_params)
    
    print(f"Running trajectory analysis with {len(alpha_values)}x{len(beta_values)} parameter combinations...")
    
    results = []
    total_runs = len(alpha_values) * len(beta_values) * runs_per_combo
    run_count = 0
    
    for alpha in alpha_values:
        for beta in beta_values:
            # Update parameters
            test_params = params.copy()
            test_params["alpha"] = alpha
            test_params["beta"] = beta
            test_params["steps"] = 5000  # Ensure enough steps for meaningful trajectories
            
            for run in range(runs_per_combo):
                run_count += 1
                print(f"Run {run_count}/{total_runs}: α={alpha:.2f}, β={beta:.2f}, run {run+1}/{runs_per_combo}")
                
                # Run model
                model = model_main.Model(test_params)
                model.run()
                
                # Record results including full trajectories
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'initial_veg_f': test_params["veg_f"],
                    'final_veg_f': model.fraction_veg[-1],
                    'fraction_veg_trajectory': model.fraction_veg,
                    'system_C_trajectory': model.system_C,
                    'run': run
                })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    ensure_output_dir()
    date_str = date.today().strftime("%Y%m%d")
    filename = f'trajectory_analysis_{date_str}.pkl'
    results_df.to_pickle(f'../model_output/{filename}')
    print(f"Results saved to ../model_output/{filename}")
    
    return results_df
def main():
    """Main function with simple menu for analysis selection - no parameter inputs"""
    print("===== Streamlined Dietary Contagion Model Analysis =====")
    
    while True:
        print("\nAnalysis Options:")
        print("[1] Emissions vs Vegetarian Fraction")
        print("[2] Tipping Point Analysis (alpha-beta heatmap)")
        print("[3] Vegetarian Growth Analysis")
        print("[4] Parameter Sweep (individual reductions)")
        print("[5] 3D Parameter Analysis (alpha, beta, veg fraction)")
        print("[6] Trajectory Analysis (for trajectory grid plots)")
        print("[0] Exit")
        
        choice = input("\nSelect option: ")
        
        if choice == '1':
            # Fixed parameters - edit these directly in the code if needed
            survey_filepath = "final_data_parameters.csv"
            veg_fractions = np.linspace(0, 1, 5)
            num_runs = 3
            print(f"Running with {len(veg_fractions)} vegetarian fractions, {num_runs} runs each")
            timer(run_emissions_analysis, num_runs=num_runs, veg_fractions=veg_fractions, survey_filepath=survey_filepath)
            
        elif choice == '2':
            survey_filepath = "final_data_parameters.csv"
            # Fixed parameters
            alpha_range = np.linspace(0.1, 0.9, 5)
            beta_range = np.linspace(0.1, 0.9, 5)
            veg_fractions = [0.2]
            print(f"Running with alpha range {alpha_range[0]:.1f}-{alpha_range[-1]:.1f}, beta range {beta_range[0]:.1f}-{beta_range[-1]:.1f}")
            print(f"Initial vegetarian fraction: {veg_fractions[0]}")
            
            timer(run_tipping_point_analysis, 
                 alpha_range=alpha_range, 
                 beta_range=beta_range,
                 veg_fractions=veg_fractions,
                 survey_filepath=survey_filepath)
            
        elif choice == '3':
            # Fixed parameters
            survey_filepath = "final_data_parameters.csv"
            max_veg_fraction = 1
            veg_fractions = np.linspace(0, 1, 5)
            print(f"Running with vegetarian fractions from {veg_fractions[0]:.1f} to {veg_fractions[-1]:.1f}")
            
            timer(run_veg_growth_analysis, 
                 veg_fractions=veg_fractions,
                 max_veg_fraction=max_veg_fraction,
                 survey_filepath=survey_filepath)
            
        elif choice == '4':
            survey_filepath = "final_data_parameters.csv"
            # Fixed parameters
            alpha_range = np.linspace(0.1, 0.9, 5)
            beta_range = np.linspace(0.1, 0.9, 5)
            runs_per_combo = 3
            print(f"Running {len(alpha_range)}×{len(beta_range)} parameter combinations, {runs_per_combo} runs each")
            
            timer(run_parameter_sweep, 
                 alpha_range=alpha_range, 
                 beta_range=beta_range,
                 runs_per_combo=runs_per_combo,
                 survey_filepath=survey_filepath)
                 
        elif choice == '5':
            survey_filepath = "final_data_parameters.csv"
            # Fixed parameters
            alpha_range = np.linspace(0.1, 0.9, 4)
            beta_range = np.linspace(0.1, 0.9, 4)
            veg_fractions = np.linspace(0.1, 0.5, 3)
            runs_per_combo = 2
            print(f"Running 3D analysis with {len(alpha_range)}×{len(beta_range)}×{len(veg_fractions)} parameter combinations")
            
            timer(run_3d_parameter_analysis,
                 alpha_range=alpha_range,
                 beta_range=beta_range,
                 veg_fractions=veg_fractions,
                 runs_per_combo=runs_per_combo,
                 survey_filepath=survey_filepath)
                 
        elif choice == '6':
            survey_filepath = "final_data_parameters.csv"
            # Fixed parameters
            alpha_values = [0.25, 0.5, 0.75]
            beta_values = [0.25, 0.5, 0.75]
            runs_per_combo = 5
            print(f"Running trajectory analysis with {len(alpha_values)}×{len(beta_values)} parameter combinations")
            
            timer(run_trajectory_analysis,
                 alpha_values=alpha_values,
                 beta_values=beta_values,
                 runs_per_combo=runs_per_combo,
                 survey_filepath=survey_filepath)
            
        elif choice == '0':
            break
        else:
            print("Invalid option")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()