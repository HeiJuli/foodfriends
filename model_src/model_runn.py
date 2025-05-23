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
from extended_model_runner import (
    run_emissions_vs_vegetarian_fraction,
    run_parameter_sensitivity,
    analyze_cluster_formation
)

# Default model parameters
DEFAULT_PARAMS = {
    "veg_CO2": 1390,
    "meat_CO2": 2054,
    "N": 300,
    "erdos_p": 3,
    "steps": 35000,
    "w_i": 5,
    "immune_n": 0.1,
    "M": 5,
    "veg_f": 0.15,
    "meat_f": 0.85,
    "n": 5,
    "v": 10,
    'topology': "CSF",
    "alpha": 0.4,
    "beta": 0.6,
    "agent_ini": "synthetic",  # Default to synthetic initialization
    "survey_file": "../data/final_data_parameters.csv"
}

def add_survey_params_to_analysis(data, a=None, b=None, t=None, v=None, mv=None):
    """Add survey params to analysis parameter spaces"""
    if a is not None and 'alpha' in data.columns:
        sa = data['alpha'].mean()
        if sa not in a: a = np.sort(np.append(a, sa))
    if b is not None and 'beta' in data.columns:
        sb = data['beta'].mean()
        if sb not in b: b = np.sort(np.append(b, sb))
    if t is not None and 'theta' in data.columns:
        st = data['theta'].mean()
        if st not in t: t = np.sort(np.append(t, st))
    if v is not None and 'diet' in data.columns:
        sv = (data['diet'] == 'veg').sum() / len(data)
        if (mv is None or sv <= mv) and sv not in v:
            v = np.sort(np.append(v, sv))
    return a, b, t, v

def load_survey_data(filepath, variables_to_include):
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
    
    # Select only required columns
    filtered_data = survey_data[variables_to_include]
    
    return filtered_data

def extract_survey_parameters(survey_data):
    """
    Extract statistical parameters from survey data for parameterized initialization.
    
    Args:
        survey_data (DataFrame): Survey data with parameter columns
        
    Returns:
        dict: Dictionary with parameter distributions
    """
    params = {}
    
    # Calculate means for parameters of interest
    if 'alpha' in survey_data.columns:
        params['alpha'] = survey_data['alpha'].mean()
    if 'beta' in survey_data.columns:
        params['beta'] = survey_data['beta'].mean()
    if 'theta' in survey_data.columns:
        params['theta'] = survey_data['theta'].mean()
    
    # Calculate diet distribution
    if 'diet' in survey_data.columns:
        veg_count = (survey_data['diet'] == 'veg').sum()
        total_count = len(survey_data)
        params['veg_f'] = veg_count / total_count
        params['meat_f'] = 1 - params['veg_f']
    
    return params

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

def run_basic_model(params=None):
    """Run a single model simulation with given parameters"""
    params = params or DEFAULT_PARAMS.copy()
    
    # Handle survey data if in parameterized mode
    if params["agent_ini"] == "parameterized":
        # For parameterized mode, extract statistical parameters from survey
        survey_data = load_survey_data(params["survey_file"], ["nomem_encr", "alpha", "beta", "theta", "diet"])
        survey_params = extract_survey_parameters(survey_data)
        params.update(survey_params)
    
    model = model_main.Model(params)
    model.run()
    return model

def run_emissions_analysis(params=None, num_runs=3, veg_fractions=None):
    """
    Run CO2 emissions vs vegetarian fraction analysis
    
    Args:
        params (dict): Model parameters
        num_runs (int): Number of runs per parameter combination
        veg_fractions (array): Array of vegetarian fractions to test
        
    Returns:
        pd.DataFrame: Results with veg_fraction and final_CO2 columns
    """
    params = DEFAULT_PARAMS.copy() if params is None else params
    if veg_fractions is None:
        veg_fractions = np.linspace(0, 1, 20)
    
    # Handle survey data if in parameterized mode
    if params["agent_ini"] == "parameterized":
        survey_data = load_survey_data(params["survey_file"], ["nomem_encr", "alpha", "beta", "theta", "diet"])
        
        # Extract parameters for simulation
        survey_params = extract_survey_parameters(survey_data)
        params.update(survey_params)
        
        # Add survey vegetarian fraction to the analysis
        _, _, _, veg_fractions = add_survey_params_to_analysis(survey_data, v=veg_fractions)
    
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

def run_tipping_point_analysis(params=None, alpha_range=None, beta_range=None, theta_range=None, veg_fractions=None):
    """
    Run parameter sensitivity analysis to find tipping points
    
    Args:
        params (dict): Model parameters
        alpha_range (array): Array of alpha values
        beta_range (array): Array of beta values
        theta_range (array): Array of theta values
        veg_fractions (array): Array of initial vegetarian fractions
        
    Returns:
        pd.DataFrame: Combined results for all vegetarian fractions
    """
    params = DEFAULT_PARAMS.copy() if params is None else params
    if alpha_range is None:
        alpha_range = np.linspace(0.1, 0.9, 10)
    if beta_range is None:
        beta_range = np.linspace(0.1, 0.9, 10)
    if theta_range is None:
        theta_range = [-0.5, 0, 0.5]
    if veg_fractions is None:
        veg_fractions = [0.2]
    
    # If in parameterized mode, include survey-derived parameters in the sweep
    if params["agent_ini"] == "parameterized":
        survey_data = load_survey_data(params["survey_file"], ["nomem_encr", "alpha", "beta", "theta", "diet"])
        
        # Add survey parameters to the analysis ranges
        alpha_range, beta_range, _, veg_fractions = add_survey_params_to_analysis(
            survey_data, 
            a=alpha_range, 
            b=beta_range, 
            v=veg_fractions
        )
    
    print(f"Running tipping point analysis with {len(alpha_range)}x{len(beta_range)}x{len(theta_range)} parameter combinations...")
    
    all_results = []
    
    for veg_f in veg_fractions:
        print(f"Testing initial vegetarian fraction: {veg_f}")
        
        for theta in theta_range:
            print(f"  Testing theta value: {theta}")
            theta_params = params.copy()
            theta_params["theta"] = theta
            
            results = run_parameter_sensitivity(
                theta_params,
                alpha_range=alpha_range,
                beta_range=beta_range,
                fixed_veg_f=veg_f
            )
            
            results['initial_veg_f'] = veg_f
            results['theta'] = theta
            all_results.append(results)
    
    combined_df = pd.concat(all_results)
    
    # Save results
    ensure_output_dir()
    date_str = date.today().strftime("%Y%m%d")
    filename = f'tipping_all_{date_str}.pkl'
    combined_df.to_pickle(f'../model_output/{filename}')
    print(f"Results saved to ../model_output/{filename}")
    
    return combined_df

def run_veg_growth_analysis(params=None, veg_fractions=None, max_veg_fraction=0.6):
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
        veg_fractions = np.linspace(0.1, max_veg_fraction, 10)
    
    # If in parameterized mode, include survey-derived veg_fraction in the analysis
    if params["agent_ini"] == "parameterized":
        survey_data = load_survey_data(params["survey_file"], ["nomem_encr", "alpha", "beta", "theta", "diet"])
        
        # Extract theta, alpha and beta for the simulation parameters
        survey_params = {}
        if 'alpha' in survey_data.columns:
            survey_params['alpha'] = survey_data['alpha'].mean()
        if 'beta' in survey_data.columns:
            survey_params['beta'] = survey_data['beta'].mean()
        if 'theta' in survey_data.columns:
            survey_params['theta'] = survey_data['theta'].mean()
            
        # Update sim parameters
        params.update(survey_params)
        
        # Add survey vegetarian fraction to the analysis
        _, _, _, veg_fractions = add_survey_params_to_analysis(
            survey_data, 
            v=veg_fractions,
            mv=max_veg_fraction
        )
    
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

def run_parameter_sweep(params=None, alpha_range=None, beta_range=None, theta_range=None, runs_per_combo=3):
    """
    Run parameter sweep focusing on individual reductions attribution
    
    Args:
        params (dict): Model parameters
        alpha_range (array): Array of alpha values
        beta_range (array): Array of beta values
        theta_range (array): Array of theta values
        runs_per_combo (int): Number of runs per parameter combination
        
    Returns:
        pd.DataFrame: Results with individual_reductions column
    """
    params = DEFAULT_PARAMS.copy() if params is None else params
    if alpha_range is None:
        alpha_range = np.linspace(0.1, 0.9, 5)
    if beta_range is None:
        beta_range = np.linspace(0.1, 0.9, 5)
    if theta_range is None:
        theta_range = [-0.5, 0, 0.5]
    
    # If in parameterized mode, include survey-derived parameters in the sweep
    if params["agent_ini"] == "parameterized":
        survey_data = load_survey_data(params["survey_file"], ["nomem_encr", "alpha", "beta", "theta", "diet"])
        
        # Add survey parameters to the analysis ranges
        alpha_range, beta_range, _, _ = add_survey_params_to_analysis(
            survey_data, 
            a=alpha_range, 
            b=beta_range
        )
    
    print(f"Running parameter sweep with {len(alpha_range)}x{len(beta_range)}x{len(theta_range)} combinations...")
    
    results = []
    total_runs = len(alpha_range) * len(beta_range) * len(theta_range) * runs_per_combo
    run_count = 0
    
    for alpha in alpha_range:
        for beta in beta_range:
            for theta in theta_range:
                run_params = params.copy()
                run_params["alpha"] = alpha
                run_params["beta"] = beta
                run_params["theta"] = theta
                
                for run in range(runs_per_combo):
                    run_count += 1
                    print(f"Run {run_count}/{total_runs}: α={alpha:.2f}, β={beta:.2f}, θ={theta:.2f}, run {run+1}/{runs_per_combo}")
                    
                    model = run_basic_model(run_params)
                    
                    results.append({
                        'alpha': alpha,
                        'beta': beta,
                        'theta': theta,
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

def run_3d_parameter_analysis(params=None, alpha_range=None, beta_range=None, theta_range=None, veg_fractions=None, runs_per_combo=3):
    """
    Run a 3D parameter analysis varying alpha, beta, theta, and initial vegetarian fraction
    
    Args:
        params (dict): Base model parameters
        alpha_range (array): Array of alpha values to test
        beta_range (array): Array of beta values to test
        theta_range (array): Array of theta values to test
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
    if theta_range is None:
        theta_range = [-0.5, 0, 0.5]
    if veg_fractions is None:
        veg_fractions = np.linspace(0.1, 0.9, 5)
    
    # If in parameterized mode, include survey-derived parameters in the sweep
    if params["agent_ini"] == "parameterized":
        survey_data = load_survey_data(params["survey_file"], ["nomem_encr", "alpha", "beta", "theta", "diet"])
        
        # Add survey parameters to the analysis ranges
        alpha_range, beta_range, _, veg_fractions = add_survey_params_to_analysis(
            survey_data, 
            a=alpha_range, 
            b=beta_range, 
            v=veg_fractions
        )
    
    print(f"Running 3D parameter analysis with {len(alpha_range)}x{len(beta_range)}x{len(theta_range)}x{len(veg_fractions)} combinations...")
    
    results = []
    total_runs = len(alpha_range) * len(beta_range) * len(theta_range) * len(veg_fractions) * runs_per_combo
    run_count = 0
    
    for alpha in alpha_range:
        for beta in beta_range:
            for theta in theta_range:
                for veg_f in veg_fractions:
                    # Update parameters
                    test_params = params.copy()
                    test_params["alpha"] = alpha
                    test_params["beta"] = beta
                    test_params["theta"] = theta
                    test_params["veg_f"] = veg_f
                    test_params["meat_f"] = 1.0 - veg_f
                    
                    for run in range(runs_per_combo):
                        run_count += 1
                        print(f"Run {run_count}/{total_runs}: α={alpha:.2f}, β={beta:.2f}, θ={theta:.2f}, veg_f={veg_f:.2f}, run {run+1}/{runs_per_combo}")
                        
                        # Run model
                        model = model_main.Model(test_params)
                        model.run()
                        
                        # Record results
                        results.append({
                            'alpha': alpha,
                            'beta': beta,
                            'theta': theta,
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

def run_trajectory_analysis(params=None, alpha_values=None, beta_values=None, theta_values=None, runs_per_combo=3):
    """Run simulations and save trajectories for different agent modes"""
    p = DEFAULT_PARAMS.copy() if params is None else params.copy()
    a = [0.25, 0.5, 0.75] if alpha_values is None else alpha_values
    b = [0.25, 0.5, 0.75] if beta_values is None else beta_values
    t = [-0.5, 0, 0.5] if theta_values is None else theta_values  # Default theta values
    r = []
    
    # Load survey data once if needed
    sd = None
    if any(m in [p.get("agent_ini"), "parameterized", "twin"] for m in ["parameterized", "twin"]):
        sd = load_survey_data(p["survey_file"], ["nomem_encr", "alpha", "beta", "theta", "diet"])
        
        veg_f, meat_f = extract_survey_parameters(sd)["veg_f"], extract_survey_parameters(sd)["meat_f"]
        
        sm = {'alpha': sd['alpha'].mean(), 'beta': sd['beta'].mean(), 'theta': sd['theta'].mean(), \
              "veg_f": veg_f, "meat_f": meat_f}
            

    
    # 1. Synthetic mode
    syn = p.copy()
    syn["agent_ini"] = "synthetic"
    total_combos = len(a) * len(b) * len(t)
    combo_count = 0
    print(f"Running synthetic mode: {total_combos} parameter combinations * {runs_per_combo} runs each")
    for ax in a:
        for bx in b:
            for tx in t:  # Added loop for theta values
                combo_count += 1
                print(f"Synthetic combo {combo_count}/{total_combos}: α={ax:.2f}, β={bx:.2f}, θ={tx:.2f}")
                syn.update({"alpha": ax, "beta": bx, "theta": tx})
                for i in range(runs_per_combo):
                    try:
                        m = model_main.Model(syn)
                        m.run()
                        r.append({
                            'agent_ini': "synthetic", 'alpha': ax, 'beta': bx, 'theta': tx,
                            'initial_veg_f': syn["veg_f"], 'final_veg_f': m.fraction_veg[-1],
                            'fraction_veg_trajectory': m.fraction_veg, 'system_C_trajectory': m.system_C,
                            'run': i, 'parameter_set': f"α={ax:.2f}, β={bx:.2f}, θ={tx:.2f}"
                        })
                    except Exception as e: print(f"Error in synthetic run: {e}")
    
    # 2+3. Survey-based modes
    if sd is not None:
        # Parameterized mode
        print(f"Running parameterized mode: {runs_per_combo} runs with survey means")
        par = p.copy()
        par.update({"agent_ini": "parameterized", **sm})
        for i in range(runs_per_combo):
            print(f"Parameterized run {i+1}/{runs_per_combo}")
            try:
                m = model_main.Model(par)
                m.run()
                r.append({
                    'agent_ini': "parameterized", **sm, 'initial_veg_f': par["veg_f"], 
                    'final_veg_f': m.fraction_veg[-1], 'fraction_veg_trajectory': m.fraction_veg,
                    'system_C_trajectory': m.system_C, 'run': i, 
                    'parameter_set': "Survey Mean Parameters"
                })
            except Exception as e: print(f"Error in parameterized run: {e}")
        
        # Twin mode
        print(f"Running twin mode: {runs_per_combo} runs with individual parameters from {len(sd)} survey respondents")
        twn = p.copy()
        twn.update({"agent_ini": "twin", "N": len(sd)})
        for i in range(runs_per_combo):
            print(f"Twin run {i+1}/{runs_per_combo}")
            try:
                m = model_main.Model(twn)
                m.run()
                agent_means = {k: np.mean([getattr(ag, k) for ag in m.agents]) for k in ['alpha', 'beta', 'theta']}
                r.append({
                    'agent_ini': "twin", **agent_means,
                    'initial_veg_f': sum(ag.diet=="veg" for ag in m.agents)/len(m.agents),
                    'final_veg_f': m.fraction_veg[-1],
                    'fraction_veg_trajectory': m.fraction_veg,
                    'system_C_trajectory': m.system_C,
                    'run': i, 'parameter_set': "Survey Individual Parameters"
                })
            except Exception as e: print(f"Error in twin run: {e}")
    
    # Save results
    df = pd.DataFrame(r)
    ensure_output_dir()
    fn = f'trajectory_analysis_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(f'../model_output/{fn}')
    print(f"Saved {len(r)} trajectories to ../model_output/{fn}")
    return df

def main():
    """Main function with simple menu for analysis selection - no parameter inputs"""
    print("===== Streamlined Dietary Contagion Model Analysis =====")
    
    # Ask for agent initialization mode
    print("\nAgent Initialization Modes:")
    print("[1] Synthetic (random parameters)")
    print("[2] Parameterized (use survey statistics)")
    print("[3] Twin (1-to-1 with survey data)")
    
    init_choice = input("\nSelect initialization mode (1-3, default is 1): ")
    
    # Set agent initialization mode
    agent_ini_mode = "synthetic"
    if init_choice == '2':
        agent_ini_mode = "parameterized"
    elif init_choice == '3':
        agent_ini_mode = "twin"
        
    # Update default parameters with chosen initialization mode
    params = DEFAULT_PARAMS.copy()
    params["agent_ini"] = agent_ini_mode
    
    # If using survey-based modes, check for survey file
    if agent_ini_mode in ["parameterized", "twin"]:
        survey_file = input(f"\nEnter survey file path (default: {params['survey_file']}): ")
        if survey_file:
            params["survey_file"] = survey_file
        print(f"Using survey data from: {params['survey_file']}")
    
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
            veg_fractions = np.linspace(0, 1, 5)
            num_runs = 3
            print(f"Running with {len(veg_fractions)} vegetarian fractions, {num_runs} runs each")
            timer(run_emissions_analysis, params=params, num_runs=num_runs, veg_fractions=veg_fractions)
            
        elif choice == '2':
            # Fixed parameters
            alpha_range = np.linspace(0.1, 0.9, 5)
            beta_range = np.linspace(0.1, 0.9, 5)
            veg_fractions = [0.2]
            print(f"Running with alpha range {alpha_range[0]:.1f}-{alpha_range[-1]:.1f}, beta range {beta_range[0]:.1f}-{beta_range[-1]:.1f}")
            print(f"Initial vegetarian fraction: {veg_fractions[0]}")
            
            timer(run_tipping_point_analysis, 
                 params=params,
                 alpha_range=alpha_range, 
                 beta_range=beta_range,
                 veg_fractions=veg_fractions)
            
        elif choice == '3':
            # Fixed parameters
            max_veg_fraction = 0.6
            veg_fractions = np.linspace(0.1, max_veg_fraction, 5)
            print(f"Running with vegetarian fractions from {veg_fractions[0]:.1f} to {veg_fractions[-1]:.1f}")
            
            timer(run_veg_growth_analysis, 
                 params=params,
                 veg_fractions=veg_fractions,
                 max_veg_fraction=max_veg_fraction)
            
        elif choice == '4':
            # Fixed parameters
            alpha_range = np.linspace(0.1, 0.9, 5)
            beta_range = np.linspace(0.1, 0.9, 5)
            runs_per_combo = 3
            print(f"Running {len(alpha_range)}×{len(beta_range)} parameter combinations, {runs_per_combo} runs each")
            
            timer(run_parameter_sweep, 
                 params=params,
                 alpha_range=alpha_range, 
                 beta_range=beta_range,
                 runs_per_combo=runs_per_combo)
                 
        elif choice == '5':
            # Fixed parameters
            alpha_range = np.linspace(0.1, 0.9, 4)
            beta_range = np.linspace(0.1, 0.9, 4)
            veg_fractions = np.linspace(0.1, 0.5, 3)
            runs_per_combo = 2
            print(f"Running 3D analysis with {len(alpha_range)}×{len(beta_range)}×{len(veg_fractions)} parameter combinations")
            
            timer(run_3d_parameter_analysis,
                 params=params,
                 alpha_range=alpha_range,
                 beta_range=beta_range,
                 veg_fractions=veg_fractions,
                 runs_per_combo=runs_per_combo)
                 
        elif choice == '6':
            # Fixed parameters
            alpha_values = [0.25, 0.5, 0.75]
            beta_values = [0.25, 0.5, 0.75]
            runs_per_combo = 5
            print(f"Running trajectory analysis with {len(alpha_values)}×{len(beta_values)} parameter combinations")
            
            timer(run_trajectory_analysis,
                 params=params,
                 alpha_values=alpha_values,
                 beta_values=beta_values,
                 runs_per_combo=runs_per_combo)
            
        elif choice == '0':
            break
        else:
            print("Invalid option")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()