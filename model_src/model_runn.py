# -*- coding: utf-8 -*-
"""
Extended version of model_runn.py focusing only on running simulations
and saving outputs to model_output directory
"""
#%% Imports
import numpy as np
import pandas as pd
import model_main_single as model_main
import itertools
from datetime import date
import time
import os
import pickle

# Import simulation functions
from extended_model_runner import (
    run_emissions_vs_vegetarian_fraction,
    run_parameter_sensitivity,
    run_topology_comparison,
    run_targeted_interventions,
    run_critical_dynamics,
    analyze_cluster_formation
)

#%% Setting parameters
params = {"veg_CO2": 1390,
          "meat_CO2": 2054,
          "N": 500,
          "erdos_p": 3,
          "steps": 5000,
          "w_i": 5, #weight of the replicator function
          "immune_n": 0.1,
          "M": 10,
          "veg_f": 0.3, #vegetarian fraction
          "meat_f": 0.7,  #meat eater fraciton
          "n": 5,
          "v": 10,
          'topology': "CSF", #can either be barabasi albert with "BA", or fully connected with "complete"
          "alpha": 0.5,
          "beta": 0.5
          }

#%% Original functions from model_runn.py

def run_model(params=params):
    """Original run_model function"""
    test_model = model_main.Model(params)
    test_model.run()
    return test_model

def parameter_sweep(params, param_ranges, num_iterations):
    """Original parameter_sweep function"""
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
    
    return results_df

def timer(func, *args):
    """Original timer function"""
    start = time.time()
    outputs = func(*args)
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime complete: {mins:5.0f} mins {sec}s\n')
    
    return outputs

#%% New integrated functions - SIMULATION ONLY

def ensure_output_dir():
    """Ensure model_output directory exists"""
    if not os.path.exists('../model_output'):
        os.makedirs('../model_output')

def run_emissions_analysis(save_prefix="emissions"):
    """
    Run CO2 emissions vs vegetarian fraction analysis and save results
    """
    print("Running emissions vs vegetarian fraction analysis...")
    ensure_output_dir()
    
    results_df = run_emissions_vs_vegetarian_fraction(
        params, 
        num_runs=3, 
        veg_fractions=np.linspace(0, 1, 20)
    )
    
    # Save results
    today = date.today()
    date_str = today.strftime("%b_%d_%Y")
    filename = f'{save_prefix}_{date_str}.pkl'
    results_df.to_pickle(f'../model_output/{filename}')
    
    print(f"Results saved to ../model_output/{filename}")
    return results_df

def run_tipping_point_analysis(save_prefix="tipping"):
    """
    Run parameter sensitivity analysis to find tipping points
    """
    print("Running tipping point analysis...")
    ensure_output_dir()
    
    # Define parameter ranges
    alpha_range = np.linspace(0.1, 0.9, 10)
    beta_range = np.linspace(0.1, 0.9, 10)
    
    # Run for multiple initial vegetarian fractions
    
    veg_fractions = np.linspace(0.1, 0.6, num = 10) 
    
    all_results = []
    
    for veg_f in veg_fractions:
        print(f"Testing initial vegetarian fraction: {veg_f}")
        
        # Run parameter sensitivity
        results = run_parameter_sensitivity(
            params,
            alpha_range=alpha_range,
            beta_range=beta_range,
            fixed_veg_f=veg_f
        )
        
        # Add vegetarian fraction to results
        results['initial_veg_f'] = veg_f
        
        # Save individual results
        individual_filename = f'{save_prefix}_veg{veg_f}_{date.today().strftime("%b_%d_%Y")}.pkl'
        results.to_pickle(f'../model_output/{individual_filename}')
        
        all_results.append(results)
    
    # Combine all results
    combined_df = pd.concat(all_results)
    
    # Save combined results
    combined_filename = f'{save_prefix}_all_{date.today().strftime("%b_%d_%Y")}.pkl'
    combined_df.to_pickle(f'../model_output/{combined_filename}')
    
    print(f"Combined results saved to ../model_output/{combined_filename}")
    return combined_df

def run_network_topology_comparison(save_prefix="topology"):
    """
    Compare different network topologies
    """
    print("Running network topology comparison...")
    ensure_output_dir()
    
    # Define topologies and vegetarian fractions
    topologies = ["CSF", "complete"]
    veg_fractions = np.linspace(0.1, 0.5, 9)
    
    # Run comparison
    results = run_topology_comparison(
        params,
        topologies=topologies,
        veg_fractions=veg_fractions,
        runs_per_config=3
    )
    
    # Save results
    filename = f'{save_prefix}_{date.today().strftime("%b_%d_%Y")}.pkl'
    results.to_pickle(f'../model_output/{filename}')
    
    print(f"Results saved to ../model_output/{filename}")
    return results

def run_intervention_analysis(save_prefix="intervention"):
    """
    Analyze the effect of targeted interventions
    """
    print("Running intervention analysis...")
    ensure_output_dir()
    
    # Run for different minority vegetarian fractions
    veg_fractions = [0.05, 0.1, 0.15, 0.2]
    all_results = []
    
    for veg_f in veg_fractions:
        print(f"Testing {veg_f*100}% vegetarian intervention...")
        
        # Run targeted interventions
        results = run_targeted_interventions(
            params,
            veg_fraction=veg_f,
            steps=25000,
            num_iterations=3
        )
        
        # Save individual results
        individual_filename = f'{save_prefix}_veg{veg_f}_{date.today().strftime("%b_%d_%Y")}.pkl'
        results.to_pickle(f'../model_output/{individual_filename}')
        
        all_results.append(results)
    
    # Combine all results
    combined_df = pd.concat(all_results)
    
    # Save combined results
    combined_filename = f'{save_prefix}_all_{date.today().strftime("%b_%d_%Y")}.pkl'
    combined_df.to_pickle(f'../model_output/{combined_filename}')
    
    print(f"Combined results saved to ../model_output/{combined_filename}")
    return combined_df

def run_cluster_analysis(save_prefix="clusters"):
    """
    Analyze vegetarian cluster formation
    """
    print("Running cluster analysis...")
    ensure_output_dir()
    
    # Run for different initial vegetarian fractions
    veg_fractions = np.linspace(0.1, 0.5, 5)
    cluster_stats = []
    cluster_results = {}
    
    for veg_f in veg_fractions:
        print(f"Analyzing clusters for {veg_f*100}% initial vegetarians...")
        
        # Modify params
        test_params = params.copy()
        test_params["veg_f"] = veg_f
        test_params["meat_f"] = 1 - veg_f
        
        # Run cluster analysis
        cluster_result = analyze_cluster_formation(
            test_params,
            veg_fraction=veg_f,
            steps=25000
        )
        
        # Extract stats for comparison
        stats = {
            'initial_veg_fraction': veg_f,
            'num_clusters': cluster_result['num_clusters'],
            'avg_cluster_size': cluster_result['avg_cluster_size'],
            'max_cluster_size': cluster_result['max_cluster_size'],
            'final_veg_fraction': cluster_result['final_veg_fraction']
        }
        
        cluster_stats.append(stats)
        
        # Store full cluster results (excluding model object for pickle compatibility)
        result_copy = cluster_result.copy()
        result_copy.pop('model', None)  # Remove model object which might cause pickle issues
        cluster_results[f"veg_{veg_f}"] = result_copy
        
        # Save individual cluster result
        individual_filename = f'{save_prefix}_veg{veg_f}_{date.today().strftime("%b_%d_%Y")}.pkl'
        with open(f'../model_output/{individual_filename}', 'wb') as f:
            pickle.dump(result_copy, f)
    
    # Convert stats to DataFrame
    stats_df = pd.DataFrame(cluster_stats)
    
    # Save stats
    stats_filename = f'{save_prefix}_stats_{date.today().strftime("%b_%d_%Y")}.pkl'
    stats_df.to_pickle(f'../model_output/{stats_filename}')
    
    # Save all cluster results
    full_filename = f'{save_prefix}_full_{date.today().strftime("%b_%d_%Y")}.pkl'
    with open(f'../model_output/{full_filename}', 'wb') as f:
        pickle.dump(cluster_results, f)
    
    print(f"Cluster stats saved to ../model_output/{stats_filename}")
    print(f"Full cluster results saved to ../model_output/{full_filename}")
    return stats_df, cluster_results

def run_critical_dynamics_analysis(save_prefix="critical"):
    """
    Analyze critical dynamics near tipping points
    """
    print("Running critical dynamics analysis...")
    ensure_output_dir()
    
    # Run for different potential tipping points
    tipping_estimates = [0.25, 0.3, 0.35]
    all_results = []
    
    for tipping_est in tipping_estimates:
        print(f"Analyzing critical dynamics near {tipping_est} vegetarian fraction...")
        
        # Modify alpha and beta to create conditions for this tipping point
        test_params = params.copy()
        test_params["alpha"] = 0.3  # Example values that might create tipping at these points
        test_params["beta"] = 0.7
        
        # Run critical dynamics analysis
        results = run_critical_dynamics(
            test_params,
            near_tipping=tipping_est,
            steps=50000
        )
        
        # Add tipping estimate to results
        results['tipping_estimate'] = tipping_est
        
        # Save individual results
        individual_filename = f'{save_prefix}_tip{tipping_est}_{date.today().strftime("%b_%d_%Y")}.pkl'
        results.to_pickle(f'../model_output/{individual_filename}')
        
        all_results.append(results)
    
    # Combine all results
    combined_df = pd.concat(all_results)
    
    # Save combined results
    combined_filename = f'{save_prefix}_all_{date.today().strftime("%b_%d_%Y")}.pkl'
    combined_df.to_pickle(f'../model_output/{combined_filename}')
    
    print(f"Combined results saved to ../model_output/{combined_filename}")
    return combined_df

def run_all_analyses():
    """Run all analyses and save results"""
    print("Running all analyses...")
    
    # Run each analysis and time it
    print("\n=== Running emissions analysis ===")
    emissions_results = timer(run_emissions_analysis)
    
    print("\n=== Running tipping point analysis ===")
    tipping_results = timer(run_tipping_point_analysis)
    
    print("\n=== Running network topology comparison ===")
    topology_results = timer(run_network_topology_comparison)
    
    print("\n=== Running intervention analysis ===")
    intervention_results = timer(run_intervention_analysis)
    
    print("\n=== Running cluster analysis ===")
    cluster_stats, cluster_results = timer(run_cluster_analysis)
    
    print("\n=== Running critical dynamics analysis ===")
    critical_results = timer(run_critical_dynamics_analysis)
    
    print("All analyses complete!")
    return {
        'emissions': emissions_results,
        'tipping': tipping_results,
        'topology': topology_results,
        'intervention': intervention_results,
        'cluster_stats': cluster_stats,
        'cluster_results': cluster_results,
        'critical': critical_results
    }

#%% Examples of running the analyses

if __name__ == "__main__":
    # Example 1: Run standard parameter sweep (original functionality)
    param_sweeps = ["alpha", "beta"]
    param_ranges = {i: np.linspace(0.0, 1.0, 10) for i in param_sweeps}
    num_iterations = 10
    
    print("=== Running original parameter sweep ===")
    results_df = timer(parameter_sweep, params, param_ranges, num_iterations)
    
    # Example 2: Run all new analyses
    # all_results = run_all_analyses()
    
    # Example 3: Run just one specific analysis
    # emissions_results = run_emissions_analysis()