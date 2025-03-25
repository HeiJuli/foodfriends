#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:23:39 2025

@author: jpoveralls
"""

# -*- coding: utf-8 -*-
"""
Extended model running functions to complement model_runn.py
"""
import numpy as np
import pandas as pd
import networkx as nx
import random
import model_main_single as model_main


#%%

def run_emissions_vs_vegetarian_fraction(base_params, num_runs=3, veg_fractions=np.linspace(0, 1, 20)):
    """
    Run multiple simulations varying vegetarian fraction and collect final CO2 consumption data
    
    Args:
        base_params (dict): Base model parameters
        num_runs (int): Number of runs per parameter combination
        veg_fractions (array): Array of vegetarian fractions to test
        
    Returns:
        pd.DataFrame: Results containing fraction and emissions data
    """
    results = []
    
    for veg_f in veg_fractions:
        params = base_params.copy()
        params["veg_f"] = veg_f
        params["meat_f"] = 1.0 - veg_f
        
        for _ in range(num_runs):
            model = model_main.Model(params)
            model.run()
            
            results.append({
                'veg_fraction': veg_f,
                'final_CO2': model.system_C[-1],
                'final_veg_fraction': model.fraction_veg[-1],
                'alpha': params['alpha'],
                'beta': params['beta'],
                'topology': params['topology']
            })
    
    return pd.DataFrame(results)

def run_parameter_sensitivity(base_params, alpha_range=np.linspace(0, 1, 5), 
                              beta_range=np.linspace(0, 1, 5), 
                              fixed_veg_f=0.2, steps=5000):
    """
    Run simulations to analyze parameter sensitivity for alpha and beta
    
    Args:
        base_params (dict): Base model parameters
        alpha_range (array): Array of alpha values to test
        beta_range (array): Array of beta values to test
        fixed_veg_f (float): Fixed initial vegetarian fraction
        steps (int): Number of simulation steps
        
    Returns:
        pd.DataFrame: Results containing parameter combinations and outcomes
    """
    tipping_data = []
    
    for alpha in alpha_range:
        for beta in beta_range:
            params = base_params.copy()
            params["alpha"] = alpha
            params["beta"] = beta
            params["veg_f"] = fixed_veg_f
            params["meat_f"] = 1.0 - fixed_veg_f
            params["steps"] = steps
            
            model = model_main.Model(params)
            model.run()
            
            # Check if tipping occurred - final veg fraction significantly higher than initial
            initial_veg_f = fixed_veg_f
            final_veg_f = model.fraction_veg[-1]
            
            # Define tipping as at least 20% increase in vegetarian fraction
            tipped = final_veg_f > (initial_veg_f * 1.2)
            
            tipping_data.append({
                'alpha': alpha,
                'beta': beta,
                'tipped': tipped,
                'final_veg_f': final_veg_f,
                'change': final_veg_f - initial_veg_f,
                'final_CO2': model.system_C[-1]
            })
    
    return pd.DataFrame(tipping_data)

def run_topology_comparison(base_params, topologies=["BA", "complete"], 
                            veg_fractions=np.linspace(0.1, 0.5, 5),
                            runs_per_config=3):
    """
    Compare how different network topologies affect the contagion dynamics
    
    Args:
        base_params (dict): Base model parameters
        topologies (list): List of topology types to test
        veg_fractions (array): Array of vegetarian fractions to test
        runs_per_config (int): Number of runs per parameter combination
        
    Returns:
        pd.DataFrame: Results containing topology comparisons
    """
    results = []
    
    for topology in topologies:
        for veg_f in veg_fractions:
            # Update parameters
            params = base_params.copy()
            params["topology"] = topology
            params["veg_f"] = veg_f
            params["meat_f"] = 1 - veg_f
            
            # Run iterations per parameter combination
            for i in range(runs_per_config):
                model = model_main.Model(params)
                model.run()
                
                results.append({
                    'topology': topology,
                    'initial_veg_f': veg_f,
                    'final_veg_f': model.fraction_veg[-1],
                    'growth': model.fraction_veg[-1] - veg_f,
                    'final_CO2': model.system_C[-1],
                    'run': i
                })
    
    return pd.DataFrame(results)

def run_targeted_interventions(base_params, veg_fraction=0.2, steps=25000, num_iterations=5):
    """
    Simulate targeted vs random interventions in the network
    
    This compares what happens when you convert the most central nodes vs random nodes
    to vegetarians, keeping the initial vegetarian fraction constant
    
    Args:
        base_params (dict): Base model parameters
        veg_fraction (float): Fraction of nodes to convert to vegetarians
        steps (int): Number of simulation steps
        num_iterations (int): Number of runs per intervention type
        
    Returns:
        pd.DataFrame: Results containing intervention outcomes
    """
    results = []
    intervention_types = ['random', 'degree_central', 'betweenness_central', 'none']
    
    for intervention in intervention_types:
        # Run multiple iterations
        for iteration in range(num_iterations):
            # Set up parameters
            params = base_params.copy()
            params["veg_f"] = 0  # Start with no vegetarians
            params["meat_f"] = 1.0
            params["steps"] = steps
            
            # Initialize model
            model = model_main.Model(params)
            model.agent_ini(params)
            
            # Create network metrics for targeted interventions
            G = model.G1
            
            # Select nodes to convert to vegetarians
            num_to_convert = int(veg_fraction * params["N"])
            
            if intervention == 'none':
                # Baseline - no interventions
                convert_indices = []
            elif intervention == 'random':
                # Random nodes
                convert_indices = np.random.choice(range(params["N"]), num_to_convert, replace=False)
            elif intervention == 'degree_central':
                # Highest degree centrality
                degree_dict = dict(G.degree())
                convert_indices = sorted(degree_dict, key=degree_dict.get, reverse=True)[:num_to_convert]
            elif intervention == 'betweenness_central':
                # Highest betweenness centrality
                betweenness_dict = nx.betweenness_centrality(G)
                convert_indices = sorted(betweenness_dict, key=betweenness_dict.get, reverse=True)[:num_to_convert]
            
            # Convert selected nodes to vegetarians
            for idx in convert_indices:
                model.agents[idx].diet = "veg"
                model.agents[idx].C = model.agents[idx].diet_emissions("veg", params)
            
            # Run the model
            model.record_fraction()  # Record initial state
            
            time_array = list(range(params["steps"]))
            for t in time_array:
                # Select random agent
                i = np.random.choice(range(len(model.agents)))

                # Update based on pairwise interaction
                model.agents[i].step(model.G1, model.agents, params)
                
                # Record system state
                model.system_C.append(model.get_attribute("C")/params["N"])
                model.record_fraction()
            
            # Record results
            results.append({
                'intervention': intervention,
                'initial_veg_f': veg_fraction if intervention != 'none' else 0,
                'final_veg_f': model.fraction_veg[-1],
                'final_CO2': model.system_C[-1],
                'veg_trajectory': model.fraction_veg.copy(),
                'iteration': iteration
            })
    
    return pd.DataFrame(results)

def run_critical_dynamics(base_params, near_tipping=0.28, steps=50000):
    """
    Run simulations to analyze critical dynamics near the tipping point
    
    Args:
        base_params (dict): Base model parameters
        near_tipping (float): Estimated tipping point value
        steps (int): Number of simulation steps
        
    Returns:
        pd.DataFrame: Results with variance and autocorrelation metrics
    """
    # Run simulations at points increasingly close to the tipping point
    distances = np.array([-0.1, -0.05, -0.02, -0.01, 0, 0.01, 0.02])
    test_points = near_tipping + distances
    
    results = []
    
    for veg_f in test_points:
        params = base_params.copy()
        params["veg_f"] = veg_f
        params["meat_f"] = 1 - veg_f
        params["steps"] = steps
        
        model = model_main.Model(params)
        model.run()
        
        # Calculate variance in the vegetarian fraction (after initial transient)
        transient = int(steps * 0.4)  # Skip first 40% as transient
        timeseries = model.fraction_veg[transient:]
        variance = np.var(timeseries)
        
        # Calculate lag-1 autocorrelation
        autocorr = np.corrcoef(timeseries[:-1], timeseries[1:])[0, 1]
        
        results.append({
            'veg_fraction': veg_f,
            'variance': variance,
            'autocorrelation': autocorr,
            'final_veg_f': model.fraction_veg[-1],
            'final_CO2': model.system_C[-1]
        })
    
    return pd.DataFrame(results)

def analyze_cluster_formation(base_params, veg_fraction=0.2, steps=25000):
    """
    Run model and analyze the formation of vegetarian clusters
    
    Args:
        base_params (dict): Base model parameters
        veg_fraction (float): Initial vegetarian fraction
        steps (int): Number of simulation steps
        
    Returns:
        dict: Cluster statistics and the model object for further analysis
    """
    # Run the model with specified parameters
    params = base_params.copy()
    params["veg_f"] = veg_fraction
    params["meat_f"] = 1 - veg_fraction
    params["steps"] = steps
    
    model = model_main.Model(params)
    model.run()
    
    # Extract the network
    G = model.G1.copy()
    
    # Get final diets
    diets = model.get_attributes("diet")
    
    # Create subgraph of only vegetarian agents
    veg_indices = [i for i, diet in enumerate(diets) if diet == 'veg']
    if len(veg_indices) > 0:
        veg_subgraph = G.subgraph(veg_indices)
        
        # Calculate the number and size of connected components (clusters)
        clusters = list(nx.connected_components(veg_subgraph))
        cluster_sizes = [len(c) for c in clusters]
        
        # Calculate clustering statistics
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
        num_clusters = len(clusters)
    else:
        clusters = []
        avg_cluster_size = 0
        max_cluster_size = 0
        num_clusters = 0
    
    # Return statistics and model for further analysis
    return {
        'model': model,
        'clusters': clusters,
        'num_clusters': num_clusters,
        'avg_cluster_size': avg_cluster_size,
        'max_cluster_size': max_cluster_size,
        'final_veg_fraction': model.fraction_veg[-1],
        'G': G,
        'diets': diets
    }