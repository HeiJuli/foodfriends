#!/usr/bin/env python3
"""Streamlined model running script"""
import os
import sys
import numpy as np
import pandas as pd
import time
import pickle
from datetime import date
import model_main_single as model_main
#import model_main_threshold as model_main
sys.path.append('..')
from auxillary.sampling_utils import load_sample_max_agents

#%%
DEFAULT_PARAMS = {"veg_CO2": 1390,
          "vegan_CO2": 1054,
          "meat_CO2": 2054,
          "N": 150,
          "erdos_p": 3,
          "steps": 250000,
          "w_i": 5, #weight of the replicator function
          "sigmoid_k": 12, #sigmoid steepness for dissonance scaling
          "immune_n": 0.10,
          "k": 8, #initial edges per node for graph generation
          "M": 8, # memory length
          "veg_f":0.1, #vegetarian fraction
          "meat_f": 0.9,  #meat eater fraction
          "p_rewire": 0.1, #probability of rewire step
          "rewire_h": 0.1, # slightly preference for same diet2
          "tc": 0.3, #probability of triadic closure for CSF, PATCH network gens
          'topology': "homophilic_emp", #can either be barabasi albert with "BA", or fully connected with "complete"
          "alpha": 0.36, #self dissonance
          "rho": 0.25, #behavioural intentions,
          "theta": 0.44, #intrinsic preference (- is for meat, + for vego)
          "agent_ini": "sample-max", #choose between "twin" "parameterized" or "synthetic"
          "survey_file": "../data/hierarchical_agents.csv",
          "adjust_veg_fraction": False, #artificially increase veg fraction to match NL demographics
          "target_veg_fraction": 0.06 #target vegetarian fraction (6% for Netherlands)
          }



#%%

def ensure_output_dir():
    if not os.path.exists('../model_output'):
        os.makedirs('../model_output')

def timer(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"Runtime: {int(elapsed/60)} mins {elapsed%60:.1f}s")
    return result

def load_survey_data(filepath, variables):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Survey file not found: {filepath}")
    data = pd.read_csv(filepath)[variables]
    print(f"Loaded survey data: {data.shape[0]} respondents")
    return data

def get_model(params):
    """Create model with PMF tables if twin mode (sample-max doesn't need them)"""
    if params.get("agent_ini") == "twin":
        pmf_tables = load_pmf_tables()
        return model_main.Model(params, pmf_tables=pmf_tables)
    elif params.get("agent_ini") == "sample-max":
        # Sample-max uses only complete cases, no PMF imputation needed
        return model_main.Model(params, pmf_tables=None)
    else:
        return model_main.Model(params)
    
def extract_survey_params(survey_data):
    params = {}
    for col in ['alpha', 'theta']:
        if col in survey_data.columns:
            params[col] = survey_data[col].mean()
    if 'diet' in survey_data.columns:
        params['veg_f'] = (survey_data['diet'] == 'veg').mean()
        params['meat_f'] = 1 - params['veg_f']
    return params

def load_pmf_tables(filepath="../data/demographic_pmfs.pkl"):
    """Load PMF tables for alpha/rho sampling"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found, using synthetic for alpha/rho")
        return None

def sample_from_pmf(demo_key, pmf_tables, param):
    """Sample single parameter from PMF"""
    if demo_key in pmf_tables[param]:
        pmf = pmf_tables[param][demo_key]
        vals, probs = pmf['values'], pmf['probabilities']
        nz = [(v,p) for v,p in zip(vals, probs) if p > 0]
        if nz:
            v, p = zip(*nz)
            return np.random.choice(v, p=np.array(p)/sum(p))
    
    # Fallback: sample from all values
    all_vals = []
    for cell in pmf_tables[param].values():
        all_vals.extend(cell['values'])
    return np.random.choice(all_vals) if all_vals else 0.5

def run_basic_model(params=None):
    params = params or DEFAULT_PARAMS.copy()
    if params["agent_ini"] == "parameterized":
        survey_data = load_survey_data(params["survey_file"], 
                                     ["nomem_encr", "alpha", "theta", "diet"])
        survey_params = extract_survey_params(survey_data)
        params.update(survey_params)
    
    model = get_model(params)
    model.run()
    return model

def run_emissions_analysis(params=None, num_runs=3, veg_fractions=None):
    params = DEFAULT_PARAMS.copy() if params is None else params
    veg_fractions = np.linspace(0, 1, 20) if veg_fractions is None else veg_fractions
    
    results = []
    for veg_f in veg_fractions:
        p = params.copy()
        p.update({"veg_f": veg_f, "meat_f": 1.0 - veg_f})
        
        for _ in range(num_runs):
            model = run_basic_model(p)
            results.append({
                'veg_fraction': veg_f, 'final_CO2': model.system_C[-1],
                'final_veg_fraction': model.fraction_veg[-1],
                'alpha': p['alpha'], 'beta': 1 - p['alpha'], 'topology': p['topology']
            })
    
    df = pd.DataFrame(results)
    ensure_output_dir()
    agent_ini = params.get('agent_ini', 'other')
    filename = f'../model_output/emissions_{agent_ini}_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(filename)
    print(f"Saved to {filename}")
    return df

def run_parameter_analysis(params=None, alpha_range=None,
                          theta_range=None, veg_fractions=None, runs_per_combo=3,
                          record_trajectories=False):
    """Unified parameter analysis - optionally records full trajectories. Supports parameterized mode."""
    params = DEFAULT_PARAMS.copy() if params is None else params
    alpha_range = np.linspace(0.1, 0.9, 5) if alpha_range is None else alpha_range
    theta_range = [-0.5, 0, 0.5] if theta_range is None else theta_range
    veg_fractions = [0.2] if veg_fractions is None else veg_fractions

    # Load survey data if parameterized mode
    survey_params = {}
    if params.get("agent_ini") == "parameterized":
        sd = load_survey_data(params["survey_file"], ["nomem_encr", "alpha", "theta", "diet"])
        survey_params = extract_survey_params(sd)

    results = []
    total = len(alpha_range) * len(theta_range) * len(veg_fractions) * runs_per_combo
    count = 0

    for a in alpha_range:
        for t in theta_range:
            for vf in veg_fractions:
                p = params.copy()
                if params.get("agent_ini") == "parameterized":
                    p.update({**survey_params, "alpha": a, "theta": t, "veg_f": vf, "meat_f": 1-vf})
                else:
                    p.update({"alpha": a, "theta": t, "veg_f": vf, "meat_f": 1-vf})

                for run in range(runs_per_combo):
                    count += 1
                    print(f"Run {count}/{total}: α={a:.2f}, θ={t:.2f}, veg_f={vf:.2f}")

                    model = get_model(p)
                    model.run()

                    result = {
                        'agent_ini': params.get("agent_ini", "other"),
                        'alpha': a, 'beta': 1 - a, 'theta': t,
                        'initial_veg_f': vf, 'final_veg_f': model.fraction_veg[-1],
                        'change': model.fraction_veg[-1] - vf,
                        'tipped': model.fraction_veg[-1] > (vf * 1.2),
                        'final_CO2': model.system_C[-1], 'run': run,
                        'individual_reductions': model.get_attributes("reduction_out")
                    }

                    if record_trajectories:
                        result.update({
                            'fraction_veg_trajectory': model.fraction_veg,
                            'system_C_trajectory': model.system_C,
                            'parameter_set': f"α={a:.2f}, β={1-a:.2f}, θ={t:.2f}"
                        })

                    results.append(result)
    
    df = pd.DataFrame(results)
    ensure_output_dir()
    suffix = "trajectories" if record_trajectories else "analysis"
    agent_ini = params.get('agent_ini', 'other')
    filename = f'../model_output/parameter_{suffix}_{agent_ini}_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(filename)
    print(f"Saved to {filename}")
    return df

def run_veg_growth_analysis(params=None, veg_fractions=None, max_veg=0.6):
    params = DEFAULT_PARAMS.copy() if params is None else params
    veg_fractions = np.linspace(0.1, max_veg, 10) if veg_fractions is None else veg_fractions
    
    results = []
    for vf in veg_fractions:
        p = params.copy()
        p.update({"veg_f": vf, "meat_f": 1-vf})
        model = run_basic_model(p)
        results.append({
            'initial_veg_fraction': vf,
            'final_veg_fraction': model.fraction_veg[-1]
        })
    
    df = pd.DataFrame(results)
    ensure_output_dir()
    agent_ini = params.get('agent_ini', 'other')
    filename = f'../model_output/veg_growth_{agent_ini}_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(filename)
    print(f"Saved to {filename}")
    return df

def run_trajectory_analysis(params=None, runs_per_combo=5):
    """Run trajectory analysis for twin or sample-max mode"""
    p = DEFAULT_PARAMS.copy() if params is None else params.copy()
    r = []

    # Use agent_ini from params (twin or sample-max)
    twn = p.copy()
    agent_ini = twn.get("agent_ini", "twin")
    print(f"INFO: Using N={twn['N']} for {agent_ini} mode")
    for i in range(runs_per_combo):
        model = get_model(twn)
        model.run()
        agent_means = {k: np.mean([getattr(ag, k) for ag in model.agents]) for k in ['alpha', 'theta']}
        agent_means['beta'] = 1 - agent_means['alpha']
        r.append({
            'agent_ini': agent_ini, **agent_means,
            'initial_veg_f': sum(ag.diet=="veg" for ag in model.agents)/len(model.agents),
            'final_veg_f': model.fraction_veg[-1],
            'fraction_veg_trajectory': model.fraction_veg, 'system_C_trajectory': model.system_C,
            'snapshots': model.snapshots if hasattr(model, 'snapshots') else None,
            'run': i, 'parameter_set': "Survey Individual Parameters"
        })

    df = pd.DataFrame(r)

    # Find median trajectory for network plots
    mode_data = df[df['agent_ini'] == agent_ini]
    if len(mode_data) > 0:
        final_veg = mode_data['final_veg_f'].values
        median_idx = mode_data.index[np.argmin(np.abs(final_veg - np.median(final_veg)))]
        df['is_median_twin'] = False
        df.loc[median_idx, 'is_median_twin'] = True
        print(f"{agent_ini} median trajectory: run {df.loc[median_idx, 'run']} with final_veg_f = {df.loc[median_idx, 'final_veg_f']:.3f}")

    ensure_output_dir()
    filename = f'../model_output/trajectory_analysis_{agent_ini}_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(filename)
    print(f"Saved {len(r)} trajectories to {filename}")
    return df

def main():
    print("===== Streamlined Dietary Contagion Model Analysis =====")

    # Agent initialization mode
    print("\n[1] Synthetic [2] Parameterized [3] Twin [4] Sample-max")
    init_choice = input("Select initialization (1-4, default 1): ")

    agent_ini = {"2": "parameterized", "3": "twin", "4": "sample-max"}.get(init_choice, "synthetic")
    params = DEFAULT_PARAMS.copy()
    params["agent_ini"] = agent_ini

    if agent_ini in ["parameterized", "twin", "sample-max"]:
        survey_file = input(f"Survey file ({params['survey_file']}): ") or params['survey_file']
        params["survey_file"] = survey_file

    if agent_ini in ["twin", "sample-max"]:
        pmf_tables = load_pmf_tables()
        params["pmf_tables"] = pmf_tables
        
    while True:
        print("\n[1] Emissions vs Veg Fraction")
        print("[2] Parameter Analysis (alpha-theta grid, end states)")
        print("[3] Vegetarian Growth Analysis")
        print("[4] Trajectory Analysis (twin mode only)")
        print("[5] Parameter Sweep with Trajectories (supplement, includes parameterized)")
        print("[0] Exit")
        
        choice = input("Select: ")
        
        if choice == '1':
            timer(run_emissions_analysis, params=params, num_runs=3, 
                  veg_fractions=np.linspace(0, 1, 5))
        elif choice == '2':
            timer(run_parameter_analysis, params=params,
                  alpha_range=np.linspace(0.1, 0.9, 5),
                  veg_fractions=[0.2], runs_per_combo=3)
        elif choice == '3':
            timer(run_veg_growth_analysis, params=params,
                  veg_fractions=np.linspace(0.1, 0.6, 10))
        elif choice == '4':
            timer(run_trajectory_analysis, params=params, runs_per_combo=10)
        elif choice == '5':
            timer(run_parameter_analysis, params=params,
                  alpha_range=np.linspace(0.1, 0.9, 3),
                  theta_range=[-0.5, 0, 0.5], veg_fractions=[0.2], runs_per_combo=2,
                  record_trajectories=True)
        elif choice == '0':
            break
        else:
            print("Invalid option")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()