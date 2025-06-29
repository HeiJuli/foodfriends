#!/usr/bin/env python3
"""Streamlined model running script"""
import os
import numpy as np
import pandas as pd
import time
from datetime import date
import model_main_single as model_main

#%%
DEFAULT_PARAMS = {"veg_CO2": 1390,
          "vegan_CO2": 1054,
          "meat_CO2": 2054,
          "N": 699,
          "erdos_p": 3,
          "steps": 45000,
          "w_i": 5, #weight of the replicator function
          "immune_n": 0.10,
          "M": 10, # memory length
          "veg_f":0.5, #vegetarian fraction
          "meat_f": 0.5,  #meat eater fraciton
          "p_rewire": 0.1, #probability of rewire step
          "rewire_h": 0.1, # slightly preference for same diet
          "tc": 0.2, #probability of triadic closure for CSF, PATCH network gens
          'topology': "complete", #can either be barabasi albert with "BA", or fully connected with "complete"
          "alpha": 0.35, #self dissonance
          "beta": 0.65, #social dissonance
          "theta": 0, #intrinsic preference (- is for meat, + for vego)
          "agent_ini": "other", #choose between "twin" "parameterized" or "synthetic" 
          "survey_file": "../data/final_data_parameters.csv"
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

def extract_survey_params(survey_data):
    params = {}
    for col in ['alpha', 'beta', 'theta']:
        if col in survey_data.columns:
            params[col] = survey_data[col].mean()
    if 'diet' in survey_data.columns:
        params['veg_f'] = (survey_data['diet'] == 'veg').mean()
        params['meat_f'] = 1 - params['veg_f']
    return params

def run_basic_model(params=None):
    params = params or DEFAULT_PARAMS.copy()
    if params["agent_ini"] == "parameterized":
        survey_data = load_survey_data(params["survey_file"], 
                                     ["nomem_encr", "alpha", "beta", "theta", "diet"])
        survey_params = extract_survey_params(survey_data)
        params.update(survey_params)
    
    model = model_main.Model(params)
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
                'alpha': p['alpha'], 'beta': p['beta'], 'topology': p['topology']
            })
    
    df = pd.DataFrame(results)
    ensure_output_dir()
    filename = f'../model_output/emissions_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(filename)
    print(f"Saved to {filename}")
    return df

def run_parameter_analysis(params=None, alpha_range=None, beta_range=None, 
                          theta_range=None, veg_fractions=None, runs_per_combo=3):
    """Unified parameter analysis - replaces both tipping point and 3D analysis"""
    params = DEFAULT_PARAMS.copy() if params is None else params
    alpha_range = np.linspace(0.1, 0.9, 5) if alpha_range is None else alpha_range
    beta_range = np.linspace(0.1, 0.9, 5) if beta_range is None else beta_range
    theta_range = [-0.5, 0, 0.5] if theta_range is None else theta_range
    veg_fractions = [0.2] if veg_fractions is None else veg_fractions
    
    results = []
    total = len(alpha_range) * len(beta_range) * len(theta_range) * len(veg_fractions) * runs_per_combo
    count = 0
    
    for a in alpha_range:
        for b in beta_range:
            for t in theta_range:
                for vf in veg_fractions:
                    p = params.copy()
                    p.update({"alpha": a, "beta": b, "theta": t, "veg_f": vf, "meat_f": 1-vf})
                    
                    for run in range(runs_per_combo):
                        count += 1
                        print(f"Run {count}/{total}: α={a:.2f}, β={b:.2f}, θ={t:.2f}, veg_f={vf:.2f}")
                        
                        model = model_main.Model(p)
                        model.run()
                        
                        results.append({
                            'alpha': a, 'beta': b, 'theta': t,
                            'initial_veg_f': vf, 'final_veg_f': model.fraction_veg[-1],
                            'change': model.fraction_veg[-1] - vf,
                            'tipped': model.fraction_veg[-1] > (vf * 1.2),
                            'final_CO2': model.system_C[-1], 'run': run,
                            'individual_reductions': model.get_attributes("reduction_out")
                        })
    
    df = pd.DataFrame(results)
    ensure_output_dir()
    filename = f'../model_output/parameter_analysis_{date.today().strftime("%Y%m%d")}.pkl'
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
    filename = f'../model_output/veg_growth_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(filename)
    print(f"Saved to {filename}")
    return df

def run_trajectory_analysis(params=None, alpha_values=None, beta_values=None, 
                           theta_values=None, runs_per_combo=3):
    p = DEFAULT_PARAMS.copy() if params is None else params.copy()
    a = [0.25, 0.5, 0.75] if alpha_values is None else alpha_values
    b = [0.25, 0.5, 0.75] if beta_values is None else beta_values
    t = [-0.5, 0, 0.5] if theta_values is None else theta_values
    r = []
    
    # Load survey data if needed
    sd = None
    if p.get("agent_ini") in ["parameterized", "twin"]:
        sd = load_survey_data(p["survey_file"], ["nomem_encr", "alpha", "beta", "theta", "diet"])
        sm = extract_survey_params(sd)
    
    # Synthetic mode
    syn = p.copy()
    syn["agent_ini"] = "synthetic"
    for ax in a:
        for bx in b:
            for tx in t:
                syn.update({"alpha": ax, "beta": bx, "theta": tx})
                for i in range(runs_per_combo):
                    m = model_main.Model(syn)
                    m.run()
                    r.append({
                        'agent_ini': "synthetic", 'alpha': ax, 'beta': bx, 'theta': tx,
                        'initial_veg_f': syn["veg_f"], 'final_veg_f': m.fraction_veg[-1],
                        'fraction_veg_trajectory': m.fraction_veg, 'system_C_trajectory': m.system_C,
                        'run': i, 'parameter_set': f"α={ax:.2f}, β={bx:.2f}, θ={tx:.2f}"
                    })
    
    # Survey-based modes
    if sd is not None:
        # Parameterized
        par = p.copy()
        par.update({"agent_ini": "parameterized", **sm})
        for i in range(runs_per_combo):
            m = model_main.Model(par)
            m.run()
            r.append({
                'agent_ini': "parameterized", **sm,
                'initial_veg_f': par["veg_f"], 'final_veg_f': m.fraction_veg[-1],
                'fraction_veg_trajectory': m.fraction_veg, 'system_C_trajectory': m.system_C,
                'run': i, 'parameter_set': "Survey Mean Parameters"
            })
        
        # Twin mode with snapshots
        twn = p.copy()
        twn.update({"agent_ini": "twin", "N": len(sd)})
        for i in range(runs_per_combo):
            m = model_main.Model(twn)
            m.run()
            agent_means = {k: np.mean([getattr(ag, k) for ag in m.agents]) for k in ['alpha', 'beta', 'theta']}
            r.append({
                'agent_ini': "twin", **agent_means,
                'initial_veg_f': sum(ag.diet=="veg" for ag in m.agents)/len(m.agents),
                'final_veg_f': m.fraction_veg[-1],
                'fraction_veg_trajectory': m.fraction_veg, 'system_C_trajectory': m.system_C,
                'snapshots': m.snapshots if hasattr(m, 'snapshots') else None,
                'run': i, 'parameter_set': "Survey Individual Parameters"
            })
    
    df = pd.DataFrame(r)
    
    # Find median twin trajectory for network plots
    twin_data = df[df['agent_ini'] == 'twin']
    if len(twin_data) > 0:
        final_veg = twin_data['final_veg_f'].values
        median_idx = twin_data.index[np.argmin(np.abs(final_veg - np.median(final_veg)))]
        df['is_median_twin'] = False
        df.loc[median_idx, 'is_median_twin'] = True
        print(f"Twin median trajectory: run {df.loc[median_idx, 'run']} with final_veg_f = {df.loc[median_idx, 'final_veg_f']:.3f}")
    
    ensure_output_dir()
    filename = f'../model_output/trajectory_analysis_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(filename)
    print(f"Saved {len(r)} trajectories to {filename}")
    return df

def main():
    print("===== Streamlined Dietary Contagion Model Analysis =====")
    
    # Agent initialization mode
    print("\n[1] Synthetic [2] Parameterized [3] Twin")
    init_choice = input("Select initialization (1-3, default 1): ")
    
    agent_ini = {"2": "parameterized", "3": "twin"}.get(init_choice, "synthetic")
    params = DEFAULT_PARAMS.copy()
    params["agent_ini"] = agent_ini
    
    if agent_ini in ["parameterized", "twin"]:
        survey_file = input(f"Survey file ({params['survey_file']}): ") or params['survey_file']
        params["survey_file"] = survey_file
    
    while True:
        print("\n[1] Emissions vs Veg Fraction")
        print("[2] Parameter Analysis (alpha-beta grid)")
        print("[3] Vegetarian Growth Analysis") 
        print("[4] Trajectory Analysis")
        print("[0] Exit")
        
        choice = input("Select: ")
        
        if choice == '1':
            timer(run_emissions_analysis, params=params, num_runs=3, 
                  veg_fractions=np.linspace(0, 1, 5))
        elif choice == '2':
            timer(run_parameter_analysis, params=params,
                  alpha_range=np.linspace(0.1, 0.9, 5), beta_range=np.linspace(0.1, 0.9, 5),
                  veg_fractions=[0.2], runs_per_combo=3)
        elif choice == '3':
            timer(run_veg_growth_analysis, params=params,
                  veg_fractions=np.linspace(0.1, 0.6, 5))
        elif choice == '4':
            timer(run_trajectory_analysis, params=params, runs_per_combo=10)
    
        elif choice == '0':
            break
        else:
            print("Invalid option")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()