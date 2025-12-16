#!/usr/bin/env python3
"""Production parallel model runner with CLI interface"""
import os
import numpy as np
import pandas as pd
import time
import argparse
from datetime import date
from multiprocessing import Pool
import model_main_single as model_main

DEFAULT_PARAMS = {"veg_CO2": 1390,
          "vegan_CO2": 1054,
          "meat_CO2": 2054,
          "N": 100,
          "erdos_p": 3,
          "steps": 30000,
          "w_i": 5,
          "sigmoid_k": 12,
          "immune_n": 0.10,
          "k": 8,
          "M": 10,
          "veg_f":0.5,
          "meat_f": 0.5,
          "p_rewire": 0.1,
          "rewire_h": 0.1,
          "tc": 0.2,
          'topology': "PATCH",
          "alpha": 0.35,
          "rho": 0.1,
          "theta": 0,
          "agent_ini": "other",
          "survey_file": "../data/hierarchical_agents.csv",
          "adjust_veg_fraction": True,
          "target_veg_fraction": 0.06
          }

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
        import pickle
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

def get_model(params):
    """Create model with PMF tables if twin mode"""
    if params.get("agent_ini") == "twin":
        pmf_tables = load_pmf_tables()
        return model_main.Model(params, pmf_tables=pmf_tables)
    else:
        return model_main.Model(params)

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

def run_single_model(params):
    """Worker function - runs one model instance"""
    model = get_model(params)
    model.run()
    return {
        'agent_ini': params.get('agent_ini', 'other'),
        'alpha': params.get('alpha', 0), 'beta': 1 - params.get('alpha', 0.35),
        'theta': params.get('theta', 0), 'initial_veg_f': params.get('veg_f', 0),
        'final_veg_f': model.fraction_veg[-1],
        'change': model.fraction_veg[-1] - params.get('veg_f', 0),
        'tipped': model.fraction_veg[-1] > (params.get('veg_f', 0) * 1.2),
        'final_CO2': model.system_C[-1],
        'individual_reductions': model.get_attributes("reduction_out"),
        'fraction_veg_trajectory': model.fraction_veg if params.get('record_trajectories') else None,
        'system_C_trajectory': model.system_C if params.get('record_trajectories') else None,
        'parameter_set': params.get('parameter_set', ''),
        'run': params.get('run', 0)
    }

def run_single_trajectory_model(params):
    """Worker function for trajectory analysis"""
    if params["agent_ini"] == "parameterized":
        survey_data = load_survey_data(params["survey_file"],
                                     ["nomem_encr", "alpha", "theta", "diet"])
        survey_params = extract_survey_params(survey_data)
        params.update(survey_params)

    model = get_model(params)
    model.run()

    result = {
        'agent_ini': params["agent_ini"],
        'alpha': params.get('alpha', 0), 'beta': 1 - params.get('alpha', 0.35), 'theta': params.get('theta', 0),
        'initial_veg_f': params["veg_f"], 'final_veg_f': model.fraction_veg[-1],
        'fraction_veg_trajectory': model.fraction_veg, 'system_C_trajectory': model.system_C,
        'run': params.get('run', 0), 'parameter_set': params.get('parameter_set', ''),
        'is_median_twin': False
    }

    if params["agent_ini"] == "twin":
        agent_means = {k: np.mean([getattr(ag, k) for ag in model.agents]) for k in ['alpha', 'theta']}
        result.update(agent_means)
        result['beta'] = 1 - agent_means['alpha']
        result['initial_veg_f'] = sum(ag.diet=="veg" for ag in model.agents)/len(model.agents)
        result['snapshots'] = model.snapshots if hasattr(model, 'snapshots') else None

    return result

def run_parameter_analysis_parallel(base_params, alpha_range, beta_range,
                                  theta_range, veg_fractions, runs_per_combo,
                                  record_trajectories, n_cores):
    """Parallel parameter analysis. Supports parameterized mode."""
    # Load survey data if parameterized mode
    survey_params = {}
    if base_params.get("agent_ini") == "parameterized":
        sd = load_survey_data(base_params["survey_file"], ["nomem_encr", "alpha", "theta", "diet"])
        survey_params = extract_survey_params(sd)

    param_list = []
    for a in alpha_range:
        for t in theta_range:
            for vf in veg_fractions:
                for run in range(runs_per_combo):
                    p = base_params.copy()
                    if base_params.get("agent_ini") == "parameterized":
                        p.update({
                            **survey_params, 'alpha': a, 'theta': t, 'veg_f': vf, 'meat_f': 1-vf,
                            'record_trajectories': record_trajectories,
                            'parameter_set': f"α={a:.2f}, β={1-a:.2f}, θ={t:.2f}",
                            'run': run
                        })
                    else:
                        p.update({
                            'alpha': a, 'theta': t, 'veg_f': vf, 'meat_f': 1-vf,
                            'record_trajectories': record_trajectories,
                            'parameter_set': f"α={a:.2f}, β={1-a:.2f}, θ={t:.2f}",
                            'run': run
                        })
                    param_list.append(p)

    print(f"Running {len(param_list)} simulations on {n_cores} cores...")

    with Pool(n_cores) as pool:
        results = pool.map(run_single_model, param_list)

    df = pd.DataFrame(results)
    ensure_output_dir()
    suffix = "trajectories" if record_trajectories else "analysis"
    filename = f'../model_output/parameter_{suffix}_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(filename)
    print(f"Saved to {filename}")
    return df

def run_emissions_analysis_parallel(base_params, num_runs, veg_fractions, n_cores):
    """Parallel emissions analysis"""
    param_list = []
    for vf in veg_fractions:
        for run in range(num_runs):
            p = base_params.copy()
            p.update({'veg_f': vf, 'meat_f': 1.0 - vf, 'run': run})
            param_list.append(p)

    print(f"Running {len(param_list)} simulations on {n_cores} cores...")

    with Pool(n_cores) as pool:
        results = pool.map(run_single_model, param_list)

    for r in results:
        r['veg_fraction'] = r['initial_veg_f']
        r['topology'] = base_params['topology']

    df = pd.DataFrame(results)
    ensure_output_dir()
    filename = f'../model_output/emissions_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(filename)
    print(f"Saved to {filename}")
    return df

def run_veg_growth_analysis(base_params, veg_fractions):
    """Sequential veg growth analysis"""
    results = []
    for vf in veg_fractions:
        p = base_params.copy()
        p.update({"veg_f": vf, "meat_f": 1-vf})
        model = get_model(p)
        model.run()
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

def run_trajectory_analysis_parallel(base_params, runs_per_combo, n_cores):
    """Parallel trajectory analysis for twin mode only"""
    param_list = []

    # Twin mode - use N from base_params
    twin_N = base_params["N"]
    print(f"INFO: Using N={twin_N} for twin mode")

    for i in range(runs_per_combo):
        p = base_params.copy()
        p.update({
            "agent_ini": "twin", "N": twin_N,
            "parameter_set": "Survey Individual Parameters",
            "run": i
        })
        param_list.append(p)

    print(f"Running {len(param_list)} trajectory simulations on {n_cores} cores...")

    with Pool(n_cores) as pool:
        results = pool.map(run_single_trajectory_model, param_list)

    df = pd.DataFrame(results)

    # Mark median twin trajectory
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
    print(f"Saved {len(results)} trajectories to {filename}")
    return df

def main():
    parser = argparse.ArgumentParser(description='Parallel Dietary Contagion Model')
    parser.add_argument('--analysis', choices=['emissions', 'parameter', 'veg_growth', 'trajectory', 'param_trajectories'],
                       required=True, help='Analysis type to run')
    parser.add_argument('--agent_ini', choices=['synthetic', 'parameterized', 'twin'],
                       default='synthetic', help='Agent initialization mode')
    parser.add_argument('--cores', type=int, help='Number of cores (default: 0.75 * available)')
    parser.add_argument('--runs', type=int, default=3, help='Runs per parameter combo')
    parser.add_argument('--alpha_points', type=int, default=5, help='Alpha grid points')
    parser.add_argument('--beta_points', type=int, default=5, help='Beta grid points')
    parser.add_argument('--veg_points', type=int, default=10, help='Veg fraction points')
    parser.add_argument('--survey_file', default='../data/hierarchical_agents.csv', help='Survey data file')

    args = parser.parse_args()

    # Setup cores (0.75x available by default)
    n_cores = args.cores or int(0.75 * os.cpu_count())
    print(f"Using {n_cores}/{os.cpu_count()} cores")

    # Setup base parameters
    params = DEFAULT_PARAMS.copy()
    params["agent_ini"] = args.agent_ini
    if args.survey_file:
        params["survey_file"] = args.survey_file

    print(f"Running {args.analysis} analysis with {args.agent_ini} initialization...")

    # Run selected analysis
    if args.analysis == 'emissions':
        veg_fractions = np.linspace(0, 1, args.veg_points)
        timer(run_emissions_analysis_parallel, params, args.runs, veg_fractions, n_cores)

    elif args.analysis == 'parameter':
        alpha_range = np.linspace(0.1, 0.9, args.alpha_points)
        beta_range = np.linspace(0.1, 0.9, args.beta_points)
        timer(run_parameter_analysis_parallel, params, alpha_range, beta_range,
              [-0.5, 0, 0.5], [0.2], args.runs, False, n_cores)

    elif args.analysis == 'veg_growth':
        veg_fractions = np.linspace(0.1, 0.6, args.veg_points)
        timer(run_veg_growth_analysis, params, veg_fractions)

    elif args.analysis == 'trajectory':
        timer(run_trajectory_analysis_parallel, params, args.runs, n_cores)

    elif args.analysis == 'param_trajectories':
        alpha_range = np.linspace(0.1, 0.9, 3)
        beta_range = np.linspace(0.1, 0.9, 3)
        timer(run_parameter_analysis_parallel, params, alpha_range, beta_range,
              [-0.5, 0, 0.5], [0.2], args.runs, True, n_cores)

    print("Analysis complete!")

if __name__ == "__main__":
    main()
