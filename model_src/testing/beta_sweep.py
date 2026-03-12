#!/usr/bin/env python3
"""Ad-hoc beta (inverse temperature) sweep: integers 14-30, parallel."""
import os, sys, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import date
from multiprocessing import Pool
sys.path.append('..')
import model_main

BASE_PARAMS = {
    "veg_CO2": 1390, "vegan_CO2": 1054, "meat_CO2": 2054,
    "N": 650, "erdos_p": 3, "steps": 38000, "k": 8,
    "immune_n": 0.15, "M": 9, "veg_f": 0, "meat_f": 0.95,
    "p_rewire": 0.01, "rewire_h": 0.1, "tc": 0.6,
    "topology": "homophilic_emp",
    "beta": 17,
    "alpha": 0.36, "rho": 0.45, "theta": 0.44,
    "agent_ini": "sample-max",
    "survey_file": "../data/hierarchical_agents.csv",
    "adjust_veg_fraction": True, "target_veg_fraction": 0.07,
    "tau": 0.015, "theta_gate_c": 0.35, "theta_gate_k": 25,
    "alpha_min": 0.05, "alpha_max": 0.80, "mu": 0.2,
    "gamma": 0.5,
    "tau_persistence": 5000,
}

RUNS_PER_BETA = 5

def run_one(params):
    seed = 42 + params['run']
    np.random.seed(seed); random.seed(seed)
    model = model_main.Model(params, pmf_tables=None)
    model.run()
    return {
        'beta_inv_temp': params['beta'],
        'run': params['run'],
        'final_veg_f': model.fraction_veg[-1],
        'fraction_veg_trajectory': model.fraction_veg,
        'system_C_trajectory': model.system_C,
    }

if __name__ == "__main__":
    beta_range = range(14, 31)
    param_list = [
        {**BASE_PARAMS, 'beta': b, 'run': r}
        for b in beta_range for r in range(RUNS_PER_BETA)
    ]
    n_cores = max(1, int(0.75 * os.cpu_count()))
    print(f"Beta sweep {list(beta_range)}, {RUNS_PER_BETA} runs each, {len(param_list)} total on {n_cores} cores")
    with Pool(n_cores) as pool:
        results = pool.map(run_one, param_list)
    df = pd.DataFrame(results)
    os.makedirs('../model_output', exist_ok=True)
    fname = f'../model_output/beta_sweep_{date.today().strftime("%Y%m%d")}.pkl'
    df.to_pickle(fname)
    print(f"Saved to {fname}")
    print(df.groupby('beta_inv_temp')['final_veg_f'].mean().to_string())

    # --- Visualisation ---
    betas = sorted(df['beta_inv_temp'].unique())
    cmap = cm.viridis
    colors = {b: cmap(i / (len(betas) - 1)) for i, b in enumerate(betas)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: trajectories (mean per beta)
    ax = axes[0]
    for b in betas:
        trajs = df[df['beta_inv_temp'] == b]['fraction_veg_trajectory'].tolist()
        mean_traj = np.mean(trajs, axis=0)
        ax.plot(mean_traj, color=colors[b], alpha=0.85, lw=1.5, label=str(b))
    ax.set_xlabel("Step")
    ax.set_ylabel("Vegetarian fraction")
    ax.set_title("Mean trajectories by beta")
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(betas), max(betas)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="beta (inverse temp)")

    # Right: final veg fraction mean +/- std across runs
    means = df.groupby('beta_inv_temp')['final_veg_f'].mean()
    stds  = df.groupby('beta_inv_temp')['final_veg_f'].std()
    ax2 = axes[1]
    ax2.plot(means.index, means.values, 'o-', color='steelblue', lw=2)
    ax2.fill_between(means.index, means - stds, means + stds, alpha=0.25, color='steelblue')
    ax2.set_xlabel("beta (inverse temp)")
    ax2.set_ylabel("Final vegetarian fraction")
    ax2.set_title("Final veg fraction vs beta")
    ax2.set_xticks(betas)

    plt.tight_layout()
    os.makedirs('../visualisations_output', exist_ok=True)
    fig_fname = f'../visualisations_output/beta_sweep_{date.today().strftime("%Y%m%d")}.png'
    fig.savefig(fig_fname, dpi=150)
    print(f"Figure saved to {fig_fname}")
    plt.show()
