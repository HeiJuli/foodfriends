#!/usr/bin/env python3
"""
Consolidated testing script for initial dynamics and uptake behavior
Combines memory sweeps, initialization strategies, and initial inertia tests
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from multiprocessing import Pool
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_main_single import Model, Agent, params
from model_runn import DEFAULT_PARAMS

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Global set to track agents with initial inertia
agents_with_inertia = set()

def mark_agents_with_initial_inertia(model):
    """Mark meat-eaters who have veg neighbors at t=0"""
    global agents_with_inertia
    agents_with_inertia.clear()
    for agent in model.agents:
        if agent.diet == "meat":
            neighbor_ids = list(model.G1.neighbors(agent.i))
            has_veg_neighbor = any(model.agents[nid].diet == "veg" for nid in neighbor_ids)
            if has_veg_neighbor:
                agents_with_inertia.add(id(agent))
    print(f"  Marked {len(agents_with_inertia)} meat-eaters with initial veg neighbors")

def create_utility_prob_calc_with_inertia(inertia_penalty=0.15):
    """Utility-based probability calculation with initial inertia penalty"""
    def new_prob_calc(self, other_agent):
        u_i = self.calc_utility(other_agent, mode="same")
        u_s = self.calc_utility(other_agent, mode="diff")
        delta = u_s - u_i
        if id(self) in agents_with_inertia:
            delta -= inertia_penalty
        if delta < -0.5:
            prob_switch = 0.01
        else:
            prob_switch = 1/(1+np.exp(-4.0*delta))
        return prob_switch
    return new_prob_calc

def modify_pure_diet_initialization():
    """Initialize memory with 100% current diet (ignore neighbors)"""
    original_init = Agent.initialize_memory_from_neighbours
    def new_init(self, G, agents):
        self.memory = [self.diet] * self.params["M"]
    Agent.initialize_memory_from_neighbours = new_init
    return original_init

def run_memory_sweep(memory_values, test_params):
    """Memory (M) parameter sweep"""
    print("\n" + "=" * 60)
    print(f"Memory (M) Parameter Sweep: {len(memory_values)} configurations")
    print("=" * 60)

    def run_single_M(M_value):
        params_with_M = copy.deepcopy(test_params)
        params_with_M['M'] = M_value
        print(f"Running: M={M_value} (2 runs)")
        trajectories = []
        for run in range(2):
            model = Model(params_with_M)
            model.run()
            trajectories.append(model.fraction_veg.copy())
        avg_trajectory = np.mean(trajectories, axis=0)
        result = {
            'M': M_value,
            'trajectory': avg_trajectory,
            'initial': avg_trajectory[0],
            'at_25k': avg_trajectory[25000],
            'final': avg_trajectory[-1]
        }
        print(f"  Done: M={M_value} -> 25k:{result['at_25k']:.3f}, final:{result['final']:.3f}")
        return result

    n_cores = min(len(memory_values), os.cpu_count() or 4)
    print(f"Using {n_cores} cores\n")
    with Pool(n_cores) as pool:
        results = pool.map(run_single_M, memory_values)

    return {r['M']: r for r in results}

def run_initialization_test(M_values, test_params):
    """Test 100% current diet vs neighbor-based initialization"""
    print("\n" + "=" * 60)
    print("TEST: 100% Current Diet vs Neighbor-Based Initialization")
    print("=" * 60)

    pure_init_results = {}
    neighbor_init_results = {}

    for M in M_values:
        print(f"\n[M={M}] Testing both initialization methods...")
        test_params_M = copy.deepcopy(test_params)
        test_params_M['M'] = M

        # 100% current diet
        original = modify_pure_diet_initialization()
        model = Model(test_params_M)
        model.run()
        pure_init_results[M] = model.fraction_veg
        Agent.initialize_memory_from_neighbours = original

        # Neighbor-based
        model2 = Model(test_params_M)
        model2.run()
        neighbor_init_results[M] = model2.fraction_veg

        print(f"  Pure: 25k={pure_init_results[M][25000]:.3f}, Neighbor: 25k={neighbor_init_results[M][25000]:.3f}")

    return pure_init_results, neighbor_init_results

def run_inertia_test(inertia_penalties, test_params):
    """Test initial inertia strategy for utility model"""
    print("\n" + "=" * 60)
    print("TESTING: Initial Inertia Strategy (Utility Model)")
    print("=" * 60)

    def run_single_variant(penalty):
        variant_name = f"Inertia penalty={penalty:.2f}" if penalty > 0 else "Baseline"
        print(f"[{variant_name}] Running...")

        original_prob_calc = Agent.prob_calc
        model = Model(test_params)

        if penalty > 0:
            Agent.prob_calc = create_utility_prob_calc_with_inertia(penalty)

        model.agent_ini()
        model.plot_params()
        model.harmonise_netIn()

        if penalty > 0:
            mark_agents_with_initial_inertia(model)

        model.record_fraction()
        model.record_snapshot(0)

        time_array = list(range(model.params["steps"]))
        for t in time_array:
            i = np.random.choice(range(len(model.agents)))
            model.agents[i].step(model.G1, model.agents, t)
            model.rewire(model.agents[i])
            model.system_C.append(model.get_attribute("C")/model.params["N"])
            model.record_fraction()
            if t in model.snapshot_times:
                model.record_snapshot(t)
            model.harmonise_netIn()

        Agent.prob_calc = original_prob_calc

        result = {
            'variant_name': variant_name,
            'penalty': penalty,
            'fraction_veg': model.fraction_veg.copy(),
            'initial': model.fraction_veg[0],
            'at_5k': model.fraction_veg[5000],
            'at_25k': model.fraction_veg[25000],
            'final': model.fraction_veg[-1]
        }
        print(f"  Done: 5k={result['at_5k']:.3f}, 25k={result['at_25k']:.3f}")
        return result

    n_cores = min(len(inertia_penalties), os.cpu_count() or 4)
    print(f"Using {n_cores} cores\n")
    with Pool(n_cores) as pool:
        results = pool.map(run_single_variant, inertia_penalties)

    return {r['penalty']: r for r in results}

def plot_memory_sweep_results(results_dict, memory_values, outfile='visualisations_output/sweep_memory_M.png'):
    """Plot memory sweep results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(memory_values)))
    for M, color in zip(memory_values, colors):
        r = results_dict[M]
        ax1.plot(r['trajectory'], label=f'M={M}', linewidth=0.5, color=color, alpha=0.9)
    ax1.axhline(y=0.016, color='red', linestyle='--', alpha=0.3, linewidth=1, label='Initial')
    ax1.set_xlabel('t (steps)')
    ax1.set_ylabel('Vegetarian Fraction')
    ax1.set_title('Memory Parameter Sweep: Trajectories')
    ax1.legend()
    ax1.grid(alpha=0.3)

    metrics_data = {
        'At 25k': [results_dict[M]['at_25k'] for M in memory_values],
        'Final': [results_dict[M]['final'] for M in memory_values]
    }
    x = np.arange(len(memory_values))
    width = 0.35
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax2.bar(x + i*width, values, width, label=metric, alpha=0.8)
    ax2.set_xlabel('Memory (M)')
    ax2.set_ylabel('Vegetarian Fraction')
    ax2.set_title('Memory Parameter Sweep: Metrics')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(memory_values)
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs("visualisations_output", exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {outfile}")

def plot_initialization_results(pure_results, neighbor_results, M_values,
                                outfile='visualisations_output/initialization_comparison.png'):
    """Plot initialization comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for M, trajectory in pure_results.items():
        ax.plot(trajectory, label=f'M={M}', alpha=0.8, linewidth=2 if M == 5 else 1)
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title('100% Current Diet Initialization')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for M, trajectory in neighbor_results.items():
        ax.plot(trajectory, label=f'M={M}', alpha=0.8, linewidth=2 if M == 5 else 1)
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title('Neighbor-Based Initialization')
    ax.legend()
    ax.grid(alpha=0.3)

    for idx, M in enumerate([5, 10]):
        ax = axes[1, idx]
        ax.plot(pure_results[M], label='100% Current Diet', alpha=0.8, linewidth=2)
        ax.plot(neighbor_results[M], label='Neighbor-Based', alpha=0.8, linewidth=2, linestyle='--')
        ax.set_xlabel('t (steps)')
        ax.set_ylabel('Vegetarian Fraction')
        ax.set_title(f'Comparison: M={M}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("visualisations_output", exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {outfile}")

def plot_inertia_results(results_dict, penalties, outfile='visualisations_output/initial_inertia_utility.png'):
    """Plot inertia test results"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, penalty in enumerate(penalties):
        r = results_dict[penalty]
        ax1.plot(r['fraction_veg'], label=r['variant_name'], alpha=0.8, linewidth=2, color=colors[i])
    ax1.set_xlabel('t (steps)')
    ax1.set_ylabel('Vegetarian Fraction')
    ax1.set_title('Initial Inertia Strategy: Full Time Series')
    ax1.legend()
    ax1.grid(alpha=0.3)

    for i, penalty in enumerate(penalties):
        r = results_dict[penalty]
        ax2.plot(r['fraction_veg'][:10000], label=r['variant_name'], alpha=0.8, linewidth=2, color=colors[i])
    ax2.set_xlabel('t (steps)')
    ax2.set_ylabel('Vegetarian Fraction')
    ax2.set_title('Initial Inertia Strategy: Early Dynamics (0-10k)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("visualisations_output", exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {outfile}")

if __name__ == '__main__':
    test_params = copy.deepcopy(params)
    test_params['steps'] = 100000
    test_params['N'] = 699
    test_params['agent_ini'] = "twin"

    print("CONSOLIDATED INITIAL DYNAMICS TESTING")
    print("=" * 70)
    print("\nAvailable tests:")
    print("1. Memory (M) parameter sweep")
    print("2. Initialization method comparison (pure vs neighbor-based)")
    print("3. Initial inertia strategy (utility model)")
    print("\nSelect test (1-3) or 'all': ", end='')

    choice = input().strip()

    if choice in ['1', 'all']:
        memory_values = list(range(5, 11))
        results = run_memory_sweep(memory_values, test_params)
        plot_memory_sweep_results(results, memory_values)

    if choice in ['2', 'all']:
        M_values = [5, 10, 15, 20]
        pure_results, neighbor_results = run_initialization_test(M_values, test_params)
        plot_initialization_results(pure_results, neighbor_results, M_values)

    if choice in ['3', 'all']:
        test_params_inertia = copy.deepcopy(DEFAULT_PARAMS)
        test_params_inertia['steps'] = 50000
        test_params_inertia['N'] = 5602
        test_params_inertia['agent_ini'] = "twin"
        inertia_penalties = [0.0, 0.10, 0.15, 0.20, 0.25]
        results = run_inertia_test(inertia_penalties, test_params_inertia)
        plot_inertia_results(results, inertia_penalties)

    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)
