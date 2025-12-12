#!/usr/bin/env python3
"""
Consolidated testing script for threshold model variants
Combines initial inertia and slowdown strategies for threshold models
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from multiprocessing import Pool
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_main_single import Model, Agent
from model_runn import DEFAULT_PARAMS

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

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

def create_threshold_prob_calc(k_value=15, variant='baseline', **kwargs):
    """Create threshold probability calculation with optional modifications

    Variants:
    - 'baseline': Standard threshold model
    - 'initial_inertia': Increase threshold for marked agents at t=0
    - 'scaled_floor': Dissonance scaling + threshold floor
    """
    def new_prob_calc(self, other_agent):
        opposite_diet = "meat" if self.diet == "veg" else "veg"
        mem = self.memory[-self.params["M"]:]
        if len(mem) == 0:
            return 0.0
        proportion = sum(d == opposite_diet for d in mem) / len(mem)

        threshold = self.rho
        dissonance_active = False

        if self.diet == "meat":
            has_veg_neighbor = any(n.diet == "veg" for n in self.neighbours)
            theta_misaligned = self.theta > 0.5  # Fixed: was <, now correctly checks if prefers veg
            dissonance_active = has_veg_neighbor and theta_misaligned

            if dissonance_active:
                dissonance = self.theta  # Fixed: was abs(theta-1.0), now distance from meat end
                if variant == 'scaled_floor':
                    scaling = kwargs.get('diss_scale', 0.5)
                else:
                    scaling = 1.0
                threshold -= scaling * self.alpha * dissonance
        else:  # veg
            theta_misaligned = self.theta < 0.5  # Fixed: was >, now correctly checks if prefers meat
            dissonance_active = theta_misaligned
            if dissonance_active:
                dissonance = 1 - self.theta  # Fixed: was abs(theta-0.0), now distance from veg end
                if variant == 'scaled_floor':
                    scaling = kwargs.get('diss_scale', 0.5)
                else:
                    scaling = 1.0
                threshold -= scaling * self.alpha * dissonance

        # Apply variant-specific modifications
        if variant == 'initial_inertia' and id(self) in agents_with_inertia:
            inertia_boost = kwargs.get('inertia_boost', 0.15)
            threshold += inertia_boost

        if variant == 'scaled_floor':
            floor = kwargs.get('floor', 0.15)
            threshold = np.clip(threshold, floor, 1)
        else:
            threshold = np.clip(threshold, 0, 1)

        social_exposure = self.beta * proportion
        prob_switch = 1 / (1 + np.exp(-k_value * (social_exposure - threshold)))
        return prob_switch

    return new_prob_calc

def run_single_variant(config):
    """Worker function to run a single model variant"""
    variant_name, variant_type, k_value, test_params, variant_kwargs = config
    print(f"[{variant_name}] Running...")

    original_prob_calc = Agent.prob_calc
    model = Model(test_params)

    if variant_type != 'utility':
        Agent.prob_calc = create_threshold_prob_calc(k_value, variant_type, **variant_kwargs)

    model.agent_ini()
    model.plot_params()
    model.harmonise_netIn()

    if variant_type == 'initial_inertia':
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
        'variant_type': variant_type,
        'fraction_veg': model.fraction_veg.copy(),
        'initial': model.fraction_veg[0],
        'at_5k': model.fraction_veg[5000],
        'at_25k': model.fraction_veg[25000],
        'final': model.fraction_veg[-1]
    }
    print(f"  Done: 5k={result['at_5k']:.3f}, 25k={result['at_25k']:.3f}, final={result['final']:.3f}")
    return result

def run_inertia_test(test_params):
    """Test initial inertia strategy for threshold model"""
    print("\n" + "=" * 70)
    print("TESTING: Initial Inertia Strategy (Threshold Model)")
    print("=" * 70)

    configs = [
        ('Utility Model', 'utility', None, test_params, {}),
        ('k=15 baseline', 'baseline', 15, test_params, {}),
        ('k=15 + Inertia(0.10)', 'initial_inertia', 15, test_params, {'inertia_boost': 0.10}),
        ('k=15 + Inertia(0.15)', 'initial_inertia', 15, test_params, {'inertia_boost': 0.15}),
        ('k=15 + Inertia(0.20)', 'initial_inertia', 15, test_params, {'inertia_boost': 0.20}),
    ]

    n_cores = min(len(configs), os.cpu_count() or 4)
    print(f"Using {n_cores} cores\n")
    with Pool(n_cores) as pool:
        results = pool.map(run_single_variant, configs)

    return {r['variant_name']: r for r in results}

def run_slowdown_test(test_params):
    """Test slowdown strategies for threshold model"""
    print("\n" + "=" * 70)
    print("TESTING: Slowdown Strategies (Threshold Model)")
    print("=" * 70)

    configs = [
        ('Utility Model', 'utility', None, test_params, {}),
        ('k=15 baseline', 'baseline', 15, test_params, {}),
        ('k=15 + Scale(0.5) + Floor(0.15)', 'scaled_floor', 15, test_params,
         {'diss_scale': 0.5, 'floor': 0.15}),
        ('k=15 + Scale(0.4) + Floor(0.2)', 'scaled_floor', 15, test_params,
         {'diss_scale': 0.4, 'floor': 0.2}),
    ]

    n_cores = min(len(configs), os.cpu_count() or 4)
    print(f"Using {n_cores} cores\n")
    with Pool(n_cores) as pool:
        results = pool.map(run_single_variant, configs)

    return {r['variant_name']: r for r in results}

def plot_comparison(results_dict, title, outfile):
    """Plot comparison of variants"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (name, r) in enumerate(results_dict.items()):
        ax1.plot(r['fraction_veg'], label=name, alpha=0.8, linewidth=2, color=colors[i % len(colors)])
    ax1.set_xlabel('t (steps)')
    ax1.set_ylabel('Vegetarian Fraction')
    ax1.set_title(f'{title}: Full Time Series')
    ax1.legend()
    ax1.grid(alpha=0.3)

    for i, (name, r) in enumerate(results_dict.items()):
        ax2.plot(r['fraction_veg'][:10000], label=name, alpha=0.8, linewidth=2, color=colors[i % len(colors)])
    ax2.set_xlabel('t (steps)')
    ax2.set_ylabel('Vegetarian Fraction')
    ax2.set_title(f'{title}: Early Dynamics (0-10k)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("visualisations_output", exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {outfile}")

if __name__ == '__main__':
    test_params = copy.deepcopy(DEFAULT_PARAMS)
    test_params['steps'] = 50000
    test_params['N'] = 5602
    test_params['agent_ini'] = "twin"

    print("CONSOLIDATED THRESHOLD MODEL TESTING")
    print("=" * 70)
    print("\nAvailable tests:")
    print("1. Initial inertia strategy")
    print("2. Slowdown strategies (scaling + floor)")
    print("\nSelect test (1-2) or 'all': ", end='')

    choice = input().strip()

    if choice in ['1', 'all']:
        results = run_inertia_test(test_params)
        plot_comparison(results, 'Initial Inertia Strategy',
                       'visualisations_output/threshold_initial_inertia.png')
        print("\nRESULTS SUMMARY:")
        for name, r in results.items():
            print(f"[{name}] Initial: {r['initial']:.3f}, At 5k: {r['at_5k']:.3f}, " +
                  f"At 25k: {r['at_25k']:.3f}, Final: {r['final']:.3f}")

    if choice in ['2', 'all']:
        test_params_slow = copy.deepcopy(test_params)
        test_params_slow['steps'] = 400000
        results = run_slowdown_test(test_params_slow)
        plot_comparison(results, 'Slowdown Strategies',
                       'visualisations_output/threshold_slowdown_strategies.png')
        print("\nRESULTS SUMMARY:")
        for name, r in results.items():
            print(f"[{name}] Initial: {r['initial']:.3f}, At 5k: {r['at_5k']:.3f}, " +
                  f"At 25k: {r['at_25k']:.3f}, Final: {r['final']:.3f}")

    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)
