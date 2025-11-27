#!/usr/bin/env python3
"""
Test threshold model variants designed to slow initial dynamics
while preserving tipping behavior
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

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

def create_threshold_prob_calc(k_value=15, variant='baseline', **kwargs):
    """Create threshold probability calculation function

    Variants:
    - 'baseline': Standard threshold model
    - 'memory_lag': Dissonance only after sustained exposure (veg_count >= threshold)
    - 'scaled_floor': Combine dissonance scaling + threshold floor
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
            theta_misaligned = self.theta < 0.5

            # VARIANT: Memory lag - require sustained exposure
            if variant == 'memory_lag':
                veg_count = sum(d == "veg" for d in mem)
                lag_threshold = kwargs.get('lag_threshold', 3)
                dissonance_active = veg_count >= lag_threshold and theta_misaligned
            else:
                dissonance_active = has_veg_neighbor and theta_misaligned

            if dissonance_active:
                dissonance = abs(self.theta - 1.0)

                # Apply scaling if variant requires it
                if variant == 'scaled_floor':
                    scaling = kwargs.get('diss_scale', 0.5)
                else:
                    scaling = 1.0

                threshold -= scaling * self.alpha * dissonance

        else:  # veg
            theta_misaligned = self.theta > 0.5
            dissonance_active = theta_misaligned

            if dissonance_active:
                dissonance = abs(self.theta - 0.0)

                if variant == 'scaled_floor':
                    scaling = kwargs.get('diss_scale', 0.5)
                else:
                    scaling = 1.0

                threshold -= scaling * self.alpha * dissonance

        # Apply floor if variant requires it
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

    # Monkey patch for threshold variants
    original_prob_calc = Agent.prob_calc
    original_run = Model.run

    if variant_type != 'utility':
        Agent.prob_calc = create_threshold_prob_calc(k_value, variant_type, **variant_kwargs)

    # Run model
    model = Model(test_params)
    model.run()

    # Restore original
    Agent.prob_calc = original_prob_calc
    Model.run = original_run

    # Extract metrics
    result = {
        'variant_name': variant_name,
        'variant_type': variant_type,
        'fraction_veg': model.fraction_veg.copy(),
        'initial': model.fraction_veg[0],
        'at_5k': model.fraction_veg[5000],
        'at_25k': model.fraction_veg[25000],
        'at_50k': model.fraction_veg[50000],
        'final': model.fraction_veg[-1]
    }

    print(f"  {variant_name} complete: Initial={result['initial']:.3f}, " +
          f"At 5k={result['at_5k']:.3f}, At 25k={result['at_25k']:.3f}, " +
          f"At 50k={result['at_50k']:.3f}, Final={result['final']:.3f}")

    return result

if __name__ == '__main__':
    # Test configuration
    test_params = copy.deepcopy(params)
    test_params['steps'] = 100000
    test_params['N'] = 800
    test_params['agent_ini'] = "twin"

    print("=" * 70)
    print("TESTING: Slowdown Strategies for Threshold Model")
    print("=" * 70)

    # Configure variants to test
    configs = [
        ('Utility Model', 'utility', None, test_params, {}),

        # Baseline for comparison
        ('k=15 baseline', 'baseline', 15, test_params, {}),

        # Strategy 1: Memory lag (require sustained exposure)
        ('k=15 + Lag(3)', 'memory_lag', 15, test_params, {'lag_threshold': 3}),
        ('k=15 + Lag(5)', 'memory_lag', 15, test_params, {'lag_threshold': 5}),
        ('k=15 + Lag(7)', 'memory_lag', 15, test_params, {'lag_threshold': 7}),

        # Strategy 2: Combined scaling + floor
        ('k=15 + Scale(0.5) + Floor(0.15)', 'scaled_floor', 15, test_params,
         {'diss_scale': 0.5, 'floor': 0.15}),
        ('k=15 + Scale(0.4) + Floor(0.2)', 'scaled_floor', 15, test_params,
         {'diss_scale': 0.4, 'floor': 0.2}),

        # Strategy 3: Very high k
        ('k=30 baseline', 'baseline', 30, test_params, {}),
        ('k=40 baseline', 'baseline', 40, test_params, {}),
    ]

    # Run in parallel
    n_cores = min(len(configs), os.cpu_count() or 4)
    print(f"\nUsing {n_cores} cores for parallel execution")
    print(f"Testing {len(configs)} variants\n")

    with Pool(n_cores) as pool:
        results = pool.map(run_single_variant, configs)

    # Organize results
    results_dict = {r['variant_name']: r for r in results}

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"\n[{r['variant_name']}]")
        print(f"  Initial: {r['initial']:.3f}")
        print(f"  At 5k:   {r['at_5k']:.3f}  (ratio vs Utility: {r['at_5k']/results[0]['at_5k']:.2f}x)")
        print(f"  At 25k:  {r['at_25k']:.3f}")
        print(f"  At 50k:  {r['at_50k']:.3f}")
        print(f"  Final:   {r['final']:.3f}")

    # Plot comparison - 1x3 layout for different strategies
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Memory lag variants
    ax = axes[0]
    ax.plot(results_dict['Utility Model']['fraction_veg'],
            label='Utility', alpha=0.8, linewidth=2, color='C0')
    ax.plot(results_dict['k=15 baseline']['fraction_veg'],
            label='k=15 baseline', alpha=0.8, linewidth=2, color='C1')
    for lag in [3, 5, 7]:
        name = f'k=15 + Lag({lag})'
        ax.plot(results_dict[name]['fraction_veg'],
                label=name, alpha=0.8, linewidth=2)
    ax.axhline(y=0.016, color='red', linestyle='--', alpha=0.3, label='Initial')
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title('Strategy 1: Memory Lag')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Scaling + floor variants
    ax = axes[1]
    ax.plot(results_dict['Utility Model']['fraction_veg'],
            label='Utility', alpha=0.8, linewidth=2, color='C0')
    ax.plot(results_dict['k=15 baseline']['fraction_veg'],
            label='k=15 baseline', alpha=0.8, linewidth=2, color='C1')
    ax.plot(results_dict['k=15 + Scale(0.5) + Floor(0.15)']['fraction_veg'],
            label='Scale(0.5)+Floor(0.15)', alpha=0.8, linewidth=2)
    ax.plot(results_dict['k=15 + Scale(0.4) + Floor(0.2)']['fraction_veg'],
            label='Scale(0.4)+Floor(0.2)', alpha=0.8, linewidth=2)
    ax.axhline(y=0.016, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title('Strategy 2: Scaling + Floor')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: High k variants
    ax = axes[2]
    ax.plot(results_dict['Utility Model']['fraction_veg'],
            label='Utility', alpha=0.8, linewidth=2, color='C0')
    ax.plot(results_dict['k=15 baseline']['fraction_veg'],
            label='k=15 baseline', alpha=0.8, linewidth=2)
    ax.plot(results_dict['k=30 baseline']['fraction_veg'],
            label='k=30', alpha=0.8, linewidth=2)
    ax.plot(results_dict['k=40 baseline']['fraction_veg'],
            label='k=40', alpha=0.8, linewidth=2)
    ax.axhline(y=0.016, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title('Strategy 3: Very High k')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("visualisations_output", exist_ok=True)
    plt.savefig('visualisations_output/threshold_slowdown_strategies.png',
                dpi=300, bbox_inches='tight')
    print("\n" + "=" * 70)
    print("Plot saved to visualisations_output/threshold_slowdown_strategies.png")
    print("=" * 70)
