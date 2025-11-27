#!/usr/bin/env python3
"""
Test complex contagion threshold model vs utility-based model
Comparing k=15, k=15+scaled_dissonance, k=15+threshold_floor
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

def create_threshold_prob_calc(variant='baseline'):
    """Create threshold probability calculation function for variant

    Args:
        variant: 'baseline' (k=15), 'scaled_diss' (k=15+0.2*dissonance),
                 'floor' (k=15+0.1 floor)
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
            dissonance_active = has_veg_neighbor and theta_misaligned

            if dissonance_active:
                dissonance = abs(self.theta - 1.0)
                scaling = 0.2 if variant == 'scaled_diss' else 1.0
                threshold -= scaling * self.alpha * dissonance
        else:  # veg
            theta_misaligned = self.theta > 0.5
            dissonance_active = theta_misaligned

            if dissonance_active:
                dissonance = abs(self.theta - 0.0)
                scaling = 0.2 if variant == 'scaled_diss' else 1.0
                threshold -= scaling * self.alpha * dissonance

        if variant == 'floor':
            threshold = np.clip(threshold, 0.1, 1)
        else:
            threshold = np.clip(threshold, 0, 1)

        social_exposure = self.beta * proportion
        k = 15
        prob_switch = 1 / (1 + np.exp(-k * (social_exposure - threshold)))

        return prob_switch

    return new_prob_calc

def run_single_variant(config):
    """Worker function to run a single model variant"""
    variant_name, variant_type, test_params = config

    print(f"[{variant_name}] Running...")

    # Monkey patch for threshold variants
    original_prob_calc = Agent.prob_calc
    if variant_type != 'utility':
        Agent.prob_calc = create_threshold_prob_calc(variant_type)

    # Run model
    model = Model(test_params)
    model.run()

    # Restore original
    Agent.prob_calc = original_prob_calc

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
    test_params['steps'] = 700000
    test_params['N'] = 800
    test_params['agent_ini'] = "twin"

    print("=" * 70)
    print("TESTING: Threshold Model Variants vs Utility Model")
    print("=" * 70)

    # Configure all variants to run
    configs = [
        ('Utility Model', 'utility', test_params),
        ('Threshold k=15', 'baseline', test_params),
        ('k=15 + Scaled Diss (0.2x)', 'scaled_diss', test_params),
        ('k=15 + Floor (0.1)', 'floor', test_params)
    ]

    # Run in parallel
    n_cores = min(len(configs), os.cpu_count() or 4)
    print(f"\nUsing {n_cores} cores for parallel execution\n")

    with Pool(n_cores) as pool:
        results = pool.map(run_single_variant, configs)

    # Organize results by variant name
    results_dict = {r['variant_name']: r for r in results}

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"\n[{r['variant_name']}]")
        print(f"  Initial: {r['initial']:.3f}")
        print(f"  At 5k:   {r['at_5k']:.3f}")
        print(f"  At 25k:  {r['at_25k']:.3f}")
        print(f"  At 50k:  {r['at_50k']:.3f}")
        print(f"  Final:   {r['final']:.3f}")

    # Extract trajectories for plotting
    model_utility = type('obj', (), {'fraction_veg': results_dict['Utility Model']['fraction_veg']})
    model_k15 = type('obj', (), {'fraction_veg': results_dict['Threshold k=15']['fraction_veg']})
    model_scaled = type('obj', (), {'fraction_veg': results_dict['k=15 + Scaled Diss (0.2x)']['fraction_veg']})
    model_floor = type('obj', (), {'fraction_veg': results_dict['k=15 + Floor (0.1)']['fraction_veg']})

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Full trajectory
    ax = axes[0, 0]
    ax.plot(model_utility.fraction_veg, label='Utility Model', alpha=0.8, linewidth=2)
    ax.plot(model_k15.fraction_veg, label='Threshold k=15', alpha=0.8, linewidth=2)
    ax.plot(model_scaled.fraction_veg, label='k=15 + Scaled Diss (0.2x)', alpha=0.8, linewidth=2)
    ax.plot(model_floor.fraction_veg, label='k=15 + Floor (0.1)', alpha=0.8, linewidth=2)
    ax.axhline(y=0.016, color='red', linestyle='--', alpha=0.3, label='Initial')
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title('Full Trajectory (0-100k steps)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Initial 10k steps
    ax = axes[0, 1]
    ax.plot(model_utility.fraction_veg[:10000], label='Utility Model', alpha=0.8, linewidth=2)
    ax.plot(model_k15.fraction_veg[:10000], label='Threshold k=15', alpha=0.8, linewidth=2)
    ax.plot(model_scaled.fraction_veg[:10000], label='k=15 + Scaled Diss', alpha=0.8, linewidth=2)
    ax.plot(model_floor.fraction_veg[:10000], label='k=15 + Floor', alpha=0.8, linewidth=2)
    ax.axhline(y=0.016, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title('Initial Dynamics (0-10k steps)')
    ax.legend()
    ax.grid(alpha=0.3)

    # 10k-50k steps
    ax = axes[1, 0]
    ax.plot(range(10000, 50000), model_utility.fraction_veg[10000:50000],
            label='Utility Model', alpha=0.8, linewidth=2)
    ax.plot(range(10000, 50000), model_k15.fraction_veg[10000:50000],
            label='Threshold k=15', alpha=0.8, linewidth=2)
    ax.plot(range(10000, 50000), model_scaled.fraction_veg[10000:50000],
            label='k=15 + Scaled Diss', alpha=0.8, linewidth=2)
    ax.plot(range(10000, 50000), model_floor.fraction_veg[10000:50000],
            label='k=15 + Floor', alpha=0.8, linewidth=2)
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title('Mid-term Dynamics (10k-50k steps)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    data = [
        ['Model', 'Initial', 'At 5k', 'At 25k', 'At 50k', 'Final'],
        ['Utility', f'{model_utility.fraction_veg[0]:.3f}',
         f'{model_utility.fraction_veg[5000]:.3f}',
         f'{model_utility.fraction_veg[25000]:.3f}',
         f'{model_utility.fraction_veg[50000]:.3f}',
         f'{model_utility.fraction_veg[-1]:.3f}'],
        ['k=15', f'{model_k15.fraction_veg[0]:.3f}',
         f'{model_k15.fraction_veg[5000]:.3f}',
         f'{model_k15.fraction_veg[25000]:.3f}',
         f'{model_k15.fraction_veg[50000]:.3f}',
         f'{model_k15.fraction_veg[-1]:.3f}'],
        ['k=15+Scaled', f'{model_scaled.fraction_veg[0]:.3f}',
         f'{model_scaled.fraction_veg[5000]:.3f}',
         f'{model_scaled.fraction_veg[25000]:.3f}',
         f'{model_scaled.fraction_veg[50000]:.3f}',
         f'{model_scaled.fraction_veg[-1]:.3f}'],
        ['k=15+Floor', f'{model_floor.fraction_veg[0]:.3f}',
         f'{model_floor.fraction_veg[5000]:.3f}',
         f'{model_floor.fraction_veg[25000]:.3f}',
         f'{model_floor.fraction_veg[50000]:.3f}',
         f'{model_floor.fraction_veg[-1]:.3f}']
    ]
    table = ax.table(cellText=data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(data[0])):
        table[(0, i)].set_facecolor('#E0E0E0')
    ax.set_title('Summary Statistics', fontsize=12, pad=20)

    plt.tight_layout()
    os.makedirs("visualisations_output", exist_ok=True)
    plt.savefig('visualisations_output/threshold_variants_comparison.png',
                dpi=300, bbox_inches='tight')
    print("\n" + "=" * 70)
    print("Plot saved to visualisations_output/threshold_variants_comparison.png")
    print("=" * 70)
