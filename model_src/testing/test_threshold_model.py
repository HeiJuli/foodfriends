#!/usr/bin/env python3
"""
Test threshold model with different w_i weights (5 to 20)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
import argparse
from multiprocessing import Pool
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_main_threshold import Model, params

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

def run_single_variant(config):
    """Worker function to run a single model variant with averaging over n_runs"""
    variant_name, w_i_value, test_params, n_runs = config

    print(f"[{variant_name}] Running {n_runs} replicate(s)...")

    # Update w_i parameter
    test_params['w_i'] = w_i_value

    # Run model n_runs times and collect trajectories
    all_trajectories = []
    for run_idx in range(n_runs):
        model = Model(test_params)
        model.run()
        all_trajectories.append(model.fraction_veg.copy())

    # Average trajectories
    avg_trajectory = np.mean(all_trajectories, axis=0)

    # Extract metrics from averaged trajectory
    result = {
        'variant_name': variant_name,
        'w_i': w_i_value,
        'fraction_veg': avg_trajectory,
        'initial': avg_trajectory[0],
        'at_5k': avg_trajectory[5000] if len(avg_trajectory) > 5000 else avg_trajectory[-1],
        'at_25k': avg_trajectory[25000] if len(avg_trajectory) > 25000 else avg_trajectory[-1],
        'at_50k': avg_trajectory[50000] if len(avg_trajectory) > 50000 else avg_trajectory[-1],
        'final': avg_trajectory[-1]
    }

    print(f"  {variant_name} complete (avg over {n_runs} runs): Initial={result['initial']:.3f}, " +
          f"At 5k={result['at_5k']:.3f}, At 25k={result['at_25k']:.3f}, " +
          f"At 50k={result['at_50k']:.3f}, Final={result['final']:.3f}")

    return result

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test threshold model with different w_i values')
    parser.add_argument('--n_runs', type=int, default=2, help='Number of runs to average over (default: 1)')
    args = parser.parse_args()

    # Test configuration
    test_params = copy.deepcopy(params)
    test_params['steps'] = 150000
    test_params['N'] = 1000
    test_params['agent_ini'] = "twin"

    print("=" * 70)
    print("TESTING: Threshold Model with w_i weights from 5 to 20")
    print("=" * 70)

    # Test w_i values from 5 to 20
    w_i_values = np.linspace(5, 11, 8)

    # Configure all variants to run
    configs = [(f'w_i={w:.1f}', w, copy.deepcopy(test_params), args.n_runs) for w in w_i_values]

    # Run in parallel
    n_cores = min(len(configs), os.cpu_count() or 4)
    print(f"\nUsing {n_cores} cores for parallel execution")
    print(f"Testing w_i values: {[f'{w:.1f}' for w in w_i_values]}")
    print(f"Averaging over {args.n_runs} run(s) per variant\n")

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

    # Plot comparison - 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Add n_runs info to title if averaging
    title_suffix = f' (avg over {args.n_runs} runs)' if args.n_runs > 1 else ''

    # Full trajectory
    ax = axes[0]
    for w in w_i_values:
        r = results_dict[f'w_i={w:.1f}']
        ax.plot(r['fraction_veg'], label=f'w_i={w:.1f}', alpha=0.8, linewidth=2)
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title(f'Full Trajectory (0-100k steps){title_suffix}')
    ax.legend()
    ax.grid(alpha=0.3)

    # Initial 10k steps
    ax = axes[1]
    for w in w_i_values:
        r = results_dict[f'w_i={w:.1f}']
        ax.plot(r['fraction_veg'][:10000], label=f'w_i={w:.1f}', alpha=0.8, linewidth=2)
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Vegetarian Fraction')
    ax.set_title(f'Initial Dynamics (0-10k steps){title_suffix}')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("visualisations_output", exist_ok=True)
    plt.savefig('visualisations_output/threshold_w_i_comparison.png',
                dpi=300, bbox_inches='tight')
    print("\n" + "=" * 70)
    print("Plot saved to visualisations_output/threshold_w_i_comparison.png")
    print("=" * 70)
