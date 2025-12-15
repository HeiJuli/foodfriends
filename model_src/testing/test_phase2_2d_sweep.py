#!/usr/bin/env python3
"""
Phase 2: 2D parameter sweep of w_i and sigmoid_k
Tests all combinations of w_i ∈ [5, 8, 10, 12, 15] and sigmoid_k ∈ [5, 8, 10, 12, 15]
Based on sigmoid literature review recommendations
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
    """Worker function to run single model variant with averaging over n_runs"""
    w_i_val, sigmoid_k_val, test_params, n_runs = config
    variant_name = f'w_i={w_i_val}_sk={sigmoid_k_val}'

    print(f"[{variant_name}] Running {n_runs} replicate(s)...")

    # Update parameters
    test_params['w_i'] = w_i_val
    test_params['sigmoid_k'] = sigmoid_k_val

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
        'w_i': w_i_val,
        'sigmoid_k': sigmoid_k_val,
        'fraction_veg': avg_trajectory,
        'initial': avg_trajectory[0],
        'at_5k': avg_trajectory[5000] if len(avg_trajectory) > 5000 else avg_trajectory[-1],
        'at_25k': avg_trajectory[25000] if len(avg_trajectory) > 25000 else avg_trajectory[-1],
        'at_50k': avg_trajectory[50000] if len(avg_trajectory) > 50000 else avg_trajectory[-1],
        'final': avg_trajectory[-1]
    }

    print(f"  {variant_name} complete: Initial={result['initial']:.3f}, " +
          f"At 5k={result['at_5k']:.3f}, Final={result['final']:.3f}")

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2: 2D parameter sweep of w_i and sigmoid_k')
    parser.add_argument('--n_runs', type=int, default=2, help='Number of runs to average (default: 2)')
    args = parser.parse_args()

    # Test configuration
    test_params = copy.deepcopy(params)
    test_params['steps'] = 150000
    test_params['N'] = 1000
    test_params['agent_ini'] = "twin"

    print("=" * 70)
    print("PHASE 2: 2D Parameter Sweep (w_i × sigmoid_k)")
    print("=" * 70)

    # Phase 2 parameter values from literature review
    w_i_values = [5, 8, 10, 12, 15]
    sigmoid_k_values = [5, 8, 10, 12, 15]

    # Configure all 25 combinations
    configs = [(w, sk, copy.deepcopy(test_params), args.n_runs)
               for w in w_i_values for sk in sigmoid_k_values]

    # Run in parallel
    n_cores = min(len(configs), os.cpu_count() or 4)
    print(f"\nUsing {n_cores} cores for parallel execution")
    print(f"Testing w_i values: {w_i_values}")
    print(f"Testing sigmoid_k values: {sigmoid_k_values}")
    print(f"Total combinations: {len(configs)}")
    print(f"Averaging over {args.n_runs} run(s) per variant\n")

    with Pool(n_cores) as pool:
        results = pool.map(run_single_variant, configs)

    # Organize results
    results_dict = {r['variant_name']: r for r in results}

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Create summary heatmaps
    final_vals = np.zeros((len(sigmoid_k_values), len(w_i_values)))
    at_5k_vals = np.zeros((len(sigmoid_k_values), len(w_i_values)))

    for i, sk in enumerate(sigmoid_k_values):
        for j, w in enumerate(w_i_values):
            r = results_dict[f'w_i={w}_sk={sk}']
            final_vals[i, j] = r['final']
            at_5k_vals[i, j] = r['at_5k']
            print(f"[w_i={w}, sigmoid_k={sk}] At 5k={r['at_5k']:.3f}, Final={r['final']:.3f}")

    # Plot results
    fig = plt.figure(figsize=(20, 12))

    # 1. Heatmap: Final vegetarian fraction
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(final_vals, cmap='viridis', aspect='auto')
    ax1.set_xticks(range(len(w_i_values)))
    ax1.set_yticks(range(len(sigmoid_k_values)))
    ax1.set_xticklabels(w_i_values)
    ax1.set_yticklabels(sigmoid_k_values)
    ax1.set_xlabel('w_i')
    ax1.set_ylabel('sigmoid_k')
    ax1.set_title('Final Veg Fraction')
    plt.colorbar(im1, ax=ax1)

    # 2. Heatmap: Vegetarian fraction at 5k steps
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(at_5k_vals, cmap='viridis', aspect='auto')
    ax2.set_xticks(range(len(w_i_values)))
    ax2.set_yticks(range(len(sigmoid_k_values)))
    ax2.set_xticklabels(w_i_values)
    ax2.set_yticklabels(sigmoid_k_values)
    ax2.set_xlabel('w_i')
    ax2.set_ylabel('sigmoid_k')
    ax2.set_title('Veg Fraction at 5k Steps')
    plt.colorbar(im2, ax=ax2)

    # 3. Trajectories for fixed sigmoid_k=12 (varying w_i)
    ax3 = plt.subplot(2, 3, 3)
    for w in w_i_values:
        r = results_dict[f'w_i={w}_sk=12']
        ax3.plot(r['fraction_veg'], label=f'w_i={w}', alpha=0.8, linewidth=2)
    ax3.set_xlabel('t (steps)')
    ax3.set_ylabel('Vegetarian Fraction')
    ax3.set_title('Trajectories (sigmoid_k=12, varying w_i)')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Trajectories for fixed w_i=10 (varying sigmoid_k)
    ax4 = plt.subplot(2, 3, 4)
    for sk in sigmoid_k_values:
        r = results_dict[f'w_i=10_sk={sk}']
        ax4.plot(r['fraction_veg'], label=f'sigmoid_k={sk}', alpha=0.8, linewidth=2)
    ax4.set_xlabel('t (steps)')
    ax4.set_ylabel('Vegetarian Fraction')
    ax4.set_title('Trajectories (w_i=10, varying sigmoid_k)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Initial dynamics for diagonal values (w_i = sigmoid_k)
    ax5 = plt.subplot(2, 3, 5)
    for val in [5, 8, 10, 12, 15]:
        r = results_dict[f'w_i={val}_sk={val}']
        ax5.plot(r['fraction_veg'][:10000], label=f'w_i=sk={val}', alpha=0.8, linewidth=2)
    ax5.set_xlabel('t (steps)')
    ax5.set_ylabel('Vegetarian Fraction')
    ax5.set_title('Initial Dynamics (0-10k, w_i=sigmoid_k)')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Identify S-curve candidates (final between 0.2-0.8, at_5k < 0.3)
    ax6 = plt.subplot(2, 3, 6)
    s_curve_candidates = []
    for r in results:
        if 0.2 <= r['final'] <= 0.8 and r['at_5k'] < 0.3:
            s_curve_candidates.append(r)
            ax6.plot(r['fraction_veg'], label=r['variant_name'], alpha=0.8, linewidth=2)
    ax6.set_xlabel('t (steps)')
    ax6.set_ylabel('Vegetarian Fraction')
    ax6.set_title(f'S-Curve Candidates (n={len(s_curve_candidates)})')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("visualisations_output", exist_ok=True)
    plt.savefig('visualisations_output/phase2_2d_sweep.png', dpi=300, bbox_inches='tight')

    print("\n" + "=" * 70)
    print(f"S-curve candidates found: {len(s_curve_candidates)}")
    for r in s_curve_candidates:
        print(f"  {r['variant_name']}: Initial={r['initial']:.3f}, " +
              f"At 5k={r['at_5k']:.3f}, Final={r['final']:.3f}")
    print("=" * 70)
    print("Plot saved to visualisations_output/phase2_2d_sweep.png")
    print("=" * 70)
