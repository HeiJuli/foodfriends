#!/usr/bin/env python3
"""
Memory (M) parameter sweep for original utility model
Tests effect of memory length on initial uptake dynamics
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from multiprocessing import Pool
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_main_single import Model, params

# Memory values to test
memory_values = list(range(5, 11))  # [5, 6, 7, 8, 9, 10]

# Test parameters
test_params = copy.deepcopy(params)
test_params['steps'] = 100000
test_params['N'] = 699
test_params['agent_ini'] = "twin"

def run_single_config(M_value):
    """Run model with specific memory value"""
    # Set M parameter
    params_with_M = copy.deepcopy(test_params)
    params_with_M['M'] = M_value

    # Run model
    print(f"Running: M={M_value}")
    model = Model(params_with_M)
    model.run()

    # Calculate metrics
    initial = model.fraction_veg[0]
    at_25k = model.fraction_veg[25000]
    at_50k = model.fraction_veg[50000]
    final = model.fraction_veg[-1]

    result = {
        'M': M_value,
        'trajectory': model.fraction_veg.copy(),
        'initial': initial,
        'at_25k': at_25k,
        'at_50k': at_50k,
        'final': final
    }

    print(f"  Done: M={M_value} -> 25k:{at_25k:.3f}, final:{final:.3f}")
    return result

if __name__ == '__main__':
    # Change to model_src directory so relative paths work
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    print("=" * 60)
    print(f"Memory (M) Parameter Sweep: {len(memory_values)} configurations")
    print(f"Memory values: {memory_values}")
    print("=" * 60)

    # Run in parallel
    n_cores = min(len(memory_values), os.cpu_count() or 4)
    print(f"\nUsing {n_cores} cores for parallel execution\n")

    with Pool(n_cores) as pool:
        results = pool.map(run_single_config, memory_values)

    # Organize results
    result_dict = {r['M']: r for r in results}

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'M':>4} {'Initial':>8} {'At 25k':>8} {'At 50k':>8} {'Final':>8}")
    print("-" * 60)
    for M in memory_values:
        r = result_dict[M]
        print(f"{M:>4} {r['initial']:>8.3f} {r['at_25k']:>8.3f} {r['at_50k']:>8.3f} {r['final']:>8.3f}")
    print("=" * 60)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(memory_values)))
    for M, color in zip(memory_values, colors):
        r = result_dict[M]
        ax1.plot(r['trajectory'], label=f'M={M}', linewidth=2, color=color, alpha=0.8)
    ax1.axhline(y=0.016, color='red', linestyle='--', alpha=0.3, linewidth=1, label='Initial')
    ax1.set_xlabel('t (steps)')
    ax1.set_ylabel('Vegetarian Fraction')
    ax1.set_title('Memory Parameter Sweep: Trajectories')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Metrics comparison
    metrics_data = {
        'At 25k': [result_dict[M]['at_25k'] for M in memory_values],
        'At 50k': [result_dict[M]['at_50k'] for M in memory_values],
        'Final': [result_dict[M]['final'] for M in memory_values]
    }
    x = np.arange(len(memory_values))
    width = 0.25
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax2.bar(x + i*width, values, width, label=metric, alpha=0.8)
    ax2.set_xlabel('Memory (M)')
    ax2.set_ylabel('Vegetarian Fraction')
    ax2.set_title('Memory Parameter Sweep: Metrics')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(memory_values)
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    os.makedirs("visualisations_output", exist_ok=True)
    outfile = 'visualisations_output/sweep_memory_M.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {outfile}")
    print("\n" + "=" * 60)
    print("Parameter sweep complete!")
    print("=" * 60)
