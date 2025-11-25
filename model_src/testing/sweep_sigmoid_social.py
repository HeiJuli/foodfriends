#!/usr/bin/env python3
"""
2D parameter sweep: sigmoid coefficient vs social pressure scaling
Tests synergistic effects on initial uptake dynamics
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

# Parameter grid
sigmoid_coeffs = [2.3, 2.5, 2.7, 3.0]
social_scalings = [3.0, 3.3, 3.6, 4.0]

# Test parameters
test_params = copy.deepcopy(params)
test_params['steps'] = 100000
test_params['N'] = 699
test_params['agent_ini'] = "twin"

def run_single_config(config):
    """Run model with specific sigmoid coeff and social scaling"""
    sigmoid_coeff, social_scale = config

    # Monkey patch Agent methods
    original_prob_calc = Agent.prob_calc
    original_calc_utility = Agent.calc_utility

    def new_prob_calc(self, other_agent):
        u_i = self.calc_utility(other_agent, mode="same")
        u_s = self.calc_utility(other_agent, mode="diff")
        prob_switch = 1/(1+np.exp(-sigmoid_coeff*(u_s-u_i)))
        return prob_switch

    def new_calc_utility(self, other_agent, mode):
        if mode == "same":
            diet = self.diet
        else:
            diet = "meat" if self.diet == "veg" else "veg"

        if len(self.memory) == 0:
            return 0.0

        mem_same = sum(1 for x in self.memory[-self.params["M"]:] if x == diet)
        ratio = mem_same/len(self.memory[-self.params["M"]:])

        util = self.beta*(social_scale*ratio - social_scale/2) + self.alpha*self.dissonance_new("simple", mode)
        return util

    Agent.prob_calc = new_prob_calc
    Agent.calc_utility = new_calc_utility

    # Run model
    print(f"Running: sigmoid={sigmoid_coeff:.1f}, social={social_scale:.1f}")
    model = Model(test_params)
    model.run()

    # Restore original methods
    Agent.prob_calc = original_prob_calc
    Agent.calc_utility = original_calc_utility

    # Calculate metrics
    initial = model.fraction_veg[0]
    at_25k = model.fraction_veg[25000]
    at_50k = model.fraction_veg[50000]
    final = model.fraction_veg[-1]

    result = {
        'sigmoid': sigmoid_coeff,
        'social': social_scale,
        'trajectory': model.fraction_veg.copy(),
        'initial': initial,
        'at_25k': at_25k,
        'at_50k': at_50k,
        'final': final
    }

    print(f"  Done: sigmoid={sigmoid_coeff:.1f}, social={social_scale:.1f} -> 25k:{at_25k:.3f}, final:{final:.3f}")
    return result

if __name__ == '__main__':
    # Generate all combinations
    configs = [(sig, soc) for sig in sigmoid_coeffs for soc in social_scalings]

    print("=" * 60)
    print(f"2D Parameter Sweep: {len(configs)} configurations")
    print(f"Sigmoid coefficients: {sigmoid_coeffs}")
    print(f"Social scalings: {social_scalings}")
    print("=" * 60)

    # Run in parallel
    n_cores = min(len(configs), os.cpu_count() or 4)
    print(f"\nUsing {n_cores} cores for parallel execution\n")

    with Pool(n_cores) as pool:
        results = pool.map(run_single_config, configs)

    # Organize results into grid
    result_grid = {}
    for r in results:
        key = (r['sigmoid'], r['social'])
        result_grid[key] = r

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Sigmoid':>8} {'Social':>8} {'Initial':>8} {'At 25k':>8} {'At 50k':>8} {'Final':>8}")
    print("-" * 80)
    for sig in sigmoid_coeffs:
        for soc in social_scalings:
            r = result_grid[(sig, soc)]
            print(f"{sig:>8.1f} {soc:>8.1f} {r['initial']:>8.3f} {r['at_25k']:>8.3f} {r['at_50k']:>8.3f} {r['final']:>8.3f}")
    print("=" * 80)

    # Create visualization grid
    fig, axes = plt.subplots(len(sigmoid_coeffs), len(social_scalings),
                             figsize=(16, 12), sharex=True, sharey=True)

    for i, sig in enumerate(sigmoid_coeffs):
        for j, soc in enumerate(social_scalings):
            ax = axes[i, j]
            r = result_grid[(sig, soc)]

            # Plot trajectory
            ax.plot(r['trajectory'], linewidth=1.5, color='C0')
            ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax.grid(alpha=0.3, linewidth=0.5)

            # Labels
            if i == 0:
                ax.set_title(f'Social={soc:.1f}', fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'Sigmoid={sig:.1f}\n\nVeg Fraction', fontsize=9, fontweight='bold')
            if i == len(sigmoid_coeffs) - 1:
                ax.set_xlabel('t (steps)', fontsize=9)

            # Add metrics text
            text = f'25k: {r["at_25k"]:.2f}\n50k: {r["at_50k"]:.2f}\nEnd: {r["final"]:.2f}'
            ax.text(0.98, 0.97, text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('2D Parameter Sweep: Sigmoid Coefficient vs Social Pressure Scaling',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    os.makedirs("../visualisations_output", exist_ok=True)
    outfile = '../visualisations_output/sweep_sigmoid_social_2d.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {outfile}")

    # Also create a heatmap of final fractions
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    metrics = ['at_25k', 'at_50k', 'final']
    titles = ['Veg Fraction at 25k steps', 'Veg Fraction at 50k steps', 'Final Veg Fraction']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes2[idx]

        # Build matrix
        matrix = np.zeros((len(sigmoid_coeffs), len(social_scalings)))
        for i, sig in enumerate(sigmoid_coeffs):
            for j, soc in enumerate(social_scalings):
                matrix[i, j] = result_grid[(sig, soc)][metric]

        # Plot heatmap
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xticks(range(len(social_scalings)))
        ax.set_yticks(range(len(sigmoid_coeffs)))
        ax.set_xticklabels([f'{s:.1f}' for s in social_scalings])
        ax.set_yticklabels([f'{s:.1f}' for s in sigmoid_coeffs])
        ax.set_xlabel('Social Scaling', fontsize=10)
        ax.set_ylabel('Sigmoid Coefficient', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Add values
        for i in range(len(sigmoid_coeffs)):
            for j in range(len(social_scalings)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    outfile2 = '../visualisations_output/sweep_sigmoid_social_heatmaps.png'
    plt.savefig(outfile2, dpi=300, bbox_inches='tight')
    print(f"Heatmaps saved to: {outfile2}")
    print("\n" + "=" * 60)
    print("Parameter sweep complete!")
    print("=" * 60)
