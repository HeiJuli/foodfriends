#!/usr/bin/env python3
"""
Systematic testing of parameters affecting initial uptake rate
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_main_single import Model, Agent, params

# Test configurations - refined ranges around current values
sigmoid_coeffs = [2.3, 2.5, 2.7, 3.0]  # Current is 2.3
social_scalings = [3.0, 3.3, 3.6, 4.0]  # Current uses 3.0 in (3*ratio - 1.5)

# Reduced run for speed - focus on initial uptake only
test_params = copy.deepcopy(params)
test_params['steps'] = 100000  # Shorter for speed, focus on initial rise
test_params['N'] = 699
test_params['agent_ini'] = "twin"  # Use individual survey parameters

def modify_agent_sigmoid(coeff):
    """Monkey patch Agent.prob_calc to use different sigmoid coefficient"""
    original_prob_calc = Agent.prob_calc
    def new_prob_calc(self, other_agent):
        u_i = self.calc_utility(other_agent, mode="same")
        u_s = self.calc_utility(other_agent, mode="diff")
        prob_switch = 1/(1+np.exp(-coeff*(u_s-u_i)))
        if self.diet == 'meat':
            return prob_switch
        else:
            return prob_switch
    Agent.prob_calc = new_prob_calc
    return original_prob_calc

def modify_agent_social_scaling(scaling):
    """Monkey patch Agent.calc_utility to use different social pressure scaling"""
    original_calc_utility = Agent.calc_utility
    def new_calc_utility(self, other_agent, mode):
        if mode == "same":
            diet = self.diet
        else:
            diet = "meat" if self.diet == "veg" else "veg"

        if len(self.memory) == 0:
            return 0.0

        mem_same = sum(1 for x in self.memory[-self.params["M"]:] if x == diet)
        ratio = mem_same/len(self.memory[-self.params["M"]:])

        # Modified social pressure term
        util = self.beta*(scaling*ratio - scaling/2) + self.alpha*self.dissonance_new("simple", mode)
        return util
    Agent.calc_utility = new_calc_utility
    return original_calc_utility

# Change to model_src directory so relative paths work
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Test 1: Sigmoid coefficient
print("=" * 60)
print("TEST 1: Sigmoid coefficient sensitivity")
print("=" * 60)

sigmoid_results = {}
for coeff in sigmoid_coeffs:
    print(f"\nTesting sigmoid coefficient = {coeff}")
    original = modify_agent_sigmoid(coeff)

    model = Model(test_params)
    model.run()

    sigmoid_results[coeff] = model.fraction_veg
    Agent.prob_calc = original  # Restore
    print(f"  Initial: {model.fraction_veg[0]:.3f}")
    print(f"  At 25k:  {model.fraction_veg[25000]:.3f}")
    print(f"  At 50k:  {model.fraction_veg[50000]:.3f}")
    print(f"  Final:   {model.fraction_veg[-1]:.3f}")

# Test 2: Social pressure scaling
print("\n" + "=" * 60)
print("TEST 2: Social pressure scaling sensitivity")
print("=" * 60)

social_results = {}
for scaling in social_scalings:
    print(f"\nTesting social scaling = {scaling} (term: {scaling}*ratio - {scaling/2})")
    original = modify_agent_social_scaling(scaling)

    model = Model(test_params)
    model.run()

    social_results[scaling] = model.fraction_veg
    Agent.calc_utility = original  # Restore
    print(f"  Initial: {model.fraction_veg[0]:.3f}")
    print(f"  At 25k:  {model.fraction_veg[25000]:.3f}")
    print(f"  At 50k:  {model.fraction_veg[50000]:.3f}")
    print(f"  Final:   {model.fraction_veg[-1]:.3f}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Sigmoid coefficient
ax = axes[0]
for coeff, trajectory in sigmoid_results.items():
    ax.plot(trajectory, label=f'coeff={coeff}', alpha=0.8)
ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, label='20% threshold')
ax.set_xlabel('t (steps)')
ax.set_ylabel('Vegetarian Fraction')
ax.set_title('Sigmoid Coefficient (Refined)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Social pressure scaling
ax = axes[1]
for scaling, trajectory in social_results.items():
    ax.plot(trajectory, label=f'scaling={scaling}', alpha=0.8)
ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, label='20% threshold')
ax.set_xlabel('t (steps)')
ax.set_ylabel('Vegetarian Fraction')
ax.set_title('Social Pressure Scaling (Refined)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
os.makedirs("visualisations_output", exist_ok=True)
plt.savefig('visualisations_output/initial_uptake_sensitivity_refined.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print("Plot saved to visualisations_output/initial_uptake_sensitivity_refined.png")
print("=" * 60)
