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
memory_lengths = [5, 10, 15, 20]  # Current is 5
init_bias_factors = [1.0, 0.7, 0.5, 0.3]  # 1.0=no bias, 0.3=strong bias toward current diet
pure_diet_init_M = [5, 10, 15, 20]  # Test 100% current diet init with different M

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

def modify_memory_initialization(bias_factor):
    """Monkey patch Agent.initialize_memory_from_neighbours to bias toward current diet"""
    original_init = Agent.initialize_memory_from_neighbours
    def new_init(self, G, agents):
        neigh_ids = list(G.neighbors(self.i))
        if len(neigh_ids) == 0:
            self.memory = [self.diet] * self.params["M"]
            return

        neigh_diets = [agents[j].diet for j in neigh_ids]
        n_veg = sum(d == "veg" for d in neigh_diets)
        n_meat = len(neigh_diets) - n_veg

        # Apply bias: reduce exposure to opposite diet
        veg_in_mem = round(self.params["M"] * n_veg / len(neigh_diets) * bias_factor)

        # If agent is veg, ensure they have more veg in memory; if meat, more meat
        if self.diet == "veg":
            veg_in_mem = max(veg_in_mem, self.params["M"] - veg_in_mem)
        else:  # meat
            veg_in_mem = min(veg_in_mem, self.params["M"] - veg_in_mem)

        meat_in_mem = self.params["M"] - veg_in_mem
        self.memory = ["veg"] * veg_in_mem + ["meat"] * meat_in_mem
        np.random.shuffle(self.memory)
    Agent.initialize_memory_from_neighbours = new_init
    return original_init

def modify_pure_diet_initialization():
    """Monkey patch to initialize memory with 100% current diet (ignore neighbors)"""
    original_init = Agent.initialize_memory_from_neighbours
    def new_init(self, G, agents):
        # Initialize with 100% current diet - maximum stability
        self.memory = [self.diet] * self.params["M"]
    Agent.initialize_memory_from_neighbours = new_init
    return original_init

# Change to model_src directory so relative paths work
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Test: 100% Current Diet Initialization with varying M
print("=" * 60)
print("TEST: 100% Current Diet Initialization (vs neighbor-based)")
print("=" * 60)

pure_init_results = {}
neighbor_init_results = {}

for M in pure_diet_init_M:
    # Test with 100% current diet initialization
    print(f"\n[100% Current Diet] Testing M = {M}")
    test_params_M = copy.deepcopy(test_params)
    test_params_M['M'] = M

    original = modify_pure_diet_initialization()
    model = Model(test_params_M)
    model.run()

    pure_init_results[M] = model.fraction_veg
    Agent.initialize_memory_from_neighbours = original  # Restore

    print(f"  Initial: {model.fraction_veg[0]:.3f}")
    print(f"  At 5k:   {model.fraction_veg[5000]:.3f}")
    print(f"  At 25k:  {model.fraction_veg[25000]:.3f}")
    print(f"  At 50k:  {model.fraction_veg[50000]:.3f}")
    print(f"  Final:   {model.fraction_veg[-1]:.3f}")

    # Test with neighbor-based initialization for comparison
    print(f"[Neighbor-Based] Testing M = {M}")
    model2 = Model(test_params_M)
    model2.run()

    neighbor_init_results[M] = model2.fraction_veg

    print(f"  Initial: {model2.fraction_veg[0]:.3f}")
    print(f"  At 5k:   {model2.fraction_veg[5000]:.3f}")
    print(f"  At 25k:  {model2.fraction_veg[25000]:.3f}")
    print(f"  At 50k:  {model2.fraction_veg[50000]:.3f}")
    print(f"  Final:   {model2.fraction_veg[-1]:.3f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: 100% Current Diet Init - All M values
ax = axes[0, 0]
for M, trajectory in pure_init_results.items():
    label = f'M={M}' + (' (current)' if M == 5 else '')
    ax.plot(trajectory, label=label, alpha=0.8, linewidth=2 if M == 5 else 1)
ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.3, label='5% reference')
ax.set_xlabel('t (steps)')
ax.set_ylabel('Vegetarian Fraction')
ax.set_title('100% Current Diet Initialization')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Neighbor-Based Init - All M values
ax = axes[0, 1]
for M, trajectory in neighbor_init_results.items():
    label = f'M={M}' + (' (current)' if M == 5 else '')
    ax.plot(trajectory, label=label, alpha=0.8, linewidth=2 if M == 5 else 1)
ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.3, label='5% reference')
ax.set_xlabel('t (steps)')
ax.set_ylabel('Vegetarian Fraction')
ax.set_title('Neighbor-Based Initialization (Current)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Direct comparison M=5
ax = axes[1, 0]
ax.plot(pure_init_results[5], label='100% Current Diet', alpha=0.8, linewidth=2)
ax.plot(neighbor_init_results[5], label='Neighbor-Based', alpha=0.8, linewidth=2, linestyle='--')
ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.3, label='5% reference')
ax.set_xlabel('t (steps)')
ax.set_ylabel('Vegetarian Fraction')
ax.set_title('Comparison: M=5')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Direct comparison M=10
ax = axes[1, 1]
ax.plot(pure_init_results[10], label='100% Current Diet', alpha=0.8, linewidth=2)
ax.plot(neighbor_init_results[10], label='Neighbor-Based', alpha=0.8, linewidth=2, linestyle='--')
ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.3, label='5% reference')
ax.set_xlabel('t (steps)')
ax.set_ylabel('Vegetarian Fraction')
ax.set_title('Comparison: M=10')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
os.makedirs("visualisations_output", exist_ok=True)
plt.savefig('visualisations_output/pure_diet_initialization_test.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print("Plot saved to visualisations_output/pure_diet_initialization_test.png")
print("=" * 60)
