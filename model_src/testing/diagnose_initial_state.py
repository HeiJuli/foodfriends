#!/usr/bin/env python3
"""
Diagnose initial state conditions to understand rapid uptake
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_main_single import Model, Agent, params

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Create model but don't run - just initialize
test_params = copy.deepcopy(params)
test_params['N'] = 699
test_params['agent_ini'] = "twin"

model = Model(test_params)
model.agent_ini()
model.harmonise_netIn()

print("=" * 60)
print("INITIAL STATE DIAGNOSTICS")
print("=" * 60)

# Analyze initial network structure
veg_agents = [a for a in model.agents if a.diet == "veg"]
meat_agents = [a for a in model.agents if a.diet == "meat"]

print(f"\nPopulation: {len(veg_agents)} veg, {len(meat_agents)} meat")
print(f"Fraction veg: {len(veg_agents)/len(model.agents):.3f}")

# Analyze neighborhood composition
veg_neighbor_counts_for_veg = []
veg_neighbor_counts_for_meat = []

for agent in veg_agents:
    neighbors = list(model.G1.neighbors(agent.i))
    n_veg_neighbors = sum(1 for n in neighbors if model.agents[n].diet == "veg")
    veg_neighbor_counts_for_veg.append(n_veg_neighbors)

for agent in meat_agents:
    neighbors = list(model.G1.neighbors(agent.i))
    n_veg_neighbors = sum(1 for n in neighbors if model.agents[n].diet == "veg")
    veg_neighbor_counts_for_meat.append(n_veg_neighbors)

print("\n" + "-" * 60)
print("NEIGHBORHOOD STRUCTURE AT t=0")
print("-" * 60)
print(f"Veg agents: avg {np.mean(veg_neighbor_counts_for_veg):.2f} veg neighbors (out of ~{np.mean([len(list(model.G1.neighbors(a.i))) for a in veg_agents]):.1f} total)")
print(f"Meat agents: avg {np.mean(veg_neighbor_counts_for_meat):.2f} veg neighbors (out of ~{np.mean([len(list(model.G1.neighbors(a.i))) for a in meat_agents]):.1f} total)")

# Analyze initial memory composition
veg_in_memory_for_veg = []
veg_in_memory_for_meat = []

for agent in veg_agents:
    veg_count = sum(1 for d in agent.memory if d == "veg")
    veg_in_memory_for_veg.append(veg_count)

for agent in meat_agents:
    veg_count = sum(1 for d in agent.memory if d == "veg")
    veg_in_memory_for_meat.append(veg_count)

print("\n" + "-" * 60)
print(f"INITIAL MEMORY (M={params['M']})")
print("-" * 60)
print(f"Veg agents: avg {np.mean(veg_in_memory_for_veg):.2f} veg in memory")
print(f"Meat agents: avg {np.mean(veg_in_memory_for_meat):.2f} veg in memory")

# Calculate initial utilities for all agents
print("\n" + "-" * 60)
print("INITIAL UTILITY DISTRIBUTION")
print("-" * 60)

utilities_same_veg = []
utilities_diff_veg = []
utilities_same_meat = []
utilities_diff_meat = []

for agent in veg_agents:
    # Temporarily set up neighbors for utility calculation
    agent.neighbours = [model.agents[n] for n in model.G1.neighbors(agent.i)]
    if agent.neighbours:
        u_same = agent.calc_utility(agent.neighbours[0], mode="same")
        u_diff = agent.calc_utility(agent.neighbours[0], mode="diff")
        utilities_same_veg.append(u_same)
        utilities_diff_veg.append(u_diff)

for agent in meat_agents:
    agent.neighbours = [model.agents[n] for n in model.G1.neighbors(agent.i)]
    if agent.neighbours:
        u_same = agent.calc_utility(agent.neighbours[0], mode="same")
        u_diff = agent.calc_utility(agent.neighbours[0], mode="diff")
        utilities_same_meat.append(u_same)
        utilities_diff_meat.append(u_diff)

print(f"Veg agents utility_same: mean={np.mean(utilities_same_veg):.3f}, std={np.std(utilities_same_veg):.3f}")
print(f"Veg agents utility_diff: mean={np.mean(utilities_diff_veg):.3f}, std={np.std(utilities_diff_veg):.3f}")
print(f"Meat agents utility_same: mean={np.mean(utilities_same_meat):.3f}, std={np.std(utilities_same_meat):.3f}")
print(f"Meat agents utility_diff: mean={np.mean(utilities_diff_meat):.3f}, std={np.std(utilities_diff_meat):.3f}")

# Calculate switch probabilities
delta_u_veg = np.array(utilities_diff_veg) - np.array(utilities_same_veg)
delta_u_meat = np.array(utilities_diff_meat) - np.array(utilities_same_meat)

prob_veg_to_meat = 1/(1+np.exp(-2.3*delta_u_veg))
prob_meat_to_veg = 1/(1+np.exp(-2.3*delta_u_meat))

print("\n" + "-" * 60)
print("INITIAL SWITCHING PROBABILITIES")
print("-" * 60)
print(f"Veg→Meat: mean={np.mean(prob_veg_to_meat):.4f}, median={np.median(prob_veg_to_meat):.4f}")
print(f"Meat→Veg: mean={np.mean(prob_meat_to_veg):.4f}, median={np.median(prob_meat_to_veg):.4f}")
print(f"Fraction of meat-eaters with prob>0.1 to switch: {np.mean(prob_meat_to_veg > 0.1):.3f}")
print(f"Fraction of meat-eaters with prob>0.5 to switch: {np.mean(prob_meat_to_veg > 0.5):.3f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Neighborhood distribution
ax = axes[0, 0]
ax.hist(veg_neighbor_counts_for_meat, bins=range(0, max(veg_neighbor_counts_for_meat)+2),
        alpha=0.7, label='Meat agents', density=True)
ax.set_xlabel('Number of veg neighbors')
ax.set_ylabel('Density')
ax.set_title('Veg Neighbors Distribution (Meat Agents)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Memory distribution
ax = axes[0, 1]
ax.hist(veg_in_memory_for_meat, bins=range(0, params['M']+2),
        alpha=0.7, label='Meat agents', density=True)
ax.set_xlabel(f'Veg diets in memory (M={params["M"]})')
ax.set_ylabel('Density')
ax.set_title('Initial Memory Distribution (Meat Agents)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Utility deltas
ax = axes[1, 0]
ax.hist(delta_u_meat, bins=30, alpha=0.7, label='Meat agents')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='No preference')
ax.set_xlabel('Δu = u_diff - u_same')
ax.set_ylabel('Count')
ax.set_title('Initial Utility Difference (Meat Agents)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Switch probabilities
ax = axes[1, 1]
ax.hist(prob_meat_to_veg, bins=30, alpha=0.7, label='Meat→Veg')
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax.set_xlabel('Probability of switching')
ax.set_ylabel('Count')
ax.set_title('Initial Switch Probabilities (Meat→Veg)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
os.makedirs("visualisations_output", exist_ok=True)
plt.savefig('visualisations_output/initial_state_diagnostics.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print("Plot saved to visualisations_output/initial_state_diagnostics.png")
print("=" * 60)
