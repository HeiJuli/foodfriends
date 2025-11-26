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
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_main_single import Model, Agent, params

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

def create_threshold_model(variant='baseline'):
    """Create threshold model with specified variant

    Args:
        variant: 'baseline' (k=15), 'scaled_diss' (k=15+0.2*dissonance),
                 'floor' (k=15+0.1 floor)
    """
    original_prob_calc = Agent.prob_calc

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
                # Apply scaling factor for 'scaled_diss' variant
                scaling = 0.2 if variant == 'scaled_diss' else 1.0
                threshold -= scaling * self.alpha * dissonance
        else:  # veg
            theta_misaligned = self.theta > 0.5
            dissonance_active = theta_misaligned

            if dissonance_active:
                dissonance = abs(self.theta - 0.0)
                scaling = 0.2 if variant == 'scaled_diss' else 1.0
                threshold -= scaling * self.alpha * dissonance

        # Apply floor for 'floor' variant, else normal [0,1] clamp
        if variant == 'floor':
            threshold = np.clip(threshold, 0.1, 1)
        else:
            threshold = np.clip(threshold, 0, 1)

        social_exposure = self.beta * proportion
        k = 15
        prob_switch = 1 / (1 + np.exp(-k * (social_exposure - threshold)))

        return prob_switch

    Agent.prob_calc = new_prob_calc
    return original_prob_calc

# Test configuration
test_params = copy.deepcopy(params)
test_params['steps'] = 700000
test_params['N'] = 800
test_params['agent_ini'] = "twin"

print("=" * 70)
print("TESTING: Threshold Model Variants vs Utility Model")
print("=" * 70)

# Run utility model
print("\n[Utility Model] Running baseline...")
model_utility = Model(test_params)
model_utility.run()
print(f"  Initial: {model_utility.fraction_veg[0]:.3f}")
print(f"  At 5k:   {model_utility.fraction_veg[5000]:.3f}")
print(f"  At 25k:  {model_utility.fraction_veg[25000]:.3f}")
print(f"  At 50k:  {model_utility.fraction_veg[50000]:.3f}")
print(f"  Final:   {model_utility.fraction_veg[-1]:.3f}")

# Run threshold model k=15 baseline
print("\n[Threshold k=15] Running...")
original = create_threshold_model('baseline')
model_k15 = Model(test_params)
model_k15.run()
Agent.prob_calc = original
print(f"  Initial: {model_k15.fraction_veg[0]:.3f}")
print(f"  At 5k:   {model_k15.fraction_veg[5000]:.3f}")
print(f"  At 25k:  {model_k15.fraction_veg[25000]:.3f}")
print(f"  At 50k:  {model_k15.fraction_veg[50000]:.3f}")
print(f"  Final:   {model_k15.fraction_veg[-1]:.3f}")

# Run threshold model k=15 + scaled dissonance (0.2x)
print("\n[Threshold k=15 + Scaled Dissonance 0.2x] Running...")
original = create_threshold_model('scaled_diss')
model_scaled = Model(test_params)
model_scaled.run()
Agent.prob_calc = original
print(f"  Initial: {model_scaled.fraction_veg[0]:.3f}")
print(f"  At 5k:   {model_scaled.fraction_veg[5000]:.3f}")
print(f"  At 25k:  {model_scaled.fraction_veg[25000]:.3f}")
print(f"  At 50k:  {model_scaled.fraction_veg[50000]:.3f}")
print(f"  Final:   {model_scaled.fraction_veg[-1]:.3f}")

# Run threshold model k=15 + threshold floor 0.1
print("\n[Threshold k=15 + Floor 0.1] Running...")
original = create_threshold_model('floor')
model_floor = Model(test_params)
model_floor.run()
Agent.prob_calc = original
print(f"  Initial: {model_floor.fraction_veg[0]:.3f}")
print(f"  At 5k:   {model_floor.fraction_veg[5000]:.3f}")
print(f"  At 25k:  {model_floor.fraction_veg[25000]:.3f}")
print(f"  At 50k:  {model_floor.fraction_veg[50000]:.3f}")
print(f"  Final:   {model_floor.fraction_veg[-1]:.3f}")

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
