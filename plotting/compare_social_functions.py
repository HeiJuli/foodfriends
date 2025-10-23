#!/usr/bin/env python3
"""
Compare different social influence formulations
"""

import numpy as np
import matplotlib.pyplot as plt

# Set beta = 0.64 (typical value from model)
beta = 0.64

# Create ratio values from 0 to 1
ratio = np.linspace(0, 1, 100)

# Calculate different formulations
original = beta * (2*ratio - 1)
simple = beta * ratio
quadratic = beta * (2*ratio - 1)**2
steeper = beta * (3*ratio - 1.5)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(ratio, original, 'b-', linewidth=2, label='Original: β(2r-1)')
plt.plot(ratio, simple, 'g--', linewidth=2, label='Simple: βr')
plt.plot(ratio, quadratic, 'r-.', linewidth=2, label='Quadratic: β(2r-1)²')
plt.plot(ratio, steeper, 'm:', linewidth=2, label='Steeper: β(3r-1.5)')

# Add reference lines
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
plt.axvline(x=0.5, color='k', linestyle='--', linewidth=0.5, alpha=0.3)

# Labels and formatting
plt.xlabel('Ratio of neighbors with same diet', fontsize=12)
plt.ylabel('Social utility contribution', fontsize=12)
plt.title(f'Comparison of Social Influence Formulations (β={beta})', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add annotations for key points
plt.text(0.05, original[5], f'  All different\n  (ratio=0)', fontsize=9, va='center')
plt.text(0.5, -0.05, 'Balanced\n(ratio=0.5)', fontsize=9, ha='center', va='top')
plt.text(0.95, original[-5], '  All same\n  (ratio=1)', fontsize=9, va='center', ha='right')

plt.tight_layout()
plt.savefig('../visualisations_output/social_function_comparison.png', dpi=300)
print("Plot saved to: ../visualisations_output/social_function_comparison.png")
plt.show()

# Print some key values for comparison
print("\nKey values at important points:")
print(f"{'Formulation':<25} {'r=0':<12} {'r=0.25':<12} {'r=0.5':<12} {'r=0.75':<12} {'r=1':<12}")
print("-" * 85)

for name, values in [
    ('Original: β(2r-1)', original),
    ('Simple: βr', simple),
    ('Quadratic: β(2r-1)²', quadratic),
    ('Steeper: β(3r-1.5)', steeper)
]:
    idx = [0, 25, 50, 75, 99]
    vals = [values[i] for i in idx]
    print(f"{name:<25} {vals[0]:>11.3f} {vals[1]:>11.3f} {vals[2]:>11.3f} {vals[3]:>11.3f} {vals[4]:>11.3f}")
