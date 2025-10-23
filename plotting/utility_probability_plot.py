import numpy as np
import matplotlib.pyplot as plt

# Generate utility difference values
delta_u = np.linspace(-3, 3, 300)

# Calculate probability using the model's logistic function
# prob = 1/(1+exp(-2.3*(u_s-u_i)))
prob = 1 / (1 + np.exp(-2.3 * delta_u))

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(delta_u, prob, 'k-', linewidth=2)
plt.xlabel('Difference in Utility', fontsize=14)
plt.ylabel('Probability of Dietary Change', fontsize=14)
plt.xlim(-3, 3)
plt.ylim(0, 1)
plt.xticks([-3, 0, 3])
plt.yticks([0, 0.5, 1])
plt.tight_layout()
plt.savefig('/home/jpoveralls/Documents/Projects_code/foodfriends/visualisations_output/utility_probability.png', dpi=300, bbox_inches='tight')
plt.show()
print("Plot saved to visualisations_output/utility_probability.png")
