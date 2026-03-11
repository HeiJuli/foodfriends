import numpy as np
import matplotlib.pyplot as plt

# Boltzmann switching probability from model_main.py:
#   P(switch) = exp(-beta*H_switch) / [exp(-beta*H_switch) + exp(-beta*H_stay)]
#             = 1 / (1 + exp(-beta * (H_stay - H_switch)))
# x-axis: delta_H = H_stay - H_switch (positive -> switch preferred)
beta = 13  # model default
delta_H = np.linspace(-1, 1, 300)
prob = 1 / (1 + np.exp(-beta * delta_H))

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(delta_H, prob, 'k-', linewidth=2)
plt.xlabel(r'$H(\mathrm{stay}) - H(\mathrm{switch})$', fontsize=14)
plt.ylabel('Probability of Dietary Change', fontsize=14)
plt.xlim(-1, 1)
plt.ylim(0, 1)
plt.xticks([-1, 0, 1])
plt.yticks([0, 0.5, 1])
plt.tight_layout()
plt.savefig('/home/jpoveralls/Documents/Projects_code/foodfriends/visualisations_output/utility_probability.png', dpi=300, bbox_inches='tight')
plt.show()
print("Plot saved to visualisations_output/utility_probability.png")
