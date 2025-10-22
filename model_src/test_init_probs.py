import sys
sys.path.append('..')
from model_main_single import Model, params
import numpy as np

# Run initialization only
params_test = params.copy()
params_test['agent_ini'] = 'twin'
model = Model(params_test)
model.agent_ini()

# Calculate initial switching probabilities
probs = []
for agent in model.agents[:100]:  # Sample first 100
    if agent.neighbours := [model.agents[n] for n in model.G1.neighbors(agent.i)]:
        other = np.random.choice(agent.neighbours)
        p = agent.prob_calc(other)
        probs.append(p)

print(f"Mean initial switching probability: {np.mean(probs):.4f}")
print(f"Median: {np.median(probs):.4f}")
print(f"High prob (>0.6): {sum(p>0.6 for p in probs)/len(probs):.2%}")
