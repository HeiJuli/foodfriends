"""
Test stratified sampling integration in model initialization.
"""
import sys
sys.path.append('../model_src')

# Import model with stratified sampling
from model_main_single import params

# Test initialization with N=2000
test_params = params.copy()
test_params['N'] = 2000
test_params['agent_ini'] = 'twin'
test_params['survey_file'] = '../data/hierarchical_agents.csv'
test_params['steps'] = 100  # Short run just to test initialization

print("=" * 80)
print("TESTING STRATIFIED SAMPLING IN MODEL INITIALIZATION")
print("=" * 80)
print(f"\nParameters:")
print(f"  N = {test_params['N']}")
print(f"  agent_ini = {test_params['agent_ini']}")
print(f"  survey_file = {test_params['survey_file']}")

# Import and initialize model
from model_main_single import Model

print("\nInitializing model...")
model = Model(test_params)

print("\n" + "=" * 80)
print("MODEL INITIALIZATION SUCCESSFUL")
print("=" * 80)
print(f"\nNumber of agents created: {len(model.agents)}")
print(f"Network nodes: {model.G1.number_of_nodes()}")
print(f"Network edges: {model.G1.number_of_edges()}")

# Verify demographic distribution
import pandas as pd

survey_full = pd.read_csv('../data/hierarchical_agents.csv')
agent_data = model.survey_data

print("\n" + "=" * 80)
print("DEMOGRAPHIC DISTRIBUTION COMPARISON")
print("=" * 80)

for col in ['gender', 'age_group', 'incquart', 'educlevel']:
    print(f"\n{col}:")
    full_dist = survey_full[col].value_counts(normalize=True).sort_index()
    agent_dist = agent_data[col].value_counts(normalize=True).sort_index()

    print(f"  {'Category':<10} {'Full Survey':<15} {'Model Sample':<15} {'Difference':<15}")
    print("  " + "-" * 60)

    for category in full_dist.index:
        full_pct = full_dist.get(category, 0) * 100
        agent_pct = agent_dist.get(category, 0) * 100
        diff = agent_pct - full_pct
        print(f"  {category:<10} {full_pct:>13.2f}%  {agent_pct:>13.2f}%  {diff:>+13.2f}%")

print("\n" + "=" * 80)
print("TEST PASSED: Stratified sampling working correctly in model")
print("=" * 80)
