#!/usr/bin/env python3
"""Quick test to verify cascade attribution tracking"""

import sys
sys.path.append('..')
from model_main_single import Model, params

# Run a single model
print("Running model to test cascade tracking...")
params_test = params.copy()
params_test.update({'topology': 'PATCH', 'steps': 10000})

test_model = Model(params_test)
test_model.run()

# Get cascade statistics
cascade_df = test_model.cascade_statistics()

print("\n=== Cascade Statistics Summary ===")
print(f"Total agents: {len(test_model.agents)}")
print(f"Agents with direct influence: {(cascade_df['direct_influence'] > 0).sum()}")
print(f"Max cascade size: {cascade_df['total_cascade'].max()}")
print(f"Mean multiplier: {cascade_df[cascade_df['multiplier'] > 0]['multiplier'].mean():.2f}")

print("\n=== Top 5 Influencers (by attributed reduction) ===")
top_5 = cascade_df.nlargest(5, 'attributed_reduction')
print(top_5[['agent_id', 'direct_influence', 'total_cascade', 'attributed_reduction', 'multiplier']].to_string())

print("\n=== Top 5 Super-spreaders (by multiplier) ===")
top_spreaders = cascade_df[cascade_df['direct_influence'] > 0].nlargest(5, 'multiplier')
print(top_spreaders[['agent_id', 'direct_influence', 'total_cascade', 'attributed_reduction', 'multiplier']].to_string())

print("\nTest completed successfully!")
