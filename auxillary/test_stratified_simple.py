"""
Simple test of stratified sampling utility.
"""
from sampling_utils import stratified_sample_agents
import pandas as pd

# Load survey data
df = pd.read_csv('../data/hierarchical_agents.csv')

print("=" * 80)
print("TESTING STRATIFIED SAMPLING UTILITY")
print("=" * 80)
print(f"\nOriginal dataset: {len(df)} participants")

# Test sampling at N=2000
sampled = stratified_sample_agents(df, n_target=2000, random_state=42, verbose=True)

print(f"\n" + "=" * 80)
print("STRATIFIED SAMPLING TEST PASSED")
print("=" * 80)
