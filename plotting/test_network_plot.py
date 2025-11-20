#!/usr/bin/env python3
"""Quick test of updated network agency evolution plot"""
import sys
sys.path.append('..')
from publication_plots import plot_network_agency_evolution

file_path = '../model_output/trajectory_analysis_20251120.pkl'
plot_network_agency_evolution(file_path=file_path, log_scale="loglog", save=True)
print("Plot generated successfully")
