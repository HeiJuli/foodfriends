#!/usr/bin/env python3
"""Quick test of updated network agency evolution plot"""
import sys
import glob
import os
sys.path.append('..')
from publication_plots import plot_network_agency_evolution

# Get most recent trajectory_analysis file
files = glob.glob('../model_output/trajectory_analysis_*.pkl')
if files:
    file_path = sorted(files, key=lambda x: os.path.getmtime(x))[-1]
    print(f"Using: {file_path}")
    plot_network_agency_evolution(file_path=file_path, log_scale="loglog", save=True)
    print("Plot generated successfully")
else:
    print("No trajectory_analysis files found")
