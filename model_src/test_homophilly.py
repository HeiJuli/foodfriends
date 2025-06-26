#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:00:41 2025

@author: jpoveralls
"""

from auxillary import network_stats
from netin import PATCH, PAH, DH, DPAH
from netin import viz
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#%%

params = {"h_MM": 0.8,
          "h_mm": 0.8,
          "f_min": 0.3,
          "m": 4,
          "n": 10,
          "tc": 0.3,
          "N": 1000}

#%% Testing inference

def gen_graph(model):
    if model == "PATCH":
        g = PATCH(params["N"], params["m"], params["f_min"], h_MM=params["h_MM"], h_mm=params["h_mm"], tc=params["tc"])
    elif model == "PAH":
        g = PAH(params["N"], params["m"], params["f_min"], h_MM=params["h_MM"], h_mm=params["h_mm"])
    elif model == "DH":
        g = DH(params["N"], d=0.1, f_m=params["f_min"], plo_M=2.5, plo_m=2.5, h_MM=params["h_MM"], h_mm=params["h_mm"])
    elif model == "DPAH":
        g = DPAH(params["N"], d=0.1, f_m=params["f_min"], plo_M=2.5, plo_m=2.5, h_MM=params["h_MM"], h_mm=params["h_mm"])
    else:
        raise ValueError(f"Unknown model: {model}")
    
    g.generate()
    return g

def test_metrics(graph):
    nominal_h_MM, nominal_h_mm = params["h_MM"], params["h_mm"]
    
    mine = network_stats.infer_homophily(graph)
    mine_new = network_stats.infer_homophily_neu(graph)
    arxiv = network_stats.homophily_inference_asymmetric(graph)
    current = graph.infer_homophily_values()
    
    methods = {'mine': mine, 'mine_new': mine_new, 'arxiv': arxiv, 'current': current}
    deltas = {}
    
    for name, result in methods.items():
        try:
            delta_MM = abs(float(result[0]) - nominal_h_MM)
            delta_mm = abs(float(result[1]) - nominal_h_mm)
            delta_overall = delta_MM + delta_mm
            deltas[name] = {'delta_overall': delta_overall, 'delta_MM': delta_MM, 'delta_mm': delta_mm}
        except:
            deltas[name] = {'delta_overall': np.nan, 'delta_MM': np.nan, 'delta_mm': np.nan}
    
    return deltas

#%%

def plot_delta_boxplot(delta_data, title="Delta Comparison", ax=None):
    methods = list(delta_data.keys())
    data = [delta_data[method] for method in methods]
    
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    ax.boxplot(data, labels=methods)
    ax.set_ylabel('Delta Overall')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    
    if ax == plt.gca():  # Only show if we created the figure
        plt.show()
        
#%% 
def run_delta_analysis(model, n_runs=10):
    delta_by_method = {'mine': [], 'mine_new': [], 'arxiv': [], 'current': []}
    
    for i in range(n_runs):
        g = gen_graph(model)
        deltas = test_metrics(g)
        for method in delta_by_method.keys():
            delta_by_method[method].append(deltas[method]['delta_overall'])
    
    plot_delta_boxplot(delta_by_method, f"Method Comparison - {model}")
    return delta_by_method

def compare_topologies(models=["PATCH", "PAH"], n_runs=10):
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        delta_by_method = {'mine': [], 'mine_new': [], 'arxiv': [], 'current': []}
        
        for _ in range(n_runs):
            g = gen_graph(model)
            deltas = test_metrics(g)
            for method in delta_by_method.keys():
                delta_by_method[method].append(deltas[method]['delta_overall'])
        
        plot_delta_boxplot(delta_by_method, model, ax=axes[i])
    
    plt.tight_layout()
    plt.show()


def compare_N_sizes(model="PATCH", N_values=[500, 1000, 1500, 2000], n_runs=5):
    methods = ['mine', 'mine_new', 'arxiv', 'current']
    
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        means, stds = [], []
        for N in N_values:
            params["N"] = N
            deltas = []
            for _ in range(n_runs):
                g = gen_graph(model)
                delta_result = test_metrics(g)
                deltas.append(delta_result[method]['delta_overall'])
            
            valid_deltas = [d for d in deltas if not np.isnan(d)]
            means.append(np.mean(valid_deltas) if valid_deltas else np.nan)
            stds.append(np.std(valid_deltas) if valid_deltas else np.nan)
        
        plt.errorbar(N_values, means, yerr=stds, marker='o', capsize=5, label=method)
    
    plt.xlabel('Network Size (N)')
    plt.ylabel('Mean Delta Overall')
    plt.title(f'Delta vs Network Size - {model}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

#%% Example usage
if __name__ == "__main__":
    # Compare methods for single topology
    # run_delta_analysis("PATCH", n_runs=10)
    
    # Compare methods across topologies  
    compare_topologies(["PATCH", "PAH"], n_runs=10)
    
    # Compare methods across network sizes
    # compare_N_sizes("PATCH", [500, 1000, 1500], n_runs=3)