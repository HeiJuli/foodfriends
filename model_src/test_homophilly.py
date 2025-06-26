#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:00:41 2025

@author: jpoveralls
"""

from auxillary import network_stats
from netin import PATCH, PAH
from netin import viz
import networkx as nx

#%%

params = {"h_MM": 0.8,
          "h_mm": 0.8,
          "f_min": 0.3,
          "m": 4,
          "n": 10,
          "tc": 0.3,
          "N": 2000}


#%% Testing inference


def gen_graph(model):
    pass 


def test_metrics(graph):
    
    print('mine', network_stats.infer_homophily(graph, params["f_min"]))
    print('mine_new', network_stats.infer_homophily_neu(graph))
    print("arxiv", network_stats.homophily_inference_asymmetric(graph))
    print('current', graph.infer_homophily_values())

    return
     


for i in range(params["n"]):
    G1 = PATCH(params["N"], params["m"], params["f_min"], h_MM=params["h_MM"], h_mm=params["h_mm"], tc=params["tc"])
    #G1 = PAH(params["N"], params["m"], params["f_min"], h_MM=params["h_MM"], h_mm=params["h_mm"])
    G1.generate()
    test_metrics(G1)
    
    
#%% Visual test

viz.handlers.plot_graph(G1, edge_width = 0.2, edge_arrows =False)
