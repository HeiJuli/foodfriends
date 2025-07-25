# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:13:30 2024

@author: Jordan
"""
import networkx as nx
from collections import Counter
from typing import Union
import pandas as pd
import numpy as np

#%%

def get_edge_type_counts(g: Union[nx.Graph, nx.DiGraph], fractions: bool = False) -> Counter:
    """
    Computes the edge type counts of the graph using the `m` attribute of each node.

    Parameters
    ----------
    g: Union[nx.Graph, nx.DiGraph]
        Graph to compute the edge type counts.

    fractions: bool
        If True, the counts are returned as fractions of the total number of edges.

    Returns
    -------
    Counter
        Counter holding the edge type counts.

    Notes
    -----
    Class labels are assumed to be binary. The majority class is assumed to be labeled as 0
    and the minority class is assumed to be labeled as 1.
    """
    class_attribute = 'm'
    majority_label = 0
    minority_label = 1
    class_values = [majority_label, minority_label]
    class_labels = {majority_label: "M", minority_label: "m"}

    counts = Counter([f"{class_labels[g.nodes[e[0]][class_attribute]]}"
                      f"{class_labels[g.nodes[e[1]][class_attribute]]}"
                      for e in g.edges if g.nodes[e[0]][class_attribute] in class_values and
                      g.nodes[e[1]][class_attribute] in class_values])

    if fractions:
        total = sum(counts.values())
        counts = Counter({k: v / total for k, v in counts.items()})

    return counts

def infer_homophily(graph) -> tuple[float, float]:
    """
    Infers analytically the homophily values for the majority and minority classes.

    Returns
    -------
    h_MM: float
        homophily within majority group

    h_mm: float
        homophily within minority group
    """
    from sympy import symbols, Eq, solve

    f_m = sum(1 for _, data in graph.nodes(data=True) if data.get('m', 0) == 1) / graph.number_of_nodes()
    f_M = 1 - f_m

    e = get_edge_type_counts(graph)
    e_MM = e.get('MM', 0)
    e_mm = e.get('mm', 0)
    e_Mm = e.get('Mm', 0)
    e_mM = e.get('mM', 0)

    if e_MM + e_Mm == 0 or e_mm + e_mM == 0:
        raise ValueError("Division by zero encountered in probability calculations")

    p_MM = e_MM / (e_MM + e_Mm) if e_MM + e_Mm != 0 else 0
    p_mm = e_mm / (e_mm + e_mM) if e_mm + e_mM != 0 else 0

    # equations
    hmm, hMM, hmM, hMm = symbols('hmm hMM hmM hMm')
    eq1 = Eq((f_m * hmm) / ((f_m * hmm) + (f_M * hmM)), p_mm)
    eq2 = Eq(hmm + hmM, 1)

    eq3 = Eq((f_M * hMM) / ((f_M * hMM) + (f_m * hMm)), p_MM)
    eq4 = Eq(hMM + hMm, 1)

    solution = solve((eq1, eq2, eq3, eq4), (hmm, hmM, hMM, hMm))
    h_MM, h_mm = solution[hMM], solution[hmm]
    return h_MM, h_mm

def infer_homophily_neu(graph) -> tuple[float, float]:
    """
    Corrected homophily calculation with power law corrections
    
    Parameters
    ----------
    graph: nx.Graph
        Undirected graph with 'm' node attribute (0=majority, 1=minority)
        
    Returns
    -------
    tuple[float, float]
        (h_MM, h_mm) - corrected homophily values for majority and minority groups
    """
    if graph.is_directed():
        raise NotImplementedError("Only supports undirected graphs")
    
    from sympy import symbols, Eq, solve
    
    e = get_edge_type_counts(graph)
    e_MM, e_mm, e_Mm, e_mM = e.get('MM', 0), e.get('mm', 0), e.get('Mm', 0), e.get('mM', 0)
    M = sum(e.values())
    
    if M == 0:
        return 0.0, 0.0
    
    # Calculate minority fraction
    f_m = sum(1 for _, data in graph.nodes(data=True) if data.get('m', 0) == 1) / graph.number_of_nodes()
    f_M = 1 - f_m
    
    # Probabilities
    p_MM, p_mm = e_MM / M, e_mm / M
    
    # Power law approximation for scale-free networks
    pl_M = pl_m = 2.5  # Typical BA value
    b_M = b_m = -1 / (pl_M + 1)
    
    # Solve NETIN equations
    hmm, hMM, hmM, hMm = symbols('hmm hMM hmM hMm')
    eqs = [
        Eq((f_m * f_m * hmm * (1 - b_M)) / ((f_m * hmm * (1 - b_M)) + (f_M * hmM * (1 - b_m))), p_mm),
        Eq(hmm + hmM, 1),
        Eq((f_M * f_M * hMM * (1 - b_m)) / ((f_M * hMM * (1 - b_m)) + (f_m * hMm * (1 - b_M))), p_MM),
        Eq(hMM + hMm, 1)
    ]
    
    try:
        sol = solve(eqs, (hmm, hmM, hMM, hMm))
        return float(sol[hMM]), float(sol[hmm])
    except:
        return 0.0, 0.0

def homophily_inference(graph) -> float:
    """
    Calculates single inferred homophily (symmetric case) based on https://arxiv.org/abs/2401.13642
    
    Parameters
    ----------
    graph: nx.Graph
        Undirected graph with 'm' node attribute
        
    Returns
    -------
    float
        Single inferred homophily value (symmetric case)
    """
    if graph.is_directed():
        raise NotImplementedError("Only supports undirected graphs")
    
    e = get_edge_type_counts(graph)
    e_rs = e.get('mM', 0) + e.get('Mm', 0)
    e_rr = 2 * e.get('MM', 0)
    e_ss = 2 * e.get('mm', 0)
    
    if e_rr == 0 or e_ss == 0:
        return 0.0
    
    alpha = e_rs / np.sqrt(e_rr * e_ss)
    return 1 / (1 + alpha)

def homophily_inference_asymmetric(graph) -> tuple[float, float]:
    """
    Calculates separate inferred homophily values for each group based on https://arxiv.org/abs/2401.13642
    Uses canonical ensemble formulation (Eq. 9 and S18)
    
    Parameters
    ----------
    graph: nx.Graph
        Undirected graph with 'm' node attribute (0=majority, 1=minority)
        
    Returns
    -------
    tuple[float, float]
        (h_MM, h_mm) - homophily values for majority and minority groups
    """
    if graph.is_directed():
        raise NotImplementedError("Only supports undirected graphs")
    
    e = get_edge_type_counts(graph)
    e_MM = e.get('MM', 0)  # majority-majority links
    e_mm = e.get('mm', 0)  # minority-minority links  
    e_Mm = e.get('Mm', 0)  # minority-majority links
    e_mM = e.get('mM', 0)  # majority-minority links
    
    # Count nodes in each group
    nodes_by_group = {}
    for node, data in graph.nodes(data=True):
        group = data.get('m', 0)
        nodes_by_group[group] = nodes_by_group.get(group, 0) + 1
    
    N_M = nodes_by_group.get(0, 0)  # majority group size
    N_m = nodes_by_group.get(1, 0)  # minority group size
    
    if N_M == 0 or N_m == 0:
        return 0.0, 0.0
    
    # Calculate asymmetric homophily values using canonical ensemble (Eq. 9)
    # For undirected graphs: e_Mm = e_mM, so we use total inter-group links
    e_inter = e_Mm + e_mM
    
    # Majority group homophily
    if e_inter == 0:
        h_MM = 1.0
    else:
        h_MM = (N_M * e_MM / N_m) / (e_inter + N_M * e_MM / N_m)
    
    # Minority group homophily  
    if e_inter == 0:
        h_mm = 1.0
    else:
        h_mm = (N_m * e_mm / N_M) / (e_inter + N_m * e_mm / N_M)
    
    return h_MM, h_mm

def graph_to_dataframes(g: nx.Graph):
    # Create a DataFrame for nodes and their attributes
    nodes_data = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient='index')
    nodes_data = nodes_data.drop(['agent'], axis = 1)
    # Create a DataFrame for edges
    edges_data = pd.DataFrame(list(g.edges(data=True)), columns=['source', 'target', 'attributes'])
    edges_data= edges_data.drop(['attributes'], axis = 1)
    return nodes_data, edges_data