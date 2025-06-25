# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:13:30 2024

@author: Jordan
"""
import networkx as nx
from collections import Counter
from typing import Union
import pandas as pd

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



def infer_homophily_values(graph, minority_frac) -> tuple[float, float]:
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

    f_m = minority_frac
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

    
def graph_to_dataframes(g: nx.Graph):
    # Create a DataFrame for nodes and their attributes
    nodes_data = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient='index')
    nodes_data = nodes_data.drop(['agent'], axis = 1)
    # Create a DataFrame for edges
    edges_data = pd.DataFrame(list(g.edges(data=True)), columns=['source', 'target', 'attributes'])
    edges_data= edges_data.drop(['attributes'], axis = 1)
    return nodes_data, edges_data
