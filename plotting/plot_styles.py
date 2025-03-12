#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 21:18:44 2025

@author: jpoveralls
"""

# plot_styles.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#%%


def set_publication_style():
    """Set global publication-quality plotting style for ecological economics"""
    plt.rcdefaults()
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.8,
        'axes.labelpad': 8,
        'pdf.fonttype': 42,  # Ensures text is editable in Adobe Illustrator
        'ps.fonttype': 42
    })

def apply_axis_style(ax):
    """Apply publication-quality styling to a matplotlib axis"""
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken left and bottom spines
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Improve tick appearance
    ax.tick_params(width=1.2, length=5)
    
    # Add subtle grid
    ax.grid(True, linestyle='--', alpha=0.3, zorder=-10)

# Color schemes optimized for ecological economics publications
COLORS = {
    'primary': '#006d77',  # Teal
    'secondary': '#e29578',  # Coral
    'vegetation': '#2a9d8f',  # Green
    'meat': '#e76f51',  # Red-orange
    'neutral': '#264653',  # Dark slate
    'highlight': '#f4a261',  # Gold
    'light_gray': '#edf6f9'
}

# Custom colormap for heatmaps
ECO_CMAP = LinearSegmentedColormap.from_list(
    'eco_cmap', 
    ['#edf6f9', '#83c5be', '#006d77', '#264653'], 
    N=256
)

# Diverging colormap for parameter effect plots
ECO_DIV_CMAP = LinearSegmentedColormap.from_list(
    'eco_div_cmap',
    ['#e76f51', '#f4a261', '#edf6f9', '#83c5be', '#006d77'],
    N=256
)