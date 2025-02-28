# -*- coding: utf-8 -*-
"""
Plotting functions for dietary emissions and basic metrics
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_co2_vs_vegetarian_fraction(results_df, save_path=None):
    """
    Create scatter plot of final CO2 consumption vs vegetarian fraction
    
    Args:
        results_df (DataFrame): Results containing veg_fraction and final_CO2 columns
        save_path (str, optional): Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=results_df, 
        x='veg_fraction', 
        y='final_CO2',
        s=80,
        alpha=0.7,
        color='#1f77b4'
    )
    
    plt.xlabel('% vegans & vegetarians', fontsize=12)
    plt.ylabel('Final average dietary consumption [kg/CO2/year]', fontsize=12)
    plt.title('Impact of Vegetarian Population on CO2 Emissions', fontsize=14)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gca()

def plot_emissions_trajectories(co2_trajectories, labels, save_path=None):
    """
    Plot emissions trajectories over time for multiple model runs
    
    Args:
        co2_trajectories (list): List of CO2 emission trajectories
        labels (list): Labels for each trajectory
        save_path (str, optional): Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    for i, trajectory in enumerate(co2_trajectories):
        plt.plot(trajectory, label=labels[i])
    
    plt.xlabel('Simulation Steps', fontsize=12)
    plt.ylabel('Average CO2 Emissions [kg/year]', fontsize=12)
    plt.title('Dietary Emissions Over Time', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gca()

def plot_vegetarian_timeseries(veg_trajectories, labels, save_path=None):
    """
    Plot vegetarian fraction over time for multiple model runs
    
    Args:
        veg_trajectories (list): List of vegetarian fraction trajectories
        labels (list): Labels for each trajectory
        save_path (str, optional): Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    for i, trajectory in enumerate(veg_trajectories):
        plt.plot(trajectory, label=labels[i])
    
    plt.xlabel('Simulation Steps', fontsize=12)
    plt.ylabel('Vegetarian Fraction', fontsize=12)
    plt.title('Vegetarian Population Dynamics', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gca()

def plot_alpha_beta_effects(veg_trajectories, alpha_values, beta_values, save_path=None):
    """
    Plot the effect of varying alpha and beta on vegetarian fraction dynamics
    
    Args:
        veg_trajectories (dict): Dictionary of trajectories keyed by (alpha, beta) tuples
        alpha_values (list): Alpha values used
        beta_values (list): Beta values used
        save_path (str, optional): Path to save figure
    """
    fig, axes = plt.subplots(len(beta_values), len(alpha_values), 
                            figsize=(4*len(alpha_values), 3*len(beta_values)),
                            sharex=True, sharey=True)
    
    for i, beta in enumerate(beta_values):
        for j, alpha in enumerate(alpha_values):
            key = (alpha, beta)
            if key in veg_trajectories:
                ax = axes[i, j] if len(beta_values) > 1 else axes[j]
                ax.plot(veg_trajectories[key])
                ax.set_title(f'α={alpha}, β={beta}')
                ax.grid(alpha=0.3)
                
                # Only add x-label for bottom row
                if i == len(beta_values) - 1:
                    ax.set_xlabel('Steps')
                
                # Only add y-label for leftmost column
                if j == 0:
                    ax.set_ylabel('Vegetarian Fraction')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_reduction_attribution(reductions, diets, top_n=10, save_path=None):
    """
    Plot the top agents by their contribution to emissions reduction
    
    Args:
        reductions (list): List of reduction values for all agents
        diets (list): List of diets for all agents
        top_n (int): Number of top agents to display
        save_path (str, optional): Path to save figure
    """
    # Create dataframe for plotting
    df = pd.DataFrame({
        'agent_id': range(len(reductions)),
        'reduction': reductions,
        'diet': diets
    })
    
    # Sort by reduction and get top contributors
    df = df.sort_values('reduction', ascending=False).head(top_n)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Set colors based on diet
    colors = ['green' if d == 'veg' else 'red' for d in df['diet']]
    
    sns.barplot(x='agent_id', y='reduction', data=df, palette=colors)
    
    plt.xlabel('Agent ID', fontsize=12)
    plt.ylabel('Emissions Reduction Attributed [kg/CO2]', fontsize=12)
    plt.title('Top Contributors to Emissions Reduction', fontsize=14)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Vegetarian'),
        Patch(facecolor='red', label='Meat Eater')
    ]
    plt.legend(handles=legend_elements)
    
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gca()

def plot_comparison_boxplot(results_df, x_var, y_var, hue=None, title=None, save_path=None):
    """
    Create a boxplot comparing different parameter values
    
    Args:
        results_df (DataFrame): Results dataframe
        x_var (str): Variable for x-axis
        y_var (str): Variable for y-axis
        hue (str, optional): Variable for color grouping
        title (str, optional): Plot title
        save_path (str, optional): Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(x=x_var, y=y_var, hue=hue, data=results_df)
    
    plt.xlabel(x_var, fontsize=12)
    plt.ylabel(y_var, fontsize=12)
    
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'{y_var} by {x_var}', fontsize=14)
    
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gca()