# -*- coding: utf-8 -*-
"""
Simplified plotting script for dietary contagion model
Loads model output files and creates visualizations
"""
import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import plotting modules - assuming they're in the same directory
try:
    from emissions_plots import plot_co2_vs_vegetarian_fraction
    from phase_diagrams import plot_tipping_point_phase_diagram, plot_multi_parameter_phase_diagrams
    from network_analysis import plot_network_influence_map, plot_vegetarian_clusters, plot_topology_comparison, plot_targeted_interventions
except ImportError:
    print("Warning: Could not import plotting modules. Make sure they're in the plotting directory.")
    sys.exit(1)

# Default plotting parameters
OUTPUT_DIR = '../visualisations_output'
DPI = 300
FIGSIZE = (12, 8)

def ensure_output_dir():
    """Ensure output directory exists"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def list_model_outputs(directory="../model_output", pattern="*all*.pkl"):
    """
    List available 'all' model output files
    
    Args:
        directory (str): Directory to search for model output files
        pattern (str): Glob pattern to match files
        
    Returns:
        list: List of model output files with full paths
    """
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found")
        return []
    
    # Get all files matching the pattern
    files = glob.glob(os.path.join(directory, pattern))
    
    if not files:
        print(f"No files found matching {pattern} in {directory}")
        # Fall back to all pickle files
        files = glob.glob(os.path.join(directory, "*.pkl"))
        if not files:
            print(f"No pickle files found in {directory}")
            return []
    
    return files

def select_file(files, prompt="Select a file"):
    """
    Simple file selection
    
    Args:
        files (list): List of files to choose from
        prompt (str): Prompt to display
        
    Returns:
        str: Selected file path or None if canceled
    """
    if not files:
        print("No files available")
        return None
    
    # Display the files with indices
    print(f"\n{prompt}:")
    for i, file in enumerate(files):
        filename = os.path.basename(file)
        print(f"[{i}] {filename}")
    
    # Ask for selection
    try:
        selection = input("Enter file number (or 'q' to cancel): ")
        if selection.lower() == 'q':
            return None
        
        index = int(selection)
        if 0 <= index < len(files):
            return files[index]
        else:
            print("Invalid selection")
            return None
    except ValueError:
        print("Invalid input")
        return None

def load_data_file(file_path):
    """
    Load a pickle file
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        data: The loaded data
    """
    try:
        # Try loading as a DataFrame first
        try:
            data = pd.read_pickle(file_path)
            return data
        except:
            # If it's not a DataFrame, try loading as a regular pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def plot_emissions_data(file_path=None):
    """Plot emissions vs vegetarian fraction"""
    if file_path is None:
        # Get emissions files
        files = list_model_outputs(pattern="*emissions*.pkl")
        if not files:
            print("No emissions data files found")
            return
        file_path = select_file(files, "Select emissions data file")
        if file_path is None:
            return
    
    # Load data
    data = load_data_file(file_path)
    if data is None:
        return
    
    # Create plot
    plt.figure(figsize=FIGSIZE)
    plot_co2_vs_vegetarian_fraction(data)
    
    # Save plot
    ensure_output_dir()
    output_file = os.path.join(OUTPUT_DIR, 'emissions_vs_veg_fraction.png')
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    plt.show()

def plot_tipping_point_data(file_path=None):
    """Plot tipping point phase diagram"""
    if file_path is None:
        # Get tipping point files
        files = list_model_outputs(pattern="*tipping*all*.pkl")
        if not files:
            print("No tipping point data files found")
            return
        file_path = select_file(files, "Select tipping point data file")
        if file_path is None:
            return
    
    # Load data
    data = load_data_file(file_path)
    if data is None:
        return
    
    # Check if there's an 'initial_veg_f' column
    if isinstance(data, pd.DataFrame) and 'initial_veg_f' in data.columns:
        # This is a combined dataset with multiple vegetarian fractions
        
        # Get unique vegetarian fractions
        veg_fractions = sorted(data['initial_veg_f'].unique())
        
        # Create multi-parameter phase diagram (showing all in one plot)
        plt.figure(figsize=FIGSIZE)
        
        # Plot overlapping contours for different veg fractions
        colors = plt.cm.viridis(np.linspace(0, 1, len(veg_fractions)))
        
        for i, veg_f in enumerate(veg_fractions):
            # Filter data for this vegetarian fraction
            vf_data = data[data['initial_veg_f'] == veg_f]
            
            # Pivot table for heatmap
            pivot_tipped = vf_data.pivot_table(index='beta', columns='alpha', values='tipped').astype(int)
            
            # Get alpha and beta ranges
            alphas = sorted(vf_data['alpha'].unique())
            betas = sorted(vf_data['beta'].unique())
            
            # Create mesh grid for contour plot
            alpha_grid, beta_grid = np.meshgrid(alphas, betas)
            
            # Plot contours
            cs = plt.contour(alpha_grid, beta_grid, pivot_tipped.values, 
                            levels=[0.5], colors=[colors[i]], linewidths=2)
            cs.collections[0].set_label(f'Veg Fraction = {veg_f}')
        
        plt.xlabel('Individual Preference Weight (α)', fontsize=12)
        plt.ylabel('Social Influence Weight (β)', fontsize=12)
        plt.title('Tipping Point Boundaries for Different Initial Vegetarian Fractions', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        # Save plot
        ensure_output_dir()
        output_file = os.path.join(OUTPUT_DIR, 'tipping_points_combined.png')
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        plt.show()
        
        # Also create separate phase diagrams for each vegetarian fraction
        for veg_f in veg_fractions:
            # Ask if user wants individual plots
            if veg_f == veg_fractions[0]:  # Only ask once
                do_individual = input("\nCreate individual phase diagrams for each vegetarian fraction? (y/n): ").lower() == 'y'
                if not do_individual:
                    break
            
            # Filter data
            vf_data = data[data['initial_veg_f'] == veg_f]
            
            # Create plot
            plt.figure(figsize=FIGSIZE)
            plot_tipping_point_phase_diagram(vf_data, fixed_veg_f=veg_f)
            
            # Save plot
            output_file = os.path.join(OUTPUT_DIR, f'tipping_phase_diagram_veg{veg_f}.png')
            plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
            
            plt.show()
    
    else:
        # Single vegetarian fraction
        plt.figure(figsize=FIGSIZE)
        plot_tipping_point_phase_diagram(data)
        
        # Save plot
        ensure_output_dir()
        output_file = os.path.join(OUTPUT_DIR, 'tipping_phase_diagram.png')
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        plt.show()

def plot_topology_data(file_path=None):
    """Plot topology comparison"""
    if file_path is None:
        # Get topology files
        files = list_model_outputs(pattern="*topology*.pkl")
        if not files:
            print("No topology data files found")
            return
        file_path = select_file(files, "Select topology data file")
        if file_path is None:
            return
    
    # Load data
    data = load_data_file(file_path)
    if data is None:
        return
    
    # Create plot
    plt.figure(figsize=FIGSIZE)
    plot_topology_comparison(data)
    
    # Save plot
    ensure_output_dir()
    output_file = os.path.join(OUTPUT_DIR, 'topology_comparison.png')
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    plt.show()

def plot_intervention_data(file_path=None):
    """Plot intervention analysis"""
    if file_path is None:
        # Get intervention files
        files = list_model_outputs(pattern="*intervention*all*.pkl")
        if not files:
            print("No intervention data files found")
            return
        file_path = select_file(files, "Select intervention data file")
        if file_path is None:
            return
    
    # Load data
    data = load_data_file(file_path)
    if data is None:
        return
    
    # Create plot
    plt.figure(figsize=FIGSIZE)
    plot_targeted_interventions(data)
    
    # Save plot
    ensure_output_dir()
    output_file = os.path.join(OUTPUT_DIR, 'intervention_effects.png')
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    plt.show()

def plot_cluster_data(file_path=None):
    """Plot cluster analysis"""
    if file_path is None:
        # Get cluster files
        files = list_model_outputs(pattern="*cluster*stats*.pkl")
        if not files:
            print("No cluster statistics files found")
            return
        file_path = select_file(files, "Select cluster statistics file")
        if file_path is None:
            return
    
    # Load data
    data = load_data_file(file_path)
    if data is None:
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    
    # Plot 1: Number of clusters vs initial fraction
    axes[0, 0].plot(data['initial_veg_fraction'], data['num_clusters'], 'o-', color='#1f77b4')
    axes[0, 0].set_xlabel('Initial Vegetarian Fraction')
    axes[0, 0].set_ylabel('Number of Clusters')
    axes[0, 0].set_title('Number of Vegetarian Clusters')
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Average cluster size vs initial fraction
    axes[0, 1].plot(data['initial_veg_fraction'], data['avg_cluster_size'], 'o-', color='#ff7f0e')
    axes[0, 1].set_xlabel('Initial Vegetarian Fraction')
    axes[0, 1].set_ylabel('Average Cluster Size')
    axes[0, 1].set_title('Average Vegetarian Cluster Size')
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Maximum cluster size vs initial fraction
    axes[1, 0].plot(data['initial_veg_fraction'], data['max_cluster_size'], 'o-', color='#2ca02c')
    axes[1, 0].set_xlabel('Initial Vegetarian Fraction')
    axes[1, 0].set_ylabel('Maximum Cluster Size')
    axes[1, 0].set_title('Maximum Vegetarian Cluster Size')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Final vs initial vegetarian fraction
    axes[1, 1].plot(data['initial_veg_fraction'], data['final_veg_fraction'], 'o-', color='#d62728')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)  # y=x reference line
    axes[1, 1].set_xlabel('Initial Vegetarian Fraction')
    axes[1, 1].set_ylabel('Final Vegetarian Fraction')
    axes[1, 1].set_title('Growth in Vegetarian Population')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    ensure_output_dir()
    output_file = os.path.join(OUTPUT_DIR, 'cluster_statistics.png')
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    plt.show()
    
    # Check for cluster visualization data
    cluster_files = list_model_outputs(pattern="*cluster*full*.pkl")
    if not cluster_files:
        return
    
    # Ask if user wants to visualize specific clusters
    if input("\nVisualize network clusters? (y/n): ").lower() == 'y':
        cluster_file = select_file(cluster_files, "Select cluster visualization file")
        if cluster_file is None:
            return
        
        cluster_data = load_data_file(cluster_file)
        if cluster_data is None:
            return
        
        # Get available vegetarian fractions
        veg_keys = list(cluster_data.keys())
        print("\nAvailable vegetarian fractions:")
        for i, key in enumerate(veg_keys):
            print(f"[{i}] {key}")
        
        selection = input("Select vegetarian fraction to visualize (or 'q' to cancel): ")
        if selection.lower() == 'q':
            return
        
        try:
            index = int(selection)
            if 0 <= index < len(veg_keys):
                cluster_result = cluster_data[veg_keys[index]]
                
                # Create cluster visualization
                plt.figure(figsize=FIGSIZE)
                plot_vegetarian_clusters(cluster_result)
                
                # Save plot
                output_file = os.path.join(OUTPUT_DIR, f'vegetarian_clusters_{veg_keys[index]}.png')
                plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
                print(f"Plot saved to {output_file}")
                
                plt.show()
                
                # Create influence map
                plt.figure(figsize=FIGSIZE)
                plot_network_influence_map(cluster_result)
                
                # Save plot
                output_file = os.path.join(OUTPUT_DIR, f'network_influence_{veg_keys[index]}.png')
                plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
                print(f"Plot saved to {output_file}")
                
                plt.show()
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")

def main():
    """Main function with simple menu"""
    print("===== Dietary Contagion Model Visualization =====")
    
    while True:
        print("\nVisualization Options:")
        print("[1] Emissions vs Vegetarian Fraction")
        print("[2] Tipping Point Phase Diagrams")
        print("[3] Network Topology Comparison")
        print("[4] Intervention Analysis")
        print("[5] Cluster Statistics")
        print("[0] Exit")
        
        choice = input("\nSelect option: ")
        
        if choice == '1':
            plot_emissions_data()
        elif choice == '2':
            plot_tipping_point_data()
        elif choice == '3':
            plot_topology_data()
        elif choice == '4':
            plot_intervention_data()
        elif choice == '5':
            plot_cluster_data()
        elif choice == '0':
            break
        else:
            print("Invalid option")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()