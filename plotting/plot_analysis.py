import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Publication-quality settings
plt.rcdefaults()
# Publication-quality settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'font.family': 'Arial',
    'axes.labelpad': 10,
    'pdf.fonttype': 42,  # Ensures text is editable in Adobe Illustrator
    'ps.fonttype': 42
})

# Color schemes
COLORS = {
    'primary': '#2271B2',  # Main blue
    'secondary': '#F15C37',  # Orange
    'tertiary': '#3B9E42',  # Green
    'gray': '#666666',
    'light_gray': '#CCCCCC'
}

# Custom color map for heatmaps that fits with nature style
nature_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
nature_cmap = LinearSegmentedColormap.from_list('nature_blues', nature_colors)

def set_publication_style(ax):
    """Apply publication-quality styling to an axis"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=-1)

def load_data():
    """
    Lists and loads parameter sweep files from the model_output directory.
    """
    output_dir = '../model_output'
    pkl_files = [f for f in os.listdir(output_dir) if f.endswith('.pkl') and f.startswith('parameter_sweep')]
    
    if not pkl_files:
        raise FileNotFoundError("No parameter sweep pickle files found in model_output directory")
    
    print("Available parameter sweep files:")
    for idx, file in enumerate(pkl_files):
        date_str = file.split('_')[2:5]
        date_str = '_'.join(date_str)
        try:
            date = datetime.strptime(date_str, '%b_%d_%Y')
            date_formatted = date.strftime('%Y-%m-%d')
        except ValueError:
            date_formatted = "Unknown date"
        print(f"[{idx}] {file} (Date: {date_formatted})")
    
    while True:
        try:
            selection = int(input("\nEnter the number of the file to analyze: "))
            if 0 <= selection < len(pkl_files):
                selected_file = pkl_files[selection]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Load the DataFrame from pickle
    df = pd.read_pickle(os.path.join(output_dir, selected_file))
    
    print("\nLoaded DataFrame info:")
    print(df.info())
    
    return df

def plot_heatmap(df, x_param, y_param, value_param, title=None):
    """Creates a publication-quality heatmap"""
    import matplotlib.ticker as ticker
    
    # Calculate mean and std for the value parameter
    pivot_df = df.groupby([x_param, y_param])[value_param].agg(['mean', 'std']).reset_index()
    pivot_table = pivot_df.pivot(index=y_param, columns=x_param, values='mean')
    pivot_table_std = pivot_df.pivot(index=y_param, columns=x_param, values='std')
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Create heatmap with annotations showing mean ± std
    sns.heatmap(pivot_table, annot=True, fmt=".2f", 
                annot_kws={'size': 8}, cmap = "crest",
                cbar_kws={'label': "Final System Emissions [kg/CO2/year]"})
    
    # Add standard deviation annotations
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            if not np.isnan(pivot_table_std.iloc[i, j]):
                text = f'±{pivot_table_std.iloc[i, j]:.2f}'
                ax.text(j + 0.5, i + 0.7, text, 
                       ha='center', va='center', 
                       color='gray', fontsize=6)
    
    # Format x and y axis tick labels to 2 decimal places
    # For the x-axis (which shows the column values)
    x_labels = [f'{float(label):.2f}' for label in pivot_table.columns]
    ax.set_xticklabels(x_labels)
    
    # For the y-axis (which shows the index values)
    y_labels = [f'{float(label):.2f}' for label in pivot_table.index]
    ax.set_yticklabels(y_labels)
    
    # Format the colorbar ticks to 2 decimal places
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    plt.title(title or f'Effect of {x_param} and {y_param} on {value_param}')
    plt.xlabel(x_param.replace('_', ' ').title())
    plt.ylabel(y_param.replace('_', ' ').title())
    plt.legend(title)
    
    set_publication_style(ax)
    plt.tight_layout()
    save_plot(f'heatmap_{x_param}_{y_param}')
    plt.show()

def plot_trajectories(df, params_to_group=None):
    """
    Plots trajectories for system_C and fraction_veg grouped by specified parameters
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if params_to_group:
        groups = df.groupby(params_to_group)
        for name, group in groups:
            # Get trajectories
            system_C_trajectories = np.array([t for t in group['system_C_trajectory']])
            veg_trajectories = np.array([t for t in group['fraction_veg_trajectory']])
            
            # Find minimum length to ensure consistent array sizes
            min_length_C = min(len(t) for t in system_C_trajectories)
            min_length_veg = min(len(t) for t in veg_trajectories)
            
            # Trim all trajectories to minimum length
            system_C_trajectories = np.array([t[:min_length_C] for t in system_C_trajectories])
            veg_trajectories = np.array([t[:min_length_veg] for t in veg_trajectories])
            
            # Calculate means and standard deviations
            mean_C = np.mean(system_C_trajectories, axis=0)
            std_C = np.std(system_C_trajectories, axis=0)
            mean_veg = np.mean(veg_trajectories, axis=0)
            std_veg = np.std(veg_trajectories, axis=0)
            
            # Create time arrays of appropriate length
            time_C = np.arange(len(mean_C))
            time_veg = np.arange(len(mean_veg))
            
            # Plot with error bands
            ax1.plot(time_C, mean_C, label=f'{params_to_group}={name:.2f}')
            ax1.fill_between(time_C, mean_C-std_C, mean_C+std_C, alpha=0.2)
            
            ax2.plot(time_veg, mean_veg, label=f'{params_to_group}={name:.2f}')
            ax2.fill_between(time_veg, mean_veg-std_veg, mean_veg+std_veg, alpha=0.2)
    else:
        # Plot individual trajectories
        for _, row in df.iterrows():
            system_C = np.array(row['system_C_trajectory'])
            veg_frac = np.array(row['fraction_veg_trajectory'])
            
            time_C = np.arange(len(system_C))
            time_veg = np.arange(len(veg_frac))
            
            ax1.plot(time_C, system_C, alpha=0.1, color='blue')
            ax2.plot(time_veg, veg_frac, alpha=0.1, color='green')
    
    # Formatting
    ax1.set_xlabel('t')
    ax1.set_ylabel('System CO₂ Emissions [kg/CO2/year]')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Vegetarian Fraction')
    
    if params_to_group:
        ax1.legend()
        ax2.legend()
    
    set_publication_style(ax1)
    set_publication_style(ax2)
    plt.tight_layout()
    save_plot('trajectories')
    plt.show()

    
def plot_distributions(df, column):
    """
    Plots both histogram and ECDF for a given column in publication quality.
    """
    # Create a figure with two subplots with more space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.3)  # Add more space between subplots
    
    # Histogram
    data = df[column].explode()
    sns.histplot(data=data, ax=ax1, color=COLORS['primary'], 
                bins=30, alpha=0.7)
    
    # Set correct label based on column type
    if column == 'final_system_C':
        ax1.set_xlabel("End state emissions [kg CO₂/year]")
    elif column == 'individual_reductions':
        ax1.set_xlabel("Total attributable emissions reduction [kg CO₂/agent]")
    ax1.set_ylabel('Frequency')
    set_publication_style(ax1)
    
    # ECDF
    sns.ecdfplot(data=data, ax=ax2, color=COLORS['primary'], 
                linewidth=2)
    # Set same label for ECDF
    if column == 'final_system_C':
        ax2.set_xlabel("End state emissions [kg CO₂/year]")
    elif column == 'individual_reductions':
        ax2.set_xlabel("Total attributable emissions reduction [kg CO₂/agent]")
    ax2.set_ylabel('Cumulative Probability')
    set_publication_style(ax2)
    
    # Use a more robust approach to layout
    plt.tight_layout(pad=2.0)  # Increase padding
    return fig, (ax1, ax2)


def create_summary_statistics(df):
    """Creates summary statistics for numerical columns"""
    # Identify numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create summary statistics
    stats = df[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median']).round(3)
    
    # Add additional statistics for trajectory data
    trajectory_cols = [col for col in df.columns if '_trajectory' in col]
    for col in trajectory_cols:
        if isinstance(df[col].iloc[0], (list, np.ndarray)):
            trajectories = np.vstack(df[col].values)
            stats.loc['final_mean', col] = np.mean(trajectories[:, -1])
            stats.loc['final_std', col] = np.std(trajectories[:, -1])
    
    return stats



def save_plot(name, dpi=300):
    """Saves the current plot with error handling"""
    output_dir = '../visualisations_output'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        plt.savefig(os.path.join(output_dir, f"{name}.pdf"), 
                   bbox_inches='tight',
                   pad_inches=0.1)  # Add some padding
    except RuntimeError as e:
        print(f"Error saving {name}: {str(e)}")
        # Try alternative renderer
        plt.savefig(os.path.join(output_dir, f"{name}.pdf"),
                   bbox_inches='tight',
                   pad_inches=0.1,
                   backend='pgf')  # Try alternative backend



    
if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Create summary statistics
    summary_stats = create_summary_statistics(df)
    print("\nSummary Statistics:")
    print(summary_stats)

    if all(param in df.columns for param in ['alpha', 'beta']):
        plot_heatmap(df, 'alpha', 'beta', 'final_system_C', 
                    'Impact of Social and Individual Dissonance on Emissions')
    
    # Plot distributions for relevant columns
    for col in ['final_system_C', 'individual_reductions']:
        plot_distributions(df, col)
    
    # # Plot parameter effects
    # if 'veg_f' in df.columns:
    #     plot_parameter_effect(df, 'veg_f', 'final_system_C')
    #     if 'beta' in df.columns:
    #         plot_parameter_effect(df, 'veg_f', 'final_system_C', 'beta')
    
    # Plot trajectories
    plot_trajectories(df, params_to_group='alpha')