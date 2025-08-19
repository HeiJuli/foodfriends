import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import sys
import os

# Get parameter from user input
parameter = input("Enter parameter to analyze (alpha, theta, or rho): ").lower().strip()
if parameter not in ['alpha', 'theta', 'rho']:
    print("Invalid parameter. Please choose: alpha, theta, or rho")
    sys.exit(1)

# File and column mapping
file_mapping = {
    'alpha': ('alpha_demographics.xlsx', 'Self-identity weight (alpha)'),
    'theta': ('theta_diet_demographics.xlsx', 'Personal Preference for Veg Diet'),
    'rho': ('rho_demographics.xlsx', 'Cost parameter (rho)')
}

filename, column_name = file_mapping[parameter]

# Load data
print(f"Loading {parameter} data from {filename}...")
try:
    data = pd.read_excel(f"./{filename}")
    if column_name not in data.columns:
        print(f"Available columns: {data.columns.tolist()}")
        sys.exit(f"Column '{column_name}' not found in {filename}")
    raw_data = data[column_name].dropna()
    
    # Clip data to appropriate range
    if parameter == 'theta':
        parameter_data = np.clip(raw_data, -1, 1)
    else:
        parameter_data = np.clip(raw_data, 0, 1)
    print(f"Loaded {len(parameter_data)} data points for {parameter}")
    print(f"Data range: [{parameter_data.min():.3f}, {parameter_data.max():.3f}]")
except Exception as e:
    sys.exit(f"Error loading data: {e}")

# Define distributions based on parameter type
if parameter == 'theta':
    distributions = {
        'skewnorm': stats.skewnorm,
        'norm': stats.norm,
        'truncnorm': stats.truncnorm,
        't': stats.t,
        'laplace': stats.laplace
    }
else:
    # Check for boundary concentration
    boundary_frac = (np.sum(parameter_data <= 0.05) + np.sum(parameter_data >= 0.95)) / len(parameter_data)
    
    if boundary_frac > 0.7:  # Boundary-heavy data
        distributions = {
            'beta_u': 'beta_u_shaped',  # U-shaped beta (a,b < 1)
            'beta': stats.beta,
            'truncnorm': stats.truncnorm
        }
    else:  # Normal data
        distributions = {
            'beta': stats.beta,
            'truncnorm': stats.truncnorm,
            'gamma_scaled': 'gamma_scaled'
        }

aic_scores = {}
fit_results = {}
fit_params = {}

if parameter == 'theta':
    x = np.linspace(-1.05, 1.05, 1000)
else:
    x = np.linspace(-0.05, 1.05, 1000)

# Fit distributions
for name, dist in distributions.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if name == 'beta_u_shaped':
                # U-shaped beta with a,b < 1
                a, b = 0.5, 0.5  # U-shaped
                eps = 1e-6
                adj_data = np.clip(parameter_data, eps, 1-eps)
                x_beta = np.linspace(eps, 1-eps, 1000)
                y = stats.beta.pdf(x_beta, a, b)
                loglik = np.sum(stats.beta.logpdf(adj_data, a, b))
                fit_results[name] = (x_beta, y)
                fit_params[name] = [a, b]
                
            elif name == 'gamma_scaled':
                # Gamma scaled to [0,1]
                scaled_data = parameter_data * 10
                a, loc, scale = stats.gamma.fit(scaled_data, floc=0)
                x_gamma = np.linspace(0.01, 0.99, 1000)
                y = stats.gamma.pdf(x_gamma * 10, a, 0, scale) * 10
                loglik = np.sum(stats.gamma.logpdf(scaled_data, a, 0, scale))
                fit_results[name] = (x_gamma, y)
                fit_params[name] = [a, scale]
                
            elif name == 'skewnorm':
                # Prioritize skewed normal for theta
                a, loc, scale = dist.fit(parameter_data)
                y = dist.pdf(x, a, loc, scale)
                loglik = np.sum(dist.logpdf(parameter_data, a, loc, scale))
                fit_results[name] = (x, y)
                fit_params[name] = [a, loc, scale]
                
            elif name == 'beta':
                # Proper beta fitting with boundary handling
                eps = 1e-6
                adj_data = np.clip(parameter_data, eps, 1-eps)
                
                # Method of moments with constraints
                m = np.mean(adj_data)
                v = np.var(adj_data)
                
                if v < m * (1 - m) and v > 0:
                    common = (m * (1 - m)) / v - 1
                    a_est = m * common
                    b_est = (1 - m) * common
                    a = np.clip(a_est, 0.1, 50)  # Reasonable bounds
                    b = np.clip(b_est, 0.1, 50)
                else:
                    a, b = 1, 1  # Fallback to uniform
                
                x_beta = np.linspace(eps, 1-eps, 1000)
                y = stats.beta.pdf(x_beta, a, b)
                loglik = np.sum(stats.beta.logpdf(adj_data, a, b))
                fit_results[name] = (x_beta, y)
                fit_params[name] = [a, b]
                
            elif name == 'truncnorm':
                m, s = np.mean(parameter_data), np.std(parameter_data)
                if parameter == 'theta':
                    a_t, b_t = (-1 - m) / s, (1 - m) / s
                else:
                    a_t, b_t = (0 - m) / s, (1 - m) / s
                
                y = dist.pdf(x, a_t, b_t, m, s)
                loglik = np.sum(dist.logpdf(parameter_data, a_t, b_t, m, s))
                fit_results[name] = (x, y)
                fit_params[name] = [a_t, b_t, m, s]
                
            else:
                # Standard fitting
                params = dist.fit(parameter_data)
                y = dist.pdf(x, *params)
                loglik = np.sum(dist.logpdf(parameter_data, *params))
                fit_results[name] = (x, y)
                fit_params[name] = list(params)

            # Calculate AIC
            k = len(fit_params[name])
            aic = 2 * k - 2 * loglik
            aic_scores[name] = aic
            
        except Exception as e:
            print(f"WARNING: {name} failed: {e}")

if not aic_scores:
    sys.exit("No distributions could be fitted successfully")

# Sort by AIC
sorted_aic = sorted(aic_scores.items(), key=lambda x: x[1])
best_fit_name = sorted_aic[0][0]

# Create output directory
os.makedirs('../visualisations_output', exist_ok=True)

# Plot results
plt.figure(figsize=(12, 8))

# Data histogram
n_bins = max(10, min(20, len(np.unique(parameter_data))))
plt.hist(parameter_data, bins=n_bins, density=True, alpha=0.6, 
         color='skyblue', edgecolor='black', 
         label=f'{parameter.capitalize()} data histogram')

# Plot top 4 distributions
colors = ['red', 'green', 'orange', 'purple']
for i, (dist_name, aic_score) in enumerate(sorted_aic[:4]):
    if dist_name in fit_results:
        fit_x, fit_y = fit_results[dist_name]
        params = fit_params[dist_name]
        
        plt.plot(fit_x, fit_y, color=colors[i], linewidth=2,
                label=f"{dist_name} (AIC={aic_score:.1f})\n"
                      f"Params: {np.round(params[:2], 3)}")

plt.title(f"Distribution Fitting for {parameter.capitalize()} Parameter")
plt.xlabel(f"{parameter.capitalize()} Value")
plt.ylabel('Probability Density')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

if parameter == 'theta':
    plt.xlim(-1.1, 1.1)
else:
    plt.xlim(-0.05, 1.05)

plt.tight_layout()

# Save plot
output_filename = f"../visualisations_output/{parameter}_distribution_fit.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_filename}")
plt.show()

# Print results
print("\nAIC Scores (lower = better):")
for dist_name, score in sorted_aic:
    print(f"{dist_name:12}: {score:.2f}")

print(f"\nBest fit: {best_fit_name}")
print(f"Parameters: {np.round(fit_params[best_fit_name], 5)}")

# Generate sampling code
print(f"\nSampling code for {best_fit_name}:")
params = fit_params[best_fit_name]

if best_fit_name == 'skewnorm':
    a, loc, scale = params[:3]
    print(f"from scipy.stats import skewnorm")
    print(f"samples = skewnorm.rvs({a:.3f}, loc={loc:.3f}, scale={scale:.3f}, size=N)")
elif best_fit_name == 'beta' or best_fit_name == 'beta_u_shaped':
    a, b = params[:2]
    print(f"samples = np.random.beta({a:.3f}, {b:.3f}, size=N)")
elif best_fit_name == 'truncnorm':
    a, b, loc, scale = params[:4]
    print(f"samples = stats.truncnorm.rvs({a:.3f}, {b:.3f}, loc={loc:.3f}, scale={scale:.3f}, size=N)")
elif best_fit_name == 'gamma_scaled':
    a, scale = params[:2]
    print(f"samples = np.random.gamma({a:.3f}, scale={scale:.3f}, size=N) / 10")
else:
    print(f"samples = stats.{best_fit_name}.rvs(*{params}, size=N)")

print(f"\nData Summary:")
print(f"N: {len(parameter_data)}, Mean: {np.mean(parameter_data):.3f}, Std: {np.std(parameter_data):.3f}")