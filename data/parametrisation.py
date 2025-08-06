import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import argparse
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fit distributions to demographics data')
parser.add_argument('parameter', choices=['alpha', 'theta', 'rho'], 
                   help='Parameter to analyze: alpha, theta, or rho')
args = parser.parse_args()

# File and column mapping
file_mapping = {
    'alpha': ('alpha_demographics.xlsx', 'Self-identity weight (alpha)'),
    'theta': ('theta_diet_demographics.xlsx', 'Personal Preference for Veg Diet'),
    'rho': ('rho_demographics.xlsx', 'Cost parameter (rho)')
}

filename, column_name = file_mapping[args.parameter]

# Load data
print(f"Loading {args.parameter} data from {filename}...")
try:
    data = pd.read_excel(f"./{filename}")
    if column_name not in data.columns:
        print(f"Available columns: {data.columns.tolist()}")
        sys.exit(f"Column '{column_name}' not found in {filename}")
    parameter_data = data[column_name].dropna()
    print(f"Loaded {len(parameter_data)} data points for {args.parameter}")
except Exception as e:
    sys.exit(f"Error loading data: {e}")

# Candidate distributions - focus on continuous distributions for 0-1 bounded data
distributions = {
    'beta': stats.beta,      # Best for 0-1 bounded data
    'norm': stats.norm,      # Normal distribution
    'gamma': stats.gamma,    # Gamma distribution
    'lognorm': stats.lognorm, # Log-normal
    'weibull_min': stats.weibull_min,
    'uniform': stats.uniform  # Uniform distribution
}

# Add discrete distributions only if data appears to be discrete
if len(np.unique(parameter_data)) < 20:  # Heuristic for discrete data
    distributions.update({
        'poisson': stats.poisson,
        'nbinom': stats.nbinom,
        'geom': stats.geom
    })

aic_scores = {}
fit_results = {}
fit_params = {}
x = np.linspace(min(parameter_data), max(parameter_data), 1000)
x_discrete = np.arange(int(min(parameter_data)), int(max(parameter_data)) + 1)

# Fit each distribution
for name, dist in distributions.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if name == 'poisson':
                # Scale data for Poisson (expects non-negative integers)
                scaled_data = (parameter_data * 10).astype(int)
                mu = np.mean(scaled_data)
                y = dist.pmf(x_discrete * 10, mu) / 10  # Scale back for plotting
                loglik = np.sum(dist.logpmf(scaled_data, mu))
                k = 1
                fit_results[name] = (x_discrete, y)
                fit_params[name] = [mu]
            elif name == 'nbinom':
                # Scale data for negative binomial
                scaled_data = (parameter_data * 10).astype(int)
                mean = np.mean(scaled_data)
                var = np.var(scaled_data)
                if var <= mean:
                    continue
                p = mean / var
                n = mean * p / (1 - p)
                y = dist.pmf(x_discrete * 10, n, p) / 10
                loglik = np.sum(dist.logpmf(scaled_data, n, p))
                k = 2
                fit_results[name] = (x_discrete, y)
                fit_params[name] = [n, p]
            elif name == 'geom':
                # Scale data for geometric
                scaled_data = (parameter_data * 10).astype(int) + 1  # Geom needs positive integers
                p = 1 / np.mean(scaled_data)
                y = dist.pmf(x_discrete * 10 + 1, p) / 10
                loglik = np.sum(dist.logpmf(scaled_data, p))
                k = 1
                fit_results[name] = (x_discrete, y)
                fit_params[name] = [p]
            elif name == 'beta':
                # Beta distribution is perfect for 0-1 bounded data
                # Use method of moments for initial guess, then MLE
                sample_mean = np.mean(parameter_data)
                sample_var = np.var(parameter_data)
                # Method of moments estimates
                alpha_est = sample_mean * ((sample_mean * (1 - sample_mean)) / sample_var - 1)
                beta_est = (1 - sample_mean) * ((sample_mean * (1 - sample_mean)) / sample_var - 1)
                # Ensure positive parameters
                if alpha_est > 0 and beta_est > 0:
                    params = (alpha_est, beta_est)
                else:
                    params = dist.fit(parameter_data)
                y = dist.pdf(x, *params)
                loglik = np.sum(dist.logpdf(parameter_data, *params))
                k = len(params)
                fit_results[name] = (x, y)
                fit_params[name] = list(params)
            elif name == 'uniform':
                # Fit uniform distribution to the range of data
                params = (np.min(parameter_data), np.max(parameter_data) - np.min(parameter_data))
                y = dist.pdf(x, *params)
                loglik = np.sum(dist.logpdf(parameter_data, *params))
                k = len(params)
                fit_results[name] = (x, y)
                fit_params[name] = list(params)
            else:
                # For other continuous distributions
                params = dist.fit(parameter_data)
                y = dist.pdf(x, *params)
                loglik = np.sum(dist.logpdf(parameter_data, *params))
                k = len(params)
                fit_results[name] = (x, y)
                fit_params[name] = list(params)

            aic = 2 * k - 2 * loglik
            aic_scores[name] = aic
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

# Determine best fit
sorted_aic = sorted(aic_scores.items(), key=lambda x: x[1])
best_fit_name = sorted_aic[0][0]
best_x, best_y = fit_results[best_fit_name]
best_params = fit_params[best_fit_name]

# Plot
plt.figure(figsize=(12, 8))
plt.hist(parameter_data, bins=30, density=True, alpha=0.6, color='skyblue', 
         edgecolor='black', label=f'{args.parameter.capitalize()} data histogram')

# Plot top 3 distributions for comparison
top_3 = sorted_aic[:3]
colors = ['red', 'green', 'orange']
for i, (dist_name, aic_score) in enumerate(top_3):
    if dist_name in fit_results:
        fit_x, fit_y = fit_results[dist_name]
        params = fit_params[dist_name]
        plt.plot(fit_x, fit_y, color=colors[i], linewidth=2,
                label=f"{dist_name} (AIC={aic_score:.1f})\nParams: {np.round(params, 3)}")

plt.title(f"Distribution Fitting for {args.parameter.capitalize()} Parameter")
plt.xlabel(f"{args.parameter.capitalize()} Value")
plt.ylabel("Density")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
output_filename = f"{args.parameter}_distribution_fit.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_filename}")
plt.show()

# Print AIC scores
print("\n📊 AIC Scores (lower = better):")
for dist_name, score in sorted_aic:
    print(f"{dist_name:12}: AIC = {score:.2f}")

# Print best distribution and its parameters
print(f"\n✅ Best fit: {best_fit_name}")
print(f"🔧 Parameters: {np.round(best_params, 5)}")

# Print summary statistics
print(f"\n📈 Data Summary for {args.parameter}:")
print(f"   Sample size: {len(parameter_data)}")
print(f"   Mean: {np.mean(parameter_data):.4f}")
print(f"   Std: {np.std(parameter_data):.4f}")
print(f"   Min: {np.min(parameter_data):.4f}")
print(f"   Max: {np.max(parameter_data):.4f}")