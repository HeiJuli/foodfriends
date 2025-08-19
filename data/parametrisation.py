import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import sys

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

# Check if data is mostly at boundaries (binary-like)
def is_boundary_heavy(data, threshold=0.8):
    at_boundaries = np.sum((data <= 0.05) | (data >= 0.95))
    return at_boundaries / len(data) > threshold

boundary_heavy = is_boundary_heavy(parameter_data) if parameter != 'theta' else False

# Distributions for different parameter ranges
if parameter == 'theta':
    distributions = {
        'norm': stats.norm,
        'uniform': stats.uniform, 
        'truncnorm': stats.truncnorm,
        'beta_scaled': 'beta_scaled'
    }
elif boundary_heavy:
    # For boundary-heavy data, use discrete/mixture approaches
    distributions = {
        'bernoulli_mix': 'bernoulli_mix',  # Custom mixture
        'beta_bimodal': 'beta_bimodal',    # Bimodal beta
        'discrete': 'discrete'             # Discrete distribution
    }
    print(f"Detected boundary-heavy data: {np.sum((parameter_data <= 0.05))}/{len(parameter_data)} at 0, {np.sum((parameter_data >= 0.95))}/{len(parameter_data)} at 1")
else:
    distributions = {
        'beta': stats.beta,
        'truncnorm': stats.truncnorm
    }

aic_scores = {}
fit_results = {}
fit_params = {}
x = np.linspace(parameter_data.min()-0.1, parameter_data.max()+0.1, 1000)

# Fit each distribution
for name, dist in distributions.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if name == 'bernoulli_mix':
                # Mixture of point masses at 0 and 1 plus uniform in between
                p_0 = np.sum(parameter_data <= 0.05) / len(parameter_data)
                p_1 = np.sum(parameter_data >= 0.95) / len(parameter_data)
                p_mid = 1 - p_0 - p_1
                
                # Create discrete representation
                x_discrete = np.array([0, 1])
                y_discrete = np.array([p_0, p_1])
                
                # Add uniform component if there's middle data
                if p_mid > 0:
                    x_mid = np.linspace(0.1, 0.9, 100)
                    y_mid = np.full_like(x_mid, p_mid / 0.8)  # Uniform over [0.1, 0.9]
                    fit_x = np.concatenate([x_discrete, x_mid])
                    fit_y = np.concatenate([y_discrete, y_mid])
                else:
                    fit_x, fit_y = x_discrete, y_discrete
                
                # Simple log-likelihood for discrete case
                loglik = p_0 * np.log(p_0 + 1e-10) + p_1 * np.log(p_1 + 1e-10)
                loglik *= len(parameter_data)
                
                fit_results[name] = (fit_x, fit_y)
                fit_params[name] = [p_0, p_1, p_mid]
                
            elif name == 'beta_bimodal':
                # Try a very low alpha, beta to get bimodal at boundaries
                a, b = 0.1, 0.1  # Forces bimodal at 0 and 1
                x_beta = np.linspace(0.001, 0.999, 1000)
                y = stats.beta.pdf(x_beta, a, b)
                loglik = np.sum(stats.beta.logpdf(np.clip(parameter_data, 0.001, 0.999), a, b))
                fit_results[name] = (x_beta, y)
                fit_params[name] = [a, b]
                
            elif name == 'discrete':
                # Pure discrete distribution based on unique values
                unique_vals, counts = np.unique(parameter_data, return_counts=True)
                probs = counts / len(parameter_data)
                
                # Only use if we have few unique values
                if len(unique_vals) <= 10:
                    fit_results[name] = (unique_vals, probs)
                    fit_params[name] = list(zip(unique_vals, probs))
                    loglik = np.sum(np.log(probs[np.searchsorted(unique_vals, parameter_data)]))
                else:
                    continue
                
            elif name == 'lognorm_scaled':
                # Log-normal scaled to [0,1] - good for right-skewed data
                # Fit lognorm to shifted/scaled data
                shifted_data = parameter_data + 0.001  # Avoid zeros
                s, loc, scale = stats.lognorm.fit(shifted_data, floc=0)
                
                # Create scaled version
                x_ln = np.linspace(0.001, 0.999, 1000)
                y_ln = stats.lognorm.pdf(x_ln + 0.001, s, 0, scale)
                
                loglik = np.sum(stats.lognorm.logpdf(shifted_data, s, 0, scale))
                fit_results[name] = (x_ln, y_ln)
                fit_params[name] = [s, scale]
                
            elif name == 'weibull_scaled':
                # Weibull scaled to [0,1] - another good option for skewed data
                c, loc, scale = stats.weibull_min.fit(parameter_data, floc=0)
                
                # Scale to [0,1] range
                x_weib = np.linspace(0.001, 0.999, 1000)
                y_weib = stats.weibull_min.pdf(x_weib, c, 0, scale)
                
                loglik = np.sum(stats.weibull_min.logpdf(parameter_data, c, 0, scale))
                fit_results[name] = (x_weib, y_weib)
                fit_params[name] = [c, scale]
                
            elif name == 'laplace':
                params = stats.laplace.fit(parameter_data)
                y = stats.laplace.pdf(x, *params)
                loglik = np.sum(stats.laplace.logpdf(parameter_data, *params))
                fit_results[name] = (x, y)
                fit_params[name] = list(params)
                
            elif name == 't':
                params = stats.t.fit(parameter_data)
                y = stats.t.pdf(x, *params)
                loglik = np.sum(stats.t.logpdf(parameter_data, *params))
                fit_results[name] = (x, y)
                fit_params[name] = list(params)
                
            elif name == 'beta_constrained':
                # Beta with reasonable parameter constraints
                # Try method of moments first, then constrain
                sample_mean = np.mean(parameter_data)
                sample_var = np.var(parameter_data)
                
                if sample_var < sample_mean * (1 - sample_mean):
                    # Method of moments
                    common = (sample_mean * (1 - sample_mean)) / sample_var - 1
                    a_est = sample_mean * common
                    b_est = (1 - sample_mean) * common
                    
                    # Constrain to reasonable ranges
                    a = np.clip(a_est, 0.1, 10)  # Avoid extreme shapes
                    b = np.clip(b_est, 0.1, 10)
                else:
                    # Fallback for high variance
                    a, b = 1, 1
                
                x_beta = np.linspace(0.001, 0.999, 1000)
                y = stats.beta.pdf(x_beta, a, b)
                loglik = np.sum(stats.beta.logpdf(np.clip(parameter_data, 0.001, 0.999), a, b))
                fit_results[name] = (x_beta, y)
                fit_params[name] = [a, b]
                
            elif name == 'gamma_scaled':
                # Gamma distribution scaled to [0,1]
                # Fit gamma to scaled data
                scaled_data = parameter_data * 10  # Scale up for fitting
                shape, loc, scale = stats.gamma.fit(scaled_data, floc=0)
                
                # Transform back to [0,1]
                x_gamma = np.linspace(0.001, 0.999, 1000)
                x_gamma_scaled = x_gamma * 10
                y_gamma = stats.gamma.pdf(x_gamma_scaled, shape, 0, scale) * 10  # Jacobian
                
                loglik = np.sum(stats.gamma.logpdf(scaled_data, shape, 0, scale))
                fit_results[name] = (x_gamma, y_gamma)
                fit_params[name] = [shape, scale]
                
            elif name == 'skewnorm':
                # Skewed normal for theta (can handle asymmetric data well)
                from scipy.stats import skewnorm
                a_skew, loc, scale = skewnorm.fit(parameter_data)
                y = skewnorm.pdf(x, a_skew, loc, scale)
                loglik = np.sum(skewnorm.logpdf(parameter_data, a_skew, loc, scale))
                fit_results[name] = (x, y)
                fit_params[name] = [a_skew, loc, scale]
                
            elif name == 'beta_scaled':
                # Beta distribution scaled to [-1,1] with reasonable constraints
                scaled_data = (parameter_data + 1) / 2  # Scale to [0,1]
                
                # Use constrained fitting similar to beta_constrained
                sample_mean = np.mean(scaled_data)
                sample_var = np.var(scaled_data)
                
                if sample_var < sample_mean * (1 - sample_mean):
                    common = (sample_mean * (1 - sample_mean)) / sample_var - 1
                    a_est = sample_mean * common
                    b_est = (1 - sample_mean) * common
                    a = np.clip(a_est, 0.1, 10)
                    b = np.clip(b_est, 0.1, 10)
                else:
                    a, b = 1, 1
                
                # Transform back to [-1,1] for plotting
                x_beta = np.linspace(-0.999, 0.999, 1000)
                x_beta_scaled = (x_beta + 1) / 2
                y_beta = stats.beta.pdf(x_beta_scaled, a, b) / 2  # Jacobian factor
                
                loglik = np.sum(stats.beta.logpdf(scaled_data, a, b)) - len(scaled_data) * np.log(2)
                fit_results[name] = (x_beta, y_beta)
                fit_params[name] = [a, b]
                
            elif name == 'beta':
                # Proper beta fitting with bounds checking
                if np.any(parameter_data <= 0) or np.any(parameter_data >= 1):
                    # Slightly adjust data to avoid boundary issues
                    adj_data = np.clip(parameter_data, 1e-6, 1-1e-6)
                else:
                    adj_data = parameter_data
                
                # Use MLE for beta distribution
                a, b, loc, scale = dist.fit(adj_data, floc=0, fscale=1)
                params = (a, b, 0, 1)  # Force loc=0, scale=1
                
                # Calculate PDF over [0,1] range
                x_beta = np.linspace(1e-6, 1-1e-6, 1000)
                y = dist.pdf(x_beta, a, b, 0, 1)
                loglik = np.sum(dist.logpdf(adj_data, a, b, 0, 1))
                
                fit_results[name] = (x_beta, y)
                fit_params[name] = [a, b]
           
            elif name == 'triang':
                # Triangular distribution
                c_est = (np.mean(parameter_data) - parameter_data.min()) / (parameter_data.max() - parameter_data.min())
                c_est = np.clip(c_est, 0.01, 0.99)  # Avoid boundary issues
                params = dist.fit(parameter_data, fc=c_est)
                y = dist.pdf(x, *params)
                loglik = np.sum(dist.logpdf(parameter_data, *params))
                fit_results[name] = (x, y)
                fit_params[name] = list(params)
                
            elif name == 'truncnorm':
                # Truncated normal to appropriate bounds
                mean_est = np.mean(parameter_data)
                std_est = np.std(parameter_data)
                if parameter == 'theta':
                    a_trunc = (-1 - mean_est) / std_est
                    b_trunc = (1 - mean_est) / std_est
                else:
                    a_trunc = (0 - mean_est) / std_est
                    b_trunc = (1 - mean_est) / std_est
                params = (a_trunc, b_trunc, mean_est, std_est)
                y = dist.pdf(x, *params)
                loglik = np.sum(dist.logpdf(parameter_data, *params))
                fit_results[name] = (x, y)
                fit_params[name] = list(params)
                
            else:
                # Standard fitting for other distributions
                params = dist.fit(parameter_data)
                y = dist.pdf(x, *params)
                loglik = np.sum(dist.logpdf(parameter_data, *params))
                fit_results[name] = (x, y)
                fit_params[name] = list(params)

            # Calculate AIC
            k = len(params)
            aic = 2 * k - 2 * loglik
            aic_scores[name] = aic
            
        except Exception as e:
            print(f"WARNING: {name} failed: {e}")

if not aic_scores:
    sys.exit("No distributions could be fitted successfully")

# Determine best fit
sorted_aic = sorted(aic_scores.items(), key=lambda x: x[1])
best_fit_name = sorted_aic[0][0]
best_x, best_y = fit_results[best_fit_name]
best_params = fit_params[best_fit_name]

# Plot with proper normalization
plt.figure(figsize=(12, 8))

# Calculate bins first
n_bins = max(10, min(20, len(np.unique(parameter_data))))

# For boundary-heavy data, use count histogram instead of density
if boundary_heavy:
    # Use counts, not density, for boundary-heavy data
    counts, bins, patches = plt.hist(parameter_data, bins=n_bins, density=False, alpha=0.6, 
                                    color='skyblue', edgecolor='black', 
                                    label=f'{parameter.capitalize()} data histogram (counts)')
    # Convert to probabilities for comparison
    total_count = len(parameter_data)
    counts_normalized = counts / total_count
    print(f"Data is mostly at boundaries - showing counts instead of density")
    
    # Create probability histogram for comparison
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.hist(parameter_data, bins=n_bins, density=False, alpha=0.6, 
             color='skyblue', edgecolor='black')
    ax2.set_ylabel('Count')
    ax2.set_xlabel(f'{parameter.capitalize()} Value')
    ax2.set_title(f'{parameter.capitalize()} Data Distribution (Counts)')
    ax2.grid(True, alpha=0.3)
    plt.figure(1)  # Return to main figure
    
else:
    # Normal density histogram for non-boundary data
    counts, bins, patches = plt.hist(parameter_data, bins=n_bins, density=True, alpha=0.6, 
                                    color='skyblue', edgecolor='black', 
                                    label=f'{parameter.capitalize()} data histogram')
    # Verify histogram integrates to 1
    bin_width = bins[1] - bins[0]
    hist_integral = np.sum(counts) * bin_width
    print(f"Histogram integral: {hist_integral:.3f} (should be ~1.0)")

# Plot top 4 distributions
top_4 = sorted_aic[:4]
colors = ['red', 'green', 'orange', 'purple']

for i, (dist_name, aic_score) in enumerate(top_4):
    if dist_name in fit_results:
        fit_x, fit_y = fit_results[dist_name]
        params = fit_params[dist_name]
        
        if dist_name in ['bernoulli_mix', 'discrete']:
            # For discrete distributions, plot as probabilities not densities
            if dist_name == 'discrete':
                # Scale to match histogram scale
                if boundary_heavy:
                    plt.bar(fit_x, fit_y * len(parameter_data), alpha=0.7, color=colors[i], width=0.02,
                           label=f"{dist_name} (AIC={aic_score:.1f})")
                else:
                    plt.bar(fit_x, fit_y, alpha=0.7, color=colors[i], width=0.02,
                           label=f"{dist_name} (AIC={aic_score:.1f})")
            else:
                # Bernoulli mixture - plot point masses
                point_masses = fit_x[:2]
                point_probs = fit_y[:2]
                
                if boundary_heavy:
                    # Scale to match count histogram
                    plt.bar(point_masses, point_probs * len(parameter_data), alpha=0.7, color=colors[i], width=0.02,
                           label=f"{dist_name} (AIC={aic_score:.1f})\nP(0)={params[0]:.2f}, P(1)={params[1]:.2f}")
                else:
                    plt.bar(point_masses, point_probs, alpha=0.7, color=colors[i], width=0.02,
                           label=f"{dist_name} (AIC={aic_score:.1f})\nP(0)={params[0]:.2f}, P(1)={params[1]:.2f}")
                
                # Add continuous part if exists
                if len(fit_x) > 2:
                    scale_factor = len(parameter_data) if boundary_heavy else 1
                    plt.plot(fit_x[2:], fit_y[2:] * scale_factor, color=colors[i], linewidth=2, alpha=0.7)
        else:
            # Continuous distributions
            scale_factor = len(parameter_data) if boundary_heavy else 1
            max_y = np.max(fit_y) * scale_factor
            
            # Skip if the PDF is too extreme (indicates poor fit)
            if max_y > 1000 and boundary_heavy:
                print(f"Skipping {dist_name} - extreme PDF values indicate poor fit for boundary data")
                continue
                
            plt.plot(fit_x, fit_y * scale_factor, color=colors[i], linewidth=2,
                    label=f"{dist_name} (AIC={aic_score:.1f})\n"
                          f"Params: {np.round(params[:2], 3)}")

# Set appropriate y-axis limits
if boundary_heavy:
    plt.ylabel('Count')
    plt.ylim(0, len(parameter_data) * 1.1)
else:
    plt.ylabel('Probability Density')
    plt.ylim(0, max(5, np.max(counts) * 1.2))  # Reasonable upper limit

plt.title(f"Distribution Fitting for {parameter.capitalize()} Parameter")
plt.xlabel(f"{parameter.capitalize()} Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Set appropriate x-axis limits
if parameter == 'theta':
    plt.xlim(-1.1, 1.1)
else:
    plt.xlim(-0.05, 1.05)

plt.tight_layout()

# Save plot
output_filename = f"{parameter}_distribution_fit_corrected.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_filename}")
plt.show()

# Print results
print("\nAIC Scores (lower = better):")
for dist_name, score in sorted_aic:
    print(f"{dist_name:12}: {score:.2f}")

print(f"\nBest fit: {best_fit_name}")
print(f"Parameters: {np.round(best_params, 5)}")

# Generate sampling code for the best distribution
print(f"\nCode to sample from best fit ({best_fit_name}):")
if best_fit_name == 'bernoulli_mix':
    p_0, p_1, p_mid = best_params
    print(f"# Mixture: P(0)={p_0:.3f}, P(1)={p_1:.3f}, P(uniform)={p_mid:.3f}")
    print(f"rand = np.random.random(N)")
    print(f"samples = np.where(rand < {p_0:.3f}, 0, np.where(rand < {p_0+p_1:.3f}, 1, np.random.uniform(0.05, 0.95, N)))")
elif best_fit_name == 'discrete':
    vals_probs = best_params
    vals = [vp[0] for vp in vals_probs]
    probs = [vp[1] for vp in vals_probs]
    print(f"values = {vals}")
    print(f"probs = {probs}")
    print(f"samples = np.random.choice(values, size=N, p=probs)")
elif best_fit_name == 'beta_bimodal':
    a, b = best_params[:2]
    print(f"samples = np.random.beta({a:.3f}, {b:.3f}, size=N)")
elif best_fit_name == 'lognorm_scaled':
    s, scale = best_params[:2]
    print(f"samples = np.random.lognormal(mean=0, sigma={s:.3f}, size=N) * {scale:.3f}")
elif best_fit_name == 'weibull_scaled':
    c, scale = best_params[:2]
    print(f"samples = np.random.weibull({c:.3f}, size=N) * {scale:.3f}")
elif best_fit_name == 'laplace':
    loc, scale = best_params[:2]
    print(f"samples = np.random.laplace(loc={loc:.3f}, scale={scale:.3f}, size=N)")
elif best_fit_name == 't':
    df, loc, scale = best_params[:3]
    print(f"samples = stats.t.rvs(df={df:.3f}, loc={loc:.3f}, scale={scale:.3f}, size=N)")
elif best_fit_name == 'beta_constrained':
    a, b = best_params[:2]
    print(f"samples = np.random.beta({a:.3f}, {b:.3f}, size=N)")
elif best_fit_name == 'gamma_scaled':
    shape, scale = best_params[:2]
    print(f"samples = np.random.gamma({shape:.3f}, scale={scale:.3f}, size=N) / 10")
elif best_fit_name == 'skewnorm':
    a_skew, loc, scale = best_params[:3]
    print(f"from scipy.stats import skewnorm")
    print(f"samples = skewnorm.rvs({a_skew:.3f}, loc={loc:.3f}, scale={scale:.3f}, size=N)")
elif best_fit_name == 'beta':
    a, b = best_params[:2]
    print(f"samples = np.random.beta({a:.3f}, {b:.3f}, size=N)")
elif best_fit_name == 'beta_scaled':
    a, b = best_params[:2]
    print(f"samples = 2 * np.random.beta({a:.3f}, {b:.3f}, size=N) - 1")
elif best_fit_name == 'uniform':
    if parameter == 'theta':
        print(f"samples = np.random.uniform(-1, 1, size=N)")
    else:
        loc, scale = best_params[:2]
        print(f"samples = np.random.uniform({loc:.3f}, {loc+scale:.3f}, size=N)")
elif best_fit_name == 'truncnorm':
    a, b, loc, scale = best_params[:4]
    print(f"samples = stats.truncnorm.rvs({a:.3f}, {b:.3f}, loc={loc:.3f}, scale={scale:.3f}, size=N)")

print(f"\nData Summary for {parameter}:")
print(f"Sample size: {len(parameter_data)}")
print(f"Mean: {np.mean(parameter_data):.4f}")
print(f"Std: {np.std(parameter_data):.4f}")
print(f"Min: {np.min(parameter_data):.4f}")
print(f"Max: {np.max(parameter_data):.4f}")