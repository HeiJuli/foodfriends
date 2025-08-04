import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# Load data
data = pd.read_excel("C:/Users/emma.thill/Dropbox/Projects/Foodfriends/Data/Collectivism/LISS/alpha_demographics.xlsx")
alpha = data['Self-identity weight (alpha)'].dropna()

# Candidate distributions
distributions = {
    'poisson': stats.poisson,
    'nbinom': stats.nbinom,
    'geom': stats.geom,
    'norm': stats.norm,
    'gamma': stats.gamma,
    'lognorm': stats.lognorm,
    'weibull_min': stats.weibull_min,
    'expon': stats.expon
}

aic_scores = {}
fit_results = {}
fit_params = {}
x = np.linspace(min(alpha), max(alpha), 1000)
x_discrete = np.arange(int(min(alpha)), int(max(alpha)) + 1)

# Fit each distribution
for name, dist in distributions.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if name == 'poisson':
                mu = np.mean(alpha)
                y = dist.pmf(x_discrete, mu)
                loglik = np.sum(dist.logpmf(alpha, mu))
                k = 1
                fit_results[name] = (x_discrete, y)
                fit_params[name] = [mu]
            elif name == 'nbinom':
                mean = np.mean(alpha)
                var = np.var(alpha)
                if var <= mean:
                    continue
                p = mean / var
                n = mean * p / (1 - p)
                y = dist.pmf(x_discrete, n, p)
                loglik = np.sum(dist.logpmf(alpha, n, p))
                k = 2
                fit_results[name] = (x_discrete, y)
                fit_params[name] = [n, p]
            elif name == 'geom':
                p = 1 / (1 + np.mean(alpha))
                y = dist.pmf(x_discrete, p)
                loglik = np.sum(dist.logpmf(alpha, p))
                k = 1
                fit_results[name] = (x_discrete, y)
                fit_params[name] = [p]
            else:
                params = dist.fit(alpha)
                y = dist.pdf(x, *params)
                loglik = np.sum(dist.logpdf(alpha, *params))
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
plt.figure(figsize=(10, 6))
plt.hist(alpha, bins=30, density=True, alpha=0.5, color='skyblue', edgecolor='black', label='Data histogram')
plt.plot(best_x, best_y, color='red', linewidth=2,
         label=f"{best_fit_name} (AIC={aic_scores[best_fit_name]:.1f})\nParams: {np.round(best_params, 3)}")
plt.title("Histogram of Data with Best Fitting Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/emma.thill/Dropbox/Projects/Foodfriends/Data/Reasons to Eat Less Meat/theta_dist_python.png", dpi=300, bbox_inches='tight')
plt.show()

# Print AIC scores
print("\n📊 AIC Scores (lower = better):")
for dist_name, score in sorted_aic:
    print(f"{dist_name:12}: AIC = {score:.2f}")

# Print best distribution and its parameters
print(f"\n✅ Best fit: {best_fit_name}")
print(f"🔧 Parameters: {np.round(best_params, 5)}")