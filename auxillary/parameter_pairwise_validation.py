import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_survey_data():
    """Load cleaned survey data with overlapping IDs"""
    
    # Load individual surveys (adjust paths as needed)
    alpha_data = pd.read_excel("../data/alpha_demographics.xlsx")
    rho_data = pd.read_excel("../data/rho_demographics.xlsx") 
    theta_data = pd.read_excel("../data/theta_diet_demographics.xlsx")
    
    # Clean and extract key columns
    alpha_clean = alpha_data[['id', 'Self-identity weight (alpha)']].copy()
    alpha_clean.columns = ['id', 'alpha']
    alpha_clean['alpha'] = pd.to_numeric(alpha_clean['alpha'], errors='coerce')
    alpha_clean = alpha_clean.dropna()
    
    rho_clean = rho_data[['id', 'Cost parameter (rho)']].copy() 
    rho_clean.columns = ['id', 'rho']
    rho_clean['rho'] = pd.to_numeric(rho_clean['rho'], errors='coerce')
    rho_clean = rho_clean.dropna()
    
    theta_clean = theta_data[['id', 'Personal Preference for Veg Diet']].copy()
    theta_clean.columns = ['id', 'theta'] 
    theta_clean['theta'] = pd.to_numeric(theta_clean['theta'], errors='coerce')
    theta_clean = theta_clean.dropna()
    
    return alpha_clean, rho_clean, theta_clean

def analyze_pairwise_correlations(alpha_df, rho_df, theta_df):
    """Analyze all pairwise parameter correlations"""
    
    print("=== Pairwise Parameter Correlations ===")
    
    # Alpha-Theta
    at_merged = pd.merge(alpha_df, theta_df, on='id')
    at_r, at_p = pearsonr(at_merged['alpha'], at_merged['theta'])
    at_rho, _ = spearmanr(at_merged['alpha'], at_merged['theta'])
    print(f"Alpha-Theta (n={len(at_merged)}):")
    print(f"  Pearson: r={at_r:.3f}, p={at_p:.4f}")
    print(f"  Spearman: rho={at_rho:.3f}")
    
    # Alpha-Rho  
    ar_merged = pd.merge(alpha_df, rho_df, on='id')
    ar_r, ar_p = pearsonr(ar_merged['alpha'], ar_merged['rho'])
    ar_rho, _ = spearmanr(ar_merged['alpha'], ar_merged['rho'])
    print(f"Alpha-Rho (n={len(ar_merged)}):")
    print(f"  Pearson: r={ar_r:.3f}, p={ar_p:.4f}")
    print(f"  Spearman: rho={ar_rho:.3f}")
    
    # Rho-Theta
    rt_merged = pd.merge(rho_df, theta_df, on='id') 
    rt_r, rt_p = pearsonr(rt_merged['rho'], rt_merged['theta'])
    rt_rho, _ = spearmanr(rt_merged['rho'], rt_merged['theta'])
    print(f"Rho-Theta (n={len(rt_merged)}):")
    print(f"  Pearson: r={rt_r:.3f}, p={rt_p:.4f}")
    print(f"  Spearman: rho={rt_rho:.3f}")
    
    return at_merged, ar_merged, rt_merged

def analyze_three_way_relationships(alpha_df, rho_df, theta_df):
    """Analyze three-way parameter relationships"""
    
    print("\n=== Three-Way Parameter Analysis ===")
    
    # Merge all three
    merged = pd.merge(pd.merge(alpha_df, rho_df, on='id'), theta_df, on='id')
    print(f"Three-way overlap: n={len(merged)}")
    
    if len(merged) < 50:
        print("Insufficient three-way overlap for robust analysis")
        return None
    
    # Correlation matrix
    corr_matrix = merged[['alpha', 'rho', 'theta']].corr()
    print("\nCorrelation matrix:")
    print(corr_matrix.round(3))
    
    # Test theta + rho -> alpha prediction
    X = merged[['theta', 'rho']]
    y = merged['alpha']
    
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    print(f"\nLinear model: alpha ~ theta + rho")
    print(f"  R-squared: {r2:.4f}")
    print(f"  Coefficients: theta={model.coef_[0]:.3f}, rho={model.coef_[1]:.3f}")
    print(f"  Intercept: {model.intercept_:.3f}")
    
    # Test individual predictors
    theta_r2 = r2_score(y, LinearRegression().fit(merged[['theta']], y).predict(merged[['theta']]))
    rho_r2 = r2_score(y, LinearRegression().fit(merged[['rho']], y).predict(merged[['rho']]))
    
    print(f"  Theta alone R²: {theta_r2:.4f}")  
    print(f"  Rho alone R²: {rho_r2:.4f}")
    print(f"  Combined improvement: {r2 - max(theta_r2, rho_r2):.4f}")
    
    return merged, model, r2

def create_correlation_plots(at_data, ar_data, rt_data, three_way_data=None):
    """Create scatter plots of parameter relationships"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Alpha-Theta
    axes[0,0].scatter(at_data['theta'], at_data['alpha'], alpha=0.6, s=20)
    axes[0,0].set_xlabel('Theta (Veg Preference)')
    axes[0,0].set_ylabel('Alpha (Individualism)')
    axes[0,0].set_title(f'Alpha vs Theta (r={pearsonr(at_data.alpha, at_data.theta)[0]:.3f})')
    
    # Alpha-Rho
    axes[0,1].scatter(ar_data['rho'], ar_data['alpha'], alpha=0.6, s=20)
    axes[0,1].set_xlabel('Rho (Behavioral Intention)')
    axes[0,1].set_ylabel('Alpha (Individualism)')
    axes[0,1].set_title(f'Alpha vs Rho (r={pearsonr(ar_data.alpha, ar_data.rho)[0]:.3f})')
    
    # Rho-Theta
    axes[1,0].scatter(rt_data['theta'], rt_data['rho'], alpha=0.6, s=20)
    axes[1,0].set_xlabel('Theta (Veg Preference)')  
    axes[1,0].set_ylabel('Rho (Behavioral Intention)')
    axes[1,0].set_title(f'Rho vs Theta (r={pearsonr(rt_data.rho, rt_data.theta)[0]:.3f})')
    
    # Three-way if available
    if three_way_data is not None and len(three_way_data) > 50:
        X = three_way_data[['theta', 'rho']]
        y = three_way_data['alpha']
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        
        axes[1,1].scatter(y, y_pred, alpha=0.6, s=20)
        axes[1,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', alpha=0.8)
        axes[1,1].set_xlabel('Actual Alpha')
        axes[1,1].set_ylabel('Predicted Alpha (theta+rho)')
        axes[1,1].set_title(f'Alpha Prediction (R²={r2_score(y, y_pred):.3f})')
    else:
        axes[1,1].text(0.5, 0.5, 'Insufficient\nthree-way data', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Alpha Prediction Model')
    
    plt.tight_layout()
    plt.show()

def compare_prediction_approaches(three_way_data):
    """Compare different approaches for predicting alpha"""
    
    if three_way_data is None or len(three_way_data) < 50:
        print("Insufficient data for prediction comparison")
        return
        
    print("\n=== Alpha Prediction Approach Comparison ===")
    
    # Split data for validation
    n = len(three_way_data)
    train_idx = np.random.choice(n, int(0.7*n), replace=False)
    test_idx = np.setdiff1d(range(n), train_idx)
    
    train_data = three_way_data.iloc[train_idx]
    test_data = three_way_data.iloc[test_idx]
    
    approaches = {}
    
    # 1. Individual parameter prediction
    model_combined = LinearRegression().fit(train_data[['theta', 'rho']], train_data['alpha'])
    pred_combined = model_combined.predict(test_data[['theta', 'rho']])
    approaches['theta+rho'] = r2_score(test_data['alpha'], pred_combined)
    
    # 2. Theta only
    model_theta = LinearRegression().fit(train_data[['theta']], train_data['alpha'])
    pred_theta = model_theta.predict(test_data[['theta']])
    approaches['theta_only'] = r2_score(test_data['alpha'], pred_theta)
    
    # 3. Rho only  
    model_rho = LinearRegression().fit(train_data[['rho']], train_data['alpha'])
    pred_rho = model_rho.predict(test_data[['rho']])
    approaches['rho_only'] = r2_score(test_data['alpha'], pred_rho)
    
    # 4. Population mean (baseline)
    mean_pred = np.full_like(test_data['alpha'], train_data['alpha'].mean())
    approaches['population_mean'] = r2_score(test_data['alpha'], mean_pred)
    
    print("Out-of-sample R² performance:")
    for name, r2 in sorted(approaches.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {r2:.4f}")
    
    return approaches

if __name__ == "__main__":
    # Load data
    alpha_df, rho_df, theta_df = load_survey_data()
    
    # Analyze correlations
    at_data, ar_data, rt_data = analyze_pairwise_correlations(alpha_df, rho_df, theta_df)
    
    # Three-way analysis
    three_way_data, model, r2 = analyze_three_way_relationships(alpha_df, rho_df, theta_df)
    
    # Create plots
    create_correlation_plots(at_data, ar_data, rt_data, three_way_data)
    
    # Compare prediction approaches
    compare_prediction_approaches(three_way_data)