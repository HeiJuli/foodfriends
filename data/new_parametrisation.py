# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 11:57:19 2025

@author: emma.thill
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Load your .xlsx file
data = pd.read_excel("C:/Users/emma.thill/Dropbox/Projects/Foodfriends/Data/Collectivism/LISS/alpha_demographics.xlsx")

# Rename columns if needed (optional check)
data.columns = data.columns.str.strip()

# Extract relevant columns
alpha = data['Self-identity weight (alpha)'].dropna()
gender = data['Gender'].dropna()
age = data['Age of the household member'].dropna()
incquart = data['Income Quartile'].dropna()
educlevel = data['Education Level'].dropna()

# Add extracted columns back to dataframe (so you work with one clean DataFrame)
data = pd.DataFrame({
    'alpha': alpha,
    'age': age,
    'incquart': incquart,
    'educlevel': educlevel,
    'gender': gender
})

# Bin age into groups
data['age_group'] = pd.cut(data['age'], bins=[17, 29, 39, 49, 59, 69, 120],
                           labels=['18–29', '30–39', '40–49', '50–59', '60–69', '70+'])

# Drop rows with missing values in any key column
data_clean = data.dropna(subset=['alpha', 'gender', 'age_group', 'incquart', 'educlevel'])

# Compute empirical PMFs
grouped_pmf = (
    data_clean.groupby(['gender', 'age_group', 'incquart', 'educlevel'])['alpha']
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)

# Rename alpha columns with "pmf for ..." labels
grouped_pmf.columns = [f"pmf for {round(val, 2)}" for val in grouped_pmf.columns]

# Save PMFs
grouped_pmf.reset_index().to_csv("alpha_empirical_pmfs_by_group.csv", index=False)
print("✅ PMF table saved as 'alpha_empirical_pmfs_by_group.csv'")

# Reset index for plotting
pmf_long = grouped_pmf.reset_index().melt(
    id_vars=['gender', 'age_group', 'incquart', 'educlevel'],
    var_name='alpha',
    value_name='probability'
)

# Convert alpha values from string to float (if necessary)
pmf_long['alpha'] = pmf_long['alpha'].str.replace("pmf for ", "").astype(float)

# List of demographic categories to plot
categories = ['gender', 'age_group', 'incquart', 'educlevel']

# Loop through each and create/save the plot
for cat in categories:
    g = sns.FacetGrid(pmf_long, col=cat, col_wrap=3, height=4, sharey=False)
    g.map_dataframe(sns.barplot, x='alpha', y='probability', color='steelblue')
    g.set_axis_labels("Alpha", "Probability")
    g.set_titles(f"{cat}: "+"{col_name}")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle(f"Empirical PMF of Alpha by {cat.capitalize()}", fontsize=16)
    plt.savefig(f"alpha_pmf_by_{cat}.png", dpi=300, bbox_inches='tight')
    plt.show()
