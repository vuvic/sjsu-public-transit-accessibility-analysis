import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

income_df = pd.read_csv('data/BG_Median_Household_Income.csv', dtype=str)

access_df = pd.read_csv('data/BG_Accessibility.csv',  dtype=str)
access_df = access_df[['Name', 'OriginID', 'Total_PublicTransitTime', 'Shape_Length']]
access_df['Name'] = access_df['Name'].str.split(' - ').str[0]

population_df = pd.read_csv('data/BG_Population.csv',  dtype=str)
population_df = population_df[['Name', 'Population']]
population_df['Name'] = population_df['Name'].str.split(' - ').str[0]

merged_df = pd.merge(access_df, income_df, left_on='Name', right_on='Name', how='inner')
merged_df = pd.merge(merged_df, population_df, left_on='Name', right_on='Name', how='inner')


plot_df = merged_df.dropna(subset=['Median_Household_Income', 'Total_PublicTransitTime']) 
plot_df['Median_Household_Income'] = pd.to_numeric(plot_df['Median_Household_Income'], errors='coerce')
plot_df['Median_Household_Income'] = plot_df['Median_Household_Income'] / 1000
plot_df['Total_PublicTransitTime'] = pd.to_numeric(plot_df['Total_PublicTransitTime'], errors='coerce')
plot_df['Population'] = pd.to_numeric(plot_df['Total_PublicTransitTime'], errors='coerce')

clean = plot_df[['Median_Household_Income','Total_PublicTransitTime', 'Population']].replace([np.inf, -np.inf], np.nan).dropna()


x = clean['Median_Household_Income'].values
y = clean['Total_PublicTransitTime'].values
w = clean['Population'].values  

X = sm.add_constant(x)

wls_mod = sm.WLS(y, X, weights=w).fit()

print(wls_mod.summary())

import matplotlib.pyplot as plt
m, b = wls_mod.params[1], wls_mod.params[0]

plt.figure(figsize=(8,6))
plt.scatter(x, y, alpha=0.6, label="Block Group")
x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, m*x_line + b,
         label=f"Fit: y = {m:.3f}x + {b:.0f}", linewidth=2)
plt.xlabel("Median Household Income (Thousands of Dollars)")
plt.ylabel("Commute Time (min)")
plt.title("WLS Regression (block‐group weights)")
plt.legend()
plt.show()