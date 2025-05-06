import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

income_df = pd.read_csv('data/BG_Median_Household_Income.csv', dtype=str)

access_df = pd.read_csv('data/BG_Accessibility.csv',  dtype=str)
access_df = access_df[['Name', 'OriginID', 'Total_PublicTransitTime', 'Shape_Length']]
access_df['Name'] = access_df['Name'].str.split(' - ').str[0]

merged_df = pd.merge(access_df, income_df, left_on='Name', right_on='name', how='inner')
merged_df = merged_df.drop(columns=['name']) 

plot_df = merged_df.dropna(subset=['median_household_income', 'Total_PublicTransitTime']) 
plot_df['median_household_income'] = pd.to_numeric(plot_df['median_household_income'], errors='coerce')
plot_df['median_household_income'] = plot_df['median_household_income'] / 1000
plot_df['Total_PublicTransitTime'] = pd.to_numeric(plot_df['Total_PublicTransitTime'], errors='coerce')

clean = plot_df[['median_household_income','Total_PublicTransitTime']].replace([np.inf, -np.inf], np.nan).dropna()
x_clean = clean['median_household_income']
y_clean = clean['Total_PublicTransitTime']

print("N samples:", len(x_clean))

alpha = 0.05
r, p = pearsonr(x_clean, y_clean)
print(f"Pearson r = {r:.3f}, p-value = {p:.3e}")

if p < alpha:
    r2 = r**2
    r2_percentage = r2 * 100
    print(
        f"There is a statistically significant correlation between the median household income of a block group "
        f"and its travel time to San Jose State University via public transportation. "
        f"{r2_percentage:.3f}% of the variance in travel time can be explained by median household income."
    )
else:
    print("There is no statistically significant correlation between the median household income of a block group.")

x = np.array(x_clean)
y = np.array(y_clean)

# 1) Compute slope (m) and intercept (b) of the best‐fit line y = m·x + b
m, b = np.polyfit(x, y, 1)

# 2) Plot scatter + regression line
plt.figure(figsize=(8,6))
plt.scatter(x, y, alpha=0.6, label="Data points")
# Create a smooth x‐axis for the line
x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, m*x_line + b, label=f"Fit: y = {m:.3f}x + {b:.0f}", linewidth=2)

plt.xlabel("Median Household Income (Thousands of Dollars)")
plt.ylabel("Commute Time (min)")
plt.title("Linear Regression (NumPy polyfit)")
plt.legend()
plt.show()
