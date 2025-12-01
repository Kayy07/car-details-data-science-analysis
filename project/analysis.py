import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.stats.anova import anova_lm

# Read File
car_details = pd.read_excel("Car_Details.xlsx")
type(car_details)

# Descriptive Statistics on height data
print("Height Data")
car_data = car_details["Height"].describe()
print(car_data)

# Quick data check
print("\nData Head")
print(car_details.head())

# Histogram on car mileage
kilometers = car_details['Kilometer']
plt.hist(kilometers, bins=[0, 20000, 40000, 60000, 80000, 100000], edgecolor='black')
plt.title("Car Kilometer Frequencies")
plt.xlabel("Kilometers")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.3)
plt.show()

# Scatterplot on length vs width
plt.figure(figsize=(8, 6))
plt.scatter(car_details['Length'], car_details['Fuel Tank Capacity'])
plt.title("Length vs Fuel Tank Capacity")
plt.xlabel("Length")
plt.ylabel("Fuel Tank Capacity")
plt.grid(alpha=0.5)
plt.show()

# Multilinear Regression Model
car_details.columns = car_details.columns.str.strip()

# Fit the initial model
MLR = sm.ols(formula='Price ~ Kilometer + Length', data=car_details).fit()
print("\nInitial Multilinear Regression Summary:")
print(MLR.summary())

# Filter significant predictors
significant_predictors = [var for var, pval in MLR.pvalues.items() if pval < 0.05 and var != 'Intercept']

if significant_predictors:
    # Refitting model
    final_formula = 'Price ~ ' + ' + '.join(significant_predictors)
    final_MLR = sm.ols(formula=final_formula, data=car_details).fit()
    print("\nFinal Multilinear Regression Summary (Significant Predictors Only):")
    print(final_MLR.summary())
else:
    print("No statistically significant predictors at the 0.05 level.")

# ANOVA
print("\n[ANOVA Table Below]")
print(anova_lm(MLR))