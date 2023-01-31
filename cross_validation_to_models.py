import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import scipy.stats as stats


# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Define the regression algorithms to be compared
models = [
    RandomForestRegressor(),
    LinearRegression(),
    KNeighborsRegressor(),
    SVR(),
    DecisionTreeRegressor()
]

# Define the number of cross-validation folds
cv = 5

# Initialize a data frame to store the results
results = pd.DataFrame(columns=["Model", "Mean CV Score", "Standard Deviation"])

# Run cross-validation for each model
for model in models:
    scores = cross_val_score(model, X, y, cv=cv)
    results = results.append({
        "Model": type(model).__name__,
        "Mean CV Score": scores.mean(),
        "Standard Deviation": scores.std(),
        "Scores":scores
    }, ignore_index=True)

# Print the results
print(results)

# Define the null hypothesis that the mean CV scores for each model are equal
null_hypothesis = "Equal Means"

# Define the alternative hypothesis that the mean CV scores for at least one model are not equal
alternative_hypothesis = "Not Equal Means"

# Perform the Kruskal-Wallis test
all_scores = [r for r in results["Scores"] ]
all_models = [r for r in results["Model"] ]
statistic, p_value = stats.kruskal(*all_scores)

# Print the test results
print(f"Null Hypothesis: {null_hypothesis}")
print(f"Alternative Hypothesis: {alternative_hypothesis}")
print(f"Test Statistic: {statistic}")
print(f"p-value: {p_value}")

# Interpret the p-value
if p_value < 0.05:
    print("Reject the null hypothesis.")
    print(f"There is evidence to suggest that the mean CV scores for at least one of the models are not equal.")
else:
    print("Fail to reject the null hypothesis.")
    print(f"There is not enough evidence to suggest that the mean CV scores for any of the models are not equal.")

import statsmodels.stats.multicomp as smm

# Perform post-hoc comparisons with the Conover-Iman test
pairwise_comparisons = smm.pairwise_tukeyhsd(all_scores, all_models)

# Print the post-hoc comparisons
print("Post-hoc comparisons:")
print(pairwise_comparisons)
