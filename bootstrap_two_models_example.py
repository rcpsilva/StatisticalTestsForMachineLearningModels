import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train two different regression models on the training data
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

lr = LinearRegression()
lr.fit(X_train, y_train)

# Define the number of bootstrap samples
n_samples = 100

# Initialize two empty arrays to store the results of each model
results_rf = np.zeros(n_samples)
results_lr = np.zeros(n_samples)

# Run bootstrap analysis for both models
for i in range(n_samples):

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train two different regression models on the training data
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Evaluate the Random Forest model on the test data
    results_rf[i] = rf.score(X_test, y_test)
    
    # Evaluate the Linear Regression model on the test data
    results_lr[i] = lr.score(X_test, y_test)

# Perform a t-test to compare the averages of the results
t, p = stats.ttest_ind(results_rf, results_lr)

# Print the t-statistic and p-value
print("t-statistic:", t)
print("p-value:", p)
