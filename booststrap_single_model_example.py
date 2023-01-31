import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest Regression model on training data
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Define the number of bootstrap samples
n_samples = 1000

# Initialize an empty array to store the results
results = np.zeros(n_samples)

# Run bootstrap analysis
for i in range(n_samples):
    # Generate bootstrapped sample
    bootstrapped_indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_bootstrapped = X_train[bootstrapped_indices]
    y_bootstrapped = y_train[bootstrapped_indices]
    
    # Train the model on the bootstrapped sample
    rf.fit(X_bootstrapped, y_bootstrapped)
    
    # Evaluate the model on the test data
    results[i] = rf.score(X_test, y_test)

# Print the mean and standard deviation of the results
print("Mean score:", np.mean(results))
print("Standard deviation:", np.std(results))
