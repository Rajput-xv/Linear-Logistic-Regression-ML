import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Fetch the Boston housing dataset from its original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(1, len(data[0])+1)])
df['target'] = target

# Inspect the first few rows
print(df.head())

# Prepare the data for modeling
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values  # Target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression object
lin_reg = LinearRegression()

# Train the model using the training sets
lin_reg.fit(X_train, y_train)

# Make predictions using the testing set
predictions = lin_reg.predict(X_test)

# Print the first few predictions
print("First 5 Predictions:", predictions[:5])

# Evaluate the model's performance
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")




'''
# # Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load Iris dataset for demonstration purposes
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression object
log_reg = LogisticRegression(max_iter=1000)

# Train the model using the training sets
log_reg.fit(X_train, y_train)

# Make predictions using the testing set
predictions = log_reg.predict(X_test)

# Print the predictions
print("Predictions:", predictions[:20])  # Display the first 10 predictions

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy Score: {accuracy:.2f}")
'''
