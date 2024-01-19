# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load Iris dataset from scikit-learn
iris = load_iris()
data = np.c_[iris.data, iris.target]
columns = np.append(iris.feature_names, ["target"])
iris_df = pd.DataFrame(data, columns=columns)

# Split the dataset into features (X) and target variable (y)
X = iris_df.drop("target", axis=1)
y = iris_df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save the datasets locally
data_folder = "data"
train_file = "iris_train.csv"
test_file = "iris_test.csv"

# Save training set
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv(f"{data_folder}/{train_file}", index=False)

# Save testing set without the target feature
test_data = X_test
test_data.to_csv(f"{data_folder}/{test_file}", index=False)

print(f"Training set saved to {data_folder}/{train_file}")
print(f"Testing set saved to {data_folder}/{test_file}")
