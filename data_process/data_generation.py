# Import necessary libraries
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
CONF_FILE = os.getenv('CONF_PATH', "settings.json")

# Load settings from JSON file
with open(CONF_FILE, "r") as file:
    settings = json.load(file)

# Load Iris dataset from scikit-learn
iris = load_iris()
data = np.c_[iris.data, iris.target]
columns = np.append(iris.feature_names, ["target"])
iris_df = pd.DataFrame(data, columns=columns)

# Split the dataset into features (X) and target variable (y)
X = iris_df.drop("target", axis=1)
y = iris_df["target"]

# Split the data into training and testing sets using settings
test_size = settings['train']['test_size']
random_state = settings['general']['random_state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Define file paths using settings
data_folder = settings['general']['data_dir']
train_file = settings['train']['table_name']
test_file = settings['inference']['inp_table_name']

# Create the data directory if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Save training set
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv(os.path.join(data_folder, train_file), index=False)

# Save testing set without the target feature
test_data = X_test
test_data.to_csv(os.path.join(data_folder, test_file), index=False)

print(f"Training set saved to {os.path.join(data_folder, train_file)}")
print(f"Testing set saved to {os.path.join(data_folder, test_file)}")