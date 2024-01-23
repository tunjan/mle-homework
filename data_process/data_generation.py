import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Constants
CONF_FILE = os.getenv('CONF_PATH', "settings.json")

def load_settings(conf_file):
    """ Load settings from JSON file """
    with open(conf_file, "r") as file:
        return json.load(file)

def load_dataset():
    """ Load Iris dataset from scikit-learn """
    iris = load_iris()
    data = np.c_[iris.data, iris.target]
    columns = np.append(iris.feature_names, ["target"])
    return pd.DataFrame(data, columns=columns)

def split_dataset(iris_df, test_size, random_state):
    """ Split the dataset into training and testing sets """
    X = iris_df.drop("target", axis=1)
    y = iris_df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_dataset(data, file_path):
    """ Save dataset to CSV """
    data.to_csv(file_path, index=False)
    print(f"Dataset saved to {file_path}")

def main():
    settings = load_settings(CONF_FILE)

    iris_df = load_dataset()
    test_size = settings['train']['test_size']
    random_state = settings['general']['random_state']
    X_train, X_test, y_train, y_test = split_dataset(iris_df, test_size, random_state)

    data_folder = settings['general']['data_dir']
    train_file = settings['train']['table_name']
    test_file = settings['inference']['inp_table_name']

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    train_data = pd.concat([X_train, y_train], axis=1)
    save_dataset(train_data, os.path.join(data_folder, train_file))

    test_data = X_test
    save_dataset(test_data, os.path.join(data_folder, test_file))

if __name__ == '__main__':
    main()