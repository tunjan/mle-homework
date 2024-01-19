"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
# Comment this lines if you have problems with MLFlow installation
import mlflow
mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH') 

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")


class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df


class Training:
    """Handles model training and evaluation."""

    def __init__(self) -> None:
        """Initializes the neural network model."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),  # Example hidden layer
            tf.keras.layers.Dense(3, activation='softmax')  # Output layer for 3 classes
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def run_training(self, training_data: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        """Runs the model training and evaluation process."""
        logging.info("Running training...")
        features, target = self.data_split(training_data, test_size=test_size)

        start_time = time.time()
        self.train(features, target)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")

        self.test(features, target)  # Assuming features and target are already split
        self.save(out_path)

    def data_split(self, data: pd.DataFrame, test_size: float = 0.33) -> tuple:
        """Splits data into features and target variables."""
        logging.info("Splitting data into features and target...")
        features = data[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
        target = data.target
        return features, target

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Trains the model on the provided features and target."""
        y_train_cat = tf.keras.utils.to_categorical(target)  # Convert labels to categorical
        self.model.fit(features, y_train_cat, epochs=100)  # Example number of epochs

    def test(self, features: pd.DataFrame, target: pd.DataFrame) -> float:
        """Evaluates the model's performance on the given test data."""
        y_pred = self.model.predict(features)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted classes
        accuracy = accuracy_score(target, y_pred_classes)
        logging.info(f"Accuracy score: {accuracy}")
        return accuracy

    def save(self, path: str) -> None:
        """Saves the trained model to the specified path in native Keras format."""
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.keras')  # Use .keras extension
        else:
            path = os.path.join(MODEL_DIR, path)

        tf.keras.models.save_model(self.model, path, save_format='tf')  # Use save_format='tf' for native Keras format
        logging.info(f"Model saved to {path}")


def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'])


if __name__ == "__main__":
    main()