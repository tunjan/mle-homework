"""
This script prepares the data, runs the training, and saves the model.
"""

import os
import sys
import argparse
import logging
import json
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Comment this lines if you have problems with MLFlow installation
import mlflow
mlflow.autolog()

# Adds the root directory to system path
# Setting up root and model directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Configuration file handling
CONF_FILE = os.getenv('CONF_PATH', "settings.json")
if not os.path.exists(CONF_FILE):
    raise FileNotFoundError(f"Configuration file {CONF_FILE} not found.")

# Load configuration
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from utils import get_project_dir, configure_logging

# Define paths from configuration
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", help="Specify inference data file", default=conf['train']['table_name'])
parser.add_argument("--model_path", help="Specify the path for the output model")

class DataProcessor:
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        try:
            return pd.read_csv(path)
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            sys.exit(1)

    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows <= 0:
            logging.info('Max_rows not defined or invalid. Skipping sampling.')
            return df

        if len(df) < max_rows:
            logging.warning('Size of dataframe is less than max_rows. Skipping sampling.')
            return df

        df_sampled = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
        logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df_sampled

class Training:
    def __init__(self) -> None:
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def run_training(self, training_data: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        logging.info("Running training...")
        features, target = self.data_split(training_data)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size)

        start_time = time.time()
        self.train(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")

        self.test(X_test, y_test)
        self.save(out_path)

    def data_split(self, data: pd.DataFrame) -> tuple:
        features = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
        target = data['target']
        return features, target

    def train(self, features: pd.DataFrame, target: pd.DataFrame, epochs: int = 20, batch_size: int = 10, validation_split: float = 0.3) -> None:
        """
        Trains the model on the provided features and target.
        """
        logging.info("Starting model training...")
        y_train_cat = tf.keras.utils.to_categorical(target)
        history = self.model.fit(features, y_train_cat, epochs=epochs, validation_split=validation_split, batch_size=batch_size)

        # Correcting the logging of training process details
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {history.history['loss'][epoch]}, Accuracy: {history.history['accuracy'][epoch]}, Validation Loss: {history.history['val_loss'][epoch]}, Validation Accuracy: {history.history['val_accuracy'][epoch]}")
        
        logging.info("Training completed.")
        
    def test(self, features: pd.DataFrame, target: pd.DataFrame) -> float:
        y_test_cat = tf.keras.utils.to_categorical(target)
        loss, accuracy = self.model.evaluate(features, y_test_cat)
        logging.info(f"Test accuracy: {accuracy}")
        return accuracy

    def save(self, path: str = None) -> None:
        logging.info("Saving the model...")
        if not path:
            path = os.path.join(MODEL_DIR, 'tensorflow_model.keras')

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        self.model.save(path, save_format='tf')
        logging.info(f"Model saved to {path}")

def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'])

if __name__ == "__main__":
    main()