"""
Script loads the latest trained model, data for inference, and predicts results.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Load configuration settings
CONF_FILE = os.getenv('CONF_PATH', 'settings.json')
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from utils import configure_logging, get_project_dir

# Define paths from settings
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])
# Initialize logging and command line arguments
configure_logging()
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", help="Specify inference data file", default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", help="Specify the path to the output table")

def get_latest_model_path() -> str:
    """Gets the path of the latest saved model (in native Keras format)"""
    latest_model_path = None
    latest_mtime = None
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".keras"):
            model_path = os.path.join(MODEL_DIR, filename)
            model_mtime = os.path.getmtime(model_path)
            if not latest_mtime or model_mtime > latest_mtime:
                latest_model_path = model_path
                latest_mtime = model_mtime

    if latest_model_path is None:
        logging.error("No trained model found.")
        sys.exit(1)

    return latest_model_path

def get_model_by_path(path: str) -> tf.keras.Model:
    """Loads and returns the specified TensorFlow model"""
    try:
        model = tf.keras.models.load_model(path)
        logging.info(f'Model loaded from {path}')
        return model
    except Exception as e:
        logging.error(f'Error loading model: {e}')
        sys.exit(1)

def get_inference_data(path: str) -> pd.DataFrame:
    """Loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        if df.empty or df.isnull().values.any():
            logging.error("Inference data is empty or contains null values.")
            sys.exit(1)
        return df
    except Exception as e:
        logging.error(f"Error loading inference data: {e}")
        sys.exit(1)

def predict_results(model: tf.keras.Model, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict results and join them with the infer_data."""
    features = infer_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    results = model.predict(features)
    infer_data['predicted_class'] = np.argmax(results, axis=1)
    return infer_data

def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    output_path = path or os.path.join(RESULTS_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.csv')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    results.to_csv(output_path, index=False)
    logging.info(f'Results saved to {output_path}')

def main():
    args = parser.parse_args()

    model_path = get_latest_model_path()
    model = get_model_by_path(model_path)
    infer_data = get_inference_data(os.path.join(DATA_DIR, args.infer_file))
    results = predict_results(model, infer_data)
    store_results(results, args.out_path)

    logging.info('Inference completed successfully.')

if __name__ == "__main__":
    main()