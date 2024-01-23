import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import tensorflow as tf
import os
import logging
from training.train import Training  # Update with the correct import path

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.training = Training()

    def test_init(self):
        self.assertIsInstance(self.training.model, tf.keras.Sequential)

    @patch('training.train.train_test_split')
    @patch('training.train.time.time')
    def test_run_training(self, mock_time, mock_train_test_split):
        mock_time.side_effect = [100, 200] 
        mock_train_test_split.return_value = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
        mock_data = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9],
            'sepal width (cm)': [3.5, 3.0],
            'petal length (cm)': [1.4, 1.4],
            'petal width (cm)': [0.2, 0.2],
            'target': [0, 1]
        })

        with patch.object(self.training, 'train') as mock_train, \
             patch.object(self.training, 'test') as mock_test, \
             patch.object(self.training, 'save') as mock_save, \
             patch('training.train.logging.info') as mock_log:
            self.training.run_training(mock_data)
            mock_train.assert_called()
            mock_test.assert_called()
            mock_save.assert_called()
            mock_log.assert_called_with("Training completed in 100 seconds.")

    def test_data_split(self):
        mock_data = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9],
            'sepal width (cm)': [3.5, 3.0],
            'petal length (cm)': [1.4, 1.4],
            'petal width (cm)': [0.2, 0.2],
            'target': [0, 1]
        })
        features, target = self.training.data_split(mock_data)
        self.assertEqual(features.shape[1], 4)
        self.assertEqual(len(target), 2)

    @patch('training.train.tf.keras.Sequential.fit')
    def test_train(self, mock_fit):
        mock_features = pd.DataFrame({'feature1': [0.1, 0.2], 'feature2': [0.3, 0.4]})
        mock_target = pd.DataFrame({'target': [1, 0]})
        with patch('training.train.logging.info') as mock_log:
            self.training.train(mock_features, mock_target)
            mock_fit.assert_called()
            mock_log.assert_called_with("Training completed.")

    @patch('training.train.tf.keras.Sequential.evaluate')
    def test_test(self, mock_evaluate):
        mock_evaluate.return_value = (0.5, 0.8)  # Loss, Accuracy
        mock_features = pd.DataFrame({'feature1': [0.1, 0.2], 'feature2': [0.3, 0.4]})
        mock_target = pd.DataFrame({'target': [1, 0]})
        accuracy = self.training.test(mock_features, mock_target)
        self.assertEqual(accuracy, 0.8)

if __name__ == '__main__':
    unittest.main()