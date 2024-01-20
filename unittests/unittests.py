import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from training.train import DataProcessor, Training

class MockConfig:
    """Mock configuration class."""
    general = {'random_state': 42}

class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_processor = DataProcessor()

    @patch('pandas.read_csv')
    def test_data_extraction(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})
        result = self.data_processor.data_extraction('data/iris_train.csv')  # Update path
        mock_read_csv.assert_called_once_with('data/iris_train.csv')  # Update path
        self.assertIsInstance(result, pd.DataFrame)

    def test_data_rand_sampling_no_sampling(self):
        df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})
        result = self.data_processor.data_rand_sampling(df, None)
        self.assertEqual(result.shape, df.shape)

    @patch('pandas.DataFrame.sample')
    def test_data_rand_sampling_with_sampling(self, mock_sample):
        df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})
        max_rows = 2
        mock_sample.return_value = pd.DataFrame({'column1': [1, 2], 'column2': ['a', 'b']})
        result = self.data_processor.data_rand_sampling(df, max_rows)
        mock_sample.assert_called_once_with(n=max_rows, replace=False, random_state=MockConfig.general['random_state'])
        self.assertIsInstance(result, pd.DataFrame)

class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = Training()

    def test_data_split(self):
        data = pd.DataFrame({'sepal length (cm)': [1, 2, 3], 'sepal width (cm)': [4, 5, 6],
                             'petal length (cm)': [7, 8, 9], 'petal width (cm)': [10, 11, 12],
                             'target': [0, 1, 0]})
        features, target = self.training.data_split(data)
        expected_features = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
        pd.testing.assert_frame_equal(features, expected_features)
        pd.testing.assert_series_equal(target, data['target'])

    @patch.object(Training, 'train', return_value=None)
    def test_run_training(self, mock_train):
        mock_data = pd.DataFrame({'sepal length (cm)': [1, 2, 3], 'sepal width (cm)': [4, 5, 6],
                                  'petal length (cm)': [7, 8, 9], 'petal width (cm)': [10, 11, 12],
                                  'target': [0, 1, 0]})
        with patch.object(Training, 'data_split', return_value=(mock_data[['sepal length (cm)', 'sepal width (cm)',
                                                                         'petal length (cm)', 'petal width (cm)']],
                                                                mock_data['target'])):
            self.training.run_training(mock_data)
        mock_train.assert_called_once()

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()
