# Machine Learning Exercise

This repository contains an attempt of a well-organized machine learning (MLE) project, focusing on achieving compatibility with different machine environments. By utilizing Docker containers, the project is able to run on any operating systems and CPU architectures.

## Overview of Functionality

1. **Data Generation:**
   - Generate the data that the model will be trained and tested on, for simplicity sake, the Iris dataset will be used.

2. **Model Training:**
   - The next phase involves the training process for the model with the `train.py` script, the `iris_train.csv` file wil be used.

3. **Inference on New Data:**
   - The trained model is then used to make predictions on the unseen data from `iris_test.csv`.

## Usage

1. Set the `CONF_PATH` environment variable to "settings.json".
   ```bash
   export CONF_PATH="settings.json"
   ```
2. Run the bash script from the parent folder (use sudo if necessary).
   ```bash
   bash script.sh
   ```
3. The generated model should be in the `/models` folder and the infered results on unseen data in the `/results` folder.

## Considerations

It's worth noting that installing TensorFlow within the Docker container is a painful requirement, as it is a substantially big package (450 MB). Patience or a high download bandwidth may be required.

## Dependencies

Obviously, Docker will need to be installed, that's all.

## Atributions

Special thanks to the original developer of the code. Any remaining mistakes, as is often inevitable, are my sole responsibility.