# Machine Learning Exercise

This repository contains an attempt of a well-organized machine learning (MLE) project, focusing on achieving compatibility with different machine environments.

## Overview of Functionality

1. **Data Generation:** Generate the data that the model will be trained and tested on. For simplicity's sake, the Iris dataset will be used.

2. **Model Training:** The model is trained using the `train.py` script, which uses the `iris_train.csv` dataset.

3. **Inference on Unseen Data:** The model is used to predict outcomes using unseen data from the `iris_test.csv` file. It ouputs the results to a `.csv` file.

## Usage

### Docker

1. Clone the repository
    ```bash
   git clone https://github.com/tunjan/mle-homework.git
   ```

2. (optional) Open VSCode, go to the File menu, and select Add Folder to Workspace.
3. (optional) Navigate to the cloned repository's directory and add it.
   
4. Set the `CONF_PATH` environment variable to "settings.json".
   ```bash
   export CONF_PATH="settings.json"
   ```
5. Run the bash script from the parent folder (use sudo if necessary).
   ```bash
   bash script.sh
   ```
6. The generated model should be in the `/models` folder and the infered results on unseen data in the `/results` folder.

**NOTE**: The bash script might require adjustments for non-Linux operating systems. All necessary commands are included in the script, and the Docker image can be run using them. Thus executing them manually is enough.

### Locally

For local execution, run these scripts in sequence: `data_process/data_generation.py`, `training/train.py`, `inference/run.py`. The results should be similar to the Docker approach. However, using Docker is advised for consistent reproducibility across different operating systems. The trained model should be in the `/models` directory, and predictions on new data in the `/results` folder.

## Considerations

It's worth noting that installing TensorFlow within the Docker container is a painful requirement, as it is a substantially big package (475 MB). Patience or a high download bandwidth may be required.

## Dependencies

Obviously, Docker will need to be installed, that's all.

## Attributions

Thanks to the original developer of the code. Any remaining mistakes, as is often inevitable, are my sole responsibility.
