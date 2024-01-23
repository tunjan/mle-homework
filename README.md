# Machine Learning Exercise

This repository contains an attempt of a well-organized machine learning (MLE) project, focusing on achieving compatibility with different machine environments.

## Overview of Functionality

1. **Data Generation:**
   - Generate the data that the model will be trained and tested on, for simplicity sake, the Iris dataset will be used.

2. **Model Training:**
   - Training  the model with the `train.py` script. The `iris_train.csv` dataset wil be used.

3. **Inference on Unseen Data:**
   - The trained model is then used to make predictions on the unseen data from `iris_test.csv`.

## Usage

### Docker

1. Clone the repository
    ```bash
   git clone https://github.com/tunjan/mle-homework.git
   ```

2. Inside VSCode, open it and navigate to the `File` menu and click `Add Folder to Workspace`. 
Navigate to the directory where you cloned the repository and add it.

3. Set the `CONF_PATH` environment variable to "settings.json".
   ```bash
   export CONF_PATH="settings.json"
   ```
4. Run the bash script from the parent folder (use sudo if necessary).
   ```bash
   bash script.sh
   ```
5. The generated model should be in the `/models` folder and the infered results on unseen data in the `/results` folder.

**NOTE**: The bash script may require modifications if it is to be run on other OS different than Linux. However, the necessary commands are inside the file and the Docker image can be ran with them.

### Locally

Run the following scripts in order `data_process/data_generation.py`, `training/train.py`, `inference/run.py`. The results achieved should be similar. However, due to the dependece on the operating system, Docker is recommended to ensure reproducibility.

## Considerations

It's worth noting that installing TensorFlow within the Docker container is a painful requirement, as it is a substantially big package (475 MB). Patience or a high download bandwidth may be required.

## Dependencies

Obviously, Docker will need to be installed, that's all.

## Attributions

Thanks to the original developer of the code. Any remaining mistakes, as is often inevitable, are my sole responsibility.