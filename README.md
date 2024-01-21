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

### Locally

Run the following scripts in order `data_process/data_generation.py`, `training/train.py`, `inference/run.py`. The results achieved should be similar. However, because of the dependence of operating system, Docker is recommended for reproducibility.

## Considerations

It's worth noting that installing TensorFlow within the Docker container is a painful requirement, as it is a substantially big package (450 MB). Patience or a high download bandwidth may be required.

## Dependencies

Obviously, Docker will need to be installed, that's all.

## Atributions

Special thanks to the original developer of the code. Any remaining mistakes, as is often inevitable, are my sole responsibility.