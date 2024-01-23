# Dockerfile for training
# Start from a base Python image
FROM python:3.11.6

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Changing the path to the working directory in the container
WORKDIR /app

# Install TensorFlow first to create a reusable layer
# Extract TensorFlow version from requirements.txt and install
COPY requirements.txt ./
RUN grep tensorflow== requirements.txt | xargs pip install --no-cache-dir

# Install any remaining necessary packages listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY data_process /app/data_process
COPY data /app/data
COPY training /app/training
COPY utils.py /app
COPY ${CONF_PATH} /app

# Train the model
RUN python3 data_process/data_generation.py
RUN python3 training/train.py