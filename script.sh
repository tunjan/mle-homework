#!/bin/bash

python ./data_process/data_generation.py

# Build training Docker image
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .

# Run training Docker container
docker run -it training_image /bin/bash

# Get the container ID of the last run container
container_id=$(docker ps -lq)

# Substitute <container_id> and <model_name> with actual values
model_name="tensorflow_model.keras"  # Replace with your actual model name

# Copy trained model from training container to local machine
docker cp $container_id:/app/models/$model_name ./models

# Build inference Docker image
docker build -f ./inference/Dockerfile --build-arg model_name=$model_name --build-arg settings_name=settings.json -t inference_image .

# Run inference Docker container with attached terminal
docker run -it inference_image /bin/bash

container_id=$(docker ps -lq)

# Copy results from inference container to local machine
docker cp $container_id:/app/results .

