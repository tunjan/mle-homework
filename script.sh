
#!/bin/bash
set -e

# Create directories if they don't exist
[ ! -d "data" ] && mkdir data
[ ! -d "models" ] && mkdir models

# Build and run training Docker container
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
container_id=$(docker run -d training_image)

# Wait for training to complete and then copy files
docker wait $container_id
docker cp $container_id:/app/models/tensorflow_model.keras ./models
docker cp $container_id:/app/data/ .

# Build and run inference Docker container
docker build -f ./inference/Dockerfile --build-arg model_name=tensorflow_model.keras --build-arg settings_name=settings.json -t inference_image .
container_id=$(docker run -d inference_image)

# Wait for inference to complete and then copy results
docker wait $container_id
docker cp $container_id:/app/results .