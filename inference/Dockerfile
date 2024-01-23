# Dockerfile for inference
# Start from a base Python image
FROM python:3.11.6

# Set environment variables
ARG model_name=tensorflow_model.keras
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}
WORKDIR /app

# Install TensorFlow in a separate layer to leverage Docker's caching
COPY requirements.txt ./
RUN grep tensorflow== requirements.txt | xargs pip install --no-cache-dir

# Final stage: Copy files and install remaining packages

# Copy necessary files
COPY data/ /app/data
COPY models/${model_name} /app/models/${model_name}
COPY inference /app/inference
COPY utils.py /app
COPY ${CONF_PATH} /app

# Install the remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the inference script
CMD ["python3", "inference/run.py"]