FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy root-level requirements if available, else standard
# Assuming dependencies are manageable. Pytorch CPU is usually best for non-GPU serving unless specified.
# For simplicity, installing manually common libs.
RUN pip install --no-cache-dir \
    torch \
    numpy \
    pydantic \
    fastapi \
    uvicorn \
    pyyaml \
    huggingface_hub \
    transformers \
    minio 
# Add other requirements from project if needed

# Copy the entire project for import resolution (since we reference offline_training)
# Ideally, we package offline_training, but direct copy works for mono-repo style deployment
COPY offline_training /app/offline_training
COPY common /app/common
COPY model-serving /app/model-serving

# Environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "model-serving.app.server:app", "--host", "0.0.0.0", "--port", "8000"]
