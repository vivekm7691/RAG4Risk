# Backend Dockerfile for RAG4Risk
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including CA certificates for SSL
RUN apt-get update && apt-get install -y \
    gcc \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first to avoid hash verification issues
RUN pip install --upgrade pip

# Install PyTorch CPU-only first to avoid pulling NVIDIA GPU libraries
# This prevents downloading cuDNN, NCCL, cuSPARSELt (very large files)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch>=2.1.0

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model using a script to avoid httpx client closure issues
# Set environment variables for SSL (corporate proxy may have self-signed certificates)
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV CURL_CA_BUNDLE=""
ENV REQUESTS_CA_BUNDLE=""
# Copy download script first
COPY download_model.py ./
# Run download script with retries
# Note: Model download may fail due to httpx client closure issues, but will retry at runtime
# This attempts to cache the model during build to avoid httpx client issues at runtime
RUN python download_model.py || (echo "WARNING: Model download failed during build, will retry at runtime" && exit 0)

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

