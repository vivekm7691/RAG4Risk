# Backend Dockerfile for RAG4Risk
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first to avoid hash verification issues
RUN pip install --upgrade pip

# Install PyTorch CPU-only first to avoid pulling NVIDIA GPU libraries
# This prevents downloading cuDNN, NCCL, cuSPARSELt (very large files)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch>=2.1.0

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

