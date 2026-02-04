# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-cloudrun.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY scripts/extract_traces_v7.py .
COPY scripts/server.py .

# Create gcs_data directory for content mapping
RUN mkdir -p gcs_data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run the HTTP server
CMD ["python", "server.py"]
