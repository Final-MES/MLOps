# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Create project directories
RUN mkdir -p data/raw data/processed data/external \
    notebooks \
    src \
    models \
    config

# Expose Jupyter Notebook port
EXPOSE 8888

# Use app.py to start Jupyter with synchronization
CMD ["python", "app.py"]