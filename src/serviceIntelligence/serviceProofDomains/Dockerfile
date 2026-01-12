FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY python/ ./python/
COPY lean4/ ./lean4/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app/python

EXPOSE 8011

CMD ["uvicorn", "python.main:app", "--host", "0.0.0.0", "--port", "8011"]
