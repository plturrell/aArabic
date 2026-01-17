FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for MarkItDown
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8005

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]