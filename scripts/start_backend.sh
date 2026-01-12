#!/bin/bash
# Start script for AI Nucleus backend

set -e

echo "🚀 Starting AI Nucleus Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Start Langflow if Docker is running
echo "🐳 Checking Docker status..."
if docker info > /dev/null 2>&1; then
    echo "✅ Docker is running. Starting Langflow container..."
    # Stop existing Langflow container if it's running
    docker rm -f langflow_container > /dev/null 2>&1 || true
    ./scripts/start_langflow.sh &
else
    echo "❌ Docker is not running. Langflow will not be started. Please start Docker if you need NucleusFlow."
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it with your settings."
    else
        echo "❌ .env.example not found. Please create .env manually."
        exit 1
    fi
fi

# Start the server
echo "🌐 Starting FastAPI server..."
python -m backend.api.server

