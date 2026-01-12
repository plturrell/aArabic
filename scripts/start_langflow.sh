#!/bin/bash

echo "Starting Langflow Docker container..."
docker run -d --rm -p 7860:7860 --name langflow_container langflowai/langflow:latest

if [ $? -eq 0 ]; then
    echo "Langflow Docker container started successfully on http://localhost:7860"
else
    echo "Failed to start Langflow Docker container."
    exit 1
fi
