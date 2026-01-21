#!/bin/bash
# Quick status check for all services

echo "=== BACKEND BUILD STATUS ==="
docker images arabic_folder-backend:latest 2>/dev/null | grep -v REPOSITORY || echo "Not built yet"

echo -e "\n=== RUNNING SERVICES ==="
docker ps --format "table {{.Names}}\t{{.Status}}" | head -20

echo -e "\n=== SERVICES WITH UIs ==="
echo "Langflow:      http://localhost:7860 (if exposed)"
echo "N8N:           http://localhost:5678 (if exposed)"
echo "Gitea:         http://localhost:3000 (if exposed)"
echo "Keycloak:      http://localhost:8080 (if exposed)"
echo "Memgraph Lab:  http://localhost:3000 (if exposed)"
echo "Nucleus Graph: http://localhost:5000 (if exposed)"
echo "Backend API:   http://localhost:8000 (if running)"

echo -e "\n=== HEALTH CHECKS ==="
docker ps --format "{{.Names}}\t{{.Status}}" | grep -E "(healthy|unhealthy|starting)"

echo -e "\n=== BUILD PROCESSES ==="
ps aux | grep "docker-compose.*build" | grep -v grep | awk '{print $2, $11, $12, $13}' || echo "No builds running"