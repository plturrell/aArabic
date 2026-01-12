#!/bin/sh
set -e

docker compose exec n8n n8n import:workflow --separate --input=/bootstrap/workflows

echo "Imported workflows from src/serviceIntelligence/serviceN8n/n8n-workflows"
