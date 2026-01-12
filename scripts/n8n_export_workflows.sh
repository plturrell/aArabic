#!/bin/sh
set -e

docker compose exec n8n n8n export:workflow --separate --output=/exports

echo "Exported workflows to ./data/n8n-exports"
