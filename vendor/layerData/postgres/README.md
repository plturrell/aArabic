# Postgres (Vendor)

This directory tracks the Postgres dependency used by platform services.

Current usage:
- Keycloak DB: `postgres:15-alpine`
- Marquez DB: `postgres:14-alpine`

Runtime configuration is defined in `docker/compose/docker-compose.yml`.
