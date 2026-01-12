# Environment & Secret Reference

This guide consolidates every environment variable consumed by the AI Nucleus Docker Compose stacks so CI/CD pipelines can source values from a central secret store.

## Core Identity & Gateway

| Variable | Purpose | Consumed By |
| --- | --- | --- |
| `KEYCLOAK_ADMIN` / `KEYCLOAK_ADMIN_PASSWORD` | Bootstrap Keycloak super-admin account. | `keycloak` (@docker-compose.yml#46-86)
| `KEYCLOAK_DB_PASSWORD` | Database password for `keycloak-db`. | `keycloak`, `keycloak-db` (@docker-compose.yml#63-105)
| `NUCLEUS_CLIENT_SECRET` | OAuth client secret for protected SPA flows. | `keycloak`, `gateway` (@docker-compose.yml#68-71 @docker-compose.yml#23-34)
| `NUCLEUS_SERVICE_SECRET` | Client credentials for service-to-service calls. | `keycloak` (@docker-compose.yml#68-71)
| `NUCLEUS_ADMIN_PASSWORD` | Default realm admin user credential. | `keycloak` (@docker-compose.yml#68-71)
| `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` | APISIX OIDC plug-in configuration. | `gateway` (@docker-compose.yml#23-34)
| `SESSION_SECRET` | Encrypts APISIX session cookies. | `gateway` (@docker-compose.yml#23-34)
| `GITEA_SECRET_KEY` | Crypto secret for Gitea server. | `gitea` (@docker-compose.yml#339-361)

## Backend & Service URLs

| Variable | Description | Default Internal URL |
| --- | --- | --- |
| `N8N_URL` | Workflow automation runtime. | `http://n8n:5678` (@docker-compose.yml#117-133 @docker-compose.yml#275-300)
| `QDRANT_URL` | Vector DB endpoint. | `http://qdrant:6333` (@docker-compose.yml#117-133)
| `MEMGRAPH_URI` | Graph DB bolt connection. | `bolt://memgraph:7687` (@docker-compose.yml#117-133)
| `SHIMMY_URL` | Rust workflow engine API. | `http://shimmy:3001` (@docker-compose.yml#117-133)
| `LANGFLOW_URL` | Langflow designer endpoint. | `http://langflow:7860` (@docker-compose.yml#117-133)
| `GITEA_URL` | Internal Git service. | `http://gitea:3000` (@docker-compose.yml#117-133)
| `HYPERBOOKLM_URL` | Research UI. | `http://hyperbooklm:3002` (@docker-compose.yml#117-133)
| `OPENCANVAS_URL` | Collaboration UI. | `http://opencanvas:3000` (@docker-compose.yml#117-133)
| `DRAGONFLY_URL` | Redis-compatible cache DSN. | `redis://dragonfly:6379` (@docker-compose.yml#117-139)
| `MARQUEZ_URL` | Data lineage API endpoint. | `http://marquez:5000` (@docker-compose.yml#117-138 @docker-compose.yml#363-409)
| `NUCLEUS_GRAPH_URL` | Flask facade for Memgraph. | `http://nucleus-graph:5000` (@docker-compose.yml#117-335)

Additional backend-required values:

- `BACKEND_SECRET_KEY` – Django/FastAPI signing secret.
- `N8N_ENCRYPTION_KEY`, `N8N_WEBHOOK_URL`, `N8N_TIMEZONE` – secure storage for n8n credentials and ingress routing (@docker-compose.yml#275-300).
- `ENVIRONMENT`, `DEBUG`, `ENABLE_AUTH` – application behavior toggles (@docker-compose.yml#117-134).

## n8n Runtime

| Variable | Purpose | Default |
| --- | --- | --- |
| `DB_TYPE` | n8n database backend. | `postgresdb` |
| `DB_POSTGRESDB_HOST` | Postgres host for n8n. | `n8n-db` |
| `DB_POSTGRESDB_PORT` | Postgres port for n8n. | `5432` |
| `DB_POSTGRESDB_DATABASE` | Postgres database name for n8n. | `n8n` |
| `DB_POSTGRESDB_USER` | Postgres user for n8n. | `n8n` |
| `DB_POSTGRESDB_PASSWORD` | Postgres password for n8n (from `N8N_DB_PASSWORD`). | `n8n` |
| `N8N_PROTOCOL` | External protocol for links/webhooks. | `http` |
| `N8N_HOST` | External hostname for links/webhooks. | `localhost` |
| `N8N_PORT` | External port for links/webhooks. | `80` |
| `N8N_PATH` | Path prefix behind the gateway. | `/n8n/` |
| `N8N_EDITOR_BASE_URL` | Public editor URL. | `http://localhost/n8n/` |
| `N8N_WEBHOOK_URL` | Public webhook base URL. | `http://localhost/n8n/` |
| `N8N_LOG_LEVEL` | Runtime log level for n8n. | `info` |
| `N8N_TEMPLATES_ENABLED` | Disable generic workflow templates. | `false` |
| `N8N_PERSONALIZATION_ENABLED` | Disable personalization prompts. | `false` |
| `N8N_DIAGNOSTICS_ENABLED` | Disable telemetry. | `false` |
| `N8N_VERSION_NOTIFICATIONS_ENABLED` | Disable version notifications. | `false` |
| `N8N_VERSION_NOTIFICATIONS_WHATS_NEW_ENABLED` | Disable "What's New" feed. | `false` |
| `N8N_BOOTSTRAP_WORKFLOWS` | Import workflows on first startup. | `true` |
| `LEAN4_PARSER_URL` | Lean4 runtime base URL for n8n workflows. | `http://lean4-runtime:8002` |
| `SERVICE_N8N_URL` | Rust orchestration service URL used by n8n workflows. | `http://service-n8n:8003` |
| `SERVICE_GITEA_URL` | Rust Gitea service URL used by n8n workflows. | `http://service-gitea:8004` |

## Service Registry

| Variable | Description | Default |
| --- | --- | --- |
| `SERVICE_REGISTRY_BIND` | Bind address for the Rust registry service. | `0.0.0.0:8100` |
| `SERVICE_REGISTRY_URL` | Registry base URL for wrapper self-registration. | `http://localhost:8100` |
| `SERVICE_REGISTRY_CONFIG` | Path to the registry catalog JSON. | `config/service_registry.json` |
| `SERVICE_REGISTRY_DB_URL` | Postgres connection string for registry persistence. | `postgres://registry:registry@registry-db:5432/service_registry` |
| `SERVICE_REGISTRY_EMBEDDING_DIM` | Qdrant vector size for service embeddings. | `384` |

## Data Store Credentials

| Variable | Purpose |
| --- | --- |
| `MARQUEZ_DB_PASSWORD` | Postgres password for Marquez and `marquez-db` (@docker-compose.yml#363-410).
| `POSTGRES_PASSWORD` | Generic Postgres default if any auxiliary services use it (@.env.example#82-85).
| `DRAGONFLY_PASSWORD` | Optional auth for Dragonfly cache (@.env.example#89-90).
| `GITEA_DB_PASSWORD` | Postgres password for Gitea (`gitea-db`).
| `N8N_DB_PASSWORD` | Postgres password for n8n (`n8n-db`).
| `SERVICE_REGISTRY_DB_PASSWORD` | Postgres password for the registry database (`registry-db`).

## Operational Toggles

| Variable | Description |
| --- | --- |
| `NETWORK_INTERNAL` | When `true`, blocks outbound internet access for containers (@docker-compose.yml#428-436).
| `APISIX_ADMIN_KEY` | Required if programmatically configuring APISIX routes (@.env.example#56-58).
| `EXTERNAL_HOSTNAME`, `SSL_ENABLED`, `SSL_CERT_PATH`, `SSL_KEY_PATH` | Configure TLS termination when exposing the gateway publicly (@.env.example#101-107).

## Usage Pattern

1. Copy `.env.example` to `.env` for local runs or ingest into your secrets manager of choice (AWS Secrets Manager, Vault, Doppler, etc.).
2. Inject variables into the deployment pipeline (e.g., `docker compose --env-file ci.env up -d`).
3. Rotate sensitive values (`*_SECRET`, `*_PASSWORD`) regularly and update the secret store, not the repository.
