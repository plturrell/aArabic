# Layered Compose Strategy

This document explains how the new vendor-aware Docker Compose setup is organized and how each service depends on others.

## Compose files

| File | Purpose |
| --- | --- |
| `docker/compose/docker-compose.yml` | Full-stack deployment with gateway, backend, data tier, and vendored UIs. Uses vendor-local build contexts and ARGs for versioning. |
| `docker/compose/docker-compose.dev.yml` | Development overlay. Adds bind mounts for vendor UIs (`open-canvas`, `hyperbooklm`) and runs `yarn dev` with hot reload. Use `docker compose -f docker/compose/docker-compose.yml -f docker/compose/docker-compose.dev.yml up opencanvas`. |
| `docker/compose/docker-compose.vendor-ui.yml` | Thin overlay that brings up only the UI-focused services by extending the base file. Useful for isolating UI testing. |
| `docker/compose/docker-compose.vendor-services.yml` | Vendor-only runtime stack (data + UI + orchestration vendors) without backend. |
| `docker/compose/docker-compose.services.yml` | Rust orchestration services (`service-n8n`, `service-gitea`) that pair with the core stack. Use alongside the base file. |
| `docker/compose/docker-compose.wrappers.yml` | Internal-only Rust wrapper services for vendor components, registering with the unified service registry. Use alongside the vendor stack. |

## Build contexts & version args

Services now build directly from their vendored folders:

- `shimmy`: `context=vendor/layerIntelligence/shimmy-ai` with `SHIMMY_VERSION` ARG.
- `opencanvas`: `context=vendor/layerUi/open-canvas` with `OPEN_CANVAS_VERSION` ARG.
- `hyperbooklm`: `context=vendor/layerIntelligence/hyperbooklm` with `HYPERBOOKLM_VERSION` ARG.

Pin a commit or tag by exporting the relevant env var before running Compose, e.g. `export OPEN_CANVAS_VERSION=v0.9.1`.

## Dependency matrix

| Service | Depends on | Notes |
| --- | --- | --- |
| Gateway (APISIX) | Keycloak, Backend | Handles OIDC auth before forwarding to backend. |
| Backend | Qdrant, Memgraph, Shimmy, Langflow, Gitea, HyperbookLM, OpenCanvas, Dragonfly, Keycloak, Marquez, NucleusGraph | Reads service URLs from `.env`. |
| Shimmy | Memgraph (via `MEMGRAPH_URI`) | Serves workflow orchestration via Rust microservice. |
| OpenCanvas | Backend API (`NEXT_PUBLIC_API_URL`) | Uses Yarn workspace build from vendored code. |
| HyperbookLM | Backend API | Next.js research assistant; now builds from vendor folder. |
| NucleusGraph | Memgraph | Provides graph visualization; packaged separately via `docker/Dockerfile.nucleusgraph`. |
| Qdrant | Persistent storage (`./data/qdrant`) | Vector DB consumed by backend. |
| Marquez | Marquez DB | Data lineage stack. |
| Gitea | Persistent volume (`./data/gitea`) | Internal Git service, referenced by backend. |

## Usage examples

- **Full stack**: `docker compose -f docker/compose/docker-compose.yml up -d`
- **UI-only**: `docker compose -f docker/compose/docker-compose.vendor-ui.yml up opencanvas hyperbooklm`
- **Dev hot reload**: `docker compose -f docker/compose/docker-compose.yml -f docker/compose/docker-compose.dev.yml up opencanvas`
- **Vendor-only + wrappers**: `scripts/start_vendor_services.sh`
- **Orchestration services**: `docker compose -f docker/compose/docker-compose.yml -f docker/compose/docker-compose.services.yml up -d service-n8n service-gitea`
- **Wrappers + registry**: `docker compose -f docker/compose/docker-compose.vendor-services.yml -f docker/compose/docker-compose.wrappers.yml up -d wrapper-apisix wrapper-qdrant wrapper-toolorchestra`

Remember to keep `.env` in sync with `docs/deployment/env-vars.md` so all services share the right secrets and URLs.

Note: the wrapper services and registry are internal-only (`expose` only). Query them via `docker compose exec service-registry curl http://localhost:8100/services`.
Note: n8n bootstraps workflows from `src/serviceIntelligence/serviceN8n/n8n-workflows` on first start. Use `scripts/n8n_export_workflows.sh` to export back to `./data/n8n-exports`.
