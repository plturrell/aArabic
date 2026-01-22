# Docker Build Cloud Setup for serviceCore

## Overview

This document describes the Docker Build Cloud configuration for building and deploying all serviceCore (first-party) services. All third-party infrastructure has been removed in favor of SAP HANA Cloud and SAP Object Store.

## Architecture

### First-Party Services Only

All services are prefixed with "n*" (nucleus) and are built with Zig, Mojo, or Rust:

- **service-registry** - Service discovery and orchestration (Rust/Actix)
- **nWebServe** - API Gateway and web server (Zig)
- **nOpenaiServer** - Local LLM inference server (Zig/Mojo)
- **nExtract** - Document extraction engine (Zig/Mojo)
- **nAudioLab** - Audio processing service (Python→Zig/Mojo)
- **nCode** - Code generation and analysis (Python→Zig/Mojo)

### Data Layer

- **SAP HANA Cloud (OData)** - All structured data, logs, metrics, traces
- **SAP Object Store** - All binary data (documents, audio, models, artifacts)

## GitHub Actions Workflows

### Backend Services (`docker-build-backend.yml`)

**Triggers:**
- Push to `main` or `master` branches
- Pull requests to `main` or `master`
- Manual workflow dispatch
- Changes to `src/serviceCore/**` or `docker/Dockerfile.*`

**Build Matrix:**
All services are built in parallel using a matrix strategy:
- service-registry
- nopenaiserver
- nwebserve
- nextract
- naudiolab
- ncode

**Features:**
- Multi-platform builds (AMD64 + ARM64)
- Docker Build Cloud integration for faster builds
- Registry caching for layer reuse
- Snyk security scanning
- Automated image tagging
- Build summaries in GitHub Actions UI

**Image Registry:**
Images are pushed to GitHub Container Registry (ghcr.io):
```
ghcr.io/<owner>/service-registry:latest
ghcr.io/<owner>/nopenaiserver:latest
ghcr.io/<owner>/nwebserve:latest
...
```

**Image Tags:**
- `latest` - Latest build from default branch
- `main` / `master` - Branch name
- `pr-123` - Pull request number
- `sha-abc123` - Git commit SHA
- `v1.0.0` - Semver tags (if tagged)

## Dockerfiles

### Zig-based Services

**Location:** `docker/Dockerfile.{service}`

**Structure:**
1. Builder stage: Install Zig, build binary
2. Runtime stage: Minimal Debian image with compiled binary

**Services:**
- `Dockerfile.nwebserve`
- `Dockerfile.nopenaiserver` (Zig + Mojo)
- `Dockerfile.nextract` (Zig + Mojo)

### Rust-based Services

**Location:** `docker/Dockerfile.service-registry`

**Structure:**
1. Builder stage: Rust 1.83, cargo build
2. Runtime stage: Minimal Debian with compiled binary

### Python-based Services (Legacy)

**Location:** `docker/Dockerfile.{service}`

**Structure:**
1. Builder stage: Install dependencies
2. Runtime stage: Python 3.11 slim with dependencies

**Services:**
- `Dockerfile.naudiolab`
- `Dockerfile.ncode`

**Note:** These will be migrated to Zig/Mojo in the future.

## Docker Compose

### serviceCore Compose File

**Location:** `docker/compose/docker-compose.servicecore.yml`

**Services:**
All first-party services with:
- SAP HANA Cloud connection (OData)
- SAP Object Store integration
- Service registry integration
- Health checks
- Internal networking

**Usage:**
```bash
# Start all services
docker-compose -f docker/compose/docker-compose.servicecore.yml up -d

# Start specific service
docker-compose -f docker/compose/docker-compose.servicecore.yml up service-registry

# View logs
docker-compose -f docker/compose/docker-compose.servicecore.yml logs -f

# Stop all services
docker-compose -f docker/compose/docker-compose.servicecore.yml down
```

## Environment Variables

### Required for All Services

```bash
# SAP HANA Cloud (OData)
HANA_ODATA_URL=https://your-hana-instance.hanacloud.ondemand.com/odata/v4
HANA_USERNAME=your_username
HANA_PASSWORD=your_password

# SAP Object Store
OBJECT_STORE_URL=https://your-object-store.s3.amazonaws.com
OBJECT_STORE_ACCESS_KEY=your_access_key
OBJECT_STORE_SECRET_KEY=your_secret_key
```

### Service-Specific

**service-registry:**
```bash
SERVICE_REGISTRY_BIND=0.0.0.0:8100
SERVICE_REGISTRY_URL=http://service-registry:8100
SERVICE_REGISTRY_CONFIG=config/service_registry.json
```

**nWebServe:**
```bash
WEBSERVE_PORT=8080
SERVICE_REGISTRY_URL=http://service-registry:8100
```

**nOpenaiServer:**
```bash
OPENAI_PORT=11434
SERVICE_REGISTRY_URL=http://service-registry:8100
```

**nExtract:**
```bash
EXTRACT_PORT=8200
SERVICE_REGISTRY_URL=http://service-registry:8100
OPENAI_SERVER_URL=http://nopenaiserver:11434
```

## GitHub Secrets Configuration

### Required Secrets

1. **GITHUB_TOKEN** (automatically provided)
   - Used for GitHub Container Registry authentication

2. **SNYK_TOKEN** (optional)
   - For security scanning
   - Get from: https://snyk.io/account

3. **DOCKER_BUILD_CLOUD_ENDPOINT** (optional)
   - For Docker Build Cloud acceleration
   - Format: `cloud://your-org/your-builder`

### Setting Secrets

```bash
# Via GitHub CLI
gh secret set SNYK_TOKEN --body "your-snyk-token"

# Via GitHub Web UI
# Settings → Secrets and variables → Actions → New repository secret
```

## Local Development

### Building Images Locally

```bash
# Build specific service
docker build -f docker/Dockerfile.nwebserve -t nwebserve:local .

# Build all services
for service in service-registry nwebserve nopenaiserver nextract; do
  docker build -f docker/Dockerfile.$service -t $service:local .
done
```

### Testing Images

```bash
# Run specific service
docker run -p 8080:8080 \
  -e HANA_ODATA_URL=... \
  -e HANA_USERNAME=... \
  -e HANA_PASSWORD=... \
  nwebserve:local

# Use docker-compose for full stack
docker-compose -f docker/compose/docker-compose.servicecore.yml up
```

## Removed Third-Party Services

The following services have been removed and replaced with SAP BTP:

### Infrastructure
- ❌ Keycloak (Auth) → SAP BTP Identity or custom
- ❌ PostgreSQL (All DBs) → SAP HANA Cloud
- ❌ Qdrant (Vectors) → SAP HANA Vector Engine
- ❌ Memgraph (Graph) → SAP HANA Graph Engine
- ❌ Dragonfly/Redis (Cache) → SAP HANA in-memory

### Observability
- ❌ Grafana (Dashboards) → SAP HANA Cloud
- ❌ Prometheus (Metrics) → SAP HANA Cloud
- ❌ Loki (Logs) → SAP HANA Cloud
- ❌ Promtail (Log shipping) → Direct to HANA
- ❌ Jaeger (Tracing) → SAP HANA Cloud

### Others
- ❌ APISIX (Gateway) → nWebServe
- ❌ Gitea (Git server) → GitHub
- ❌ Marquez (Lineage) → Removed
- ❌ Portainer (Container UI) → Native tools

## Troubleshooting

### Build Failures

**Check workflow logs:**
```bash
gh run list --workflow=docker-build-backend.yml
gh run view <run-id> --log
```

**Common issues:**
- Missing dependencies in Dockerfile
- Zig/Mojo installation failures
- Network timeouts during builds
- Insufficient permissions

### Image Pull Failures

**Authenticate to registry:**
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

**Pull specific image:**
```bash
docker pull ghcr.io/<owner>/service-registry:latest
```

### Runtime Issues

**Check service logs:**
```bash
docker logs servicecore_registry
docker logs servicecore_nwebserve
```

**Verify SAP HANA connectivity:**
```bash
curl -u username:password https://your-hana-instance.hanacloud.ondemand.com/odata/v4
```

## Migration from Third-Party Services

### Data Migration

If migrating from PostgreSQL, Qdrant, etc.:

1. Export data from old services
2. Transform to SAP HANA schema
3. Import via OData or SQL
4. Verify data integrity
5. Update service configurations
6. Test thoroughly
7. Decommission old services

### Configuration Updates

Update all service configs to use:
- SAP HANA Cloud endpoints
- SAP Object Store URIs
- New authentication methods

## Future Enhancements

1. **Service Mesh** - Add Istio/Linkerd for advanced networking
2. **Canary Deployments** - Gradual rollouts with traffic splitting
3. **Auto-scaling** - HPA based on metrics in HANA
4. **Multi-region** - Deploy across multiple SAP regions
5. **Disaster Recovery** - Automated backups and failover

## Support

For issues or questions:
- GitHub Issues: https://github.com/<owner>/<repo>/issues
- Documentation: `/docs`
- Service Core README: `/src/serviceCore/README.md`

---

**Last Updated:** January 22, 2026  
**Version:** 2.0.0 (First-Party Only)
