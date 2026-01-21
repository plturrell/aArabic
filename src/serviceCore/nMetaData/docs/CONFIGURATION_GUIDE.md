# nMetaData Configuration Guide

**Complete guide to configuring nMetaData for all deployment scenarios**

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration File Format](#configuration-file-format)
3. [Environment Variables](#environment-variables)
4. [Configuration Sections](#configuration-sections)
5. [Deployment Scenarios](#deployment-scenarios)
6. [Security Best Practices](#security-best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

nMetaData uses JSON configuration files with support for environment variable substitution. This allows:

- **Version control**: Store configuration templates in Git
- **Environment-specific overrides**: Use environment variables for secrets
- **12-factor app compliance**: Separate config from code
- **Validation**: Catch configuration errors at startup

### Configuration Priority

1. Configuration file (JSON)
2. Environment variable overrides (`${VAR_NAME}`)
3. Default values (hardcoded)

---

## Configuration File Format

### Basic Structure

```json
{
  "server": { ... },
  "database": { ... },
  "auth": { ... },
  "logging": { ... },
  "metrics": { ... },
  "cors": { ... }
}
```

### Environment Variable Syntax

```json
{
  "database": {
    "connection": "${DATABASE_URL}"
  }
}
```

### Default Values

```json
{
  "database": {
    "connection": "${DATABASE_URL:-postgresql://localhost:5432/default}"
  }
}
```

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `postgresql://user:pass@host:5432/db` |
| `JWT_SECRET` | JWT signing secret (min 32 chars) | `your-super-secret-jwt-key-here` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SERVER_PORT` | HTTP server port | `8080` |
| `SERVER_HOST` | Bind address | `0.0.0.0` |
| `LOG_LEVEL` | Logging level | `info` |
| `WORKERS` | Thread pool size | `4` |

### Setting Environment Variables

**Linux/macOS:**
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/nmetadata"
export JWT_SECRET="my-super-secret-jwt-key-at-least-32-characters"
```

**Docker:**
```yaml
services:
  nmetadata:
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/nmetadata
      - JWT_SECRET=${JWT_SECRET}
```

**Kubernetes:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: nmetadata-secrets
type: Opaque
stringData:
  jwt-secret: "your-secret-here"
  database-url: "postgresql://..."
```

---

## Configuration Sections

### Server Configuration

Controls HTTP server behavior.

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 8,
    "read_timeout_ms": 30000,
    "write_timeout_ms": 30000,
    "max_request_size_bytes": 10485760
  }
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `host` | string | Bind address | `0.0.0.0` |
| `port` | integer | HTTP port | `8080` |
| `workers` | integer | Thread pool size | `4` |
| `read_timeout_ms` | integer | Read timeout (ms) | `30000` |
| `write_timeout_ms` | integer | Write timeout (ms) | `30000` |
| `max_request_size_bytes` | integer | Max request body size | `10485760` (10MB) |

**Tuning Guidelines:**
- **workers**: Set to CPU core count for CPU-bound workloads
- **timeouts**: Increase for slow clients/networks
- **max_request_size_bytes**: Adjust based on expected payload sizes

---

### Database Configuration

Configures database connection and pooling.

```json
{
  "database": {
    "type": "postgres",
    "connection": "${DATABASE_URL}",
    "pool": {
      "min_size": 20,
      "max_size": 100,
      "acquire_timeout_ms": 5000,
      "idle_timeout_ms": 300000,
      "max_lifetime_ms": 1800000,
      "health_check_interval_ms": 30000
    }
  }
}
```

#### Database Types

| Type | Description | Connection String Format |
|------|-------------|-------------------------|
| `postgres` | PostgreSQL | `postgresql://user:pass@host:5432/db` |
| `hana` | SAP HANA | `hana://user:pass@host:30015/SYSTEMDB` |
| `sqlite` | SQLite | `/path/to/file.db` or `:memory:` |

#### Pool Configuration

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `min_size` | integer | Minimum connections | `5` |
| `max_size` | integer | Maximum connections | `20` |
| `acquire_timeout_ms` | integer | Connection acquire timeout | `5000` |
| `idle_timeout_ms` | integer | Idle connection timeout | `300000` (5 min) |
| `max_lifetime_ms` | integer | Max connection lifetime | `1800000` (30 min) |
| `health_check_interval_ms` | integer | Health check interval | `30000` (30 sec) |

**Pool Sizing Formula:**
```
optimal_size = (concurrent_users × avg_query_time_ms) / 1000
```

**Examples:**
- 100 users, 50ms queries: ~5 connections
- 1000 users, 100ms queries: ~100 connections

---

### Authentication Configuration

Configures JWT authentication.

```json
{
  "auth": {
    "jwt_secret": "${JWT_SECRET}",
    "token_expiry_minutes": 60,
    "refresh_token_expiry_days": 30,
    "bcrypt_rounds": 12
  }
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `jwt_secret` | string | JWT signing secret (min 32 chars) | *required* |
| `token_expiry_minutes` | integer | Access token TTL | `60` |
| `refresh_token_expiry_days` | integer | Refresh token TTL | `30` |
| `bcrypt_rounds` | integer | Password hashing rounds | `12` |

**Security Notes:**
- **jwt_secret**: Must be at least 32 characters
- **bcrypt_rounds**: Higher = slower but more secure (10-14 recommended)
- **token_expiry**: Balance security vs user experience

---

### Logging Configuration

Controls application logging.

```json
{
  "logging": {
    "level": "info",
    "format": "json",
    "output": "stdout"
  }
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `level` | enum | Log level: `debug`, `info`, `warn`, `error` | `info` |
| `format` | enum | Format: `json`, `text` | `json` |
| `output` | string | Output: `stdout`, `/path/to/file.log` | `stdout` |

**Level Guidelines:**
- **debug**: Development only (verbose)
- **info**: Production default
- **warn**: Production (reduced logging)
- **error**: Critical issues only

**Format Guidelines:**
- **json**: Production (machine-readable, structured)
- **text**: Development (human-readable)

---

### Metrics Configuration

Configures Prometheus metrics endpoint.

```json
{
  "metrics": {
    "enabled": true,
    "port": 9090,
    "path": "/metrics"
  }
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `enabled` | boolean | Enable metrics | `true` |
| `port` | integer | Metrics port | `9090` |
| `path` | string | Metrics endpoint path | `/metrics` |

**Metrics Exposed:**
- HTTP request count/duration
- Database query count/duration
- Connection pool statistics
- Custom business metrics

---

### CORS Configuration

Configures Cross-Origin Resource Sharing.

```json
{
  "cors": {
    "enabled": true
  }
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `enabled` | boolean | Enable CORS | `true` |

**Note:** Currently uses permissive defaults. Future versions will support detailed configuration.

---

## Deployment Scenarios

### Development

**File:** `config/development.json`

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "workers": 2
  },
  "database": {
    "type": "sqlite",
    "connection": "./data/nmetadata-dev.db",
    "pool": {
      "min_size": 1,
      "max_size": 5
    }
  },
  "auth": {
    "jwt_secret": "${JWT_SECRET:-dev-secret-key-min-32-characters-long}",
    "bcrypt_rounds": 10
  },
  "logging": {
    "level": "debug",
    "format": "text"
  }
}
```

**Usage:**
```bash
./nmetadata --config config/development.json
```

---

### Staging/Testing

**File:** `config/staging.json`

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 4
  },
  "database": {
    "type": "postgres",
    "connection": "${DATABASE_URL}",
    "pool": {
      "min_size": 10,
      "max_size": 30
    }
  },
  "auth": {
    "jwt_secret": "${JWT_SECRET}",
    "token_expiry_minutes": 120
  },
  "logging": {
    "level": "info",
    "format": "json"
  }
}
```

**Environment:**
```bash
export DATABASE_URL="postgresql://user:pass@staging-db:5432/nmetadata"
export JWT_SECRET="staging-jwt-secret-key-min-32-chars"
```

---

### Production (PostgreSQL)

**File:** `config/production.json`

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 8
  },
  "database": {
    "type": "postgres",
    "connection": "${DATABASE_URL}",
    "pool": {
      "min_size": 20,
      "max_size": 100,
      "acquire_timeout_ms": 5000
    }
  },
  "auth": {
    "jwt_secret": "${JWT_SECRET}",
    "token_expiry_minutes": 60,
    "bcrypt_rounds": 12
  },
  "logging": {
    "level": "info",
    "format": "json"
  },
  "metrics": {
    "enabled": true,
    "port": 9090
  }
}
```

**Kubernetes Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nmetadata
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: nmetadata
        image: nmetadata:latest
        args: ["--config", "/config/production.json"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: nmetadata-secrets
              key: database-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: nmetadata-secrets
              key: jwt-secret
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: nmetadata-config
```

---

### Production (SAP HANA)

**File:** `config/hana-production.json`

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 16
  },
  "database": {
    "type": "hana",
    "connection": "${HANA_URL}",
    "pool": {
      "min_size": 30,
      "max_size": 150
    }
  },
  "auth": {
    "jwt_secret": "${JWT_SECRET}",
    "token_expiry_minutes": 60
  },
  "logging": {
    "level": "info",
    "format": "json"
  }
}
```

**HANA Connection String:**
```
hana://username:password@hana-host:30015/SYSTEMDB
```

---

## Security Best Practices

### 1. Secrets Management

**❌ DON'T:**
```json
{
  "auth": {
    "jwt_secret": "hardcoded-secret-in-config"
  }
}
```

**✅ DO:**
```json
{
  "auth": {
    "jwt_secret": "${JWT_SECRET}"
  }
}
```

### 2. JWT Secret Strength

**Minimum Requirements:**
- At least 32 characters
- Mix of letters, numbers, symbols
- Randomly generated

**Generate Strong Secret:**
```bash
openssl rand -base64 48
```

### 3. Database Credentials

**Never commit:**
- Database passwords
- Connection strings with credentials
- API keys

**Use:**
- Environment variables
- Secret management systems (Vault, AWS Secrets Manager)
- Kubernetes secrets

### 4. File Permissions

```bash
chmod 600 config/production.json
chown nmetadata:nmetadata config/production.json
```

### 5. Network Security

- Bind to `127.0.0.1` for local-only access
- Use `0.0.0.0` only behind firewall/load balancer
- Enable TLS at load balancer level

---

## Troubleshooting

### Configuration Validation Errors

**Error:** `ValidationFailed: port cannot be 0`

**Solution:** Set valid port number (1-65535)

```json
{
  "server": {
    "port": 8080
  }
}
```

---

**Error:** `ValidationFailed: pool min_size > max_size`

**Solution:** Ensure min_size ≤ max_size

```json
{
  "database": {
    "pool": {
      "min_size": 10,
      "max_size": 50
    }
  }
}
```

---

**Error:** `MissingRequiredField: database`

**Solution:** Add required database section

```json
{
  "database": {
    "type": "postgres",
    "connection": "${DATABASE_URL}"
  }
}
```

---

### Environment Variable Errors

**Error:** `EnvironmentVariableNotFound: DATABASE_URL`

**Solution:** Set the environment variable

```bash
export DATABASE_URL="postgresql://localhost:5432/nmetadata"
```

Or use default value:

```json
{
  "database": {
    "connection": "${DATABASE_URL:-postgresql://localhost:5432/nmetadata}"
  }
}
```

---

### Connection Issues

**Error:** `Failed to connect to database`

**Checklist:**
1. Verify connection string format
2. Check database is running
3. Verify network connectivity
4. Check firewall rules
5. Validate credentials

**Test Connection:**
```bash
# PostgreSQL
psql "postgresql://user:pass@host:5432/db"

# SAP HANA
hdbsql -n host:30015 -u user -p pass
```

---

### Performance Issues

**Symptom:** High connection wait times

**Solution:** Increase pool size

```json
{
  "database": {
    "pool": {
      "max_size": 100
    }
  }
}
```

---

**Symptom:** Slow requests

**Solution:** Increase timeouts

```json
{
  "server": {
    "read_timeout_ms": 60000,
    "write_timeout_ms": 60000
  }
}
```

---

## Configuration Validation

### Manual Validation

```bash
# Test configuration loading
./nmetadata --config config/production.json --validate

# Check environment variable expansion
./nmetadata --config config/production.json --print-config
```

### Automated Testing

```zig
const config = @import("config");

test "validate production config" {
    const cfg = try config.loadFromFile(allocator, "config/production.json");
    defer cfg.deinit();
    
    try cfg.validate();
}
```

---

## Configuration Reference

### Complete Example

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 8,
    "read_timeout_ms": 30000,
    "write_timeout_ms": 30000,
    "max_request_size_bytes": 10485760
  },
  "database": {
    "type": "postgres",
    "connection": "${DATABASE_URL}",
    "pool": {
      "min_size": 20,
      "max_size": 100,
      "acquire_timeout_ms": 5000,
      "idle_timeout_ms": 300000,
      "max_lifetime_ms": 1800000,
      "health_check_interval_ms": 30000
    }
  },
  "auth": {
    "jwt_secret": "${JWT_SECRET}",
    "token_expiry_minutes": 60,
    "refresh_token_expiry_days": 30,
    "bcrypt_rounds": 12
  },
  "logging": {
    "level": "info",
    "format": "json",
    "output": "stdout"
  },
  "metrics": {
    "enabled": true,
    "port": 9090,
    "path": "/metrics"
  },
  "cors": {
    "enabled": true
  }
}
```

---

## Additional Resources

- [Database Selection Guide](DATABASE_SELECTION_GUIDE.md)
- [Performance Tuning Guide](DATABASE_PERFORMANCE_GUIDE.md)
- [Operations Guide](DATABASE_OPERATIONS_GUIDE.md)
- [Troubleshooting Guide](DATABASE_TROUBLESHOOTING_GUIDE.md)

---

**Last Updated:** Day 43 - January 20, 2026
