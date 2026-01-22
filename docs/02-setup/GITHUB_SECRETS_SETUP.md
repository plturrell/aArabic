# GitHub Secrets Configuration

## Required Secrets for Docker Build Cloud

To enable the Docker Build Cloud workflows, you need to configure the following GitHub Secrets.

## Setting Up Secrets

### Via GitHub Web UI

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret below

### Via GitHub CLI

```bash
# Docker Hub credentials
gh secret set DOCKERHUB_USERNAME --body "plturrell"
gh secret set DOCKERHUB_TOKEN --body "your-docker-hub-token"

# Snyk security scanning (optional)
gh secret set SNYK_TOKEN --body "your-snyk-token"

# AWS/S3 for DVC (if needed in CI)
gh secret set AWS_ACCESS_KEY_ID --body "YOUR_AWS_ACCESS_KEY_ID"
gh secret set AWS_SECRET_ACCESS_KEY --body "YOUR_AWS_SECRET_ACCESS_KEY"
```

## Required Secrets

### 1. DOCKERHUB_USERNAME
- **Value**: `plturrell`
- **Purpose**: Docker Hub username for pushing images
- **Used in**: `.github/workflows/docker-build-backend.yml`

### 2. DOCKERHUB_TOKEN
- **Value**: Your Docker Hub access token
- **Purpose**: Authentication for Docker Hub
- **How to get**:
  1. Log in to Docker Hub
  2. Go to Account Settings → Security
  3. Click "New Access Token"
  4. Copy the token
- **Used in**: `.github/workflows/docker-build-backend.yml`

### 3. SNYK_TOKEN (Optional)
- **Value**: Your Snyk API token
- **Purpose**: Security scanning of Docker images
- **How to get**:
  1. Sign up at https://snyk.io
  2. Go to Account Settings
  3. Copy your API token
- **Used in**: `.github/workflows/docker-build-backend.yml`
- **Note**: If not set, security scanning will be skipped

## Optional Secrets (for CI/CD DVC integration)

### 4. AWS_ACCESS_KEY_ID
- **Value**: `YOUR_AWS_ACCESS_KEY_ID`
- **Purpose**: AWS S3 access for DVC model pulling
- **Used in**: DVC operations in CI/CD

### 5. AWS_SECRET_ACCESS_KEY
- **Value**: `YOUR_AWS_SECRET_ACCESS_KEY`
- **Purpose**: AWS S3 secret key for DVC
- **Used in**: DVC operations in CI/CD

## Docker Build Cloud Configuration

The workflow is pre-configured to use:
- **Builder**: `plturrell/anewmodel`
- **Registry**: `docker.io` (Docker Hub)
- **Image prefix**: `plturrell`

Images will be pushed to:
```
docker.io/plturrell/service-registry:latest
docker.io/plturrell/nopenaiserver:latest
docker.io/plturrell/nwebserve:latest
docker.io/plturrell/nextract:latest
docker.io/plturrell/naudiolab:latest
docker.io/plturrell/ncode:latest
```

## Verification

### Test Docker Hub Authentication

```bash
# Test login locally
echo "YOUR_TOKEN" | docker login -u plturrell --password-stdin

# Verify
docker info | grep Username
```

### Test Docker Build Cloud

```bash
# Create/use the cloud builder
docker buildx create --driver cloud plturrell/anewmodel --use

# List builders
docker buildx ls

# Test build
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.nwebserve \
  -t plturrell/nwebserve:test \
  --push .
```

### Verify GitHub Secrets

```bash
# List configured secrets (names only, not values)
gh secret list

# Expected output:
# DOCKERHUB_USERNAME
# DOCKERHUB_TOKEN
# SNYK_TOKEN (optional)
# AWS_ACCESS_KEY_ID (optional)
# AWS_SECRET_ACCESS_KEY (optional)
```

## Troubleshooting

### "Authentication required" error

**Problem**: Workflow fails with authentication error

**Solution**: 
1. Verify DOCKERHUB_USERNAME and DOCKERHUB_TOKEN are set
2. Regenerate Docker Hub token if expired
3. Update the secret in GitHub

### "Builder not found" error

**Problem**: Docker Build Cloud builder not accessible

**Solution**:
1. Ensure you have access to `plturrell/anewmodel` builder
2. Create the builder if it doesn't exist:
```bash
docker buildx create --driver cloud plturrell/anewmodel
```

### "Rate limit exceeded" error

**Problem**: Docker Hub rate limits reached

**Solution**:
1. Authenticate with Docker Hub (increases rate limits)
2. Use Docker Build Cloud caching to reduce pulls
3. Consider Docker Hub Pro account for higher limits

## Security Best Practices

### ✅ Do:
- Use access tokens, not passwords
- Rotate tokens regularly
- Use least-privilege tokens (read/write only for needed repos)
- Review GitHub Actions logs for exposed secrets

### ❌ Don't:
- Commit secrets to repository
- Share secrets in public channels
- Use personal passwords as tokens
- Log secret values in workflows

## Next Steps

After configuring secrets:

1. **Test the workflow**:
```bash
# Trigger workflow manually
gh workflow run docker-build-backend.yml

# Monitor the run
gh run watch
```

2. **Verify images are pushed**:
```bash
# Check Docker Hub
docker pull plturrell/service-registry:latest
docker pull plturrell/nwebserve:latest
```

3. **Check build logs**:
```bash
# View latest workflow run
gh run view --log
```

## Support

- **Docker Build Cloud**: https://docs.docker.com/build/cloud/
- **GitHub Actions Secrets**: https://docs.github.com/en/actions/security-guides/encrypted-secrets
- **Docker Hub Tokens**: https://docs.docker.com/docker-hub/access-tokens/

---

**Last Updated**: January 22, 2026  
**Workflow**: `.github/workflows/docker-build-backend.yml`
