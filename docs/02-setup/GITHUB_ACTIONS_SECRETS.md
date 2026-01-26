# GitHub Actions Secrets Setup

This document describes the required GitHub Actions secrets for the Docker build workflow.

## Required Secrets

The workflow `.github/workflows/docker-build-backend.yml` requires the following secrets to be configured in your GitHub repository:

### 1. DOCKERHUB_USERNAME
- **Purpose**: Docker Hub username for authentication
- **Used in**: Docker login step to authenticate with Docker Hub
- **How to get**: Your Docker Hub account username
- **Example**: `plturrell`

### 2. DOCKERHUB_TOKEN
- **Purpose**: Docker Hub access token for secure authentication
- **Used in**: Docker login step (more secure than using password)
- **How to get**: 
  1. Log in to [Docker Hub](https://hub.docker.com)
  2. Go to Account Settings → Security
  3. Click "New Access Token"
  4. Give it a description (e.g., "GitHub Actions - aArabic")
  5. Copy the generated token (you won't be able to see it again!)
- **Note**: This is an access token, NOT your Docker Hub password

### 3. SNYK_TOKEN (Optional)
- **Purpose**: Snyk API token for security scanning of Docker images
- **Used in**: Security scanning step (optional, will be skipped if not set)
- **How to get**:
  1. Log in to [Snyk](https://snyk.io)
  2. Go to Account Settings
  3. Copy your API token
- **Note**: The workflow has `continue-on-error: true`, so builds will succeed even without this secret

## How to Add Secrets to GitHub

1. Go to your GitHub repository: https://github.com/plturrell/aArabic
2. Click on **Settings** (top menu)
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. For each secret:
   - Enter the **Name** exactly as shown above (case-sensitive)
   - Enter the **Secret** value
   - Click **Add secret**

## Verification Steps

After adding the secrets, you can verify they're set up correctly:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. You should see the secrets listed (values are hidden)
3. The next time the workflow runs, it should be able to:
   - Authenticate with Docker Hub
   - Build and push images to `plturrell/nservices`
   - Optionally scan with Snyk (if token is provided)

## Workflow Trigger

The workflow will automatically run when:
- Code is pushed to `main` or `master` branch
- Changes are made to:
  - `src/serviceCore/**` (any service code)
  - `docker/Dockerfile.*` (any Dockerfile)
  - `.github/workflows/docker-build-backend.yml` (workflow itself)
- Manually triggered via **Actions** → **Docker Build - Backend Services** → **Run workflow**

## Expected Output

Once secrets are configured, the workflow will:
1. Build 5 services: nAgentFlow, nAgentMeta, nLocalModels, nGrounding, nWebServe
2. Push images to Docker Hub as:
   - `plturrell/nservices:nAgentFlow-latest`
   - `plturrell/nservices:nAgentMeta-latest`
   - `plturrell/nservices:nLocalModels-latest`
   - `plturrell/nservices:nGrounding-latest`
   - `plturrell/nservices:nWebServe-latest`
3. Create additional tags with branch name, commit SHA, and timestamp

## Troubleshooting

### "Error: Username and password required"
- **Cause**: DOCKERHUB_USERNAME or DOCKERHUB_TOKEN is missing or incorrect
- **Solution**: Verify both secrets are set correctly

### "Error: denied: requested access to the resource is denied"
- **Cause**: The Docker Hub token doesn't have permission to push to `plturrell/nservices`
- **Solution**: Ensure the token has Read & Write permissions

### "Snyk scan failed"
- **Cause**: SNYK_TOKEN is invalid or expired
- **Solution**: This is optional and won't stop the build. Update the token if you want security scanning.

## Security Best Practices

1. **Never commit secrets to code** - Always use GitHub Actions secrets
2. **Use access tokens** - Don't use your Docker Hub password directly
3. **Rotate tokens regularly** - Generate new tokens periodically for security
4. **Limit token scope** - Only give tokens the minimum required permissions
5. **Monitor usage** - Check Docker Hub for unexpected image pulls/pushes

## Quick Setup Checklist

- [ ] Create Docker Hub access token
- [ ] Add DOCKERHUB_USERNAME secret to GitHub
- [ ] Add DOCKERHUB_TOKEN secret to GitHub
- [ ] (Optional) Add SNYK_TOKEN secret for security scanning
- [ ] Trigger workflow manually or push code to test
- [ ] Verify images appear in Docker Hub under `plturrell/nservices`

---

**Last Updated**: January 26, 2026
**Workflow File**: `.github/workflows/docker-build-backend.yml`
**Docker Repository**: `plturrell/nservices`