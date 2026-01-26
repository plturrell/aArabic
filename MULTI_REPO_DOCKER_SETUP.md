# Multi-Repository Docker Build Setup

## Architecture Overview

This project uses a **multi-repository architecture** for building Docker images:

```
┌─────────────────────────────────────────────────────────────┐
│ SDK Repositories (Build & Publish SDK Images)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐     │
│  │  n-c-sdk Repo        │    │  n-python-sdk Repo   │     │
│  │  Custom Zig SDK      │    │  Custom Mojo SDK     │     │
│  └──────────────────────┘    └──────────────────────┘     │
│           │                            │                    │
│           ▼                            ▼                    │
│  plturrell/n-c-sdk:latest   plturrell/n-python-sdk:latest │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Main Repository (aArabic) - Use SDK Images                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Backend Services:           Process Apps:                 │
│  ├─ nAgentFlow              ├─ TrialBalance                │
│  ├─ nAgentMeta              └─ (future apps...)            │
│  ├─ nLocalModels                                            │
│  ├─ nGrounding                                              │
│  └─ nWebServe                                               │
│                                                             │
│  All pull from: plturrell/n-c-sdk:latest                   │
│                 plturrell/n-python-sdk:latest               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### Part 1: SDK Repositories Setup

#### 1.1 Setup n-c-sdk Repository

1. **Clone the repository:**
   ```bash
   git clone https://github.com/plturrell/n-c-sdk.git
   cd n-c-sdk
   ```

2. **Add Dockerfile** (copy from `../n-c-sdk-Dockerfile`):
   ```bash
   # Copy the Dockerfile to the root of n-c-sdk repo
   cp /path/to/n-c-sdk-Dockerfile ./Dockerfile
   ```

3. **Add GitHub Actions workflow:**
   ```bash
   mkdir -p .github/workflows
   # Copy the workflow file
   cp /path/to/n-c-sdk-github-workflow.yml ./.github/workflows/docker-build.yml
   ```

4. **Add GitHub Secrets:**
   - Go to: https://github.com/plturrell/n-c-sdk/settings/secrets/actions
   - Add secrets:
     - `DOCKERHUB_USERNAME` = `plturrell`
     - `DOCKERHUB_TOKEN` = your Docker Hub access token
     - `SNYK_TOKEN` = your Snyk token (optional)

5. **Commit and push:**
   ```bash
   git add Dockerfile .github/workflows/docker-build.yml
   git commit -m "Add Docker build workflow for custom Zig SDK"
   git push origin main
   ```

6. **Verify build:**
   - Check: https://github.com/plturrell/n-c-sdk/actions
   - Image will be pushed to: `plturrell/n-c-sdk:latest`

#### 1.2 Setup n-python-sdk Repository

1. **Clone the repository:**
   ```bash
   git clone https://github.com/plturrell/n-python-sdk.git
   cd n-python-sdk
   ```

2. **Add Dockerfile** (copy from `../n-python-sdk-Dockerfile`):
   ```bash
   cp /path/to/n-python-sdk-Dockerfile ./Dockerfile
   ```

3. **Add GitHub Actions workflow:**
   ```bash
   mkdir -p .github/workflows
   cp /path/to/n-python-sdk-github-workflow.yml ./.github/workflows/docker-build.yml
   ```

4. **Add GitHub Secrets** (same as n-c-sdk):
   - Go to: https://github.com/plturrell/n-python-sdk/settings/secrets/actions
   - Add: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `SNYK_TOKEN`

5. **Commit and push:**
   ```bash
   git add Dockerfile .github/workflows/docker-build.yml
   git commit -m "Add Docker build workflow for custom Mojo SDK"
   git push origin main
   ```

6. **Verify build:**
   - Check: https://github.com/plturrell/n-python-sdk/actions
   - Image will be pushed to: `plturrell/n-python-sdk:latest`

### Part 2: Main Repository (aArabic) - Already Done!

The main repository has been updated to:
- ✅ Remove SDK base image build
- ✅ Update all Dockerfiles to pull from `plturrell/n-c-sdk:latest` and `plturrell/n-python-sdk:latest`
- ✅ Remove submodule initialization (not needed anymore)
- ✅ Simplify build workflows

## Docker Images Produced

### SDK Images (from separate repos):
- `plturrell/n-c-sdk:latest` - Custom Zig compiler
- `plturrell/n-python-sdk:latest` - Custom Mojo SDK

### Service Images (from aArabic repo):
- `plturrell/nservices:nAgentFlow-latest`
- `plturrell/nservices:nAgentMeta-latest`
- `plturrell/nservices:nLocalModels-latest`
- `plturrell/nservices:nGrounding-latest`
- `plturrell/nservices:nWebServe-latest`
- `plturrell/nservices:TrialBalance-latest`

## Build Order

### Initial Setup (one-time):
1. Build SDK images first:
   - Push to n-c-sdk repo → triggers build → publishes `plturrell/n-c-sdk:latest`
   - Push to n-python-sdk repo → triggers build → publishes `plturrell/n-python-sdk:latest`

2. Then build services in aArabic repo:
   - Services automatically pull the published SDK images
   - No need to build SDKs in this repo anymore!

### Ongoing Development:
- **SDK changes**: Push to n-c-sdk or n-python-sdk repos
- **Service changes**: Push to aArabic repo
- Each builds independently!

## Benefits of This Architecture

✅ **Clean Separation**: SDKs built in their own repos  
✅ **No Submodule Issues**: Main repo just pulls Docker images  
✅ **Independent Versioning**: Each SDK can version independently  
✅ **Faster Builds**: Services don't rebuild SDKs every time  
✅ **Reusability**: SDK images can be used by other projects  
✅ **Simpler CI**: No complex dependency checks needed  

## Troubleshooting

### Issue: Service build fails with "image not found"
**Solution**: Ensure SDK images are built and published first:
```bash
docker pull plturrell/n-c-sdk:latest
docker pull plturrell/n-python-sdk:latest
```

### Issue: SDK build fails
**Solution**: Check the SDK repo's GitHub Actions logs and ensure:
- `build.zig` exists in the repo
- Bootstrap Zig can compile the SDK
- Docker Hub credentials are correct

### Issue: Changes to SDK not reflected in services
**Solution**: SDK images are cached. Either:
- Wait for services to rebuild (they pull `:latest` each time)
- Manually trigger service rebuild in GitHub Actions
- Use specific SDK version tags instead of `:latest`

## Monitoring Builds

### n-c-sdk builds:
https://github.com/plturrell/n-c-sdk/actions

### n-python-sdk builds:
https://github.com/plturrell/n-python-sdk/actions

### aArabic builds:
https://github.com/plturrell/aArabic/actions

## Docker Hub Images

View all images:
https://hub.docker.com/u/plturrell

- SDK images: plturrell/n-c-sdk, plturrell/n-python-sdk
- Service images: plturrell/nservices (with service-specific tags)