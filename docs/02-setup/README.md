# Setup & Configuration

Complete guides for setting up the serviceCore platform and all its components.

## Quick Start

New to serviceCore? Start here:
1. Check [Prerequisites](#prerequisites)
2. Follow [SAP BTP Setup](./SAP_BTP_SETUP.md)
3. Configure [Docker Build Cloud](./DOCKER_BUILD_CLOUD_SETUP.md)
4. Set up [DVC with SAP S3](./DVC_SAP_S3_SETUP.md)
5. Configure [GitHub Secrets](./GITHUB_SECRETS_SETUP.md)

## Setup Guides

### [SAP BTP Setup](./SAP_BTP_SETUP.md)
Configure SAP HANA Cloud and Object Store integration.

**Topics covered:**
- SAP HANA Cloud connection setup
- OData endpoint configuration
- Object Store (AWS S3) configuration
- Schema creation for logs, metrics, traces
- Authentication setup

**Time required**: ~30 minutes

### [Docker Build Cloud Setup](./DOCKER_BUILD_CLOUD_SETUP.md)
Set up Docker Build Cloud for multi-platform image builds.

**Topics covered:**
- Docker Build Cloud configuration
- GitHub Actions workflow setup
- Multi-platform builds (AMD64, ARM64)
- Registry caching strategies
- Security scanning with Snyk

**Time required**: ~20 minutes

### [DVC with SAP S3 Setup](./DVC_SAP_S3_SETUP.md)
Configure Data Version Control with SAP Object Store.

**Topics covered:**
- DVC remote storage configuration
- SAP S3 bucket setup
- Model versioning workflow
- Team collaboration with DVC
- CI/CD integration

**Time required**: ~15 minutes

### [GitHub Secrets Setup](./GITHUB_SECRETS_SETUP.md)
Configure GitHub repository secrets for CI/CD.

**Topics covered:**
- Docker Hub credentials
- Snyk security tokens
- AWS credentials for DVC
- Secret management best practices

**Time required**: ~10 minutes

## Prerequisites

Before starting setup, ensure you have:

### Required Access
- ✅ SAP BTP account with HANA Cloud access
- ✅ SAP Object Store credentials
- ✅ Docker Hub account
- ✅ GitHub repository access

### Required Tools
- ✅ Docker Desktop or Docker Engine
- ✅ Git
- ✅ GitHub CLI (`gh`)
- ✅ DVC (`pip install dvc[s3]`)

### Optional Tools
- Docker Build Cloud account
- Snyk account (for security scanning)
- AWS CLI (for S3 operations)

## Configuration Files

After setup, you'll have these configured:

```
.env                           # Environment variables
.dvc/config                    # DVC remote configuration
docker-compose.yml             # Service definitions
config/service_registry.json   # Service registry config
```

## Setup Order

Follow this order for smooth setup:

1. **SAP BTP** (Foundation)
   - HANA Cloud
   - Object Store
   
2. **Local Environment** (Development)
   - Install tools
   - Configure .env file
   - Set up DVC
   
3. **Docker** (Containerization)
   - Docker Build Cloud
   - Build images locally
   - Test with docker-compose
   
4. **CI/CD** (Automation)
   - Configure GitHub Secrets
   - Test workflows
   - Monitor builds

## Verification

After setup, verify everything works:

### 1. Test SAP HANA Connection
```bash
curl -u DBADMIN:Initial@1 \
  https://d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com/odata/v4
```

### 2. Test Object Store
```bash
aws s3 ls s3://hcp-055af4b0-2344-40d2-88fe-ddc1c4aad6c5/ \
  --region us-east-1
```

### 3. Test DVC
```bash
dvc status --remote
```

### 4. Test Docker Build
```bash
docker build -f docker/Dockerfile.nwebserve -t nwebserve:test .
```

### 5. Test Services
```bash
docker-compose -f docker/compose/docker-compose.servicecore.yml up -d
docker-compose -f docker/compose/docker-compose.servicecore.yml ps
```

## Troubleshooting

Common setup issues and solutions:

### SAP HANA Connection Fails
- Verify hostname and port
- Check firewall rules
- Confirm credentials
- Test with SQL client first

### Object Store Access Denied
- Verify AWS credentials
- Check bucket permissions
- Confirm region setting
- Test with AWS CLI

### DVC Push/Pull Fails
- Check S3 credentials in `.dvc/config`
- Verify bucket exists
- Test network connectivity
- Check DVC remote configuration

### Docker Build Fails
- Verify Dockerfile paths
- Check source code locations
- Ensure build dependencies installed
- Review build logs

## Next Steps

After completing setup:

1. **Explore Services**: [Service Documentation](../03-services/)
2. **Development**: [Contributing Guide](../05-development/CONTRIBUTING.md)
3. **Operations**: [Operator Runbook](../04-operations/OPERATOR_RUNBOOK.md)

## Support

- **Documentation Issues**: File a GitHub issue
- **Setup Questions**: Check [FAQ](../09-reference/FAQ.md)
- **Technical Support**: Contact platform team

---

**Last Updated**: January 22, 2026  
**Setup Version**: 2.0.0
