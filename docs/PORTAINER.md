# Portainer CE - Container Management UI

![Portainer](https://www.portainer.io/hubfs/portainer-logo-black.svg)

## Overview

Portainer Community Edition (CE) is a lightweight container management UI that provides an intuitive web interface for managing your Docker containers, images, volumes, networks, and more.

---

## 🚀 Quick Start

### Access Portainer

**URL:** http://localhost:9000

### First-Time Setup

1. **Start Portainer:**
   ```bash
   cd docker/compose
   docker compose -f docker-compose.core.yml up -d portainer
   ```

2. **Open Browser:**
   ```bash
   open http://localhost:9000
   ```

3. **Create Admin Account:**
   - On first visit, you'll be prompted to create an admin username and password
   - Choose a strong password (minimum 12 characters)
   - Click "Create user"

4. **Connect to Local Docker:**
   - Select "Get Started" or "Local"
   - Portainer will automatically connect to your local Docker environment

---

## 📊 Features

### Container Management
- ✅ **Start/Stop/Restart** containers with one click
- ✅ **View logs** in real-time
- ✅ **Inspect containers** - see environment variables, volumes, networks
- ✅ **Execute commands** inside running containers (terminal access)
- ✅ **Monitor resources** - CPU, memory, network usage

### Image Management
- ✅ **Pull images** from Docker Hub or private registries
- ✅ **Build images** from Dockerfiles
- ✅ **Tag and push** images
- ✅ **Delete unused** images

### Volume Management
- ✅ **Create volumes** for persistent data
- ✅ **Browse volume** contents
- ✅ **Backup/restore** volumes
- ✅ **Delete unused** volumes

### Network Management
- ✅ **Create networks** (bridge, overlay, etc.)
- ✅ **Connect/disconnect** containers to networks
- ✅ **Inspect network** topology

### Stack Management
- ✅ **Deploy stacks** from docker-compose files
- ✅ **Update stacks** via UI
- ✅ **View stack** resources

---

## 🎯 Common Tasks

### View All Containers

1. Navigate to **Containers** in the left sidebar
2. See all running and stopped containers
3. Filter by name, status, or labels

### View Container Logs

1. Go to **Containers**
2. Click on a container name
3. Click **Logs** tab
4. Use auto-refresh to tail logs in real-time

### Access Container Terminal

1. Go to **Containers**
2. Click on a container name
3. Click **Console** tab
4. Select shell (usually `/bin/bash` or `/bin/sh`)
5. Click "Connect"

### Monitor Resource Usage

1. Go to **Containers**
2. Click on a container name
3. Click **Stats** tab
4. View real-time CPU, memory, network, and I/O stats

### Deploy a Stack

1. Go to **Stacks** in the left sidebar
2. Click **Add stack**
3. Choose "Upload" and select a `docker-compose.yml` file
4. Or paste the compose file content
5. Click **Deploy the stack**

---

## 🔐 Security

### Current Configuration

- **Port:** 9000 (HTTP)
- **HTTPS Port:** 9443 (optional, self-signed cert)
- **Docker Socket:** Read-only access (`/var/run/docker.sock:ro`)
- **Security Options:** `no-new-privileges:true`

### Best Practices

1. ✅ **Use HTTPS in production** - Access via port 9443
2. ✅ **Strong passwords** - Use 12+ character passwords
3. ✅ **Limit access** - Only expose to trusted networks
4. ✅ **Regular updates** - Run `docker compose pull portainer` monthly
5. ✅ **Audit logs** - Review activity logs regularly

### Enable HTTPS (Optional)

Access Portainer via: https://localhost:9443

The first time you access HTTPS, you'll see a certificate warning (self-signed cert). This is normal for development.

---

## 📦 Integration with AI Nucleus Platform

### Manage All Services

Portainer can manage all AI Nucleus services:

- ✅ **Keycloak** - Identity & Access Management
- ✅ **Qdrant** - Vector Database
- ✅ **Memgraph** - Graph Database
- ✅ **DragonflyDB** - In-memory Cache
- ✅ **Gitea** - Git Server
- ✅ **Marquez** - Data Lineage
- ✅ **Langflow** - AI Workflow Builder
- ✅ **n8n** - Workflow Automation
- ✅ **Kafka** - Message Broker
- ✅ **Backend Services** - Translation, Embedding, RAG
- ✅ **And 20+ more...**

### View Network Topology

1. Go to **Networks** → `compose_nucleus_internal`
2. See all containers connected to the internal network
3. Visualize service dependencies

### Quick Actions from Portainer

```bash
# Restart a service
Containers → ai_nucleus_backend → Restart

# View logs
Containers → ai_nucleus_translation → Logs

# Access terminal
Containers → ai_nucleus_qdrant → Console → /bin/sh

# Check resource usage
Containers → ai_nucleus_memgraph → Stats
```

---

## 🔧 Configuration

### Environment Variables

Located in `docker/compose/docker-compose.core.yml`:

```yaml
portainer:
  image: portainer/portainer-ce:latest
  container_name: ai_nucleus_portainer
  ports:
    - "9000:9000"    # HTTP
    - "9443:9443"    # HTTPS
  environment:
    - PORTAINER_HTTP_ENABLED=true
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro  # Read-only Docker socket
    - ./data/portainer:/data                          # Persistent data
```

### Data Persistence

All Portainer data (users, settings, templates) is stored in:
```
docker/compose/data/portainer/
```

This directory is **git-ignored** and persists across container restarts.

---

## 🆘 Troubleshooting

### Cannot Access Portainer

**Check if container is running:**
```bash
docker ps | grep portainer
```

**Check logs:**
```bash
docker logs ai_nucleus_portainer
```

**Restart Portainer:**
```bash
docker restart ai_nucleus_portainer
```

### "Cannot connect to Docker endpoint"

**Verify Docker socket is accessible:**
```bash
ls -la /var/run/docker.sock
```

**Check Portainer has socket access:**
```bash
docker inspect ai_nucleus_portainer | grep docker.sock
```

### Reset Portainer Admin Password

**Stop container and remove data:**
```bash
docker stop ai_nucleus_portainer
rm -rf docker/compose/data/portainer/*
docker start ai_nucleus_portainer
```

Then access http://localhost:9000 and create a new admin account.

---

## 📚 Resources

### Official Documentation
- **Main Site:** https://www.portainer.io
- **Docs:** https://docs.portainer.io
- **GitHub:** https://github.com/portainer/portainer

### License
- **Community Edition:** Free & Open Source (Zlib License)
- **Business Edition:** Commercial ($10-20/node/month)
- **Your Setup:** CE is sufficient for all your needs

### Support
- **Community Forum:** https://community.portainer.io
- **GitHub Issues:** https://github.com/portainer/portainer/issues
- **Slack:** https://portainer.io/slack

---

## 🎓 Tips & Tricks

### 1. Create Templates

Save frequently used compose files as templates for quick deployment.

### 2. Use Labels

Add labels to containers for better organization:
```yaml
labels:
  - "com.project=ai-nucleus"
  - "com.environment=dev"
```

### 3. Backup Configuration

Regular backups of Portainer data:
```bash
tar -czf portainer-backup-$(date +%Y%m%d).tar.gz docker/compose/data/portainer/
```

### 4. Monitor Resources

Set up alerts for high CPU/memory usage (Business Edition feature).

### 5. Use Stacks

Deploy related services as stacks for easier management.

---

## ✅ Next Steps

1. ✅ Access Portainer at http://localhost:9000
2. ✅ Create your admin account
3. ✅ Explore your 30+ running containers
4. ✅ Try restarting a service
5. ✅ View logs of a container
6. ✅ Check resource usage

**Happy Container Management!** 🐳
