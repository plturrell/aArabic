# Portainer Complete Setup Guide

**Status:** ✅ Portainer is running at http://localhost:9000  
**Admin Account:** admin / Standard2026

---

## 📋 Step-by-Step Setup

### ✅ Step 1: Admin Account Created
You've already created your admin account!
- **Username:** admin
- **Password:** Standard2026

### 🔄 Step 2: Connect to Docker Environment

After logging in, you should see the "Quick Setup" page. Follow these steps:

#### **Option A: Get Started (Recommended)**
1. Click the blue **"Get Started"** button
2. Portainer will automatically connect to your local Docker environment
3. You'll be taken to the Home dashboard

#### **Option B: Manual Connection**
1. Click **"Add environment"**
2. Select **"Docker Standalone"**
3. Choose **"Socket"** as the connection type
4. Socket path: `/var/run/docker.sock` (already configured)
5. Name: `local` or `docker-local`
6. Click **"Connect"**

---

## 🎯 Step 3: Explore Your Platform

Once connected, you'll see the Portainer dashboard. Here's what to do:

### **A. View All Containers**
1. Click **"Home"** in the left sidebar
2. Click on your **"local"** environment
3. Click **"Containers"** in the left menu
4. You should see 30+ containers including:
   - `ai_nucleus_portainer` (yourself!)
   - `ai_nucleus_keycloak`
   - `ai_nucleus_qdrant`
   - `ai_nucleus_memgraph`
   - `ai_nucleus_dragonfly`
   - And many more...

### **B. Test Container Management**
1. Find any container (e.g., `ai_nucleus_portainer`)
2. Click on the container name
3. You'll see tabs: **Stats, Logs, Inspect, Console**
4. Try these features:
   - **Stats:** See real-time CPU, memory usage
   - **Logs:** View container output (enable auto-refresh)
   - **Console:** Access terminal (select `/bin/sh` or `/bin/bash`)

### **C. View Networks**
1. Click **"Networks"** in the left sidebar
2. Find `compose_nucleus_internal`
3. Click on it to see all connected containers
4. Verify it shows `"Internal": false`

### **D. View Volumes**
1. Click **"Volumes"** in the left sidebar
2. See all persistent data volumes:
   - `compose_portainer_data`
   - `compose_qdrant_data`
   - `compose_memgraph_data`
   - And more...

---

## 🛠️ Step 4: Useful First Actions

### **1. Create a Dashboard Widget (Optional)**
1. Go to **Home**
2. Click **"Customize"**
3. Add widgets for quick access to:
   - Container count
   - Running/stopped containers
   - System resources

### **2. Set Up Notifications (Optional)**
1. Go to **Settings** (gear icon)
2. Click **"Notifications"**
3. Configure webhook or email alerts for:
   - Container stopped
   - High CPU usage
   - Low disk space

### **3. Create Container Templates (Optional)**
1. Go to **App Templates**
2. Create custom templates for frequently deployed services
3. Makes future deployments one-click easy

---

## 🔍 Common Tasks

### **Restart a Service**
```
1. Go to Containers
2. Find the container (e.g., ai_nucleus_qdrant)
3. Click checkbox next to it
4. Click "Restart" button at top
```

### **View Container Logs**
```
1. Go to Containers
2. Click container name
3. Click "Logs" tab
4. Toggle "Auto-refresh" for live tail
5. Use search box to filter logs
```

### **Access Container Shell**
```
1. Go to Containers
2. Click container name
3. Click "Console" tab
4. Select shell: /bin/bash or /bin/sh
5. Click "Connect"
6. You now have terminal access!
```

### **Check Resource Usage**
```
1. Go to Containers
2. Click container name
3. Click "Stats" tab
4. View real-time metrics:
   - CPU percentage
   - Memory usage
   - Network I/O
   - Block I/O
```

### **Deploy a New Stack**
```
1. Go to Stacks
2. Click "Add stack"
3. Option A: Upload docker-compose.yml
4. Option B: Paste compose content
5. Click "Deploy the stack"
```

---

## 🚨 Troubleshooting

### **Can't see any containers?**
**Solution:** Make sure you clicked "Get Started" or connected to the Docker environment.

### **Container shows as "unhealthy"?**
**Cause:** Healthcheck is failing  
**Check:** Click container → Logs tab to see what's wrong

### **Want to reset admin password?**
```bash
docker stop ai_nucleus_portainer
docker rm ai_nucleus_portainer
rm -rf docker/compose/data/portainer/*
docker compose -f docker-compose.core.yml up -d portainer
# Create new admin account at http://localhost:9000
```

### **Portainer not accessible?**
**Check:**
```bash
# Verify container is running
docker ps | grep portainer

# Check network
docker network inspect compose_nucleus_internal | grep "Internal"
# Should show: "Internal": false

# Test access
curl -I http://localhost:9000
# Should show: HTTP/1.1 200 OK
```

---

## 📱 Mobile Access

Portainer has a responsive design that works great on tablets and phones!

1. Access from mobile browser: `http://YOUR_MACHINE_IP:9000`
2. Login with: admin / Standard2026
3. Manage containers on the go!

---

## 🔐 Security Best Practices

### **1. Change Default Password**
```
1. Click user icon (top right)
2. Click "My account"
3. Change password to something stronger
4. Click "Update password"
```

### **2. Enable HTTPS (Production)**
Access via: https://localhost:9443
- Accept self-signed certificate warning
- More secure for production use

### **3. Create Additional Users (Optional)**
```
1. Go to Settings → Users
2. Click "Add user"
3. Set username, password
4. Assign role: Administrator or User
5. Click "Create user"
```

### **4. Enable Audit Logs**
```
1. Go to Settings
2. Click "Authentication logs"
3. View all login attempts
4. Monitor for suspicious activity
```

---

## 🎓 Learning Resources

### **Official Portainer Docs**
- Main: https://docs.portainer.io
- User Guide: https://docs.portainer.io/user/home
- API Docs: https://docs.portainer.io/api/docs

### **Video Tutorials**
- YouTube: Search "Portainer tutorial"
- Official channel: https://www.youtube.com/@portainerio

### **Community**
- Forum: https://community.portainer.io
- Slack: https://portainer.io/slack
- GitHub: https://github.com/portainer/portainer

---

## ✅ Setup Verification Checklist

Check off each item as you complete it:

- [ ] Logged in to Portainer
- [ ] Clicked "Get Started" or connected to Docker
- [ ] Can see list of containers (30+)
- [ ] Viewed logs of at least one container
- [ ] Checked Stats tab for resource usage
- [ ] Viewed Networks section
- [ ] Viewed Volumes section
- [ ] Tested Console access (terminal)
- [ ] Changed admin password (recommended)
- [ ] Bookmarked http://localhost:9000

---

## 🎯 What's Next?

### **Immediate Actions:**
1. ✅ Click "Get Started" in Portainer
2. ✅ Explore Containers section
3. ✅ View logs of a service
4. ✅ Check resource usage
5. ✅ Access a container terminal

### **Advanced Features:**
- Create custom container templates
- Set up webhooks for automation
- Deploy new stacks from UI
- Create backups of configurations
- Monitor resource limits

---

## 🆘 Need Help?

**Quick Commands:**
```bash
# Start Portainer
cd docker/compose
docker compose -f docker-compose.core.yml up -d portainer

# Stop Portainer
docker compose -f docker-compose.core.yml stop portainer

# View logs
docker logs ai_nucleus_portainer -f

# Restart
docker restart ai_nucleus_portainer

# Check status
docker ps | grep portainer
```

**Files:**
- Full docs: `docs/PORTAINER.md`
- This guide: `docs/PORTAINER_SETUP_GUIDE.md`
- Config: `docker/compose/docker-compose.core.yml`

---

## 🎉 You're Ready!

Portainer is now fully set up and ready to use. Start by:

1. **Going to:** http://localhost:9000
2. **Logging in:** admin / Standard2026
3. **Clicking:** "Get Started"
4. **Exploring:** Your 30+ containers!

**Happy Container Managing!** 🐳
