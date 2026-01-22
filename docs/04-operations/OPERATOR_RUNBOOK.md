# LLM Inference Server - Production Operator Runbook
## Day 10: Week 2 Production Hardening Complete

**Version:** 1.0  
**Last Updated:** 2026-01-19  
**Maintained By:** Platform Engineering Team

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [System Architecture](#system-architecture)
3. [Common Operational Tasks](#common-operational-tasks)
4. [Incident Response](#incident-response)
5. [Failure Scenarios & Recovery](#failure-scenarios--recovery)
6. [Monitoring & Alerts](#monitoring--alerts)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Emergency Procedures](#emergency-procedures)
9. [Maintenance Windows](#maintenance-windows)
10. [Runbook Updates](#runbook-updates)

---

## Quick Reference

### System Status Dashboard
- **Grafana**: http://grafana.example.com/d/health-monitoring
- **Prometheus**: http://prometheus.example.com
- **Jaeger Tracing**: http://jaeger.example.com
- **Logs (Loki)**: http://grafana.example.com/explore

### Emergency Contacts
- **On-Call Engineer**: See PagerDuty rotation
- **Platform Lead**: platform-lead@example.com
- **Incident Channel**: #incidents-llm-inference

### Critical Metrics Thresholds
| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Error Rate | >1% | >5% | Page on-call |
| Request Queue | >30 | >45 | Scale up pods |
| Memory Usage | >85% | >95% | Check for leaks |
| SSD Free Space | <20GB | <10GB | Clean up or expand |
| Pod Restarts | >3/hour | >10/hour | Investigate root cause |
| Circuit Breaker | Open >5min | Open >15min | Check dependencies |

### Quick Commands

```bash
# Check pod status
kubectl get pods -n production -l app=llm-inference

# View logs
kubectl logs -n production -l app=llm-inference --tail=100 -f

# Check health
kubectl exec -n production llm-inference-xxx -- curl localhost:8080/health/readiness

# Scale deployment
kubectl scale deployment llm-inference-server -n production --replicas=5

# Restart deployment (last resort)
kubectl rollout restart deployment/llm-inference-server -n production

# Check circuit breaker state
curl http://prometheus.example.com/api/v1/query?query=circuit_breaker_state
```

---

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer (K8s Service)           │
└────────────────┬──────────────────────────┬─────────────────┘
                 │                          │
        ┌────────▼────────┐        ┌────────▼────────┐
        │   Pod 1         │        │   Pod 2         │
        │ ┌─────────────┐ │        │ ┌─────────────┐ │
        │ │ Startup     │ │        │ │ Startup     │ │
        │ │ Probe       │ │        │ │ Probe       │ │
        │ └─────────────┘ │        │ └─────────────┘ │
        │ ┌─────────────┐ │        │ ┌─────────────┐ │
        │ │ Liveness    │ │        │ │ Liveness    │ │
        │ │ Probe       │ │        │ │ Probe       │ │
        │ └─────────────┘ │        │ └─────────────┘ │
        │ ┌─────────────┐ │        │ ┌─────────────┐ │
        │ │ Readiness   │ │        │ │ Readiness   │ │
        │ │ Probe       │ │        │ │ Probe       │ │
        │ └─────────────┘ │        │ └─────────────┘ │
        │                 │        │                 │
        │ ┌─────────────┐ │        │ ┌─────────────┐ │
        │ │ Load        │ │        │ │ Load        │ │
        │ │ Shedder     │ │        │ │ Shedder     │ │
        │ └─────────────┘ │        │ └─────────────┘ │
        │ ┌─────────────┐ │        │ ┌─────────────┐ │
        │ │ Circuit     │ │        │ │ Circuit     │ │
        │ │ Breaker     │ │        │ │ Breaker     │ │
        │ └─────────────┘ │        │ └─────────────┘ │
        │                 │        │                 │
        │ RAM: 16-32GB    │        │ RAM: 16-32GB    │
        │ SSD: 50-100GB   │        │ SSD: 50-100GB   │
        └─────────────────┘        └─────────────────┘
                 │                          │
                 └──────────┬───────────────┘
                            │
                ┌───────────▼──────────────┐
                │   Monitoring Stack       │
                │ ┌────────┐ ┌──────────┐ │
                │ │Prometh│ │ Grafana  │ │
                │ │eus    │ │          │ │
                │ └────────┘ └──────────┘ │
                │ ┌────────┐ ┌──────────┐ │
                │ │Jaeger │ │   Loki   │ │
                │ │        │ │          │ │
                │ └────────┘ └──────────┘ │
                └──────────────────────────┘
```

### Key Components

1. **Health Checks** (Day 9)
   - SSD Health Checker
   - RAM Health Checker
   - Model Integrity Checker

2. **K8s Probes** (Day 9)
   - Startup Probe: 5 min tolerance for model loading
   - Liveness Probe: Checks process responsiveness
   - Readiness Probe: Checks ability to serve traffic

3. **Load Management** (Day 9)
   - Load Shedder: Max 100 active, 50 queued
   - Priority Queue: Higher priority first
   - Backpressure: Automatic rejection

4. **Error Handling** (Day 8)
   - Circuit Breaker: 3 states (closed/open/half-open)
   - Retry Logic: Exponential backoff with jitter
   - Graceful Degradation: 4 modes

5. **Observability** (Days 6-7)
   - Structured Logging: JSON, 5 levels
   - Distributed Tracing: OpenTelemetry + Jaeger
   - Metrics: Prometheus + Grafana

---

## Common Operational Tasks

### 1. Scaling the Deployment

**When to Scale:**
- Request queue consistently >30
- Average latency >500ms
- CPU usage >70% across all pods

**Manual Scaling:**
```bash
# Scale up to 5 replicas
kubectl scale deployment llm-inference-server -n production --replicas=5

# Verify scaling
kubectl get pods -n production -l app=llm-inference

# Check HPA status
kubectl get hpa llm-inference-hpa -n production
```

**HPA Configuration:**
- Min replicas: 3
- Max replicas: 10
- Target CPU: 70%
- Target Memory: 80%
- Custom metric: Queue length (target: 10)

### 2. Deploying a New Version

**Zero-Downtime Deployment:**
```bash
# Update image
kubectl set image deployment/llm-inference-server \
  llm-inference=llm-inference-server:v1.1 -n production

# Monitor rollout
kubectl rollout status deployment/llm-inference-server -n production

# Check new pods
kubectl get pods -n production -l app=llm-inference -w

# Rollback if needed
kubectl rollout undo deployment/llm-inference-server -n production
```

**Pre-Deployment Checklist:**
- [ ] Review changes in staging
- [ ] Check current error rate (<1%)
- [ ] Verify HPA not actively scaling
- [ ] Confirm backup pods available (PDB: min 2)
- [ ] Alert on-call engineer
- [ ] Monitor Grafana during rollout

### 3. Log Analysis

**View Recent Logs:**
```bash
# All pods, last 100 lines
kubectl logs -n production -l app=llm-inference --tail=100

# Follow logs in real-time
kubectl logs -n production -l app=llm-inference -f

# Specific pod
kubectl logs -n production llm-inference-xxx-yyy

# Previous container (after crash)
kubectl logs -n production llm-inference-xxx-yyy --previous
```

**Common Log Queries (Loki):**
```logql
# Error logs in last hour
{namespace="production", app="llm-inference"} |= "ERROR" | json

# Circuit breaker events
{namespace="production", app="llm-inference"} |= "circuit_breaker" | json

# Load shedding events
{namespace="production", app="llm-inference"} |= "load_shedding" | json

# Health check failures
{namespace="production", app="llm-inference"} |= "health_check" | json | status="unhealthy"
```

### 4. Metrics Analysis

**Key Prometheus Queries:**
```promql
# Error rate (last 5 min)
rate(error_count_total{namespace="production"}[5m])

# Request latency (p95)
histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))

# Circuit breaker state (0=closed, 1=open, 2=half-open)
circuit_breaker_state{namespace="production"}

# Active requests
load_shedding_active_requests{namespace="production"}

# Memory usage
container_memory_usage_bytes{namespace="production",pod=~"llm-inference.*"}

# SSD free space
health_ssd_free_space_gb{namespace="production"}
```

---

## Incident Response

### Incident Severity Levels

**P0 - Critical**
- Complete service outage
- Data loss imminent
- **Response Time:** 15 minutes
- **Example:** All pods down, circuit breaker stuck open

**P1 - High**
- Significant degradation
- >5% error rate
- **Response Time:** 1 hour
- **Example:** Memory leak, SSD full, high latency

**P2 - Medium**
- Partial degradation
- 1-5% error rate
- **Response Time:** 4 hours
- **Example:** Single pod crashloop, slow queries

**P3 - Low**
- Minor issues
- <1% error rate
- **Response Time:** Next business day
- **Example:** Elevated warnings, minor config issues

### Incident Response Process

1. **Acknowledge** (within 5 min)
   - Acknowledge alert in PagerDuty
   - Join #incidents-llm-inference Slack channel
   - Check Grafana dashboard

2. **Assess** (within 10 min)
   - Determine severity (P0-P3)
   - Check recent changes (deployments, config)
   - Review error logs and metrics
   - Identify affected users/requests

3. **Mitigate** (ASAP)
   - Apply immediate fix (see failure scenarios below)
   - Escalate if needed
   - Update incident channel every 15 min

4. **Resolve** (varies by severity)
   - Verify metrics returned to normal
   - Confirm error rate <1%
   - Monitor for 30 min post-fix

5. **Post-Mortem** (within 3 days for P0/P1)
   - Write incident report
   - Identify root cause
   - Create action items
   - Update runbook

---

## Failure Scenarios & Recovery

### Scenario 1: SSD Failure

**Symptoms:**
- Alert: `CircuitBreakerOpen` (SSD circuit)
- Alert: `ServiceDegraded`
- Logs: `"SSD I/O error"`, `"Falling back to RAM"`
- Metrics: `health_ssd_io_error_rate` spiking

**Impact:**
- Degraded performance (RAM-only mode)
- Reduced cache capacity
- Higher latency for cache misses

**Recovery (Automatic):**
1. Circuit breaker opens immediately
2. System falls back to RAM-only mode
3. Continues serving traffic (degraded)
4. Circuit tests recovery every 30s
5. Automatically recovers when SSD accessible

**Manual Recovery (if needed):**
```bash
# Check SSD mount
kubectl exec -n production llm-inference-xxx -- df -h /mnt/ssd

# Check I/O errors
kubectl exec -n production llm-inference-xxx -- dmesg | grep -i "I/O error"

# Remount SSD if needed
kubectl exec -n production llm-inference-xxx -- mount -o remount /mnt/ssd

# Force circuit breaker reset (emergency only)
curl -X POST http://localhost:8080/admin/circuit-breaker/reset
```

**Expected Recovery Time:** <1 minute (automatic)

---

### Scenario 2: Disk Full

**Symptoms:**
- Alert: `DiskSpaceLow` or `DiskSpaceCritical`
- Alert: `ServiceDegraded`
- Logs: `"No space left on device"`
- Metrics: `health_ssd_free_space_gb` <10GB

**Impact:**
- Cannot cache new data
- Potential write failures
- Degraded mode activated

**Recovery:**
```bash
# Check disk usage
kubectl exec -n production llm-inference-xxx -- du -sh /mnt/ssd/*

# Clear old cache files (if safe)
kubectl exec -n production llm-inference-xxx -- \
  find /mnt/ssd/cache -type f -mtime +7 -delete

# Expand PVC (if supported)
kubectl edit pvc llm-model-pvc -n production
# Increase storage: 200Gi -> 300Gi

# Monitor recovery
watch kubectl exec -n production llm-inference-xxx -- df -h /mnt/ssd
```

**Expected Recovery Time:** <5 minutes (cleanup) or <30 minutes (PVC expansion)

---

### Scenario 3: Out of Memory (OOM)

**Symptoms:**
- Alert: `MemoryPressureMode`
- Pod restart visible in `kubectl get events`
- Logs: `"OOMKilled"` in pod status
- Metrics: `container_memory_usage_bytes` at limit

**Impact:**
- Pod killed and restarted by K8s
- 1-2 minute downtime for that pod
- Other pods handle traffic (if PDB respected)

**Recovery (Automatic):**
1. K8s kills pod (OOMKilled)
2. K8s starts new pod
3. Startup probe waits for model load (up to 5 min)
4. Readiness probe adds pod back to service
5. HPA may scale up if load high

**Manual Investigation:**
```bash
# Check memory usage
kubectl top pods -n production -l app=llm-inference

# Check for memory leaks
kubectl exec -n production llm-inference-xxx -- \
  cat /proc/meminfo

# Review recent memory trends
# Grafana -> Memory Usage by Pod panel

# If leak suspected, restart all pods gradually
kubectl rollout restart deployment/llm-inference-server -n production
```

**Expected Recovery Time:** 2-5 minutes (automatic restart)

**Prevention:**
- Monitor memory trends weekly
- Increase memory limits if consistently >85%
- Review code for leaks if restarts frequent

---

### Scenario 4: High Load (Overload)

**Symptoms:**
- Alert: `HighErrorRate` (429 errors)
- Alert: `RequestQueueHigh`
- Logs: `"Load shedding active"`, `"Request rejected"`
- Metrics: `load_shedding_rejected_requests_total` increasing

**Impact:**
- Some requests rejected (HTTP 429)
- Increased latency for accepted requests
- System remains stable (no crashes)

**Recovery:**
```bash
# Check current load
curl http://localhost:9090/api/v1/query?query=load_shedding_active_requests

# Check queue
curl http://localhost:9090/api/v1/query?query=load_shedding_queued_requests

# Scale up immediately
kubectl scale deployment llm-inference-server -n production --replicas=8

# Monitor HPA
kubectl get hpa llm-inference-hpa -n production -w

# Check load shedding stops
watch 'curl -s http://localhost:9090/api/v1/query?query=rate(load_shedding_rejected_requests_total[1m])'
```

**Expected Recovery Time:** 5-10 minutes (scale up + pod startup)

**Prevention:**
- Set HPA to scale proactively (lower thresholds)
- Implement request prioritization
- Add caching layer upstream

---

### Scenario 5: Circuit Breaker Stuck Open

**Symptoms:**
- Alert: `CircuitBreakerOpen` >15 minutes
- Logs: `"Circuit breaker state: open"`, `"Fast-failing requests"`
- Metrics: `circuit_breaker_state=1` (open)

**Impact:**
- All requests to affected component fail immediately
- No recovery testing occurring
- Degraded or unavailable service

**Investigation:**
```bash
# Check circuit breaker state
curl http://localhost:8080/admin/circuit-breaker/status

# Check what's failing
kubectl logs -n production llm-inference-xxx | grep "circuit_breaker"

# Check dependency health (e.g., SSD)
kubectl exec -n production llm-inference-xxx -- \
  curl localhost:8080/health/components
```

**Recovery:**
```bash
# If underlying issue fixed but circuit still open:
# Wait for automatic recovery (tests every 30s in half-open)

# If emergency and issue confirmed fixed:
curl -X POST http://localhost:8080/admin/circuit-breaker/reset

# Monitor recovery
watch 'curl -s http://localhost:9090/api/v1/query?query=circuit_breaker_state'
```

**Expected Recovery Time:** 30-60 seconds (automatic) or immediate (manual reset)

---

### Scenario 6: Model Corruption

**Symptoms:**
- Alert: `ModelIntegrityCheckFailed`
- Logs: `"Model file not accessible"` or `"Checksum mismatch"`
- Pod fails startup probe
- Metrics: `health_check_status{component="model_integrity"}=3` (critical)

**Impact:**
- Pod cannot start
- Reduced capacity (other pods serve traffic)
- New deployments fail

**Recovery:**
```bash
# Check model file
kubectl exec -n production llm-inference-xxx -- \
  ls -lh /models/model.gguf

# Check checksum
kubectl exec -n production llm-inference-xxx -- \
  sha256sum /models/model.gguf

# If corrupted, re-download from backup
kubectl exec -n production llm-inference-xxx -- \
  cp /backups/model.gguf /models/model.gguf

# Or delete PVC and recreate (will re-download)
kubectl delete pod llm-inference-xxx -n production

# Verify startup succeeds
kubectl get pod llm-inference-xxx -n production -w
```

**Expected Recovery Time:** 2-10 minutes (copy) or 30+ minutes (re-download)

---

## Monitoring & Alerts

### Critical Alerts (Page Immediately)

1. **CircuitBreakerOpen**
   - **Trigger:** Circuit breaker open >5 minutes
   - **Action:** Check dependency health, investigate root cause
   - **Runbook:** See Scenario 5

2. **CriticalErrorRate**
   - **Trigger:** >100 errors/sec or >5% error rate
   - **Action:** Check logs, rollback if recent deployment
   - **Runbook:** See Incident Response

3. **EmergencyDegradation**
   - **Trigger:** System in emergency mode
   - **Action:** Check all health checks, investigate resource exhaustion
   - **Runbook:** Check memory, SSD, model integrity

4. **PodRestartLoop**
   - **Trigger:** >10 restarts in 1 hour
   - **Action:** Check logs for OOM, crash, investigate root cause
   - **Runbook:** See Scenario 3

### Warning Alerts (Investigate During Business Hours)

1. **HighErrorRate**
   - **Trigger:** >50 errors/sec or >1% error rate
   - **Action:** Monitor, investigate if sustained >15 min

2. **ServiceDegraded**
   - **Trigger:** Non-normal degradation mode
   - **Action:** Check component health

3. **DiskSpaceLow**
   - **Trigger:** <20GB free space
   - **Action:** Plan cleanup or expansion

4. **MemoryPressureMode**
   - **Trigger:** Aggressive eviction active
   - **Action:** Check for memory leaks, plan scale-up

### Dashboards

**Main Dashboard:** `http://grafana.example.com/d/health-monitoring`
- Overall system health
- Active/queued/rejected requests
- Error rates by category
- Resource usage
- Circuit breaker states

**SLO Dashboard:** `http://grafana.example.com/d/slo-tracking`
- Availability (target: 99.9%)
- Error budget consumption
- Latency percentiles (p50/p95/p99)

---

## Troubleshooting Guide

### High Latency (p95 >500ms)

**Possible Causes:**
1. Cache misses (cold cache)
2. SSD slow I/O
3. Memory pressure
4. High concurrent load

**Diagnostic Steps:**
```bash
# Check cache hit rate
curl http://prometheus.example.com/api/v1/query?query=cache_hit_rate

# Check SSD I/O wait
kubectl exec -n production llm-inference-xxx -- iostat -x 1 5

# Check memory pressure
kubectl top pod -n production llm-inference-xxx

# Check concurrent requests
curl http://prometheus.example.com/api/v1/query?query=load_shedding_active_requests
```

**Solutions:**
- Warm up cache with common requests
- Check SSD health, replace if degraded
- Scale up pods to reduce per-pod load
- Optimize queries to reduce context length

---

### Frequent Pod Restarts

**Possible Causes:**
1. OOM kills
2. Liveness probe failures
3. Application crashes
4. Node issues

**Diagnostic Steps:**
```bash
# Check restart count
kubectl get pods -n production -l app=llm-inference

# Check events
kubectl get events -n production --field-selector involvedObject.name=llm-inference-xxx

# Check logs before crash
kubectl logs -n production llm-inference-xxx --previous

# Check resource usage
kubectl top pod -n production llm-inference-xxx
```

**Solutions:**
- Increase memory limits if OOM
- Review liveness probe thresholds
- Fix application bugs causing crashes
- Check node health, cordon if faulty

---

### No Traffic to Pod (Readiness Failing)

**Possible Causes:**
1. Health checks failing
2. Pod not ready
3. Service misconfiguration

**Diagnostic Steps:**
```bash
# Check readiness probe
kubectl describe pod llm-inference-xxx -n production | grep -A5 "Readiness"

# Check health endpoint manually
kubectl exec -n production llm-inference-xxx -- \
  curl localhost:8080/health/readiness

# Check service endpoints
kubectl get endpoints llm-inference-service -n production
```

**Solutions:**
- Fix underlying health issue (SSD, RAM, model)
- Wait for startup to complete (<5 min)
- Verify service selector matches pod labels

---

## Emergency Procedures

### Complete Service Outage

**If all pods down:**
```bash
# 1. Check cluster status
kubectl get nodes
kubectl get pods -n production

# 2. Check for cluster-wide issues
kubectl get events -n production --sort-by='.lastTimestamp'

# 3. Force restart all pods
kubectl delete pods -n production -l app=llm-inference

# 4. If PVC issue, remount
kubectl get pvc -n production
kubectl delete pod -n production llm-inference-xxx  # Will recreate

# 5. If deployment issue, rollback
kubectl rollout undo deployment/llm-inference-server -n production

# 6. Monitor recovery
kubectl get pods -n production -l app=llm-inference -w
```

---

### Emergency Rollback

**If new deployment causing issues:**
```bash
# Quick rollback
kubectl rollout undo deployment/llm-inference-server -n production

# Verify rollback
kubectl rollout status deployment/llm-inference-server -n production

# Check previous version
kubectl rollout history deployment/llm-inference-server -n production

# Rollback to specific revision
kubectl rollout undo deployment/llm-inference-server -n production --to-revision=2
```

---

### Emergency Scale-Down

**If system overloaded and cascading failures:**
```bash
# Temporarily reduce to min replicas
kubectl scale deployment llm-inference-server -n production --replicas=3

# Disable HPA temporarily
kubectl patch hpa llm-inference-hpa -n production -p '{"spec":{"minReplicas":3,"maxReplicas":3}}'

# Enable load shedding (if not automatic)
# This is automatic in our system

# After stabilization, re-enable HPA
kubectl patch hpa llm-inference-hpa -n production -p '{"spec":{"minReplicas":3,"maxReplicas":10}}'
```

---

## Maintenance Windows

### Planned Maintenance Process

**Pre-Maintenance (T-24h):**
- [ ] Announce in #announcements channel
- [ ] Update status page
- [ ] Verify backup procedures
- [ ] Review rollback plan

**During Maintenance:**
- [ ] Scale up 20% extra capacity
- [ ] Perform changes on 1 pod at a time
- [ ] Monitor error rates continuously
- [ ] Keep incident channel active

**Post-Maintenance:**
- [ ] Verify all pods healthy
- [ ] Check error rate <1%
- [ ] Monitor for 1 hour
- [ ] Update documentation if needed

### Recommended Maintenance Schedule

- **Weekly:** Review metrics, check for anomalies
- **Monthly:** Update dependencies, security patches
- **Quarterly:** Load testing, disaster recovery drill
- **Annually:** Full system audit, capacity planning

---

## Runbook Updates

This runbook should be updated:
- After every incident (add new scenarios)
- After system changes (update commands/configs)
- Quarterly (review all procedures)

**Version History:**
- v1.0 (2026-01-19): Initial version (Week 2 completion)

**Contributors:**
- Platform Engineering Team
- Site Reliability Engineers
- On-Call Rotation

---

**Questions or Issues?**
- Slack: #platform-engineering
- Email: platform-team@example.com
- Docs: https://docs.example.com/llm-inference
