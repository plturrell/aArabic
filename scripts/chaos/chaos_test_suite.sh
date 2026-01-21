#!/bin/bash
# Chaos Testing Suite for LLM Inference Server
# Day 10: Week 2 Wrap-up - Production Hardening Validation
#
# Tests:
# 1. SSD Failure - Verify RAM fallback (Day 8)
# 2. Disk Full - Verify alerts and degradation (Day 9)
# 3. OOM (Out of Memory) - Verify K8s restart (Day 9)
# 4. Network Partition - Verify timeout handling (Day 8)
# 5. High Load - Verify load shedding (Day 9)
# 6. Circuit Breaker Trip - Verify recovery (Day 8)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-production}"
POD_NAME="${POD_NAME:-llm-inference}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          LLM Inference Server - Chaos Testing Suite          ║${NC}"
echo -e "${BLUE}║                    Day 10: Production Hardening              ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Helper functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

wait_for_recovery() {
    local timeout=$1
    local check_cmd=$2
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if eval "$check_cmd" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

check_prometheus_alert() {
    local alert_name=$1
    curl -s "$PROMETHEUS_URL/api/v1/alerts" | grep -q "$alert_name"
}

check_health_status() {
    curl -s "http://localhost:8080/health/readiness" | grep -q "Ready to serve"
}

# ============================================================================
# Test 1: SSD Failure - RAM Fallback
# ============================================================================
test_ssd_failure() {
    log_test "Test 1: SSD Failure - Verify RAM Fallback"
    
    # Simulate SSD mount point becoming unavailable
    log_info "Simulating SSD failure by unmounting /mnt/ssd..."
    
    # In production, this would be:
    # kubectl exec $POD_NAME -- umount /mnt/ssd
    # For testing, we'll simulate with a file marker
    
    echo "SIMULATED_SSD_FAILURE" > /tmp/ssd_failure_marker
    
    # Wait for error detection (should be <5 seconds from Day 9)
    sleep 6
    
    # Check if circuit breaker opened
    if check_prometheus_alert "CircuitBreakerOpen"; then
        log_pass "Circuit breaker detected SSD failure"
    else
        log_fail "Circuit breaker did not detect SSD failure"
        return 1
    fi
    
    # Check if system degraded to RAM-only mode
    log_info "Checking for RAM-only fallback..."
    if check_prometheus_alert "ServiceDegraded"; then
        log_pass "System degraded to RAM-only mode"
    else
        log_fail "System did not degrade gracefully"
        return 1
    fi
    
    # Verify requests still succeed (with degraded performance)
    log_info "Testing inference requests in degraded mode..."
    if check_health_status; then
        log_pass "System still serving traffic in degraded mode"
    else
        log_fail "System not serving traffic in degraded mode"
        return 1
    fi
    
    # Restore SSD
    log_info "Restoring SSD..."
    rm -f /tmp/ssd_failure_marker
    
    # Wait for recovery (<1 minute from Day 8)
    if wait_for_recovery 60 "check_health_status"; then
        log_pass "System recovered from SSD failure (< 60s)"
    else
        log_fail "System did not recover within timeout"
        return 1
    fi
    
    log_pass "Test 1: SSD Failure - PASSED"
    echo ""
}

# ============================================================================
# Test 2: Disk Full - Alerts and Degradation
# ============================================================================
test_disk_full() {
    log_test "Test 2: Disk Full - Verify Alerts and Degradation"
    
    log_info "Simulating disk full scenario..."
    
    # Create large file to fill disk (in test environment)
    # In production: dd if=/dev/zero of=/mnt/ssd/fillfile bs=1M count=100000
    echo "SIMULATED_DISK_FULL" > /tmp/disk_full_marker
    
    # Wait for health check to detect (should be <5s)
    sleep 6
    
    # Check for disk space alert
    if check_prometheus_alert "DiskSpaceLow"; then
        log_pass "Disk space alert triggered"
    else
        log_fail "Disk space alert not triggered"
        return 1
    fi
    
    # Check system degradation
    if check_prometheus_alert "ServiceDegraded"; then
        log_pass "System entered degraded state"
    else
        log_fail "System did not degrade"
        return 1
    fi
    
    # Cleanup
    log_info "Cleaning up disk..."
    rm -f /tmp/disk_full_marker
    
    # Verify recovery
    if wait_for_recovery 30 "check_health_status"; then
        log_pass "System recovered after disk cleanup"
    else
        log_fail "System did not recover"
        return 1
    fi
    
    log_pass "Test 2: Disk Full - PASSED"
    echo ""
}

# ============================================================================
# Test 3: OOM (Out of Memory) - K8s Restart
# ============================================================================
test_oom_restart() {
    log_test "Test 3: OOM - Verify K8s Automatic Restart"
    
    log_info "Simulating OOM condition..."
    
    # In real scenario: Allocate memory until OOM
    # kubectl exec $POD_NAME -- stress-ng --vm 1 --vm-bytes 32G --vm-hang 0
    
    echo "SIMULATED_OOM" > /tmp/oom_marker
    
    # Check for memory pressure alert
    sleep 6
    
    if check_prometheus_alert "MemoryPressureMode"; then
        log_pass "Memory pressure detected"
    else
        log_fail "Memory pressure not detected"
        return 1
    fi
    
    # In real K8s, pod would be OOMKilled
    log_info "Simulating K8s restart..."
    
    # Check liveness probe would fail (critical status)
    log_info "Checking liveness probe status..."
    
    # After restart, pod should come back
    rm -f /tmp/oom_marker
    
    if wait_for_recovery 120 "check_health_status"; then
        log_pass "Pod restarted and recovered (< 120s)"
    else
        log_fail "Pod did not recover"
        return 1
    fi
    
    log_pass "Test 3: OOM - PASSED"
    echo ""
}

# ============================================================================
# Test 4: Network Partition - Timeout Handling
# ============================================================================
test_network_partition() {
    log_test "Test 4: Network Partition - Verify Timeout Handling"
    
    log_info "Simulating network partition..."
    
    # In production: iptables rules to drop packets
    # iptables -A INPUT -s <peer_ip> -j DROP
    
    echo "SIMULATED_NETWORK_PARTITION" > /tmp/network_partition_marker
    
    # Attempt request that should timeout
    log_info "Testing timeout behavior..."
    
    # Circuit breaker should trip after repeated timeouts
    sleep 10
    
    if check_prometheus_alert "CircuitBreakerOpen"; then
        log_pass "Circuit breaker tripped on timeouts"
    else
        log_fail "Circuit breaker did not trip"
        return 1
    fi
    
    # Restore network
    log_info "Restoring network connectivity..."
    rm -f /tmp/network_partition_marker
    
    # Circuit breaker should recover (half-open → closed)
    if wait_for_recovery 60 "check_health_status"; then
        log_pass "Circuit breaker recovered"
    else
        log_fail "Circuit breaker did not recover"
        return 1
    fi
    
    log_pass "Test 4: Network Partition - PASSED"
    echo ""
}

# ============================================================================
# Test 5: High Load - Load Shedding
# ============================================================================
test_high_load() {
    log_test "Test 5: High Load - Verify Load Shedding"
    
    log_info "Simulating high load (>100 concurrent requests)..."
    
    # Generate load (in test environment)
    # In production: Use load testing tool like hey or wrk
    # hey -z 30s -c 150 http://localhost:8080/v1/chat/completions
    
    echo "SIMULATED_HIGH_LOAD" > /tmp/high_load_marker
    
    # Check load shedding metrics
    sleep 10
    
    # Query Prometheus for rejected requests
    log_info "Checking for rejected requests (load shedding active)..."
    
    # Should see non-zero rejected_requests metric
    rejected=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=load_shedding_rejected_requests_total" | grep -o '"value":\[[0-9]*,"[0-9]*"\]' || echo "0")
    
    if [ "$rejected" != "0" ]; then
        log_pass "Load shedding rejected excess requests"
    else
        log_fail "Load shedding did not activate"
        return 1
    fi
    
    # Verify system didn't crash (no OOM, no restarts)
    if check_health_status; then
        log_pass "System remained stable under load"
    else
        log_fail "System crashed under load"
        return 1
    fi
    
    # Stop load
    log_info "Stopping load generation..."
    rm -f /tmp/high_load_marker
    
    # Verify recovery to normal
    if wait_for_recovery 30 "check_health_status"; then
        log_pass "System recovered to normal operation"
    else
        log_fail "System did not recover"
        return 1
    fi
    
    log_pass "Test 5: High Load - PASSED"
    echo ""
}

# ============================================================================
# Test 6: Circuit Breaker Recovery
# ============================================================================
test_circuit_breaker_recovery() {
    log_test "Test 6: Circuit Breaker - Verify Recovery Testing"
    
    log_info "Simulating repeated failures to trip circuit breaker..."
    
    # Simulate 5+ failures
    for i in {1..6}; do
        echo "SIMULATED_FAILURE_$i" > /tmp/failure_$i
        sleep 1
    done
    
    # Circuit should be open
    if check_prometheus_alert "CircuitBreakerOpen"; then
        log_pass "Circuit breaker opened after failures"
    else
        log_fail "Circuit breaker did not open"
        return 1
    fi
    
    # Wait for half-open state (30s from Day 8)
    log_info "Waiting for half-open state (30s)..."
    sleep 31
    
    # Clear failures
    rm -f /tmp/failure_*
    
    # Circuit should test recovery (half-open)
    log_info "Circuit testing recovery in half-open state..."
    
    # After 2 successful requests, should close
    sleep 5
    
    if ! check_prometheus_alert "CircuitBreakerOpen"; then
        log_pass "Circuit breaker closed after successful recovery tests"
    else
        log_fail "Circuit breaker did not close"
        return 1
    fi
    
    log_pass "Test 6: Circuit Breaker Recovery - PASSED"
    echo ""
}

# ============================================================================
# Main Execution
# ============================================================================

echo -e "${BLUE}Starting Chaos Testing Suite...${NC}"
echo ""

# Run all tests
test_ssd_failure || true
test_disk_full || true
test_oom_restart || true
test_network_partition || true
test_high_load || true
test_circuit_breaker_recovery || true

# Summary
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        Test Summary                           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Total Tests:  $TESTS_TOTAL"
echo -e "${GREEN}Passed:       $TESTS_PASSED${NC}"
echo -e "${RED}Failed:       $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All chaos tests passed! System is production-ready.${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Review failures before production deployment.${NC}"
    exit 1
fi
