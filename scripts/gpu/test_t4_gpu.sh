#!/bin/bash
# Test T4 GPU Features on Brev Shell
# Usage: ./scripts/test_t4_gpu.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

echo "=== T4 GPU Testing Suite ==="
echo ""

# Step 1: Verify GPU
print_step "Step 1: Verifying T4 GPU availability"
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Please ensure CUDA drivers are installed."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)

if [[ $GPU_NAME == *"T4"* ]]; then
    print_success "T4 GPU detected"
else
    print_warning "GPU detected: $GPU_NAME (expected T4)"
fi
echo ""

# Step 2: Check CUDA version
print_step "Step 2: Checking CUDA installation"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_success "CUDA version: $CUDA_VERSION"
else
    print_warning "nvcc not found. Some features may not compile."
fi
echo ""

# Step 3: Navigate to project
print_step "Step 3: Navigating to project directory"
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
print_success "Project root: $PROJECT_ROOT"
echo ""

# Step 4: Check Zig installation
print_step "Step 4: Checking Zig installation"
if ! command -v zig &> /dev/null; then
    print_error "Zig not found. Installing Zig..."
    # Install Zig if needed
    curl -L https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz | tar -xJ
    export PATH="$PWD/zig-linux-x86_64-0.11.0:$PATH"
fi

ZIG_VERSION=$(zig version)
print_success "Zig version: $ZIG_VERSION"
echo ""

# Step 5: Build inference engine
print_step "Step 5: Building inference engine with CUDA support"
cd "$PROJECT_ROOT/src/serviceCore/nOpenaiServer/inference/engine"

if [ -f "build.zig" ]; then
    print_step "Running: zig build -Doptimize=ReleaseFast"
    if zig build -Doptimize=ReleaseFast 2>&1 | tee /tmp/zig_build.log; then
        print_success "Build completed successfully"
    else
        print_error "Build failed. Check /tmp/zig_build.log for details"
        exit 1
    fi
else
    print_warning "build.zig not found. Skipping build step."
fi
echo ""

# Step 6: Run CUDA tests
print_step "Step 6: Running CUDA unit tests"

test_components=(
    "test-nvidia-smi:NVIDIA SMI Integration"
    "test-cuda-context:CUDA Context Management"
    "test-cuda-memory:CUDA Memory Management"
    "test-cuda-streams:CUDA Stream Management"
    "test-gpu-cache:GPU KV Cache"
)

TESTS_PASSED=0
TESTS_FAILED=0

for test_spec in "${test_components[@]}"; do
    IFS=':' read -r test_name test_desc <<< "$test_spec"
    print_step "Testing: $test_desc"
    
    if zig build "$test_name" 2>&1 | tee "/tmp/${test_name}.log"; then
        print_success "$test_desc passed"
        ((TESTS_PASSED++))
    else
        print_error "$test_desc failed (see /tmp/${test_name}.log)"
        ((TESTS_FAILED++))
    fi
    echo ""
done

# Step 7: GPU Memory Test
print_step "Step 7: Testing GPU memory allocation"
python3 << 'EOF'
import subprocess
import json

try:
    # Get GPU memory info
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True
    )
    
    total, used, free = map(int, result.stdout.strip().split(','))
    
    print(f"GPU Memory:")
    print(f"  Total: {total} MB")
    print(f"  Used:  {used} MB")
    print(f"  Free:  {free} MB")
    
    if free < 1000:
        print("⚠ Warning: Less than 1GB free GPU memory")
    else:
        print("✓ Sufficient GPU memory available")
        
except Exception as e:
    print(f"✗ Error checking GPU memory: {e}")
EOF
echo ""

# Step 8: Performance benchmark
print_step "Step 8: Running performance benchmark"
cat > /tmp/gpu_benchmark.py << 'EOF'
import time
import subprocess

def benchmark_gpu_operation():
    """Simple GPU benchmark using nvidia-smi"""
    start_time = time.time()
    
    # Query GPU stats multiple times to measure overhead
    for _ in range(100):
        subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, check=True
        )
    
    elapsed = time.time() - start_time
    ops_per_sec = 100 / elapsed
    
    print(f"GPU Query Benchmark:")
    print(f"  Operations: 100")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {ops_per_sec:.2f} ops/sec")

if __name__ == "__main__":
    benchmark_gpu_operation()
EOF

python3 /tmp/gpu_benchmark.py
echo ""

# Summary
print_step "=== Test Summary ==="
echo ""
print_success "Tests Passed: $TESTS_PASSED"
if [ $TESTS_FAILED -gt 0 ]; then
    print_error "Tests Failed: $TESTS_FAILED"
else
    print_success "All tests passed!"
fi
echo ""

print_step "Next Steps:"
echo "1. Review test logs in /tmp/ directory"
echo "2. Run inference server: cd src/serviceCore/nOpenaiServer && zig build run"
echo "3. Monitor GPU usage: watch -n 1 nvidia-smi"
echo "4. Check documentation: docs/T4/T4_OPTIMIZATION_GUIDE.md"
echo ""

exit $TESTS_FAILED
