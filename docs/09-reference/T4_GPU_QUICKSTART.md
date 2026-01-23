# T4 GPU Testing Quick Start Guide

Quick guide for testing the T4 GPU optimization features on Brev shell `awesome-gpu-nucleus`.

## Prerequisites

- Access to Brev shell with T4 GPU
- CUDA drivers installed
- Git and basic development tools

## Quick Start (3 Commands)

```bash
# 1. Connect to your Brev instance
brev shell awesome-gpu-nucleus

# 2. Clone/update the repository
cd ~ && git clone https://github.com/plturrell/aArabic.git arabic_folder || (cd arabic_folder && git pull)

# 3. Run the test suite
cd arabic_folder && ./scripts/test_t4_gpu.sh
```

## What Gets Tested

The test suite (`test_t4_gpu.sh`) will:

1. ✓ Verify T4 GPU is available
2. ✓ Check CUDA installation
3. ✓ Build inference engine with CUDA support
4. ✓ Run unit tests:
   - NVIDIA SMI integration
   - CUDA context management
   - CUDA memory management
   - CUDA stream management
   - GPU KV cache operations
5. ✓ Test GPU memory allocation
6. ✓ Run performance benchmarks

## Monitoring GPU During Tests

Open a second terminal and run:

```bash
./scripts/monitor_t4_gpu.sh
```

This provides a real-time dashboard showing:
- GPU utilization
- Memory usage
- Temperature
- Power consumption
- Active processes
- Clock speeds

## Manual Testing Steps

If you prefer manual testing:

```bash
# Verify GPU
nvidia-smi

# Navigate to project
cd ~/arabic_folder/src/serviceCore/nLocalModels/inference/engine

# Build
zig build -Doptimize=ReleaseFast

# Run specific tests
zig build test-cuda-context
zig build test-cuda-memory
zig build test-gpu-cache
```

## Expected Results

**Test Suite:**
- All 5 CUDA tests should pass
- GPU memory allocation test succeeds
- Benchmark shows reasonable throughput

**GPU Monitoring:**
- T4 GPU detected (15GB VRAM)
- Temperature under 80°C during tests
- Memory usage spikes during cache tests
- GPU utilization varies 0-100% during different test phases

## Troubleshooting

### "nvidia-smi not found"
```bash
# Check CUDA installation
ls /usr/local/cuda*
# Reinstall NVIDIA drivers if needed
sudo apt-get update && sudo apt-get install nvidia-driver-535
```

### "Zig not found"
```bash
# Install Zig
curl -L https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz | tar -xJ
export PATH="$PWD/zig-linux-x86_64-0.11.0:$PATH"
```

### Build Failures
- Check CUDA toolkit is installed: `nvcc --version`
- Ensure you have sufficient GPU memory available
- Review build logs in `/tmp/zig_build.log`

### Test Failures
- Individual test logs are in `/tmp/test-*.log`
- Verify GPU is not being used by other processes: `nvidia-smi`
- Check GPU temperature is normal (< 85°C)

## Performance Expectations

**T4 GPU Specifications:**
- 16GB GDDR6 Memory
- 2560 CUDA Cores
- 320 Tensor Cores
- 8.1 TFLOPS FP32
- 65 TFLOPS FP16 (with Tensor Cores)

**Expected Test Performance:**
- Context creation: < 100ms
- Memory allocation (1GB): < 50ms
- Cache operations: < 10ms per operation
- Tensor operations: Varies by size

## Next Steps

After successful testing:

1. **Run inference server:**
   ```bash
   cd ~/arabic_folder/src/serviceCore/nLocalModels
   zig build run -- --backend cuda --model-path ../../vendor/layerModels/google-gemma-3-270m-it
   ```

2. **Review documentation:**
   - Implementation details: `docs/T4/IMPLEMENTATION_PLAN.md`
   - Optimization guide: `docs/T4/T4_OPTIMIZATION_GUIDE.md`

3. **Benchmark performance:**
   ```bash
   ./scripts/benchmark_gpu_tier.sh
   ```

## Useful Commands

```bash
# Quick GPU check
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Watch GPU in real-time
watch -n 1 nvidia-smi

# Check CUDA version
nvcc --version

# Monitor GPU power usage
nvidia-smi -q -d POWER

# List all GPU processes
nvidia-smi pmon

# Reset GPU (if needed)
sudo nvidia-smi -r
```

## Support

For issues or questions:
- Check test logs in `/tmp/` directory
- Review CUDA error codes in test output
- Consult T4 documentation: `docs/T4/`
- GitHub issues: https://github.com/plturrell/aArabic/issues
