// CUDA Runtime API bindings
// FFI layer for interfacing with CUDA runtime library
//
// Links against: libcuda.so, libcudart.so
// Requires: CUDA Toolkit 11.8+ or 12.x

const std = @import("std");

// ============================================================================
// CUDA Error Codes
// ============================================================================

pub const cudaSuccess: c_int = 0;
pub const cudaErrorMemoryAllocation: c_int = 2;
pub const cudaErrorInitializationError: c_int = 3;
pub const cudaErrorLaunchFailure: c_int = 4;
pub const cudaErrorInvalidValue: c_int = 11;
pub const cudaErrorInvalidDevicePointer: c_int = 17;
pub const cudaErrorInvalidMemcpyDirection: c_int = 21;
pub const cudaErrorNoDevice: c_int = 38;
pub const cudaErrorInsufficientDriver: c_int = 35;

// ============================================================================
// Memory Copy Kinds
// ============================================================================

pub const cudaMemcpyHostToHost: c_int = 0;
pub const cudaMemcpyHostToDevice: c_int = 1;
pub const cudaMemcpyDeviceToHost: c_int = 2;
pub const cudaMemcpyDeviceToDevice: c_int = 3;
pub const cudaMemcpyDefault: c_int = 4;

// ============================================================================
// Device Management
// ============================================================================

/// Get the number of CUDA-capable devices
pub extern "cudart" fn cudaGetDeviceCount(count: *c_int) c_int;

/// Set the current CUDA device
pub extern "cudart" fn cudaSetDevice(device: c_int) c_int;

/// Get the current CUDA device
pub extern "cudart" fn cudaGetDevice(device: *c_int) c_int;

/// Reset the current CUDA device
pub extern "cudart" fn cudaDeviceReset() c_int;

/// Synchronize the current device
pub extern "cudart" fn cudaDeviceSynchronize() c_int;

// ============================================================================
// Device Properties
// ============================================================================

/// CUDA device properties structure - CUDA 12.x compatible
/// Must be large enough to hold all fields written by cudaGetDeviceProperties
pub const CudaDeviceProp = extern struct {
    name: [256]u8,
    uuid: [16]u8,
    luid: [8]u8,
    luidDeviceNodeMask: c_uint,
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    regsPerBlock: c_int,
    warpSize: c_int,
    memPitch: usize,
    maxThreadsPerBlock: c_int,
    maxThreadsDim: [3]c_int,
    maxGridSize: [3]c_int,
    clockRate: c_int,
    totalConstMem: usize,
    major: c_int,
    minor: c_int,
    textureAlignment: usize,
    texturePitchAlignment: usize,
    deviceOverlap: c_int,
    multiProcessorCount: c_int,
    kernelExecTimeoutEnabled: c_int,
    integrated: c_int,
    canMapHostMemory: c_int,
    computeMode: c_int,
    maxTexture1D: c_int,
    maxTexture1DMipmap: c_int,
    maxTexture1DLinear: c_int,
    maxTexture2D: [2]c_int,
    maxTexture2DMipmap: [2]c_int,
    maxTexture2DLinear: [3]c_int,
    maxTexture2DGather: [2]c_int,
    maxTexture3D: [3]c_int,
    maxTexture3DAlt: [3]c_int,
    maxTextureCubemap: c_int,
    maxTexture1DLayered: [2]c_int,
    maxTexture2DLayered: [3]c_int,
    maxTextureCubemapLayered: [2]c_int,
    maxSurface1D: c_int,
    maxSurface2D: [2]c_int,
    maxSurface3D: [3]c_int,
    maxSurface1DLayered: [2]c_int,
    maxSurface2DLayered: [3]c_int,
    maxSurfaceCubemap: c_int,
    maxSurfaceCubemapLayered: [2]c_int,
    surfaceAlignment: usize,
    concurrentKernels: c_int,
    ECCEnabled: c_int,
    pciBusID: c_int,
    pciDeviceID: c_int,
    pciDomainID: c_int,
    tccDriver: c_int,
    asyncEngineCount: c_int,
    unifiedAddressing: c_int,
    memoryClockRate: c_int,
    memoryBusWidth: c_int,
    l2CacheSize: c_int,
    persistingL2CacheMaxSize: c_int,
    maxThreadsPerMultiProcessor: c_int,
    streamPrioritiesSupported: c_int,
    globalL1CacheSupported: c_int,
    localL1CacheSupported: c_int,
    sharedMemPerMultiprocessor: usize,
    regsPerMultiprocessor: c_int,
    managedMemory: c_int,
    isMultiGpuBoard: c_int,
    multiGpuBoardGroupID: c_int,
    hostNativeAtomicSupported: c_int,
    singleToDoublePrecisionPerfRatio: c_int,
    pageableMemoryAccess: c_int,
    concurrentManagedAccess: c_int,
    computePreemptionSupported: c_int,
    canUseHostPointerForRegisteredMem: c_int,
    cooperativeLaunch: c_int,
    cooperativeMultiDeviceLaunch: c_int,
    sharedMemPerBlockOptin: usize,
    pageableMemoryAccessUsesHostPageTables: c_int,
    directManagedMemAccessFromHost: c_int,
    // CUDA 12.x additional fields
    maxBlocksPerMultiProcessor: c_int,
    accessPolicyMaxWindowSize: c_int,
    reservedSharedMemPerBlock: usize,
    hostRegisterSupported: c_int,
    sparseCudaArraySupported: c_int,
    hostRegisterReadOnlySupported: c_int,
    timelineSemaphoreInteropSupported: c_int,
    memoryPoolsSupported: c_int,
    gpuDirectRDMASupported: c_int,
    gpuDirectRDMAFlushWritesOptions: c_uint,
    gpuDirectRDMAWritesOrdering: c_int,
    memoryPoolSupportedHandleTypes: c_uint,
    deferredMappingCudaArraySupported: c_int,
    ipcEventSupported: c_int,
    clusterLaunch: c_int,
    unifiedFunctionPointers: c_int,
    // Reserved padding for future CUDA versions (512 bytes extra)
    _reserved: [512]u8,
};

/// Get device properties
pub extern "cudart" fn cudaGetDeviceProperties(prop: *CudaDeviceProp, device: c_int) c_int;

// ============================================================================
// Memory Management
// ============================================================================

/// Allocate memory on the device
pub extern "cudart" fn cudaMalloc(ptr: **anyopaque, size: usize) c_int;

/// Free memory on the device
pub extern "cudart" fn cudaFree(ptr: *anyopaque) c_int;

/// Allocate pinned host memory (page-locked)
pub extern "cudart" fn cudaMallocHost(ptr: **anyopaque, size: usize) c_int;

/// Free pinned host memory
pub extern "cudart" fn cudaFreeHost(ptr: *anyopaque) c_int;

/// Get free and total device memory
pub extern "cudart" fn cudaMemGetInfo(free: *usize, total: *usize) c_int;

/// Copy memory between host and device
pub extern "cudart" fn cudaMemcpy(dst: *anyopaque, src: *const anyopaque, size: usize, kind: c_int) c_int;

/// Asynchronous memory copy
pub extern "cudart" fn cudaMemcpyAsync(
    dst: *anyopaque,
    src: *const anyopaque,
    size: usize,
    kind: c_int,
    stream: ?*anyopaque,
) c_int;

/// Set device memory to a value
pub extern "cudart" fn cudaMemset(ptr: *anyopaque, value: c_int, size: usize) c_int;

/// Asynchronous memset
pub extern "cudart" fn cudaMemsetAsync(ptr: *anyopaque, value: c_int, size: usize, stream: ?*anyopaque) c_int;

// ============================================================================
// Stream Management
// ============================================================================

/// Create a CUDA stream
pub extern "cudart" fn cudaStreamCreate(stream: **anyopaque) c_int;

/// Create a CUDA stream with flags
pub extern "cudart" fn cudaStreamCreateWithFlags(stream: **anyopaque, flags: c_uint) c_int;

/// Destroy a CUDA stream
pub extern "cudart" fn cudaStreamDestroy(stream: *anyopaque) c_int;

/// Synchronize a CUDA stream
pub extern "cudart" fn cudaStreamSynchronize(stream: *anyopaque) c_int;

/// Query stream status
pub extern "cudart" fn cudaStreamQuery(stream: *anyopaque) c_int;

// Stream flags
pub const cudaStreamDefault: c_uint = 0x00;
pub const cudaStreamNonBlocking: c_uint = 0x01;

/// Create stream with priority
pub extern "cudart" fn cudaStreamCreateWithPriority(stream: **anyopaque, flags: c_uint, priority: c_int) c_int;

/// Make stream wait for event
pub extern "cudart" fn cudaStreamWaitEvent(stream: *anyopaque, event: *anyopaque, flags: c_uint) c_int;

// Error code for operations not ready
pub const cudaErrorNotReady: c_int = 600;

// ============================================================================
// Event Management
// ============================================================================

/// Create a CUDA event
pub extern "cudart" fn cudaEventCreate(event: **anyopaque) c_int;

/// Create a CUDA event with flags
pub extern "cudart" fn cudaEventCreateWithFlags(event: **anyopaque, flags: c_uint) c_int;

/// Destroy a CUDA event
pub extern "cudart" fn cudaEventDestroy(event: *anyopaque) c_int;

/// Record an event in a stream
pub extern "cudart" fn cudaEventRecord(event: *anyopaque, stream: *anyopaque) c_int;

/// Synchronize on an event
pub extern "cudart" fn cudaEventSynchronize(event: *anyopaque) c_int;

/// Query event status
pub extern "cudart" fn cudaEventQuery(event: *anyopaque) c_int;

/// Calculate elapsed time between two events
pub extern "cudart" fn cudaEventElapsedTime(ms: *f32, start: *anyopaque, end: *anyopaque) c_int;

// Event flags
pub const cudaEventDefault: c_uint = 0x00;
pub const cudaEventBlockingSync: c_uint = 0x01;
pub const cudaEventDisableTiming: c_uint = 0x02;

// ============================================================================
// Error Handling
// ============================================================================

/// Get the last CUDA error
pub extern "cudart" fn cudaGetLastError() c_int;

/// Peek at the last CUDA error (doesn't clear it)
pub extern "cudart" fn cudaPeekAtLastError() c_int;

/// Get error string from error code
pub extern "cudart" fn cudaGetErrorString(err: c_int) [*:0]const u8;

/// Get error name from error code
pub extern "cudart" fn cudaGetErrorName(err: c_int) [*:0]const u8;

// ============================================================================
// Helper Functions
// ============================================================================

/// Check CUDA error and return Zig error if failed
pub fn checkCudaError(err: c_int, comptime context: []const u8) !void {
    if (err != cudaSuccess) {
        const err_str = cudaGetErrorString(err);
        const err_name = cudaGetErrorName(err);
        std.debug.print("CUDA Error in {s}: {s} ({s})\n", .{ context, err_str, err_name });
        return error.CudaError;
    }
}

/// Initialize CUDA and verify it's working
pub fn initCUDA() !void {
    var device_count: c_int = 0;
    try checkCudaError(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    
    if (device_count == 0) {
        std.debug.print("No CUDA devices found\n", .{});
        return error.NoGPUFound;
    }
    
    std.debug.print("Found {d} CUDA device(s)\n", .{device_count});
}

/// Get a human-readable name for compute capability
pub fn getComputeCapabilityName(major: i32, minor: i32) []const u8 {
    return switch (major) {
        7 => switch (minor) {
            0 => "Volta",
            2 => "Volta",
            5 => "Turing (T4)",
            else => "Volta/Turing",
        },
        8 => switch (minor) {
            0 => "Ampere (A100)",
            6 => "Ampere (A40/A10)",
            9 => "Ada Lovelace (L4/L40)",
            else => "Ampere/Ada",
        },
        9 => switch (minor) {
            0 => "Hopper (H100)",
            else => "Hopper",
        },
        else => "Unknown",
    };
}

// ============================================================================
// Tests
// ============================================================================

test "CUDA initialization" {
    // This test will fail if CUDA is not available, which is expected
    initCUDA() catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
}

test "CUDA device count" {
    var device_count: c_int = 0;
    const err = cudaGetDeviceCount(&device_count);
    
    if (err == cudaErrorNoDevice or err == cudaErrorInsufficientDriver) {
        std.debug.print("Test skipped: No CUDA device or driver\n", .{});
        return;
    }
    
    try checkCudaError(err, "cudaGetDeviceCount");
    std.debug.print("Device count: {d}\n", .{device_count});
}
