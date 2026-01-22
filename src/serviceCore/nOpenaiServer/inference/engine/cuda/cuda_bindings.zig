// CUDA Runtime API bindings
// FFI layer for interfacing with CUDA runtime library
//
// Links against: libcudart.so
// Requires: CUDA Toolkit 11.8+ or 12.x

const std = @import("std");

// Use @cImport to properly import CUDA headers
const c = @cImport({
    @cInclude("cuda_runtime_api.h");
});

// Re-export types and functions from C import
pub const cudaError_t = c.cudaError_t;

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
// Device Management - Using C import for proper linking
// ============================================================================

/// Get the number of CUDA-capable devices
pub fn cudaGetDeviceCount(count: *c_int) c_int {
    return @intCast(c.cudaGetDeviceCount(count));
}

/// Set the current CUDA device
pub fn cudaSetDevice(device: c_int) c_int {
    return @intCast(c.cudaSetDevice(device));
}

/// Get the current CUDA device
pub fn cudaGetDevice(device: *c_int) c_int {
    return @intCast(c.cudaGetDevice(device));
}

/// Reset the current CUDA device
pub fn cudaDeviceReset() c_int {
    return @intCast(c.cudaDeviceReset());
}

/// Synchronize the current device
pub fn cudaDeviceSynchronize() c_int {
    return @intCast(c.cudaDeviceSynchronize());
}

// ============================================================================
// Device Properties
// ============================================================================

/// Re-export the C cudaDeviceProp struct directly
pub const CudaDeviceProp = c.struct_cudaDeviceProp;

/// Get device properties
pub fn cudaGetDeviceProperties(prop: *CudaDeviceProp, device: c_int) c_int {
    return @intCast(c.cudaGetDeviceProperties(prop, device));
}

// ============================================================================
// Memory Management - Using C import wrappers
// ============================================================================

/// Allocate memory on the device
pub fn cudaMalloc(ptr: **anyopaque, size: usize) c_int {
    return @intCast(c.cudaMalloc(@ptrCast(ptr), size));
}

/// Free memory on the device
pub fn cudaFree(ptr: *anyopaque) c_int {
    return @intCast(c.cudaFree(ptr));
}

/// Allocate pinned host memory (page-locked)
pub fn cudaMallocHost(ptr: **anyopaque, size: usize) c_int {
    return @intCast(c.cudaMallocHost(@ptrCast(ptr), size));
}

/// Free pinned host memory
pub fn cudaFreeHost(ptr: *anyopaque) c_int {
    return @intCast(c.cudaFreeHost(ptr));
}

/// Get free and total device memory
pub fn cudaMemGetInfo(free_mem: *usize, total: *usize) c_int {
    return @intCast(c.cudaMemGetInfo(free_mem, total));
}

/// Copy memory between host and device
pub fn cudaMemcpy(dst: *anyopaque, src: *const anyopaque, size: usize, kind: c_int) c_int {
    return @intCast(c.cudaMemcpy(dst, @constCast(src), size, @intCast(kind)));
}

/// Asynchronous memory copy
pub fn cudaMemcpyAsync(
    dst: *anyopaque,
    src: *const anyopaque,
    size: usize,
    kind: c_int,
    stream: ?*anyopaque,
) c_int {
    return @intCast(c.cudaMemcpyAsync(dst, @constCast(src), size, @intCast(kind), @ptrCast(stream)));
}

/// Set device memory to a value
pub fn cudaMemset(ptr: *anyopaque, value: c_int, size: usize) c_int {
    return @intCast(c.cudaMemset(ptr, value, size));
}

/// Asynchronous memset
pub fn cudaMemsetAsync(ptr: *anyopaque, value: c_int, size: usize, stream: ?*anyopaque) c_int {
    return @intCast(c.cudaMemsetAsync(ptr, value, size, @ptrCast(stream)));
}

// ============================================================================
// Stream Management - Using C import wrappers
// ============================================================================

/// Create a CUDA stream
pub fn cudaStreamCreate(stream: **anyopaque) c_int {
    return @intCast(c.cudaStreamCreate(@ptrCast(stream)));
}

/// Create a CUDA stream with flags
pub fn cudaStreamCreateWithFlags(stream: **anyopaque, flags: c_uint) c_int {
    return @intCast(c.cudaStreamCreateWithFlags(@ptrCast(stream), flags));
}

/// Destroy a CUDA stream
pub fn cudaStreamDestroy(stream: *anyopaque) c_int {
    return @intCast(c.cudaStreamDestroy(@ptrCast(stream)));
}

/// Synchronize a CUDA stream
pub fn cudaStreamSynchronize(stream: *anyopaque) c_int {
    return @intCast(c.cudaStreamSynchronize(@ptrCast(stream)));
}

/// Query stream status
pub fn cudaStreamQuery(stream: *anyopaque) c_int {
    return @intCast(c.cudaStreamQuery(@ptrCast(stream)));
}

// Stream flags
pub const cudaStreamDefault: c_uint = 0x00;
pub const cudaStreamNonBlocking: c_uint = 0x01;

/// Create stream with priority
pub fn cudaStreamCreateWithPriority(stream: **anyopaque, flags: c_uint, priority: c_int) c_int {
    return @intCast(c.cudaStreamCreateWithPriority(@ptrCast(stream), flags, priority));
}

/// Make stream wait for event
pub fn cudaStreamWaitEvent(stream: *anyopaque, event: *anyopaque, flags: c_uint) c_int {
    return @intCast(c.cudaStreamWaitEvent(@ptrCast(stream), @ptrCast(event), flags));
}

// Error code for operations not ready
pub const cudaErrorNotReady: c_int = 600;

// ============================================================================
// Event Management - Using C import wrappers
// ============================================================================

/// Create a CUDA event
pub fn cudaEventCreate(event: **anyopaque) c_int {
    return @intCast(c.cudaEventCreate(@ptrCast(event)));
}

/// Create a CUDA event with flags
pub fn cudaEventCreateWithFlags(event: **anyopaque, flags: c_uint) c_int {
    return @intCast(c.cudaEventCreateWithFlags(@ptrCast(event), flags));
}

/// Destroy a CUDA event
pub fn cudaEventDestroy(event: *anyopaque) c_int {
    return @intCast(c.cudaEventDestroy(@ptrCast(event)));
}

/// Record an event in a stream
pub fn cudaEventRecord(event: *anyopaque, stream: *anyopaque) c_int {
    return @intCast(c.cudaEventRecord(@ptrCast(event), @ptrCast(stream)));
}

/// Synchronize on an event
pub fn cudaEventSynchronize(event: *anyopaque) c_int {
    return @intCast(c.cudaEventSynchronize(@ptrCast(event)));
}

/// Query event status
pub fn cudaEventQuery(event: *anyopaque) c_int {
    return @intCast(c.cudaEventQuery(@ptrCast(event)));
}

/// Calculate elapsed time between two events
pub fn cudaEventElapsedTime(ms: *f32, start_event: *anyopaque, end_event: *anyopaque) c_int {
    return @intCast(c.cudaEventElapsedTime(ms, @ptrCast(start_event), @ptrCast(end_event)));
}

// Event flags
pub const cudaEventDefault: c_uint = 0x00;
pub const cudaEventBlockingSync: c_uint = 0x01;
pub const cudaEventDisableTiming: c_uint = 0x02;

// ============================================================================
// CUDA Graph API - For capturing and replaying kernel sequences
// ============================================================================

/// Opaque graph types
pub const cudaGraph_t = *anyopaque;
pub const cudaGraphExec_t = *anyopaque;

/// Stream capture modes
pub const cudaStreamCaptureMode = c_int;
pub const cudaStreamCaptureModeGlobal: cudaStreamCaptureMode = 0;
pub const cudaStreamCaptureModeThreadLocal: cudaStreamCaptureMode = 1;
pub const cudaStreamCaptureModeRelaxed: cudaStreamCaptureMode = 2;

/// Begin stream capture
pub extern "cudart" fn cudaStreamBeginCapture(stream: *anyopaque, mode: cudaStreamCaptureMode) c_int;

/// End stream capture and get graph
pub extern "cudart" fn cudaStreamEndCapture(stream: *anyopaque, graph: *cudaGraph_t) c_int;

/// Instantiate graph into executable
pub extern "cudart" fn cudaGraphInstantiate(graph_exec: *cudaGraphExec_t, graph: cudaGraph_t, error_node: ?*anyopaque, log_buffer: ?[*]u8, buffer_size: usize) c_int;

/// Launch graph
pub extern "cudart" fn cudaGraphLaunch(graph_exec: cudaGraphExec_t, stream: *anyopaque) c_int;

/// Destroy graph
pub extern "cudart" fn cudaGraphDestroy(graph: cudaGraph_t) c_int;

/// Destroy executable graph
pub extern "cudart" fn cudaGraphExecDestroy(graph_exec: cudaGraphExec_t) c_int;

// ============================================================================
// Error Handling - Using C import wrappers
// ============================================================================

/// Get the last CUDA error
pub fn cudaGetLastError() c_int {
    return @intCast(c.cudaGetLastError());
}

/// Peek at the last CUDA error (doesn't clear it)
pub fn cudaPeekAtLastError() c_int {
    return @intCast(c.cudaPeekAtLastError());
}

/// Get error string from error code
pub fn cudaGetErrorString(err: c_int) [*:0]const u8 {
    return c.cudaGetErrorString(@intCast(err));
}

/// Get error name from error code
pub fn cudaGetErrorName(err: c_int) [*:0]const u8 {
    return c.cudaGetErrorName(@intCast(err));
}

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
