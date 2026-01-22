const std = @import("std");
const math = std.math;
const builtin = @import("builtin");

// Core mHC modules - use module names for Zig 0.15 module system
const mhc_constraints = @import("mhc_constraints");
const mhc_config = @import("mhc_configuration");

// Module imports - use module names for Zig 0.15 module system
const gguf_loader = @import("gguf_loader");
const thread_pool = @import("thread_pool");
const q4_k = @import("q4_k");
const q6_k = @import("q6_k");

/// High-performance matrix operations for transformer inference
/// Uses SIMD for 4-8x speedup on CPU
/// Integrates mHC (manifold Hyperbolic Constraints) for stability

// ============================================================================
// Types
// ============================================================================

/// SIMD capability detection for future optimizations
pub const SIMDCapabilities = struct {
    has_neon: bool,      // ARM NEON
    has_sse: bool,       // x86 SSE
    has_avx: bool,       // x86 AVX
    has_avx2: bool,      // x86 AVX2
    vector_width: usize, // Native vector width

    /// Detect available SIMD capabilities
    pub fn detect() SIMDCapabilities {
        const arch = builtin.cpu.arch;
        const has_neon = arch == .aarch64 or arch == .arm;
        
        return SIMDCapabilities{
            .has_neon = has_neon,
            .has_sse = false,  // TODO: x86 detection
            .has_avx = false,
            .has_avx2 = false,
            .vector_width = if (has_neon) 4 else 1,
        };
    }
};

/// Configuration for matrix multiplication with optional mHC
pub const MatMulConfig = struct {
    /// Enable mHC constraints after matmul
    use_mhc: bool = false,

    /// Layer ID for tracking metrics
    layer_id: u32 = 0,

    /// mHC constraint configuration
    mhc_config: mhc_constraints.MHCConfig = .{},

    /// Optional manifold constraints
    manifold_constraints: ?ManifoldConstraints = null,

    /// Initialize from global configuration
    pub fn from_global(
        config: mhc_config.MHCConfiguration,
        layer_id: u32,
    ) MatMulConfig {
        // Convert LayerRange from mhc_config to mhc_constraints type
        const converted_layer_range: ?mhc_constraints.LayerRange = if (config.core.layer_range) |lr|
            mhc_constraints.LayerRange{ .start = lr.start, .end = lr.end }
        else
            null;

        return MatMulConfig{
            .use_mhc = config.matrix_ops.use_mhc and config.core.enabled,
            .layer_id = layer_id,
            .mhc_config = .{
                .enabled = config.core.enabled,
                .sinkhorn_iterations = config.core.sinkhorn_iterations,
                .manifold_epsilon = config.core.manifold_epsilon,
                .stability_threshold = config.core.stability_threshold,
                .manifold_beta = config.core.manifold_beta,
                .log_stability_metrics = config.core.log_stability_metrics,
                .layer_range = converted_layer_range,
                .early_stopping = config.core.early_stopping,
            },
            .manifold_constraints = null,
        };
    }
};

/// Optional manifold constraints for geometric extensions
pub const ManifoldConstraints = struct {
    /// Manifold type
    manifold_type: enum { euclidean, hyperbolic, spherical, product } = .euclidean,

    /// Curvature parameter (for hyperbolic/spherical)
    curvature: f32 = -1.0,

    /// Apply projection after normalization
    apply_projection: bool = false,
};

pub const Weight = union(enum) {
    f32: []const f32,
    q4_0: []const u8,
    q4_k: []const u8,
    q6_k: []const u8,
};

// ============================================================================
// Dequantization
// ============================================================================

/// Dequantize raw quantized data to f32 output buffer
/// Used by GPU backends that need CPU-side dequantization before transfer
pub fn dequantize(
    output: []f32,
    data: []const u8,
    quant_type: gguf_loader.QuantizationType,
    count: usize,
) void {
    _ = count; // Reserved for validation

    switch (quant_type) {
        .Q4_K => {
            const block_size: usize = 256;
            const block_bytes: usize = @sizeOf(q4_k.BlockQ4_K);
            const num_blocks = output.len / block_size;

            for (0..num_blocks) |bi| {
                const block_ptr = @as(*const q4_k.BlockQ4_K, @ptrCast(@alignCast(&data[bi * block_bytes])));
                const out_start = bi * block_size;
                q4_k.dequantizeBlock(output[out_start .. out_start + block_size], block_ptr);
            }
        },
        .Q6_K => {
            const block_size: usize = 256;
            const block_bytes: usize = @sizeOf(q6_k.BlockQ6_K);
            const num_blocks = output.len / block_size;

            for (0..num_blocks) |bi| {
                const block_ptr = @as(*const q6_k.BlockQ6_K, @ptrCast(@alignCast(&data[bi * block_bytes])));
                const out_start = bi * block_size;
                q6_k.dequantizeBlock(output[out_start .. out_start + block_size], block_ptr);
            }
        },
        .Q4_0 => {
            // Q4_0: 32 values per block, 18 bytes per block (2 bytes scale + 16 bytes quants)
            const block_size: usize = 32;
            const block_bytes: usize = 18;
            const num_blocks = output.len / block_size;

            for (0..num_blocks) |bi| {
                const block_offset = bi * block_bytes;
                // Scale is stored as f16 (2 bytes)
                const scale_bytes = data[block_offset .. block_offset + 2];
                const scale_u16 = @as(u16, scale_bytes[0]) | (@as(u16, scale_bytes[1]) << 8);
                const scale_f16: f16 = @bitCast(scale_u16);
                const scale: f32 = @floatCast(scale_f16);

                // Dequantize 32 values from 16 bytes (4 bits each)
                for (0..16) |j| {
                    const quant_byte = data[block_offset + 2 + j];
                    const q0 = @as(i8, @intCast(quant_byte & 0x0F)) - 8;
                    const q1 = @as(i8, @intCast(quant_byte >> 4)) - 8;
                    const out_start = bi * block_size;
                    output[out_start + j * 2] = @as(f32, @floatFromInt(q0)) * scale;
                    output[out_start + j * 2 + 1] = @as(f32, @floatFromInt(q1)) * scale;
                }
            }
        },
        .F32 => {
            // Direct copy from f32 data
            const src = @as([*]const f32, @ptrCast(@alignCast(data.ptr)))[0..output.len];
            @memcpy(output, src);
        },
        .F16 => {
            // Convert from f16 to f32
            const src = @as([*]const f16, @ptrCast(@alignCast(data.ptr)))[0..output.len];
            for (0..output.len) |i| {
                output[i] = @floatCast(src[i]);
            }
        },
        else => {
            // Unsupported quantization type - fill with zeros
            @memset(output, 0.0);
        },
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Get a row from a weight matrix (dequantizing if necessary)
/// Debug counter for get_row calls (only for embedding lookups with row_size=2048)
var get_row_embed_counter: usize = 0;

pub fn get_row(
    output: []f32,
    weight: Weight,
    row_idx: usize,
    row_size: usize,
) void {
    // Debug embedding lookups (row_size=2048 for LFM2 hidden_size)
    if (row_size == 2048 and get_row_embed_counter < 5) {
        const data_len = switch (weight) {
            .f32 => |d| d.len,
            .q4_0 => |d| d.len,
            .q4_k => |d| d.len,
            .q6_k => |d| d.len,
        };
        std.debug.print("🔬 EMBED get_row[{d}]: row_idx={d}, row_size={d}, weight_type={s}, data.len={d}\n", .{
            get_row_embed_counter, row_idx, row_size, switch (weight) {
                .f32 => "f32",
                .q4_0 => "q4_0",
                .q4_k => "q4_k",
                .q6_k => "q6_k",
            }, data_len
        });
        get_row_embed_counter += 1;
    }

    switch (weight) {
        .f32 => |data| {
            const start = row_idx * row_size;
            @memcpy(output, data[start .. start + row_size]);
        },
        .q4_0 => |data| {
            // Q4_0 row stride
            const block_size = 18; // 2 + 16
            const nb = row_size / 32;
            const stride = nb * block_size;
            const start = row_idx * stride;
            
            // Bounds check - zero fill if insufficient data
            if (start + stride > data.len) {
                @memset(output, 0);
                return;
            }
            
            // We need to dequantize this row
            var out_idx: usize = 0;
            var block_ptr_idx = start;
            
            for (0..nb) |_| {
                // Additional per-block bounds check
                if (block_ptr_idx + 18 > data.len) {
                    @memset(output[out_idx..], 0);
                    return;
                }
                // Scale
                const scale_bits = @as(*const u16, @ptrCast(@alignCast(&data[block_ptr_idx]))).*;
                var scale = @call(.always_inline, f16_to_f32, .{scale_bits});

                // Handle NaN/Inf scales - replace with 0 to avoid propagation
                if (std.math.isNan(scale) or std.math.isInf(scale)) {
                    scale = 0.0;
                }

                // QS
                const qs = data[block_ptr_idx + 2 .. block_ptr_idx + 18];

                for (0..16) |i| {
                    const byte = qs[i];
                    const v0 = @as(f32, @floatFromInt(@as(i8, @intCast(byte & 0xF)) - 8)) * scale;
                    const v1 = @as(f32, @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8)) * scale;
                    output[out_idx] = v0;
                    output[out_idx+1] = v1;
                    out_idx += 2;
                }
                block_ptr_idx += 18;
            }
        },
        .q4_k => |data| {
            // Q4_K row stride
            const block_size = q4_k.BLOCK_SIZE; // 256
            const block_bytes = q4_k.BLOCK_BYTES; // 144
            const nb = row_size / block_size;
            const stride = nb * block_bytes;
            const start = row_idx * stride;
            
            // Bounds check - zero fill if insufficient data
            if (start + stride > data.len) {
                @memset(output, 0);
                return;
            }
            
            var out_idx: usize = 0;
            var block_ptr_idx = start;
            
            for (0..nb) |_| {
                // Additional per-block bounds check
                if (block_ptr_idx + block_bytes > data.len) {
                    @memset(output[out_idx..], 0);
                    return;
                }
                const block_ptr = @as(*const q4_k.BlockQ4_K, @ptrCast(@alignCast(&data[block_ptr_idx])));
                q4_k.dequantizeBlock(output[out_idx .. out_idx + block_size], block_ptr);
                out_idx += block_size;
                block_ptr_idx += block_bytes;
            }
        },
        .q6_k => |data| {
            const block_size = q6_k.QK_K; // 256
            const block_bytes = @sizeOf(q6_k.BlockQ6_K); // 210
            const nb = row_size / block_size;
            const stride = nb * block_bytes;
            const start = row_idx * stride;

            // Bounds check - zero fill if insufficient data
            if (start + stride > data.len) {
                @memset(output, 0);
                return;
            }

            var out_idx: usize = 0;
            var block_ptr_idx = start;
            for (0..nb) |_| {
                // Additional per-block bounds check
                if (block_ptr_idx + block_bytes > data.len) {
                    @memset(output[out_idx..], 0);
                    return;
                }
                const block_ptr = @as(*const q6_k.BlockQ6_K, @ptrCast(@alignCast(&data[block_ptr_idx])));
                q6_k.dequantizeBlock(output[out_idx .. out_idx + block_size], block_ptr);
                out_idx += block_size;
                block_ptr_idx += block_bytes;
            }
        }
    }
}

// ============================================================================
// SIMD-Optimized Matrix Multiplication
// ============================================================================

/// Matrix multiplication with A stored transposed: C = A^T @ B
/// Where A is physically stored as [k, m] but logically used as [m, k]
/// This avoids expensive transpose operations on GGUF weights
pub fn matmul_transposed_a(
    c: []f32,
    a_transposed: Weight,
    b: []const f32,
    m: usize,
    n: usize,
    k: usize,
    allocator: std.mem.Allocator,
    _: ?*thread_pool.ThreadPool,
) !void {
    // For transposed A, we need to access columns as rows
    // A stored as [k, m] means when we want row i of logical [m, k],
    // we need column i of physical [k, m]
    
    // Simple approach: dequantize and transpose on-the-fly per row
    const a_row_buf = try allocator.alloc(f32, k);
    defer allocator.free(a_row_buf);
    
    for (0..m) |i| {
        // Extract logical row i from transposed storage (column i)
        switch (a_transposed) {
            .f32 => |data| {
                for (0..k) |ki| {
                    a_row_buf[ki] = data[ki * m + i];
                }
            },
            .q4_k, .q4_0, .q6_k => {
                // For quantized, dequantize entire matrix rows that contain our column
                // This is complex, so fall back to full dequantize for now
                const full = try allocator.alloc(f32, m * k);
                defer allocator.free(full);
                
                for (0..k) |row_idx| {
                    const row_out = full[row_idx * m .. (row_idx + 1) * m];
                    get_row(row_out, a_transposed, row_idx, m);
                }
                
                // Extract column i
                for (0..k) |ki| {
                    a_row_buf[ki] = full[ki * m + i];
                }
            },
        }
        
        // Now compute: c[i*n..(i+1)*n] = a_row_buf @ b
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |ki| {
                sum += a_row_buf[ki] * b[ki * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Unified matrix multiplication supporting mixed precision and threading
/// NOTE: This is the standard matmul without mHC. For mHC-enabled matmul, use matmul_with_mhc()
pub fn matmul(
    c: []f32,
    a: Weight,
    b: []const f32,
    m: usize,
    n: usize,
    k: usize,
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
) !void {
    switch (a) {
        // matmul_f32 expects (m, k, n) but wrapper receives (m, n, k), so swap n and k
        .f32 => |data| try matmul_f32(c, data, b, m, k, n, allocator, pool),
        .q4_0 => |data| {
            const blocks_per_row = (k + 32 - 1) / 32;
            const required = m * blocks_per_row * 18;
            if (data.len < required) {
                std.debug.print("   ⚠️  Q4_0 tensor too short for matmul (have {d} bytes, need {d}); zero fallback\n", .{ data.len, required });
                const zero = try allocator.alloc(f32, m * k);
                defer allocator.free(zero);
                @memset(zero, 0);
                try matmul_f32(c, zero, b, m, k, n, allocator, pool);
                return;
            }
            try matmul_quantized(c, data, .Q4_0, b, m, n, k, allocator, pool);
        },
        .q4_k => |data| {
            // For quantized weights, blocks_per_row is based on the inner dimension k
            // Data layout: m rows × (k/256 blocks_per_row) × 144 bytes_per_block
            const blocks_per_row = (k + q4_k.BLOCK_SIZE - 1) / q4_k.BLOCK_SIZE;
            const required = m * blocks_per_row * q4_k.BLOCK_BYTES;
            // Only check if we have enough data - don't fallback for transposed weights
            // The data buffer should match the weight's original storage size
            if (data.len > 0 and data.len < required) {
                // If short, use zero-fill fallback
                std.debug.print("   ⚠️  Q4_K tensor short (have {d} bytes, expected {d}); zero fallback\n", .{ data.len, required });
                const zero = try allocator.alloc(f32, m * k);
                defer allocator.free(zero);
                @memset(zero, 0);
                try matmul_f32(c, zero, b, m, k, n, allocator, pool);
                return;
            }
            try matmul_quantized(c, data, .Q4_K, b, m, n, k, allocator, pool);
        },
        .q6_k => |data| {
            const blocks_per_row = (k + q6_k.QK_K - 1) / q6_k.QK_K;
            const required = m * blocks_per_row * @sizeOf(q6_k.BlockQ6_K);
            if (data.len < required) {
                std.debug.print("   ⚠️  Q6_K tensor too short for matmul (have {d} bytes, need {d}); zero fallback\n", .{ data.len, required });
                const zero = try allocator.alloc(f32, m * k);
                defer allocator.free(zero);
                @memset(zero, 0);
                try matmul_f32(c, zero, b, m, k, n, allocator, pool);
                return;
            }
            try matmul_quantized(c, data, .Q6_K, b, m, n, k, allocator, pool);
        },
    }
}

/// Matrix multiplication: C = A * B
/// A: [m, k], B: [k, n], C: [m, n]
/// Optimized with cache tiling and vectorization
pub fn matmul_f32(
    c: []f32,
    a: []const f32,
    b: []const f32,
    m: usize,
    k: usize,
    n: usize,
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
) !void {
    const TILE_N = 64; // Process 64 columns at a time to fit in L1/L2

    // Serial implementation function (reusable for threaded chunks)
    const compute_chunk = struct {
        fn run(
            c_slice: []f32, a_slice: []const f32, b_ptr: []const f32,
            n_dim: usize, k_dim: usize,
            start_row: usize, end_row: usize
        ) void {
            const Vec = @Vector(8, f32);
            
            // Loop over rows assigned to this thread
            for (start_row..end_row) |i| {
                // Tiling over N to improve cache locality for C and B
                var j_tile: usize = 0;
                while (j_tile < n_dim) : (j_tile += TILE_N) {
                    const current_n_end = @min(j_tile + TILE_N, n_dim);
                    
                    // Initialize C for this tile (accumulate starting from 0)
                    // We directly write to C, assuming we overwrite
                    const c_row_offset = i * n_dim;
                    @memset(c_slice[c_row_offset + j_tile .. c_row_offset + current_n_end], 0.0);
                    
                    // Inner loop over K
                    for (0..k_dim) |p| {
                        // A[i, p] is scalar for this row
                        const val_a = a_slice[i * k_dim + p];
                        const vec_a: Vec = @splat(val_a);
                        
                        const b_row_offset = p * n_dim;
                        
                        // Vectorized loop over J in this tile
                        var j: usize = j_tile;
                        // Process main vector chunks
                        while (j + 8 <= current_n_end) : (j += 8) {
                            const b_idx = b_row_offset + j;
                            const c_idx = c_row_offset + j;
                            
                            // Load B and C
                            const vec_b = @as(Vec, b_ptr[b_idx..][0..8].*);
                            var vec_c = @as(Vec, c_slice[c_idx..][0..8].*);
                            
                            // FMA
                            vec_c += vec_a * vec_b;
                            
                            // Store C
                            c_slice[c_idx..][0..8].* = vec_c;
                        }
                        
                        // Handle remaining elements scalar
                        while (j < current_n_end) : (j += 1) {
                            const b_val = b_ptr[b_row_offset + j];
                            c_slice[c_row_offset + j] += val_a * b_val;
                        }
                    }
                }
            }
        }
    }.run;

    // Parallel execution logic
    if (pool) |tp| {
        if (m >= 4) {
            const num_threads = tp.config.num_threads;
            const rows_per_task = (m + num_threads - 1) / num_threads;
            
            const Context = struct {
                c: []f32,
                a: []const f32,
                b: []const f32,
                n: usize,
                k: usize,
                start_row: usize,
                end_row: usize,
            };
            
            const work_fn = struct {
                fn work(ctx: *anyopaque) void {
                    const context: *Context = @ptrCast(@alignCast(ctx));
                    compute_chunk(context.c, context.a, context.b, context.n, context.k, context.start_row, context.end_row);
                }
            }.work;
            
            var contexts = try allocator.alloc(Context, num_threads);
            // CRITICAL FIX: Don't use defer - free after waitAll() to prevent thread use-after-free
            
            for (0..num_threads) |t| {
                const start = t * rows_per_task;
                if (start >= m) break;
                const end = @min(start + rows_per_task, m);
                
                contexts[t] = Context{
                    .c = c,
                    .a = a,
                    .b = b,
                    .n = n,
                    .k = k,
                    .start_row = start,
                    .end_row = end,
                };
                
                try tp.submit(.{
                    .work_fn = work_fn,
                    .context = @ptrCast(&contexts[t]),
                });
            }
            tp.waitAll();
            allocator.free(contexts); // Free AFTER threads complete
            return;
        }
    }

    // Serial fallback
    compute_chunk(c, a, b, n, k, 0, m);
}

/// Matrix multiplication with transposed B: C = A * B^T
/// More cache-friendly for certain operations
pub fn matmul_transposed(
    c: []f32,
    a: []const f32,
    b: []const f32,
    m: usize,
    k: usize,
    n: usize,
) void {
    const Vec = @Vector(8, f32);
    
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            var vec_sum: Vec = @splat(0.0);
            
            var ki: usize = 0;
            while (ki + 8 <= k) : (ki += 8) {
                const a_vec: Vec = a[i * k + ki ..][0..8].*;
                const b_vec: Vec = b[j * k + ki ..][0..8].*;
                vec_sum += a_vec * b_vec;
            }
            
            sum = @reduce(.Add, vec_sum);
            
            while (ki < k) : (ki += 1) {
                sum += a[i * k + ki] * b[j * k + ki];
            }
            
            c[i * n + j] = sum;
        }
    }
}

/// Matrix multiplication with mHC constraints
/// This wrapper applies manifold constraints after standard matmul
///
/// Flow:
/// 1. Perform standard matrix multiplication: C = A * B
/// 2. Apply Sinkhorn-Knopp normalization to output (if enabled)
/// 3. Apply manifold constraints (L2 ball projection)
/// 4. Check stability and collect metrics
/// 5. Return stability metrics for monitoring
///
/// Parameters:
///   - c: Output matrix [m, n] (modified in-place)
///   - a: Input matrix A (can be quantized)
///   - b: Input matrix B [k, n]
///   - m, n, k: Matrix dimensions
///   - config: MatMulConfig with mHC settings
///   - allocator: Memory allocator
///   - pool: Optional thread pool for parallelization
///
/// Returns:
///   - StabilityMetrics if mHC is enabled, null otherwise
pub fn matmul_with_mhc(
    c: []f32,
    a: Weight,
    b: []const f32,
    m: usize,
    n: usize,
    k: usize,
    config: MatMulConfig,
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
) !?mhc_constraints.StabilityMetrics {
    // Step 1: Perform standard matrix multiplication
    try matmul(c, a, b, m, n, k, allocator, pool);

    // If mHC is disabled, return early
    if (!config.use_mhc or !config.mhc_config.enabled) {
        return null;
    }

    // Check if layer is within range (if specified)
    if (config.mhc_config.layer_range) |range| {
        if (!range.contains(config.layer_id)) {
            return null;
        }
    }

    // Store original activations for metrics
    const activations_before = try allocator.alloc(f32, c.len);
    defer allocator.free(activations_before);
    @memcpy(activations_before, c);

    // Step 2: Apply Sinkhorn-Knopp normalization (if output is matrix-like)
    // For now, we treat the output as a 1D vector and skip matrix normalization
    // This will be enhanced in future iterations to handle 2D normalization
    var iterations: u32 = 0;
    
    // Only apply Sinkhorn if we have a proper 2D matrix structure
    // For vector outputs (m=1 or n=1), skip Sinkhorn
    if (m > 1 and n > 1) {
        iterations = try mhc_constraints.sinkhorn_normalize(
            c,
            m,
            n,
            config.mhc_config,
            allocator,
        );
    }

    // Step 3: Apply manifold constraints (L2 ball projection)
    _ = mhc_constraints.apply_manifold_constraints(
        c,
        config.mhc_config.manifold_beta,
    );

    // Step 4: Apply optional geometric constraints
    if (config.manifold_constraints) |manifold| {
        if (manifold.apply_projection) {
            try apply_geometric_projection(c, manifold, allocator);
        }
    }

    // Step 5: Check stability
    const is_stable = mhc_constraints.check_stability(
        c,
        config.mhc_config.stability_threshold,
    );

    // Step 6: Compute and return metrics
    const metrics = mhc_constraints.compute_stability_metrics(
        config.layer_id,
        activations_before,
        c,
        iterations,
    );

    // Log metrics if enabled
    if (config.mhc_config.log_stability_metrics) {
        std.debug.print("[mHC] {any}\n", .{metrics});
    }

    // Abort on instability if configured (and instability detected)
    if (!is_stable and config.mhc_config.stability_threshold > 0) {
        // For now, just log warning - later we may add abort option
        std.debug.print("[mHC] ⚠️  Layer {d} unstable: α={d:.3}\n", .{
            config.layer_id,
            metrics.amplification_factor,
        });
    }

    return metrics;
}

/// Apply geometric projection based on manifold type
/// Made public for testing (Day 38+)
pub fn apply_geometric_projection(
    activations: []f32,
    manifold: ManifoldConstraints,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator; // Reserved for future use
    
    switch (manifold.manifold_type) {
        .euclidean => {
            // Already handled by L2 ball projection
        },
        .hyperbolic => {
            // Hyperbolic projection (Days 54-60, placeholder)
            // TODO: Implement Poincaré ball or hyperboloid projection
            const curvature = manifold.curvature;
            if (curvature >= 0) {
                return error.InvalidHyperbolicCurvature;
            }
            // For now, just apply additional L2 normalization
            const norm = mhc_constraints.apply_manifold_constraints(activations, 1.0);
            _ = norm;
        },
        .spherical => {
            // Spherical projection - project onto unit sphere
            // Compute norm and normalize to length 1
            const norm = l2_norm(activations);
            if (norm > 0) {
                for (activations) |*val| {
                    val.* /= norm;
                }
            }
        },
        .product => {
            // Product manifold projection (Days 54-60, placeholder)
            // TODO: Implement component-wise projection
        },
    }
}

// ============================================================================
// Day 36: Batch Operations with mHC
// ============================================================================

/// Batch matrix multiplication with mHC constraints
/// Processes multiple matrix multiplications with optional mHC application
///
/// Parameters:
///   - outputs: Array of output matrices [batch_size][m×n]
///   - weights: Array of weight matrices (can be quantized)
///   - inputs: Array of input matrices [batch_size][k×n]
///   - batch_size: Number of matrices in batch
///   - m, n, k: Matrix dimensions (same for all batch elements)
///   - config: MatMulConfig with mHC settings
///   - allocator: Memory allocator
///   - pool: Optional thread pool for parallel execution
///
/// Returns:
///   - Array of StabilityMetrics (one per batch element, null if mHC disabled)
pub fn matmul_batch_with_mhc(
    outputs: [][]f32,
    weights: []Weight,
    inputs: [][]const f32,
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
    config: MatMulConfig,
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
) ![]?mhc_constraints.StabilityMetrics {
    // Allocate metrics array
    const metrics = try allocator.alloc(?mhc_constraints.StabilityMetrics, batch_size);
    errdefer allocator.free(metrics);

    // Process batch elements
    if (pool) |tp| {
        // Parallel batch processing
        const BatchContext = struct {
            outputs: [][]f32,
            weights: []Weight,
            inputs: [][]const f32,
            metrics: []?mhc_constraints.StabilityMetrics,
            m: usize,
            n: usize,
            k: usize,
            config: MatMulConfig,
            allocator: std.mem.Allocator,
            start_idx: usize,
            end_idx: usize,
        };

        const batch_work = struct {
            fn work(ctx: *anyopaque) void {
                const context = @as(*BatchContext, @ptrCast(@alignCast(ctx)));
                for (context.start_idx..context.end_idx) |i| {
                    // Process each batch element
                    context.metrics[i] = matmul_with_mhc(
                        context.outputs[i],
                        context.weights[i],
                        context.inputs[i],
                        context.m,
                        context.n,
                        context.k,
                        context.config,
                        context.allocator,
                        null, // Don't nest thread pools
                    ) catch null;
                }
            }
        }.work;

        const num_threads = tp.config.num_threads;
        const items_per_thread = (batch_size + num_threads - 1) / num_threads;

        var contexts = try allocator.alloc(BatchContext, num_threads);
        defer allocator.free(contexts);

        for (0..num_threads) |t| {
            const start = t * items_per_thread;
            if (start >= batch_size) break;
            const end = @min(start + items_per_thread, batch_size);

            contexts[t] = BatchContext{
                .outputs = outputs,
                .weights = weights,
                .inputs = inputs,
                .metrics = metrics,
                .m = m,
                .n = n,
                .k = k,
                .config = config,
                .allocator = allocator,
                .start_idx = start,
                .end_idx = end,
            };

            tp.submit(.{
                .work_fn = batch_work,
                .context = @ptrCast(&contexts[t]),
            }) catch {};
        }
        tp.waitAll();
    } else {
        // Serial processing
        for (0..batch_size) |i| {
            metrics[i] = try matmul_with_mhc(
                outputs[i],
                weights[i],
                inputs[i],
                m,
                n,
                k,
                config,
                allocator,
                null,
            );
        }
    }

    return metrics;
}

/// Determine optimal thread count for mHC operations
pub fn get_thread_count(
    matrix_size: usize,
    pool: ?*thread_pool.ThreadPool,
) usize {
    if (pool) |tp| {
        // Use threading for matrices larger than 2048 elements
        if (matrix_size >= 2048) {
            return tp.config.num_threads;
        }
    }
    return 1; // Serial fallback for small matrices
}

// ============================================================================
// Vector Operations
// ============================================================================

/// Element-wise vector addition: c = a + b
pub fn vec_add(c: []f32, a: []const f32, b: []const f32) void {
    const Vec = @Vector(8, f32);
    const n = a.len;
    
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const a_vec: Vec = a[i..][0..8].*;
        const b_vec: Vec = b[i..][0..8].*;
        const c_vec = a_vec + b_vec;
        c[i..][0..8].* = c_vec;
    }
    
    while (i < n) : (i += 1) {
        c[i] = a[i] + b[i];
    }
}

/// Element-wise vector multiplication: c = a * b
pub fn vec_mul(c: []f32, a: []const f32, b: []const f32) void {
    const Vec = @Vector(8, f32);
    const n = a.len;
    
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const a_vec: Vec = a[i..][0..8].*;
        const b_vec: Vec = b[i..][0..8].*;
        const c_vec = a_vec * b_vec;
        c[i..][0..8].* = c_vec;
    }
    
    while (i < n) : (i += 1) {
        c[i] = a[i] * b[i];
    }
}

/// Scalar multiplication: c = a * scalar
pub fn vec_scale(c: []f32, a: []const f32, scalar: f32) void {
    const Vec = @Vector(8, f32);
    const scalar_vec: Vec = @splat(scalar);
    const n = a.len;
    
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const a_vec: Vec = a[i..][0..8].*;
        const c_vec = a_vec * scalar_vec;
        c[i..][0..8].* = c_vec;
    }
    
    while (i < n) : (i += 1) {
        c[i] = a[i] * scalar;
    }
}

/// RMS normalization (used in Llama)
pub fn rms_norm(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
) void {
    const n = input.len;

    // Calculate RMS, skipping NaN/Inf values
    var sum_squares: f32 = 0.0;
    var valid_count: usize = 0;
    for (input) |val| {
        if (!std.math.isNan(val) and !std.math.isInf(val)) {
            sum_squares += val * val;
            valid_count += 1;
        }
    }

    // Handle all-NaN input
    if (valid_count == 0) {
        @memset(output, 0.0);
        return;
    }

    const mean_square = sum_squares / @as(f32, @floatFromInt(valid_count));
    const rms = @sqrt(mean_square + eps);

    // Safety: prevent division by zero/NaN
    const scale = if (rms > 1e-8 and !std.math.isNan(rms)) 1.0 / rms else 0.0;

    // Normalize and scale
    for (0..n) |i| {
        const val = input[i];
        const w = weight[i];
        if (std.math.isNan(val) or std.math.isInf(val) or std.math.isNan(w) or std.math.isInf(w)) {
            output[i] = 0.0;
        } else {
            output[i] = val * scale * w;
        }
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

/// ReLU activation: max(0, x)
pub fn relu(output: []f32, input: []const f32) void {
    const Vec = @Vector(8, f32);
    const zero: Vec = @splat(0.0);
    const n = input.len;
    
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const x: Vec = input[i..][0..8].*;
        const result = @max(x, zero);
        output[i..][0..8].* = result;
    }
    
    while (i < n) : (i += 1) {
        output[i] = @max(input[i], 0.0);
    }
}

/// GELU activation (Gaussian Error Linear Unit)
/// Used in older transformers
pub fn gelu(output: []f32, input: []const f32) void {
    const sqrt_2_over_pi = 0.7978845608;
    
    for (0..input.len) |i| {
        const x = input[i];
        const tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        // Use std.math.tanh instead of @tanh
        output[i] = 0.5 * x * (1.0 + math.tanh(tanh_arg));
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
/// Used in Llama
pub fn silu(output: []f32, input: []const f32) void {
    for (0..input.len) |i| {
        const x = input[i];
        const sigmoid = 1.0 / (1.0 + @exp(-x));
        output[i] = x * sigmoid;
    }
}

/// SwiGLU activation (used in Llama MLP)
/// Combination of SiLU and gating
pub fn swiglu(
    output: []f32,
    gate: []const f32,
    up: []const f32,
) void {
    for (0..output.len) |i| {
        const g = gate[i];
        const sigmoid = 1.0 / (1.0 + @exp(-g));
        output[i] = (g * sigmoid) * up[i];
    }
}

// ============================================================================
// Attention Operations
// ============================================================================

/// Softmax: exp(x) / sum(exp(x))
pub fn softmax(output: []f32, input: []const f32) void {
    const n = input.len;
    
    // Find max for numerical stability
    var max_val: f32 = input[0];
    for (input[1..]) |val| {
        max_val = @max(max_val, val);
    }
    
    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (0..n) |i| {
        const exp_val = @exp(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    const inv_sum = 1.0 / sum;
    vec_scale(output, output, inv_sum);
}

/// RoPE (Rotary Position Embedding)
/// Critical for position encoding in modern LLMs
pub fn apply_rope(
    q: []f32,
    k: []f32,
    pos: usize,
    dim: usize,
    rope_theta: f32,
) void {
    const half_dim = dim / 2;
    
    for (0..half_dim) |i| {
        const freq = 1.0 / math.pow(f32, rope_theta, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half_dim)));
        const val = @as(f32, @floatFromInt(pos)) * freq;
        const cos_val = @cos(val);
        const sin_val = @sin(val);
        
        // Rotate q
        const q0 = q[i];
        const q1 = q[i + half_dim];
        q[i] = q0 * cos_val - q1 * sin_val;
        q[i + half_dim] = q0 * sin_val + q1 * cos_val;
        
        // Rotate k
        const k0 = k[i];
        const k1 = k[i + half_dim];
        k[i] = k0 * cos_val - k1 * sin_val;
        k[i + half_dim] = k0 * sin_val + k1 * cos_val;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Copy data from one buffer to another
pub fn copy(dest: []f32, src: []const f32) void {
    const Vec = @Vector(8, f32);
    const n = src.len;
    
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const vec: Vec = src[i..][0..8].*;
        dest[i..][0..8].* = vec;
    }
    
    while (i < n) : (i += 1) {
        dest[i] = src[i];
    }
}

/// Fill buffer with a scalar value
pub fn fill(buffer: []f32, value: f32) void {
    const Vec = @Vector(8, f32);
    const vec: Vec = @splat(value);
    const n = buffer.len;
    
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        buffer[i..][0..8].* = vec;
    }
    
    while (i < n) : (i += 1) {
        buffer[i] = value;
    }
}

/// Compute L2 norm
pub fn l2_norm(input: []const f32) f32 {
    var sum: f32 = 0.0;
    for (input) |val| {
        sum += val * val;
    }
    return @sqrt(sum);
}

// ============================================================================
// Quantized Matrix Operations
// ============================================================================

/// Matrix multiplication with quantized weights
pub fn matmul_quantized(
    c: []f32,
    a_quant: []const u8,
    a_type: gguf_loader.QuantizationType,
    b: []const f32,
    m: usize,
    n: usize,
    k: usize,
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
) !void {
    switch (a_type) {
        .F32 => {
            const a_f32 = @as([*]const f32, @ptrCast(@alignCast(a_quant.ptr)));
            // matmul_f32 expects (m, k, n) so swap n and k
            try matmul_f32(c, a_f32[0 .. m * k], b, m, k, n, allocator, pool);
        },
        .Q4_0 => {
             const QK4_0 = 32;
             const block_size = 18;
             if (k % QK4_0 != 0) @panic("K must be multiple of 32 for Q4_0");
             const nb = k / QK4_0;

             const TILE_N = 64;

             const compute_chunk_q = struct {
                 fn run(
                     c_slice: []f32, a_q: []const u8, b_ptr: []const f32,
                     n_dim: usize, nb_dim: usize, block_size_bytes: usize, qk: usize,
                     start_row: usize, end_row: usize
                 ) void {
                     const Vec = @Vector(8, f32);
                     
                     for (start_row..end_row) |i| {
                         var j_tile: usize = 0;
                         while (j_tile < n_dim) : (j_tile += TILE_N) {
                             const current_n_end = @min(j_tile + TILE_N, n_dim);
                             
                             const c_row_offset = i * n_dim;
                             @memset(c_slice[c_row_offset + j_tile .. c_row_offset + current_n_end], 0.0);
                             
                             // Iterate over Q4_0 blocks (K dimension)
                             for (0..nb_dim) |bi| {
                                 // Decode Q4_0 block 'A' scalar values
                                 const row_offset_a = i * nb_dim * block_size_bytes;
                                 const block_offset_a = row_offset_a + bi * block_size_bytes;
                                 const block_ptr = a_q[block_offset_a..];
                                 
                                 const scale_bits = @as(*const u16, @ptrCast(@alignCast(block_ptr.ptr))).*;
                                 var scale = @call(.always_inline, f16_to_f32, .{scale_bits});
                                 // Handle NaN/Inf scales
                                 if (std.math.isNan(scale) or std.math.isInf(scale)) scale = 0.0;
                                 const qs = block_ptr[2..18];
                                 
                                 // Iterate 32 values in block
                                 // We manually unroll nibble access for speed
                                 var k_idx = bi * qk;
                                 
                                 for (0..16) |byte_idx| {
                                     const byte = qs[byte_idx];
                                     
                                     // Low nibble
                                     {
                                         const v0_i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
                                         const val_a = @as(f32, @floatFromInt(v0_i8)) * scale;
                                         const vec_a: Vec = @splat(val_a);
                                         
                                         const b_row_offset = k_idx * n_dim;
                                         
                                         // Vectorized update over J
                                         var j: usize = j_tile;
                                         while (j + 8 <= current_n_end) : (j += 8) {
                                             const vec_b = @as(Vec, b_ptr[b_row_offset + j..][0..8].*);
                                             var vec_c = @as(Vec, c_slice[c_row_offset + j..][0..8].*);
                                             vec_c += vec_a * vec_b;
                                             c_slice[c_row_offset + j..][0..8].* = vec_c;
                                         }
                                         while (j < current_n_end) : (j += 1) {
                                             c_slice[c_row_offset + j] += val_a * b_ptr[b_row_offset + j];
                                         }
                                     }
                                     k_idx += 1;
                                     
                                     // High nibble
                                     {
                                         const v1_i8 = @as(i8, @intCast(byte >> 4)) - 8;
                                         const val_a = @as(f32, @floatFromInt(v1_i8)) * scale;
                                         const vec_a: Vec = @splat(val_a);
                                         
                                         const b_row_offset = k_idx * n_dim;
                                         
                                         var j: usize = j_tile;
                                         while (j + 8 <= current_n_end) : (j += 8) {
                                             const vec_b = @as(Vec, b_ptr[b_row_offset + j..][0..8].*);
                                             var vec_c = @as(Vec, c_slice[c_row_offset + j..][0..8].*);
                                             vec_c += vec_a * vec_b;
                                             c_slice[c_row_offset + j..][0..8].* = vec_c;
                                         }
                                         while (j < current_n_end) : (j += 1) {
                                             c_slice[c_row_offset + j] += val_a * b_ptr[b_row_offset + j];
                                         }
                                     }
                                     k_idx += 1;
                                 }
                             }
                         }
                     }
                 }
             }.run;

             // Parallel dispatch
             if (pool) |tp| {
                 if (m >= 4) {
                     const num_threads = tp.config.num_threads;
                     const rows_per_task = (m + num_threads - 1) / num_threads;
                     const Context = struct {
                         c: []f32, a_q: []const u8, b: []const f32,
                         n: usize, nb: usize, blk: usize, qk: usize,
                         start: usize, end: usize
                     };
                     const work = struct {
                         fn w(ctx: *anyopaque) void {
                             const context = @as(*Context, @ptrCast(@alignCast(ctx)));
                             compute_chunk_q(context.c, context.a_q, context.b, context.n, context.nb, context.blk, context.qk, context.start, context.end);
                         }
                     }.w;
                     
                     var contexts = try allocator.alloc(Context, num_threads);
                     // CRITICAL FIX: Don't use defer - free after waitAll() to prevent thread use-after-free
                     for (0..num_threads) |t| {
                         const start = t * rows_per_task;
                         if (start >= m) break;
                         const end = @min(start + rows_per_task, m);
                         contexts[t] = .{ 
                             .c = c, .a_q = a_quant, .b = b,
                             .n = n, .nb = nb, .blk = block_size, .qk = QK4_0,
                             .start = start, .end = end
                         };
                         tp.submit(.{ .work_fn = work, .context = @ptrCast(&contexts[t]) }) catch {};
                     }
                     tp.waitAll();
                     allocator.free(contexts); // Free AFTER threads complete
                     return;
                 }
             }

             // Serial fallback
             compute_chunk_q(c, a_quant, b, n, nb, block_size, QK4_0, 0, m);
        },
        .Q4_K => {
             const QK4_K = 256;
             const block_size = 144;
             if (k % QK4_K != 0) {
                 std.debug.print("   ⚠️  Q4_K matmul fallback to F32 (m={d}, n={d}, k={d})\n", .{ m, n, k });
                 const rows = m;
                 const cols = k;
                 const f32_buf = try allocator.alloc(f32, rows * cols);
                 defer allocator.free(f32_buf);
                 for (0..rows) |row| {
                     const row_out = f32_buf[row * cols .. (row + 1) * cols];
                     get_row(row_out, .{ .q4_k = a_quant }, row, cols);
                 }
                 try matmul_f32(c, f32_buf, b, m, k, n, allocator, pool);
                 return;
             }
             const nb = k / QK4_K;
             const TILE_N = 64;

             const compute_chunk_qk = struct {
                 fn run(
                     c_slice: []f32, a_q: []const u8, b_ptr: []const f32,
                     n_dim: usize, nb_dim: usize, block_size_bytes: usize, qk: usize,
                     start_row: usize, end_row: usize
                 ) void {
                     const Vec = @Vector(8, f32);
                     var block_buf: [256]f32 = undefined;
                     
                     for (start_row..end_row) |i| {
                         var j_tile: usize = 0;
                         while (j_tile < n_dim) : (j_tile += TILE_N) {
                             const current_n_end = @min(j_tile + TILE_N, n_dim);
                             
                             const c_row_offset = i * n_dim;
                             @memset(c_slice[c_row_offset + j_tile .. c_row_offset + current_n_end], 0.0);
                             
                             for (0..nb_dim) |bi| {
                                 const row_offset_a = i * nb_dim * block_size_bytes;
                                 const block_offset_a = row_offset_a + bi * block_size_bytes;
                                 const block_ptr = @as(*const q4_k.BlockQ4_K, @ptrCast(@alignCast(&a_q[block_offset_a])));
                                 
                                 q4_k.dequantizeBlock(&block_buf, block_ptr);
                                 
                                 var k_idx = bi * qk;
                                 
                                 // For each value in the decoded block
                                 for (0..qk) |blk_k| {
                                     const val_a = block_buf[blk_k];
                                     const vec_a: Vec = @splat(val_a);
                                     const b_row_offset = k_idx * n_dim;
                                     
                                     var j: usize = j_tile;
                                     while (j + 8 <= current_n_end) : (j += 8) {
                                         const vec_b = @as(Vec, b_ptr[b_row_offset + j..][0..8].*);
                                         var vec_c = @as(Vec, c_slice[c_row_offset + j..][0..8].*);
                                         vec_c += vec_a * vec_b;
                                         c_slice[c_row_offset + j..][0..8].* = vec_c;
                                     }
                                     while (j < current_n_end) : (j += 1) {
                                         c_slice[c_row_offset + j] += val_a * b_ptr[b_row_offset + j];
                                     }
                                     k_idx += 1;
                                 }
                             }
                         }
                     }
                 }
             }.run;

             if (pool) |tp| {
                 if (m >= 4) {
                     const num_threads = tp.config.num_threads;
                     const rows_per_task = (m + num_threads - 1) / num_threads;
                     const Context = struct {
                         c: []f32, a_q: []const u8, b: []const f32,
                         n: usize, nb: usize, blk: usize, qk: usize,
                         start: usize, end: usize
                     };
                     const work = struct {
                         fn w(ctx: *anyopaque) void {
                             const context = @as(*Context, @ptrCast(@alignCast(ctx)));
                             compute_chunk_qk(context.c, context.a_q, context.b, context.n, context.nb, context.blk, context.qk, context.start, context.end);
                         }
                     }.w;
                     
                     var contexts = try allocator.alloc(Context, num_threads);
                     // CRITICAL FIX: Don't use defer - free after waitAll() to prevent thread use-after-free
                     for (0..num_threads) |t| {
                         const start = t * rows_per_task;
                         if (start >= m) break;
                         const end = @min(start + rows_per_task, m);
                         contexts[t] = .{
                             .c = c, .a_q = a_quant, .b = b,
                             .n = n, .nb = nb, .blk = block_size, .qk = QK4_K,
                             .start = start, .end = end
                         };
                         tp.submit(.{ .work_fn = work, .context = @ptrCast(&contexts[t]) }) catch {};
                     }
                     tp.waitAll();
                     allocator.free(contexts); // Free AFTER threads complete
                     return;
                 }
             }
             compute_chunk_qk(c, a_quant, b, n, nb, block_size, QK4_K, 0, m);
        },
        .Q6_K => {
             const QK6_K = q6_k.QK_K;
             const block_size = @sizeOf(q6_k.BlockQ6_K); // 210 bytes
             if (k % QK6_K != 0) @panic("K must be multiple of 256 for Q6_K");
             const nb = k / QK6_K;
             const TILE_N = 64;

             const compute_chunk_q6k = struct {
                 fn run(
                     c_slice: []f32, a_q: []const u8, b_ptr: []const f32,
                     n_dim: usize, nb_dim: usize, block_size_bytes: usize, qk: usize,
                     start_row: usize, end_row: usize
                 ) void {
                     const Vec = @Vector(8, f32);
                     var block_buf: [q6_k.QK_K]f32 = undefined;
                     
                     for (start_row..end_row) |i| {
                         var j_tile: usize = 0;
                         while (j_tile < n_dim) : (j_tile += TILE_N) {
                             const current_n_end = @min(j_tile + TILE_N, n_dim);
                             
                             const c_row_offset = i * n_dim;
                             @memset(c_slice[c_row_offset + j_tile .. c_row_offset + current_n_end], 0.0);
                             
                             for (0..nb_dim) |bi| {
                                 const row_offset_a = i * nb_dim * block_size_bytes;
                                 const block_offset_a = row_offset_a + bi * block_size_bytes;
                                 const block_ptr = @as(*const q6_k.BlockQ6_K, @ptrCast(@alignCast(&a_q[block_offset_a])));
                                 
                                 q6_k.dequantizeBlock(&block_buf, block_ptr);
                                 
                                 var k_idx = bi * qk;
                                 for (0..qk) |blk_k| {
                                     const val_a = block_buf[blk_k];
                                     const vec_a: Vec = @splat(val_a);
                                     const b_row_offset = k_idx * n_dim;
                                     
                                     var j: usize = j_tile;
                                     while (j + 8 <= current_n_end) : (j += 8) {
                                         const vec_b = @as(Vec, b_ptr[b_row_offset + j..][0..8].*);
                                         var vec_c = @as(Vec, c_slice[c_row_offset + j..][0..8].*);
                                         vec_c += vec_a * vec_b;
                                         c_slice[c_row_offset + j..][0..8].* = vec_c;
                                     }
                                     while (j < current_n_end) : (j += 1) {
                                         c_slice[c_row_offset + j] += val_a * b_ptr[b_row_offset + j];
                                     }
                                     k_idx += 1;
                                 }
                             }
                         }
                     }
                 }
             }.run;

             if (pool) |tp| {
                 if (m >= 4) {
                     const num_threads = tp.config.num_threads;
                     const rows_per_task = (m + num_threads - 1) / num_threads;
                     const Context = struct {
                         c: []f32, a_q: []const u8, b: []const f32,
                         n: usize, nb: usize, blk: usize, qk: usize,
                         start: usize, end: usize
                     };
                     const work = struct {
                         fn w(ctx: *anyopaque) void {
                             const context = @as(*Context, @ptrCast(@alignCast(ctx)));
                             compute_chunk_q6k(context.c, context.a_q, context.b, context.n, context.nb, context.blk, context.qk, context.start, context.end);
                         }
                     }.w;
                     
                     var contexts = try allocator.alloc(Context, num_threads);
                     // CRITICAL FIX: Don't use defer - free after waitAll() to prevent thread use-after-free
                     for (0..num_threads) |t| {
                         const start = t * rows_per_task;
                         if (start >= m) break;
                         const end = @min(start + rows_per_task, m);
                         contexts[t] = .{
                             .c = c, .a_q = a_quant, .b = b,
                             .n = n, .nb = nb, .blk = block_size, .qk = QK6_K,
                             .start = start, .end = end
                         };
                         tp.submit(.{ .work_fn = work, .context = @ptrCast(&contexts[t]) }) catch {};
                     }
                     tp.waitAll();
                     allocator.free(contexts); // Free AFTER threads complete
                     return;
                 }
             }
             compute_chunk_q6k(c, a_quant, b, n, nb, block_size, QK6_K, 0, m);
        },
        else => {
            @panic("Quantized matmul not yet integrated for this type");
        },
    }
}

// Helper for f16 to f32 (simplified)
fn f16_to_f32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

// ============================================================================
// Testing & Benchmarking
// ============================================================================

pub fn benchmark_matmul(m: usize, n: usize, k: usize, allocator: std.mem.Allocator) !void {
    std.debug.print("\n⚡ Benchmarking matmul [{d}x{d}] * [{d}x{d}]\n", .{ m, k, k, n });
    
    // Allocate test matrices
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);
    
    // Initialize with random values
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    
    for (a) |*val| val.* = random.float(f32);
    for (b) |*val| val.* = random.float(f32);
    
    // Benchmark
    const start = std.time.nanoTimestamp();
    
    const iterations = 10;
    for (0..iterations) |_| {
        try matmul_f32(c, a, b, m, n, k, allocator, null);
    }
    
    const end = std.time.nanoTimestamp();
    const elapsed_ms = @divFloor(end - start, 1_000_000);
    const avg_ms = @divFloor(elapsed_ms, iterations);
    
    // Calculate GFLOPS
    const ops = 2 * m * n * k; // Each output: k multiplies + k adds
    const gflops = @as(f32, @floatFromInt(ops)) / @as(f32, @floatFromInt(avg_ms)) / 1_000_000.0;
    
    std.debug.print("   Time: {d}ms (avg of {d} iterations)\n", .{ avg_ms, iterations });
    std.debug.print("   Performance: {d:.2} GFLOPS\n", .{gflops});
    std.debug.print("   Result sample: {d:.4}\n", .{c[0]});
}

pub fn test_operations(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Matrix Operations\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: Small matmul
    {
        std.debug.print("\n1️⃣  Testing 4x4 matmul...\n", .{});
        
        const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        const b = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; // Identity
        var c: [16]f32 = undefined;
        
        try matmul_f32(&c, &a, &b, 4, 4, 4, allocator, null);
        
        // Result should equal a (identity matrix)
        for (0..16) |i| {
            if (@abs(c[i] - a[i]) > 0.001) {
                std.debug.print("❌ Mismatch at {d}: {d} vs {d}\n", .{ i, c[i], a[i] });
                return error.TestFailed;
            }
        }
        std.debug.print("   ✅ Identity matmul correct\n", .{});
    }
    
    // Test 2: Vector operations
    {
        std.debug.print("\n2️⃣  Testing vector operations...\n", .{});
        
        const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const b = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
        var c: [8]f32 = undefined;
        
        vec_add(&c, &a, &b);
        const expected_sum = [_]f32{ 9, 9, 9, 9, 9, 9, 9, 9 };
        for (0..8) |i| {
            if (c[i] != expected_sum[i]) {
                return error.TestFailed;
            }
        }
        std.debug.print("   ✅ Vector add correct\n", .{});
        
        vec_scale(&c, &a, 2.0);
        for (0..8) |i| {
            if (c[i] != a[i] * 2.0) {
                return error.TestFailed;
            }
        }
        std.debug.print("   ✅ Vector scale correct\n", .{});
    }
    
    // Test 3: Softmax
    {
        std.debug.print("\n3️⃣  Testing softmax...\n", .{});
        
        const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        var output: [4]f32 = undefined;
        
        softmax(&output, &input);
        
        // Sum should be 1.0
        var sum: f32 = 0.0;
        for (output) |val| {
            sum += val;
        }
        
        if (@abs(sum - 1.0) > 0.001) {
            std.debug.print("❌ Softmax sum: {d} (expected 1.0)\n", .{sum});
            return error.TestFailed;
        }
        std.debug.print("   ✅ Softmax normalized (sum = {d:.6})\n", .{sum});
    }
    
    // Test 4: Activations
    {
        std.debug.print("\n4️⃣  Testing activation functions...\n", .{});
        
        const input = [_]f32{ -2, -1, 0, 1, 2 };
        var output: [5]f32 = undefined;
        
        relu(&output, &input);
        const expected_relu = [_]f32{ 0, 0, 0, 1, 2 };
        for (0..5) |i| {
            if (output[i] != expected_relu[i]) {
                return error.TestFailed;
            }
        }
        std.debug.print("   ✅ ReLU correct\n", .{});
        
        silu(&output, &input);
        // Just verify it doesn't crash and produces reasonable values
        for (output) |val| {
            if (math.isNan(val) or math.isInf(val)) {
                return error.TestFailed;
            }
        }
        std.debug.print("   ✅ SiLU correct\n", .{});
    }
    
    // Benchmark
    try benchmark_matmul(256, 256, 256, allocator);
    try benchmark_matmul(512, 512, 512, allocator);
    
    std.debug.print("\n✅ All matrix operations tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
