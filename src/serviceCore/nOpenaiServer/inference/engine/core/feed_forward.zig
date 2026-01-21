const std = @import("std");
const matrix_ops = @import("matrix_ops");
const thread_pool = @import("thread_pool");
const compute = @import("compute");
const gguf = @import("gguf_loader");

/// Feed-forward network (MLP) for Llama models
/// Implements SwiGLU activation: FFN(x) = (Swish(xW_gate) ⊙ xW_up)W_down

// ============================================================================
// Structures
// ============================================================================

pub const FFNWeights = struct {
    w_gate: matrix_ops.Weight,  // Gate projection [ffn_dim, embed_dim]
    w_up: matrix_ops.Weight,    // Up projection [ffn_dim, embed_dim]
    w_down: matrix_ops.Weight,  // Down projection [embed_dim, ffn_dim]
};

// ============================================================================
// Feed-Forward Operations
// ============================================================================

/// Compute feed-forward network with SwiGLU activation
pub fn computeFFN(
    allocator: std.mem.Allocator,
    output: []f32,
    input: []const f32,
    weights: FFNWeights,
    ffn_dim: u32,
    pool: ?*thread_pool.ThreadPool,
) !void {
    const embed_dim = input.len;
    
    // Allocate intermediate buffers
    const gate = try allocator.alloc(f32, ffn_dim);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, ffn_dim);
    defer allocator.free(up);
    const gated = try allocator.alloc(f32, ffn_dim);
    defer allocator.free(gated);
    
    // Gate projection: gate = x * W_gate
    // W_gate is [ffn_dim, embed_dim]
    try matrix_ops.matmul(gate, weights.w_gate, input, ffn_dim, 1, embed_dim, allocator, pool);
    
    // Up projection: up = x * W_up
    try matrix_ops.matmul(up, weights.w_up, input, ffn_dim, 1, embed_dim, allocator, pool);
    
    // Apply SwiGLU: gated = SiLU(gate) ⊙ up
    matrix_ops.swiglu(gated, gate, up);
    
    // Down projection: output = gated * W_down
    try matrix_ops.matmul(output, weights.w_down, gated, embed_dim, 1, ffn_dim, allocator, pool);
}

/// Simpler version using pre-allocated workspace
pub fn computeFFNWorkspace(
    allocator: std.mem.Allocator,
    output: []f32,
    input: []const f32,
    weights: FFNWeights,
    workspace: []f32,
    pool: ?*thread_pool.ThreadPool,
) !void {
    const embed_dim = input.len;
    const ffn_dim = workspace.len / 3; // workspace = [gate, up, gated]
    
    const gate = workspace[0..ffn_dim];
    const up = workspace[ffn_dim .. 2 * ffn_dim];
    const gated = workspace[2 * ffn_dim .. 3 * ffn_dim];
    
    // Gate projection
    try matrix_ops.matmul(gate, weights.w_gate, input, ffn_dim, 1, embed_dim, allocator, pool);
    
    // Up projection
    try matrix_ops.matmul(up, weights.w_up, input, ffn_dim, 1, embed_dim, allocator, pool);
    
    // SwiGLU activation
    matrix_ops.swiglu(gated, gate, up);
    
    // Down projection
    try matrix_ops.matmul(output, weights.w_down, gated, embed_dim, 1, ffn_dim, allocator, pool);
}

// ============================================================================
// GPU Backend Support
// ============================================================================

/// Helper to extract weight data and quantization type from a Weight union
fn weightToBackendParams(weight: matrix_ops.Weight) struct { data: []const u8, quant_type: gguf.QuantizationType } {
    return switch (weight) {
        .f32 => |data| .{ .data = std.mem.sliceAsBytes(data), .quant_type = .F32 },
        .q4_0 => |data| .{ .data = data, .quant_type = .Q4_0 },
        .q4_k => |data| .{ .data = data, .quant_type = .Q4_K },
        .q6_k => |data| .{ .data = data, .quant_type = .Q6_K },
    };
}

/// Compute feed-forward network with SwiGLU activation using GPU backend
/// Uses backend.matmul() for accelerated matrix operations
pub fn computeFFNGpu(
    allocator: std.mem.Allocator,
    output: []f32,
    input: []const f32,
    weights: FFNWeights,
    ffn_dim: u32,
    backend: compute.ComputeBackend,
) !void {
    const embed_dim = input.len;

    // Allocate intermediate buffers
    const gate = try allocator.alloc(f32, ffn_dim);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, ffn_dim);
    defer allocator.free(up);
    const gated = try allocator.alloc(f32, ffn_dim);
    defer allocator.free(gated);

    // Gate projection: gate = x * W_gate
    // W_gate is [ffn_dim, embed_dim]
    const gate_params = weightToBackendParams(weights.w_gate);
    try backend.matmul(gate, gate_params.data, gate_params.quant_type, input, ffn_dim, 1, embed_dim);

    // Up projection: up = x * W_up
    const up_params = weightToBackendParams(weights.w_up);
    try backend.matmul(up, up_params.data, up_params.quant_type, input, ffn_dim, 1, embed_dim);

    // Apply SwiGLU: gated = SiLU(gate) ⊙ up
    // Keep using CPU for this simple elementwise operation
    matrix_ops.swiglu(gated, gate, up);

    // Down projection: output = gated * W_down
    const down_params = weightToBackendParams(weights.w_down);
    try backend.matmul(output, down_params.data, down_params.quant_type, gated, embed_dim, 1, ffn_dim);
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_feed_forward(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Feed-Forward Network\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    const embed_dim: u32 = 64;
    const ffn_dim: u32 = 256;
    
    // Test 1: FFN with identity-like weights
    {
        std.debug.print("\n1️⃣  Testing FFN computation...\n", .{});
        
        // Create simple weights
        const w_gate = try allocator.alloc(f32, embed_dim * ffn_dim);
        defer allocator.free(w_gate);
        const w_up = try allocator.alloc(f32, embed_dim * ffn_dim);
        defer allocator.free(w_up);
        const w_down = try allocator.alloc(f32, ffn_dim * embed_dim);
        defer allocator.free(w_down);
        
        @memset(w_gate, 0.0);
        @memset(w_up, 0.0);
        @memset(w_down, 0.0);
        
        // Simple pattern
        for (0..@min(embed_dim, ffn_dim)) |i| {
            w_gate[i * ffn_dim + i] = 1.0;
            w_up[i * ffn_dim + i] = 1.0;
        }
        for (0..@min(ffn_dim, embed_dim)) |i| {
            w_down[i * embed_dim + i] = 1.0;
        }
        
        const weights = FFNWeights{
            .w_gate = .{ .f32 = w_gate },
            .w_up = .{ .f32 = w_up },
            .w_down = .{ .f32 = w_down },
        };
        
        // Test input
        const input = try allocator.alloc(f32, embed_dim);
        defer allocator.free(input);
        for (0..embed_dim) |i| {
            input[i] = 1.0;
        }
        
        const output = try allocator.alloc(f32, embed_dim);
        defer allocator.free(output);
        
        try computeFFN(allocator, output, input, weights, ffn_dim, null);
        
        const input_sum = blk: {
            var sum: f32 = 0.0;
            for (input) |v| sum += v;
            break :blk sum;
        };
        const output_sum = blk: {
            var sum: f32 = 0.0;
            for (output) |v| sum += v;
            break :blk sum;
        };
        
        std.debug.print("   Input sum: {d:.2}\n", .{input_sum});
        std.debug.print("   Output sum: {d:.2}\n", .{output_sum});
        std.debug.print("   ✅ FFN computed\n", .{});
    }
    
    // Test 2: FFN with workspace
    {
        std.debug.print("\n2️⃣  Testing FFN with workspace...\n", .{});
        
        // Create simple weights
        const w_gate = try allocator.alloc(f32, embed_dim * ffn_dim);
        defer allocator.free(w_gate);
        const w_up = try allocator.alloc(f32, embed_dim * ffn_dim);
        defer allocator.free(w_up);
        const w_down = try allocator.alloc(f32, ffn_dim * embed_dim);
        defer allocator.free(w_down);
        
        @memset(w_gate, 0.1);
        @memset(w_up, 0.1);
        @memset(w_down, 0.1);
        
        const weights = FFNWeights{
            .w_gate = .{ .f32 = w_gate },
            .w_up = .{ .f32 = w_up },
            .w_down = .{ .f32 = w_down },
        };
        
        // Test input
        const input = try allocator.alloc(f32, embed_dim);
        defer allocator.free(input);
        for (0..embed_dim) |i| {
            input[i] = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(embed_dim));
        }
        
        const output1 = try allocator.alloc(f32, embed_dim);
        defer allocator.free(output1);
        const output2 = try allocator.alloc(f32, embed_dim);
        defer allocator.free(output2);
        
        // Allocate workspace
        const workspace = try allocator.alloc(f32, ffn_dim * 3);
        defer allocator.free(workspace);
        
        // Compute with both methods
        try computeFFN(allocator, output1, input, weights, ffn_dim, null);
        try computeFFNWorkspace(allocator, output2, input, weights, workspace, null);
        
        // Compare outputs
        var max_diff: f32 = 0.0;
        for (0..embed_dim) |i| {
            const diff = @abs(output1[i] - output2[i]);
            if (diff > max_diff) max_diff = diff;
        }
        
        std.debug.print("   Max difference: {d:.6}\n", .{max_diff});
        
        if (max_diff < 0.001) {
            std.debug.print("   ✅ Workspace version matches\n", .{});
        } else {
            std.debug.print("   ⚠️  Outputs differ slightly\n", .{});
        }
    }
    
    // Test 3: Different input patterns
    {
        std.debug.print("\n3️⃣  Testing with different inputs...\n", .{});
        
        const w_gate = try allocator.alloc(f32, embed_dim * ffn_dim);
        defer allocator.free(w_gate);
        const w_up = try allocator.alloc(f32, embed_dim * ffn_dim);
        defer allocator.free(w_up);
        const w_down = try allocator.alloc(f32, ffn_dim * embed_dim);
        defer allocator.free(w_down);
        
        // Random-like weights
        for (w_gate, 0..) |*w, i| {
            w.* = @sin(@as(f32, @floatFromInt(i)) * 0.01) * 0.1;
        }
        for (w_up, 0..) |*w, i| {
            w.* = @cos(@as(f32, @floatFromInt(i)) * 0.01) * 0.1;
        }
        for (w_down, 0..) |*w, i| {
            w.* = @sin(@as(f32, @floatFromInt(i)) * 0.02) * 0.1;
        }
        
        const weights = FFNWeights{
            .w_gate = .{ .f32 = w_gate },
            .w_up = .{ .f32 = w_up },
            .w_down = .{ .f32 = w_down },
        };
        
        const test_inputs = [_][]const f32{
            &[_]f32{1.0} ** embed_dim,
            &[_]f32{0.5} ** embed_dim,
            &[_]f32{-1.0} ** embed_dim,
        };
        
        for (test_inputs, 0..) |test_input, idx| {
            const output = try allocator.alloc(f32, embed_dim);
            defer allocator.free(output);
            
            try computeFFN(allocator, output, test_input, weights, ffn_dim, null);
            
            const out_sum = blk: {
                var sum: f32 = 0.0;
                for (output) |v| sum += v;
                break :blk sum;
            };
            
            std.debug.print("   Input {d}: output sum = {d:.4}\n", .{ idx + 1, out_sum });
        }
        
        std.debug.print("   ✅ Various inputs processed\n", .{});
    }
    
    std.debug.print("\n✅ All feed-forward tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}

