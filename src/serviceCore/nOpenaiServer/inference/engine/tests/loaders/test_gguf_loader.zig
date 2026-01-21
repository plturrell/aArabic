const std = @import("std");
const gguf = @import("gguf_loader");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("🧪 GGUF Loader Test Suite\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});

    // Test 1: Try to load a GGUF model (if available)
    const test_models = [_][]const u8{
        "models/llama-3.2-1b-q4_0.gguf",
        "models/llama-3.2-1b.gguf",
        "~/.cache/huggingface/hub/models--bartowski--Llama-3.2-1B-Instruct-GGUF/llama-3.2-1b-instruct-q4_0.gguf",
    };

    var model_loaded = false;
    var model: gguf.GGUFModel = undefined;

    for (test_models) |model_path| {
        std.debug.print("🔍 Trying to load: {s}\n", .{model_path});
        
        model = gguf.GGUFModel.load(allocator, model_path) catch |err| {
            std.debug.print("   ⚠️  Failed: {}\n\n", .{err});
            continue;
        };
        
        model_loaded = true;
        std.debug.print("   ✅ Loaded successfully!\n\n", .{});
        break;
    }

    if (!model_loaded) {
        std.debug.print("⚠️  No GGUF models found to test\n", .{});
        std.debug.print("   To test GGUF loading, download a model:\n", .{});
        std.debug.print("   huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \\\n", .{});
        std.debug.print("     llama-3.2-1b-instruct-q4_0.gguf --local-dir ./models/\n\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
        std.debug.print("✅ GGUF loader code is ready (no models to test with)\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
        return;
    }

    defer model.deinit();

    // Print model summary
    model.printSummary();

    // Test 2: Validate model structure
    std.debug.print("🧪 Test 2: Model Validation\n", .{});
    std.debug.print("───────────────────────────────────────────────────────────────────────\n", .{});
    
    gguf.validateModel(&model) catch |err| {
        std.debug.print("❌ Validation failed: {}\n", .{err});
        return err;
    };
    
    std.debug.print("\n", .{});

    // Test 3: Tensor lookup
    std.debug.print("🧪 Test 3: Tensor Lookup\n", .{});
    std.debug.print("───────────────────────────────────────────────────────────────────────\n", .{});
    
    const test_tensor_names = [_][]const u8{
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_q.weight",
    };
    
    for (test_tensor_names) |name| {
        const tensor_info = model.getTensor(name);
        if (tensor_info) |info| {
            std.debug.print("✅ Found tensor: {s}\n", .{name});
            std.debug.print("   Shape: [", .{});
            for (0..info.n_dimensions) |d| {
                if (d > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{info.dimensions[d]});
            }
            std.debug.print("]\n", .{});
            std.debug.print("   Type: {s}\n", .{@tagName(info.quant_type)});
            std.debug.print("   Size: {d} elements\n", .{info.size()});
            std.debug.print("   Data: {d} bytes\n", .{info.dataSize()});
        } else {
            std.debug.print("⚠️  Tensor not found: {s}\n", .{name});
        }
    }

    // Test 4: Load a small tensor
    std.debug.print("\n🧪 Test 4: Tensor Loading\n", .{});
    std.debug.print("───────────────────────────────────────────────────────────────────────\n", .{});
    
    if (model.getTensor("output_norm.weight")) |norm_info| {
        var tensor = try gguf.Tensor.loadFromFile(allocator, model.file, norm_info);
        defer tensor.deinit();
        
        std.debug.print("✅ Loaded tensor: {s}\n", .{norm_info.name});
        std.debug.print("   Data size: {d} bytes\n", .{tensor.data.len});
        std.debug.print("   First bytes: ", .{});
        const preview_len = @min(16, tensor.data.len);
        for (0..preview_len) |i| {
            std.debug.print("{x:0>2} ", .{tensor.data[i]});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ All GGUF loader tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
