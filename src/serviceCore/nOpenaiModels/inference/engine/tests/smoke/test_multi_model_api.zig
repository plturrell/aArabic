const std = @import("std");

extern fn inference_load_model_v2(model_id: [*:0]const u8, model_path: [*:0]const u8) i32;
extern fn inference_is_loaded_v2(model_id: [*:0]const u8) i32;
extern fn inference_generate_v2(
    model_id: [*:0]const u8,
    prompt_ptr: [*]const u8,
    prompt_len: usize,
    max_tokens: u32,
    temperature: f32,
    result_buffer: [*]u8,
    buffer_size: usize,
) i32;
extern fn inference_unload_v2(model_id: [*:0]const u8) void;

const RequiredModel = struct {
    id: [:0]const u8,
    env: [:0]const u8,
};

const requested_models = [_]RequiredModel{
    .{ .id = "unit-model-a", .env = "SHIMMY_TEST_MODEL_A" },
    .{ .id = "unit-model-b", .env = "SHIMMY_TEST_MODEL_B" },
};

const Model = struct {
    id: [:0]const u8,
    path: [:0]const u8,
};

pub fn main() !void {
    std.debug.print("\n═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  Multi-model Inference API Tests\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

    var models_buf: [requested_models.len]Model = undefined;
    const models = try loadEnvModels(&models_buf);
    if (models.len == 0) {
        std.debug.print("⚠️  Skipping multi-model tests (set SHIMMY_TEST_MODEL_A/B)\n", .{});
        return;
    }

    defer {
        for (models) |model| {
            inference_unload_v2(model.id);
        }
    }

    try testLoadMultipleModels(models);
    try testIndependentGeneration(models);

    std.debug.print("\n✅ All multi-model tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
}

fn loadEnvModels(buf: []Model) ![]Model {
    var count: usize = 0;
    for (requested_models) |req| {
        const path = std.posix.getenv(req.env) orelse {
            std.debug.print("⚠️  Missing env {s}; cannot load model {s}\n", .{ req.env, req.id });
            return buf[0..0];
        };
        buf[count] = .{ .id = req.id, .path = path };
        count += 1;
    }
    return buf[0..count];
}

fn testLoadMultipleModels(models: []const Model) !void {
    std.debug.print("\n🧪 Test: Load multiple models\n", .{});

    for (models) |model| {
        const rc = inference_load_model_v2(model.id, model.path);
        if (rc != 0) return error.LoadFailed;
        try expectLoaded(model.id);
    }
}

fn testIndependentGeneration(models: []const Model) !void {
    std.debug.print("\n🧪 Test: Independent generation per model\n", .{});

    const prompt_a = "Hello from model A!";
    const prompt_b = "Hello from model B!";

    var buffer: [2048]u8 = undefined;

    const len_a = inference_generate_v2(
        models[0].id,
        prompt_a.ptr,
        prompt_a.len,
        4,
        0.0,
        &buffer,
        buffer.len,
    );
    if (len_a <= 0) return error.GenerationFailed;
    std.debug.print("   Model A output ({d} bytes)\n", .{len_a});

    const len_b = inference_generate_v2(
        models[1].id,
        prompt_b.ptr,
        prompt_b.len,
        4,
        0.0,
        &buffer,
        buffer.len,
    );
    if (len_b <= 0) return error.GenerationFailed;
    std.debug.print("   Model B output ({d} bytes)\n", .{len_b});
}

fn expectLoaded(id: [:0]const u8) !void {
    if (inference_is_loaded_v2(id) != 1) {
        std.debug.print("❌ Model not loaded: {s}\n", .{id});
        return error.NotLoaded;
    }
    std.debug.print("   ✅ Loaded {s}\n", .{id});
}
