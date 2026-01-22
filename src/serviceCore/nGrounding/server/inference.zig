const std = @import("std");

const c = @cImport({
    @cInclude("stdint.h");
});

pub const InferenceEngine = struct {
    allocator: std.mem.Allocator,
    lib: ?std.DynLib = null,
    loaded: bool = false,
    model_path: []u8 = &.{},
    mutex: std.Thread.Mutex = .{},
    max_output_bytes: usize = 8192,
    embedding_dims: usize = 128,

    // Function pointers for the native inference library (if present).
    load_fn: ?*const fn ([*:0]const u8) callconv(.c) c_int = null,
    is_loaded_fn: ?*const fn () callconv(.c) c_int = null,
    get_info_fn: ?*const fn ([*]u8, usize) callconv(.c) c_int = null,
    generate_fn: ?*const fn ([*]const u8, usize, u32, f32, [*]u8, usize) callconv(.c) c_int = null,
    embed_fn: ?*const fn ([*]const u8, usize, [*]f32, usize) callconv(.c) c_int = null,

    pub fn init(allocator: std.mem.Allocator) InferenceEngine {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *InferenceEngine) void {
        if (self.model_path.len > 0) {
            self.allocator.free(self.model_path);
        }
        if (self.lib) |*dl| {
            dl.close();
        }
        self.* = .{
            .allocator = self.allocator,
        };
    }

    pub fn loadLibrary(self: *InferenceEngine, paths: []const []const u8) void {
        if (self.lib != null) return;
        for (paths) |p| {
            if (std.fs.cwd().openFile(p, .{})) |f| {
                f.close();
                if (std.DynLib.open(p)) |dl| {
                    self.lib = dl;
                    self.bindFunctions();
                    self.refreshInfo();
                    return;
                } else |_| {}
            } else |_| {}
        }
    }

    fn bindFunctions(self: *InferenceEngine) void {
        if (self.lib) |*dl| {
            self.load_fn = dl.lookup(*const fn ([*:0]const u8) callconv(.c) c_int, "inference_load_model") orelse null;
            self.is_loaded_fn = dl.lookup(*const fn () callconv(.c) c_int, "inference_is_loaded") orelse null;
            self.get_info_fn = dl.lookup(*const fn ([*]u8, usize) callconv(.c) c_int, "inference_get_info") orelse null;
            self.generate_fn = dl.lookup(*const fn ([*]const u8, usize, u32, f32, [*]u8, usize) callconv(.c) c_int, "inference_generate") orelse null;
            self.embed_fn = dl.lookup(*const fn ([*]const u8, usize, [*]f32, usize) callconv(.c) c_int, "inference_embed") orelse null;
        }
    }

    fn refreshInfo(self: *InferenceEngine) void {
        if (self.get_info_fn) |f| {
            var buf: [256]u8 = undefined;
            const rc = f(&buf, buf.len);
            if (rc > 0 and rc <= buf.len) {
                const slice = buf[0..@intCast(rc)];
                var it = std.mem.splitSequence(u8, slice, ";");
                while (it.next()) |part| {
                    if (std.mem.startsWith(u8, part, "max_output=")) {
                        if (std.fmt.parseInt(usize, part["max_output=".len..], 10)) |v| {
                            self.max_output_bytes = v;
                        } else |_| {}
                    }
                    if (std.mem.startsWith(u8, part, "embedding_dims=")) {
                        if (std.fmt.parseInt(usize, part["embedding_dims=".len..], 10)) |v| {
                            self.embedding_dims = v;
                        } else |_| {}
                    }
                }
            }
        }
    }

    pub fn applyEnvOverrides(self: *InferenceEngine) void {
        if (std.posix.getenv("LEANSHIMMY_MAX_OUTPUT")) |v| {
            self.max_output_bytes = std.fmt.parseInt(usize, v, 10) catch self.max_output_bytes;
        }
        if (std.posix.getenv("LEANSHIMMY_EMBED_DIMS")) |v| {
            self.embedding_dims = std.fmt.parseInt(usize, v, 10) catch self.embedding_dims;
        }
    }

    pub fn loadModel(self: *InferenceEngine, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.model_path.len > 0) {
            self.allocator.free(self.model_path);
        }
        self.model_path = try self.allocator.dupe(u8, path);

        if (self.load_fn) |f| {
            const c_path = try self.allocator.dupeZ(u8, path);
            defer self.allocator.free(c_path);
            const rc = f(c_path);
            self.loaded = (rc == 0);
        } else {
            // Fallback stub mode if native lib not present.
            self.loaded = true;
        }
    }

    pub fn isLoaded(self: *InferenceEngine) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.is_loaded_fn) |f| {
            return f() == 1;
        }
        return self.loaded;
    }

    pub fn modelId(self: *InferenceEngine) []const u8 {
        if (self.model_path.len == 0) return "stub-model";
        return self.model_path;
    }

    pub const GenResult = struct { buf: []u8, used: usize };

    pub fn generate(self: *InferenceEngine, prompt: []const u8, max_tokens: u32, temperature: f32) !GenResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.generate_fn) |f| {
            const out_buf = try self.allocator.alloc(u8, self.max_output_bytes);
            const rc = f(prompt.ptr, prompt.len, max_tokens, temperature, out_buf.ptr, out_buf.len);
            if (rc <= 0) {
                self.allocator.free(out_buf);
                return error.GenerationFailed;
            }
            return GenResult{ .buf = out_buf, .used = @intCast(rc) };
        }
        // Stub
        const text = try std.fmt.allocPrint(self.allocator, "stub generation for: {s}", .{prompt});
        return GenResult{ .buf = text, .used = text.len };
    }

    pub const EmbedResult = struct { buf: []f32, used: usize };

    pub fn embed(self: *InferenceEngine, input: []const u8) !EmbedResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.embed_fn) |f| {
            const buf = try self.allocator.alloc(f32, self.embedding_dims);
            const rc = f(input.ptr, input.len, buf.ptr, buf.len);
            if (rc <= 0) {
                self.allocator.free(buf);
                return error.EmbeddingFailed;
            }
            return EmbedResult{ .buf = buf, .used = @intCast(rc) };
        }
        // Stub embedding.
        const dims: usize = 4;
        var buf = try self.allocator.alloc(f32, dims);
        buf[0] = 0.1;
        buf[1] = 0.2;
        buf[2] = 0.3;
        buf[3] = 0.4;
        return EmbedResult{ .buf = buf, .used = dims };
    }
};
