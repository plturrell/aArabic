//! HuggingFace Model Card Extractor
//! Extracts model specifications, benchmarks, and metadata from HuggingFace model cards
//!
//! Features:
//! - Parse model card README for specifications and benchmarks
//! - Extract structured metadata from HF API
//! - Map pipeline tags to orchestration categories
//! - Download and validate benchmark scores
//! - Support batch processing of MODEL_REGISTRY.json

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Orchestration category mappings
const PIPELINE_TAG_MAPPING = std.ComptimeStringMap([]const []const u8, .{
    .{ "math", &[_][]const u8{"math"} },
    .{ "reasoning", &[_][]const u8{ "reasoning", "math" } },
    .{ "question-answering", &[_][]const u8{"reasoning"} },
    .{ "text-generation", &[_][]const u8{"code"} },
    .{ "code", &[_][]const u8{"code"} },
    .{ "code-generation", &[_][]const u8{"code"} },
    .{ "summarization", &[_][]const u8{"summarization"} },
    .{ "translation", &[_][]const u8{"relational"} },
    .{ "multilingual", &[_][]const u8{"relational"} },
    .{ "feature-extraction", &[_][]const u8{"vector_search"} },
    .{ "sentence-similarity", &[_][]const u8{"vector_search"} },
});

const NAME_PATTERNS = std.ComptimeStringMap([]const []const u8, .{
    .{ "math", &[_][]const u8{ "gsm", "math", "calc" } },
    .{ "code", &[_][]const u8{ "code", "coder", "starcoder", "codegen" } },
    .{ "reasoning", &[_][]const u8{ "reason", "think", "chain-of-thought", "cot" } },
    .{ "vector_search", &[_][]const u8{ "embed", "retriev", "rag" } },
});

pub const ModelInfo = struct {
    allocator: Allocator,
    hf_metadata: std.StringHashMap(std.json.Value),
    specifications: std.StringHashMap([]const u8),
    benchmarks: std.StringHashMap(BenchmarkScore),
    hardware: std.StringHashMap([]const u8),
    orchestration_categories: std.ArrayList([]const u8),
    agent_types: std.ArrayList([]const u8),
    
    pub const BenchmarkScore = struct {
        score: f64,
        date: []const u8,
    };
    
    pub fn init(allocator: Allocator) ModelInfo {
        return .{
            .allocator = allocator,
            .hf_metadata = std.StringHashMap(std.json.Value).init(allocator),
            .specifications = std.StringHashMap([]const u8).init(allocator),
            .benchmarks = std.StringHashMap(BenchmarkScore).init(allocator),
            .hardware = std.StringHashMap([]const u8).init(allocator),
            .orchestration_categories = std.ArrayList([]const u8){},
            .agent_types = std.ArrayList([]const u8){},
        };
    }
    
    pub fn deinit(self: *ModelInfo) void {
        self.hf_metadata.deinit();
        
        var spec_iter = self.specifications.iterator();
        while (spec_iter.next()) |entry| {
            self.specifications.allocator.free(entry.key_ptr.*);
            self.specifications.allocator.free(entry.value_ptr.*);
        }
        self.specifications.deinit();
        
        var bench_iter = self.benchmarks.iterator();
        while (bench_iter.next()) |entry| {
            self.benchmarks.allocator.free(entry.key_ptr.*);
            self.benchmarks.allocator.free(entry.value_ptr.date);
        }
        self.benchmarks.deinit();
        
        var hw_iter = self.hardware.iterator();
        while (hw_iter.next()) |entry| {
            self.hardware.allocator.free(entry.key_ptr.*);
            self.hardware.allocator.free(entry.value_ptr.*);
        }
        self.hardware.deinit();
        
        for (self.orchestration_categories.items) |cat| {
            self.allocator.free(cat);
        }
        self.orchestration_categories.deinit();
        
        for (self.agent_types.items) |agent| {
            self.allocator.free(agent);
        }
        self.agent_types.deinit();
    }
};

pub const HFModelCardExtractor = struct {
    allocator: Allocator,
    verbose: bool,
    http_client: std.http.Client,
    
    pub fn init(allocator: Allocator, verbose: bool) HFModelCardExtractor {
        return .{
            .allocator = allocator,
            .verbose = verbose,
            .http_client = std.http.Client{ .allocator = allocator },
        };
    }
    
    pub fn deinit(self: *HFModelCardExtractor) void {
        self.http_client.deinit();
    }
    
    fn log(self: *HFModelCardExtractor, comptime fmt: []const u8, args: anytype) void {
        if (self.verbose) {
            std.debug.print("[INFO] " ++ fmt ++ "\n", args);
        }
    }
    
    /// Extract comprehensive model information from HuggingFace
    pub fn extractModelInfo(self: *HFModelCardExtractor, hf_repo: []const u8) !ModelInfo {
        self.log("Extracting info for: {s}", .{hf_repo});
        
        var info = ModelInfo.init(self.allocator);
        errdefer info.deinit();
        
        // Fetch API info
        if (self.fetchAPIInfo(hf_repo)) |api_data| {
            defer self.allocator.free(api_data);
            try self.parseAPIInfo(&info, api_data);
        } else |err| {
            std.debug.print("[WARN] Failed to fetch API info: {any}\n", .{err});
        }
        
        // Fetch model card README
        if (self.fetchModelCard(hf_repo)) |readme| {
            defer self.allocator.free(readme);
            try self.extractSpecifications(&info, readme, hf_repo);
            try self.extractBenchmarks(&info, readme);
            try self.extractHardwareInfo(&info, readme);
        } else |err| {
            std.debug.print("[WARN] Failed to fetch model card: {any}\n", .{err});
        }
        
        // Determine categories and agent types
        try self.determineCategories(&info, hf_repo);
        try self.determineAgentTypes(&info);
        
        // Always include inference
        const inference = try self.allocator.dupe(u8, "inference");
        try info.agent_types.insert(0, inference);
        
        return info;
    }
    
    fn fetchAPIInfo(self: *HFModelCardExtractor, hf_repo: []const u8) ![]const u8 {
        _ = self;
        _ = hf_repo;
        // TODO: Re-implement with Zig 0.15.2 HTTP API
        return error.HTTPNotImplemented;
    }
    
    fn fetchModelCard(self: *HFModelCardExtractor, hf_repo: []const u8) ![]const u8 {
        _ = self;
        _ = hf_repo;
        // TODO: Re-implement with Zig 0.15.2 HTTP API
        return error.HTTPNotImplemented;
    }
    
    fn parseAPIInfo(self: *HFModelCardExtractor, info: *ModelInfo, json_data: []const u8) !void {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json_data,
            .{}
        );
        defer parsed.deinit();
        
        const root = parsed.value.object;
        
        // Extract key metadata
        if (root.get("downloads")) |downloads| {
            try info.hf_metadata.put("downloads", downloads);
        }
        if (root.get("likes")) |likes| {
            try info.hf_metadata.put("likes", likes);
        }
        if (root.get("pipeline_tag")) |pipeline| {
            try info.hf_metadata.put("pipeline_tag", pipeline);
        }
    }
    
    fn extractSpecifications(
        self: *HFModelCardExtractor,
        info: *ModelInfo,
        readme: []const u8,
        hf_repo: []const u8
    ) !void {
        // Extract parameter count from repo name
        if (std.mem.indexOf(u8, hf_repo, "B")) |_| {
            // Look for patterns like "7B", "1.5B"
            var it = std.mem.tokenizeAny(u8, hf_repo, "-_/");
            while (it.next()) |token| {
                if (std.mem.endsWith(u8, token, "B")) {
                    const key = try self.allocator.dupe(u8, "parameters");
                    const value = try self.allocator.dupe(u8, token);
                    try info.specifications.put(key, value);
                    break;
                }
            }
        }
        
        // Detect quantization
        if (std.mem.indexOf(u8, hf_repo, "GGUF")) |_| {
            const key = try self.allocator.dupe(u8, "quantization");
            var value: []const u8 = "unknown";
            
            if (std.mem.indexOf(u8, hf_repo, "Q4")) |_| {
                value = "Q4";
            } else if (std.mem.indexOf(u8, hf_repo, "Q8")) |_| {
                value = "Q8";
            } else if (std.mem.indexOf(u8, hf_repo, "Q6")) |_| {
                value = "Q6";
            }
            
            const value_copy = try self.allocator.dupe(u8, value);
            try info.specifications.put(key, value_copy);
        }
        
        _ = readme; // TODO: Parse README for more specs
    }
    
    fn extractBenchmarks(self: *HFModelCardExtractor, info: *ModelInfo, readme: []const u8) !void {
        // Simple regex-like search for benchmark patterns
        const benchmarks = [_]struct { name: []const u8, pattern: []const u8 }{
            .{ .name = "gsm8k", .pattern = "GSM8K" },
            .{ .name = "humaneval", .pattern = "HumanEval" },
            .{ .name = "mbpp", .pattern = "MBPP" },
            .{ .name = "mmlu", .pattern = "MMLU" },
        };
        
        for (benchmarks) |bench| {
            if (std.mem.indexOf(u8, readme, bench.pattern)) |idx| {
                // Look for a number after the benchmark name
                const search_start = idx + bench.pattern.len;
                if (search_start < readme.len) {
                    const remaining = readme[search_start..];
                    
                    // Simple number extraction
                    var score_str = std.ArrayList(u8){};
                    defer score_str.deinit();
                    
                    for (remaining) |c| {
                        if (std.ascii.isDigit(c) or c == '.') {
                            try score_str.append(c);
                        } else if (score_str.items.len > 0) {
                            break;
                        }
                    }
                    
                    if (score_str.items.len > 0) {
                        if (std.fmt.parseFloat(f64, score_str.items)) |score| {
                            const name_copy = try self.allocator.dupe(u8, bench.name);
                            const date_copy = try self.allocator.dupe(u8, "2026-01");
                            try info.benchmarks.put(name_copy, .{
                                .score = score,
                                .date = date_copy,
                            });
                            self.log("Found {s}: {d:.2}", .{ bench.name, score });
                        } else |_| {}
                    }
                }
            }
        }
    }
    
    fn extractHardwareInfo(self: *HFModelCardExtractor, info: *ModelInfo, readme: []const u8) !void {
        // Look for GPU memory patterns
        if (std.mem.indexOf(u8, readme, "GB")) |_| {
            // Simple heuristic: look for patterns like "16GB GPU" or "requires 8GB"
            var it = std.mem.tokenizeAny(u8, readme, " \t\n");
            while (it.next()) |token| {
                if (std.mem.endsWith(u8, token, "GB")) {
                    const key = try self.allocator.dupe(u8, "min_gpu_memory");
                    const value = try self.allocator.dupe(u8, token);
                    try info.hardware.put(key, value);
                    break;
                }
            }
        }
    }
    
    fn determineCategories(self: *HFModelCardExtractor, info: *ModelInfo, hf_repo: []const u8) !void {
        var categories = std.StringHashMap(void).init(self.allocator);
        defer categories.deinit();
        
        // Check repo name patterns
        var it = NAME_PATTERNS.iterator();
        while (it.next()) |entry| {
            const category = entry.key;
            const patterns = entry.value;
            
            for (patterns) |pattern| {
                if (std.mem.indexOf(u8, hf_repo, pattern)) |_| {
                    try categories.put(category, {});
                    break;
                }
            }
        }
        
        // Default to code if nothing found
        if (categories.count() == 0) {
            try categories.put("code", {});
        }
        
        // Convert to array
        var cat_iter = categories.iterator();
        while (cat_iter.next()) |entry| {
            const cat_copy = try self.allocator.dupe(u8, entry.key_ptr.*);
            try info.orchestration_categories.append(cat_copy);
        }
    }
    
    fn determineAgentTypes(self: *HFModelCardExtractor, info: *ModelInfo) !void {
        _ = self;
        _ = info;
        // Agent types determined by categories
        // Always include "inference" which is added in extractModelInfo
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2) {
        const stderr = std.fs.File.stderr();
        try stderr.writeAll("Usage: ");
        try stderr.writeAll(args[0]);
        try stderr.writeAll(" <registry_path> [--test HF_REPO] [--verbose]\n");
        std.process.exit(1);
    }
    
    var verbose = false;
    var test_repo: ?[]const u8 = null;
    
    // Parse options
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "--test")) {
            // Next arg should be repo
            continue;
        } else if (test_repo == null and args.len > 2) {
            test_repo = arg;
        }
    }
    
    var extractor = HFModelCardExtractor.init(allocator, verbose);
    defer extractor.deinit();
    
    if (test_repo) |repo| {
        std.debug.print("\n{'=':**<60}\n", .{});
        std.debug.print("Testing model: {s}\n", .{repo});
        std.debug.print("{'=':**<60}\n\n", .{});
        
        var info = try extractor.extractModelInfo(repo);
        defer info.deinit();
        
        std.debug.print("Categories: ", .{});
        for (info.orchestration_categories.items) |cat| {
            std.debug.print("{s} ", .{cat});
        }
        std.debug.print("\n", .{});
        
        std.debug.print("Benchmarks: {d} found\n", .{info.benchmarks.count()});
        var bench_iter = info.benchmarks.iterator();
        while (bench_iter.next()) |entry| {
            std.debug.print("  {s}: {d:.2}\n", .{ entry.key_ptr.*, entry.value_ptr.score });
        }
    } else {
        std.debug.print("Registry enrichment not yet implemented\n", .{});
        std.debug.print("Use --test HF_REPO to test single model extraction\n", .{});
    }
}
