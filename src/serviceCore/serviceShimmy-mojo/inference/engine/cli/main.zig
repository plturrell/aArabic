const std = @import("std");
const gguf_loader = @import("gguf_loader");
const llama_model = @import("llama_model");
const gguf_model_loader = @import("gguf_model_loader");
const batch_processor = @import("batch_processor");
const performance = @import("performance");
const sampler = @import("sampler");

/// CLI Interface for Zig Inference Engine
/// Provides command-line access to model loading and generation

const VERSION = "0.2.0";

const SamplingStrategy = enum {
    greedy,
    temperature,
    top_k,
    top_p,
    
    pub fn fromString(s: []const u8) ?SamplingStrategy {
        if (std.mem.eql(u8, s, "greedy")) return .greedy;
        if (std.mem.eql(u8, s, "temperature")) return .temperature;
        if (std.mem.eql(u8, s, "top-k") or std.mem.eql(u8, s, "topk")) return .top_k;
        if (std.mem.eql(u8, s, "top-p") or std.mem.eql(u8, s, "topp") or std.mem.eql(u8, s, "nucleus")) return .top_p;
        return null;
    }
};

const CliArgs = struct {
    model_path: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    max_tokens: u32 = 100,
    strategy: SamplingStrategy = .greedy,
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    batch_size: u32 = 8,
    show_stats: bool = false,
    help: bool = false,
    version: bool = false,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Parse arguments
    const args = try parseArgs(allocator);
    
    if (args.help) {
        printHelp();
        return;
    }
    
    if (args.version) {
        printVersion();
        return;
    }
    
    if (args.model_path == null) {
        std.debug.print("Error: --model <path> is required\n\n", .{});
        printHelp();
        return error.MissingModelPath;
    }
    
    // Run inference
    try runInference(allocator, args);
}

fn parseArgs(allocator: std.mem.Allocator) !CliArgs {
    var args = CliArgs{};
    
    var arg_iter = try std.process.argsWithAllocator(allocator);
    defer arg_iter.deinit();
    
    // Skip program name
    _ = arg_iter.skip();
    
    while (arg_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            args.help = true;
        } else if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            args.version = true;
        } else if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            if (arg_iter.next()) |path| {
                args.model_path = path;
            }
        } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
            if (arg_iter.next()) |prompt| {
                args.prompt = prompt;
            }
        } else if (std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) {
            if (arg_iter.next()) |val| {
                args.max_tokens = try std.fmt.parseInt(u32, val, 10);
            }
        } else if (std.mem.eql(u8, arg, "--strategy") or std.mem.eql(u8, arg, "-s")) {
            if (arg_iter.next()) |val| {
                args.strategy = SamplingStrategy.fromString(val) orelse .greedy;
            }
        } else if (std.mem.eql(u8, arg, "--temperature") or std.mem.eql(u8, arg, "-t")) {
            if (arg_iter.next()) |val| {
                args.temperature = try std.fmt.parseFloat(f32, val);
            }
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            if (arg_iter.next()) |val| {
                args.top_k = try std.fmt.parseInt(u32, val, 10);
            }
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            if (arg_iter.next()) |val| {
                args.top_p = try std.fmt.parseFloat(f32, val);
            }
        } else if (std.mem.eql(u8, arg, "--batch-size") or std.mem.eql(u8, arg, "-b")) {
            if (arg_iter.next()) |val| {
                args.batch_size = try std.fmt.parseInt(u32, val, 10);
            }
        } else if (std.mem.eql(u8, arg, "--stats")) {
            args.show_stats = true;
        }
    }
    
    return args;
}

fn printHelp() void {
    std.debug.print(
        \\Zig Inference Engine - CLI Interface
        \\
        \\USAGE:
        \\    zig-inference [OPTIONS]
        \\
        \\OPTIONS:
        \\    -m, --model <path>           Path to GGUF model file (required)
        \\    -p, --prompt <text>          Input prompt text
        \\    -n, --max-tokens <num>       Maximum tokens to generate (default: 100)
        \\    -s, --strategy <name>        Sampling strategy: greedy, temperature, top-k, top-p (default: greedy)
        \\    -t, --temperature <float>    Sampling temperature (default: 0.7)
        \\    --top-k <num>                Top-k value for top-k sampling (default: 40)
        \\    --top-p <float>              Top-p value for nucleus sampling (default: 0.9)
        \\    -b, --batch-size <num>       Batch size for prompt processing (default: 8)
        \\    --stats                      Show performance statistics
        \\    -h, --help                   Show this help message
        \\    -v, --version                Show version information
        \\
        \\EXAMPLES:
        \\    # Greedy sampling (deterministic)
        \\    zig-inference -m model.gguf -p "Hello, world!" -s greedy
        \\
        \\    # Temperature sampling
        \\    zig-inference -m model.gguf -p "Once upon a time" -s temperature -t 0.8
        \\
        \\    # Top-k sampling
        \\    zig-inference -m model.gguf -p "The quick brown fox" -s top-k --top-k 40 -t 1.0
        \\
        \\    # Top-p (nucleus) sampling for best quality
        \\    zig-inference -m model.gguf -p "Explain quantum computing" -s top-p --top-p 0.9 -t 0.7
        \\
        \\    # Show performance stats
        \\    zig-inference -m model.gguf -p "Test" --stats
        \\
    , .{});
}

fn printVersion() void {
    std.debug.print("Zig Inference Engine v{s}\n", .{VERSION});
}

fn runInference(allocator: std.mem.Allocator, args: CliArgs) !void {
    std.debug.print("\n🚀 Zig Inference Engine v{s}\n", .{VERSION});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
    
    var total_timer = performance.Timer.start_timer();
    
    // Load model
    std.debug.print("📂 Loading model: {s}\n", .{args.model_path.?});
    var load_timer = performance.Timer.start_timer();
    
    var loader = gguf_model_loader.GGUFModelLoader.init(
        allocator,
        .OnTheFly,
    );
    
    var model = try loader.loadModel(args.model_path.?);
    defer model.deinit();
    
    const load_time = load_timer.elapsed_ms();
    std.debug.print("   ✅ Model loaded in {d:.2} ms\n", .{load_time});
    std.debug.print("   📊 Model info:\n", .{});
    std.debug.print("      - Vocabulary: {d} tokens\n", .{model.config.vocab_size});
    std.debug.print("      - Layers: {d}\n", .{model.config.n_layers});
    std.debug.print("      - Embedding: {d}\n", .{model.config.embed_dim});
    std.debug.print("      - Context: {d} tokens\n\n", .{model.config.max_seq_len});
    
    // Get prompt
    const prompt = args.prompt orelse "Hello, world!";
    std.debug.print("💬 Prompt: \"{s}\"\n\n", .{prompt});
    
    // Tokenize prompt
    std.debug.print("🔤 Tokenizing prompt...\n", .{});
    const tokens = try model.tok.encode(prompt, allocator);
    defer allocator.free(tokens);
    
    std.debug.print("   ✅ Tokenized to {d} tokens\n\n", .{tokens.len});
    
    // Process prompt with batch processor
    if (tokens.len > 1) {
        std.debug.print("⚡ Processing prompt in batches (batch_size={d})...\n", .{args.batch_size});
        
        const batch_config = batch_processor.BatchConfig{
            .max_batch_size = args.batch_size,
            .enable_parallel = false,
        };
        
        var batch_model = try batch_processor.BatchLlamaModel.init(
            allocator,
            &model,
            batch_config,
        );
        defer batch_model.deinit();
        
        var prompt_timer = performance.Timer.start_timer();
        try batch_model.processPromptBatch(tokens, args.batch_size);
        const prompt_time = prompt_timer.elapsed_ms();
        
        std.debug.print("   ✅ Prompt processed in {d:.2} ms\n", .{prompt_time});
        std.debug.print("   ⚡ Speed: {d:.1} tokens/sec\n\n", .{
            @as(f64, @floatFromInt(tokens.len)) / (prompt_time / 1000.0),
        });
    }
    
    // Set up sampler
    const sampling_config = switch (args.strategy) {
        .greedy => sampler.SamplingConfig.greedy(),
        .temperature => sampler.SamplingConfig.withTemperature(args.temperature),
        .top_k => sampler.SamplingConfig.topK(args.top_k, args.temperature),
        .top_p => sampler.SamplingConfig.topP(args.top_p, args.temperature),
    };
    
    var token_sampler = sampler.Sampler.init(allocator, sampling_config);
    
    // Display sampling info
    const strategy_name = switch (args.strategy) {
        .greedy => "Greedy (deterministic)",
        .temperature => "Temperature",
        .top_k => "Top-k",
        .top_p => "Top-p (nucleus)",
    };
    
    std.debug.print("✨ Generating {d} tokens (strategy: {s})\n", .{args.max_tokens, strategy_name});
    if (args.strategy != .greedy) {
        std.debug.print("   Temperature: {d:.2}\n", .{args.temperature});
    }
    if (args.strategy == .top_k) {
        std.debug.print("   Top-k: {d}\n", .{args.top_k});
    }
    if (args.strategy == .top_p) {
        std.debug.print("   Top-p: {d:.2}\n", .{args.top_p});
    }
    std.debug.print("\n", .{});
    
    var generated_count: u32 = 0;
    var gen_timer = performance.Timer.start_timer();
    
    std.debug.print("Output: ", .{});
    
    var current_pos: u32 = @intCast(tokens.len - 1);
    var last_token: u32 = tokens[tokens.len - 1];
    
    while (generated_count < args.max_tokens) : (generated_count += 1) {
        // Forward pass
        const logits = try model.forward(last_token, current_pos);
        defer allocator.free(logits);
        
        // Sample next token using configured strategy
        const next_token = try token_sampler.sample(logits);
        
        // Decode token
        const token_text = try model.tok.decode(&[_]u32{next_token}, allocator);
        defer allocator.free(token_text);
        
        std.debug.print("{s}", .{token_text});
        
        // Update for next iteration
        last_token = next_token;
        current_pos += 1;
        
        // Check for EOS
        if (next_token == 2) break; // Assume token 2 is EOS
    }
    
    std.debug.print("\n\n", .{});
    
    const gen_time = gen_timer.elapsed_ms();
    const total_time = total_timer.elapsed_ms();
    
    // Show statistics
    if (args.show_stats) {
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
        std.debug.print("📊 Performance Statistics\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
        
        std.debug.print("Model Loading:     {d:.2} ms\n", .{load_time});
        std.debug.print("Prompt Processing: {d:.2} ms ({d} tokens)\n", .{
            0.0, // Would need to track this separately
            tokens.len,
        });
        std.debug.print("Token Generation:  {d:.2} ms ({d} tokens)\n", .{
            gen_time,
            generated_count,
        });
        std.debug.print("Total Time:        {d:.2} ms\n\n", .{total_time});
        
        std.debug.print("Generation Speed:  {d:.1} tokens/sec\n", .{
            @as(f64, @floatFromInt(generated_count)) / (gen_time / 1000.0),
        });
        std.debug.print("Overall Speed:     {d:.1} tokens/sec\n\n", .{
            @as(f64, @floatFromInt(tokens.len + generated_count)) / (total_time / 1000.0),
        });
    }
    
    std.debug.print("✅ Inference complete!\n\n", .{});
}
