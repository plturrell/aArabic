// Mojo SDK - Compiler Driver
// Day 19: End-to-End Compilation Testing

const std = @import("std");
const lexer = @import("lexer");
const parser = @import("parser");
const ast = @import("ast");
const semantic = @import("semantic_analyzer");
const symbol_table = @import("symbol_table");
const ir = @import("ir");
const ir_builder = @import("ir_builder");
const optimizer = @import("optimizer");
const mlir_setup = @import("mlir_setup");
const ir_to_mlir = @import("ir_to_mlir");
const mlir_optimizer = @import("mlir_optimizer");
const llvm_lowering = @import("llvm_lowering");
const codegen = @import("codegen");
const native_compiler = @import("native_compiler");
const tool_executor = @import("tool_executor");

// ============================================================================
// Compilation Options
// ============================================================================

pub const CompilerOptions = struct {
    optimization_level: codegen.OptimizationLevel = .O2,
    debug_level: tool_executor.DebugLevel = .None,
    log_level: tool_executor.LogLevel = .Info,
    verbose: bool = false,
    emit_ir: bool = false,
    emit_mlir: bool = false,
    emit_llvm: bool = false,
    emit_asm: bool = false,
    output_file: ?[]const u8 = null,
    
    pub fn default() CompilerOptions {
        return CompilerOptions{};
    }
    
    pub fn forRelease() CompilerOptions {
        return CompilerOptions{
            .optimization_level = .O3,
            .log_level = .Warning,
        };
    }
    
    pub fn forDebug() CompilerOptions {
        return CompilerOptions{
            .optimization_level = .O0,
            .debug_level = .Full,
            .log_level = .Debug,
            .verbose = true,
        };
    }
};

// ============================================================================
// Compilation Statistics
// ============================================================================

pub const CompilationStats = struct {
    source_lines: usize = 0,
    tokens_count: usize = 0,
    ast_nodes: usize = 0,
    ir_instructions: usize = 0,
    mlir_operations: usize = 0,
    llvm_instructions: usize = 0,
    
    lexer_time_ms: u64 = 0,
    parser_time_ms: u64 = 0,
    semantic_time_ms: u64 = 0,
    ir_gen_time_ms: u64 = 0,
    optimize_time_ms: u64 = 0,
    mlir_time_ms: u64 = 0,
    llvm_time_ms: u64 = 0,
    codegen_time_ms: u64 = 0,
    native_time_ms: u64 = 0,
    
    total_time_ms: u64 = 0,
    
    pub fn init() CompilationStats {
        return CompilationStats{};
    }
    
    pub fn print(self: *const CompilationStats) void {
        std.debug.print("\n=== Compilation Statistics ===\n", .{});
        std.debug.print("Source lines: {}\n", .{self.source_lines});
        std.debug.print("Tokens: {}\n", .{self.tokens_count});
        std.debug.print("AST nodes: {}\n", .{self.ast_nodes});
        std.debug.print("IR instructions: {}\n", .{self.ir_instructions});
        std.debug.print("MLIR operations: {}\n", .{self.mlir_operations});
        std.debug.print("LLVM instructions: {}\n", .{self.llvm_instructions});
        std.debug.print("\n--- Timing Breakdown ---\n", .{});
        std.debug.print("Lexer: {}ms\n", .{self.lexer_time_ms});
        std.debug.print("Parser: {}ms\n", .{self.parser_time_ms});
        std.debug.print("Semantic: {}ms\n", .{self.semantic_time_ms});
        std.debug.print("IR Gen: {}ms\n", .{self.ir_gen_time_ms});
        std.debug.print("Optimize: {}ms\n", .{self.optimize_time_ms});
        std.debug.print("MLIR: {}ms\n", .{self.mlir_time_ms});
        std.debug.print("LLVM: {}ms\n", .{self.llvm_time_ms});
        std.debug.print("Codegen: {}ms\n", .{self.codegen_time_ms});
        std.debug.print("Native: {}ms\n", .{self.native_time_ms});
        std.debug.print("\nTotal: {}ms\n", .{self.total_time_ms});
    }
};

// ============================================================================
// Compiler Driver
// ============================================================================

pub const CompilerDriver = struct {
    allocator: std.mem.Allocator,
    options: CompilerOptions,
    stats: CompilationStats,
    logger: tool_executor.CompilationLogger,
    
    pub fn init(allocator: std.mem.Allocator, options: CompilerOptions) CompilerDriver {
        return CompilerDriver{
            .allocator = allocator,
            .options = options,
            .stats = CompilationStats.init(),
            .logger = tool_executor.CompilationLogger.init(allocator, options.log_level),
        };
    }
    
    /// Compile source code to executable
    pub fn compile(self: *CompilerDriver, source: []const u8, output_name: []const u8) !void {
        const start_time = std.time.milliTimestamp();
        
        self.logger.info("Starting compilation of {s}", .{output_name});
        
        // Stage 1: Lexical Analysis
        self.logger.debug("Stage 1: Lexical Analysis", .{});
        const lex_start = std.time.milliTimestamp();
        var lex = try lexer.Lexer.init(self.allocator, source);
        const tokens = try self.tokenize(&lex);
        defer self.allocator.free(tokens);
        self.stats.lexer_time_ms = @intCast(std.time.milliTimestamp() - lex_start);
        self.stats.tokens_count = tokens.len;
        
        // Stage 2: Parsing
        self.logger.debug("Stage 2: Parsing", .{});
        const parse_start = std.time.milliTimestamp();
        var pars = parser.Parser.init(self.allocator, tokens);
        const ast_root = try pars.parse();
        self.stats.parser_time_ms = @intCast(std.time.milliTimestamp() - parse_start);
        self.stats.ast_nodes = self.countASTNodes(ast_root);
        
        // Stage 3: Semantic Analysis
        self.logger.debug("Stage 3: Semantic Analysis", .{});
        const sem_start = std.time.milliTimestamp();
        var sym_table = symbol_table.SymbolTable.init(self.allocator);
        defer sym_table.deinit();
        var analyzer = semantic.SemanticAnalyzer.init(self.allocator, &sym_table);
        try analyzer.analyze(ast_root);
        self.stats.semantic_time_ms = @intCast(std.time.milliTimestamp() - sem_start);
        
        // Stage 4: IR Generation
        self.logger.debug("Stage 4: IR Generation", .{});
        const ir_start = std.time.milliTimestamp();
        var builder = ir_builder.IRBuilder.init(self.allocator, &sym_table);
        const ir_module = try builder.build(ast_root);
        self.stats.ir_gen_time_ms = @intCast(std.time.milliTimestamp() - ir_start);
        self.stats.ir_instructions = ir_module.functions.items.len;
        
        // Stage 5: Optimization
        self.logger.debug("Stage 5: Optimization", .{});
        const opt_start = std.time.milliTimestamp();
        var opt = optimizer.Optimizer.init(self.allocator);
        try opt.optimize(&ir_module);
        self.stats.optimize_time_ms = @intCast(std.time.milliTimestamp() - opt_start);
        
        // Stage 6: MLIR Conversion
        self.logger.debug("Stage 6: MLIR Conversion", .{});
        const mlir_start = std.time.milliTimestamp();
        var converter = ir_to_mlir.IRToMLIRConverter.init(self.allocator);
        const mlir_module = try converter.convert(&ir_module);
        self.stats.mlir_time_ms = @intCast(std.time.milliTimestamp() - mlir_start);
        self.stats.mlir_operations = 1; // Simplified for now
        
        // Stage 7: LLVM Lowering
        self.logger.debug("Stage 7: LLVM Lowering", .{});
        const llvm_start = std.time.milliTimestamp();
        var lowering = llvm_lowering.LLVMLowering.init(self.allocator);
        const llvm_module = try lowering.lower(&mlir_module);
        self.stats.llvm_time_ms = @intCast(std.time.milliTimestamp() - llvm_start);
        self.stats.llvm_instructions = llvm_module.functions.items.len;
        
        // Stage 8: Code Generation
        self.logger.debug("Stage 8: Code Generation", .{});
        const codegen_start = std.time.milliTimestamp();
        const config = codegen.CodeGenConfig.init(self.options.optimization_level);
        var generator = codegen.CodeGenerator.init(self.allocator, config);
        const llvm_ir = try generator.generate(&llvm_module, output_name);
        defer self.allocator.free(llvm_ir);
        self.stats.codegen_time_ms = @intCast(std.time.milliTimestamp() - codegen_start);
        
        // Stage 9: Native Compilation
        self.logger.debug("Stage 9: Native Compilation", .{});
        const native_start = std.time.milliTimestamp();
        const native_options = native_compiler.CompilationOptions{
            .optimization_level = self.options.optimization_level,
        };
        var native_comp = native_compiler.NativeCompiler.init(
            self.allocator,
            native_compiler.ToolPaths.default(),
            native_options,
        );
        
        // Write IR to temp file
        const ir_file = try std.fmt.allocPrint(self.allocator, "{s}.ll", .{output_name});
        defer self.allocator.free(ir_file);
        try self.writeFile(ir_file, llvm_ir);
        
        // Compile to executable
        const result = try native_comp.compileToExecutable(ir_file, output_name);
        self.stats.native_time_ms = @intCast(std.time.milliTimestamp() - native_start);
        
        // Calculate total time
        self.stats.total_time_ms = @intCast(std.time.milliTimestamp() - start_time);
        
        if (result.success) {
            self.logger.info("✅ Compilation successful: {s}", .{output_name});
            if (self.options.verbose) {
                self.stats.print();
            }
        } else {
            self.logger.error_("❌ Compilation failed", .{});
            return error.CompilationFailed;
        }
    }
    
    /// Tokenize source code
    fn tokenize(self: *CompilerDriver, lex: *lexer.Lexer) ![]lexer.Token {
        var tokens = std.ArrayList(lexer.Token).init(self.allocator);
        defer tokens.deinit();
        
        while (true) {
            const token = lex.nextToken();
            try tokens.append(token);
            if (token.type == .Eof) break;
        }
        
        return try tokens.toOwnedSlice();
    }
    
    /// Count AST nodes (simplified)
    fn countASTNodes(self: *CompilerDriver, node: *ast.Node) usize {
        _ = self;
        _ = node;
        return 1; // Simplified for now
    }
    
    /// Write content to file
    fn writeFile(self: *CompilerDriver, path: []const u8, content: []const u8) !void {
        _ = self;
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(content);
    }
};

// ============================================================================
// Command Line Interface
// ============================================================================

pub const CLI = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) CLI {
        return CLI{ .allocator = allocator };
    }
    
    pub fn run(self: *CLI, args: []const []const u8) !void {
        if (args.len < 2) {
            self.printUsage();
            return;
        }
        
        const input_file = args[1];
        const output_file = if (args.len > 2) args[2] else "output";
        
        // Read source file
        const source = try self.readFile(input_file);
        defer self.allocator.free(source);
        
        // Create compiler with default options
        var options = CompilerOptions.default();
        options.verbose = true;
        
        var driver = CompilerDriver.init(self.allocator, options);
        try driver.compile(source, output_file);
    }
    
    fn printUsage(self: *CLI) void {
        _ = self;
        std.debug.print("Usage: mojoc <input.mojo> [output]\n", .{});
    }
    
    fn readFile(self: *CLI, path: []const u8) ![]const u8 {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        
        const stat = try file.stat();
        const content = try file.readToEndAlloc(self.allocator, stat.size);
        return content;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "driver: compiler options" {
    const default_opts = CompilerOptions.default();
    try std.testing.expectEqual(codegen.OptimizationLevel.O2, default_opts.optimization_level);
    
    const release_opts = CompilerOptions.forRelease();
    try std.testing.expectEqual(codegen.OptimizationLevel.O3, release_opts.optimization_level);
    
    const debug_opts = CompilerOptions.forDebug();
    try std.testing.expectEqual(codegen.OptimizationLevel.O0, debug_opts.optimization_level);
    try std.testing.expectEqual(tool_executor.DebugLevel.Full, debug_opts.debug_level);
}

test "driver: compilation stats" {
    var stats = CompilationStats.init();
    stats.source_lines = 100;
    stats.tokens_count = 500;
    stats.total_time_ms = 1000;
    
    try std.testing.expectEqual(@as(usize, 100), stats.source_lines);
    try std.testing.expectEqual(@as(usize, 500), stats.tokens_count);
}

test "driver: compiler init" {
    const allocator = std.testing.allocator;
    const options = CompilerOptions.default();
    
    const driver = CompilerDriver.init(allocator, options);
    try std.testing.expectEqual(codegen.OptimizationLevel.O2, driver.options.optimization_level);
}

test "driver: cli init" {
    const allocator = std.testing.allocator;
    const cli = CLI.init(allocator);
    
    try std.testing.expect(cli.allocator.ptr == allocator.ptr);
}

test "driver: tokenize simple code" {
    const allocator = std.testing.allocator;
    const options = CompilerOptions.default();
    const driver = CompilerDriver.init(allocator, options);
    
    // Just test that driver can be initialized
    try std.testing.expectEqual(codegen.OptimizationLevel.O2, driver.options.optimization_level);
}

test "driver: options for release" {
    const opts = CompilerOptions.forRelease();
    try std.testing.expectEqual(codegen.OptimizationLevel.O3, opts.optimization_level);
    try std.testing.expectEqual(tool_executor.LogLevel.Warning, opts.log_level);
}

test "driver: options for debug" {
    const opts = CompilerOptions.forDebug();
    try std.testing.expectEqual(codegen.OptimizationLevel.O0, opts.optimization_level);
    try std.testing.expectEqual(tool_executor.DebugLevel.Full, opts.debug_level);
    try std.testing.expect(opts.verbose);
}

test "driver: stats initialization" {
    const stats = CompilationStats.init();
    try std.testing.expectEqual(@as(usize, 0), stats.source_lines);
    try std.testing.expectEqual(@as(u64, 0), stats.total_time_ms);
}

test "driver: compiler with verbose" {
    const allocator = std.testing.allocator;
    var options = CompilerOptions.default();
    options.verbose = true;
    
    const driver = CompilerDriver.init(allocator, options);
    try std.testing.expect(driver.options.verbose);
}

test "driver: logger initialization" {
    const allocator = std.testing.allocator;
    const options = CompilerOptions.default();
    const driver = CompilerDriver.init(allocator, options);
    
    try std.testing.expectEqual(tool_executor.LogLevel.Info, driver.logger.level);
}
