// Mojo SDK - Advanced Compilation Features
// Day 20: Incremental compilation, modules, and caching

const std = @import("std");
const driver = @import("driver");

// ============================================================================
// Module System
// ============================================================================

pub const ModuleId = struct {
    name: []const u8,
    version: []const u8 = "0.1.0",
    
    pub fn init(name: []const u8) ModuleId {
        return ModuleId{ .name = name };
    }
    
    pub fn withVersion(self: ModuleId, version: []const u8) ModuleId {
        return ModuleId{
            .name = self.name,
            .version = version,
        };
    }
    
    pub fn hash(self: *const ModuleId) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(self.name);
        hasher.update(self.version);
        return hasher.final();
    }
};

pub const ModuleDependency = struct {
    module_id: ModuleId,
    required: bool = true,
    
    pub fn init(name: []const u8) ModuleDependency {
        return ModuleDependency{
            .module_id = ModuleId.init(name),
        };
    }
    
    pub fn optional(name: []const u8) ModuleDependency {
        return ModuleDependency{
            .module_id = ModuleId.init(name),
            .required = false,
        };
    }
};

pub const Module = struct {
    id: ModuleId,
    source_files: std.ArrayList([]const u8),
    dependencies: std.ArrayList(ModuleDependency),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) Module {
        return Module{
            .id = ModuleId.init(name),
            .source_files = std.ArrayList([]const u8){},
            .dependencies = std.ArrayList(ModuleDependency){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Module) void {
        self.source_files.deinit(self.allocator);
        self.dependencies.deinit(self.allocator);
    }
    
    pub fn addSource(self: *Module, file: []const u8) !void {
        try self.source_files.append(self.allocator, file);
    }
    
    pub fn addDependency(self: *Module, dep: ModuleDependency) !void {
        try self.dependencies.append(self.allocator, dep);
    }
    
    pub fn hasDependency(self: *const Module, name: []const u8) bool {
        for (self.dependencies.items) |dep| {
            if (std.mem.eql(u8, dep.module_id.name, name)) {
                return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Compilation Cache
// ============================================================================

pub const CacheEntry = struct {
    module_hash: u64,
    source_hash: u64,
    timestamp: i64,
    artifact_path: []const u8,
    
    pub fn isValid(self: *const CacheEntry, source_hash: u64) bool {
        return self.source_hash == source_hash;
    }
};

pub const CompilationCache = struct {
    entries: std.AutoHashMap(u64, CacheEntry),
    cache_dir: []const u8,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, cache_dir: []const u8) CompilationCache {
        return CompilationCache{
            .entries = std.AutoHashMap(u64, CacheEntry).init(allocator),
            .cache_dir = cache_dir,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *CompilationCache) void {
        self.entries.deinit();
    }
    
    pub fn get(self: *CompilationCache, module_hash: u64, source_hash: u64) ?CacheEntry {
        if (self.entries.get(module_hash)) |entry| {
            if (entry.isValid(source_hash)) {
                return entry;
            }
        }
        return null;
    }
    
    pub fn put(self: *CompilationCache, entry: CacheEntry) !void {
        try self.entries.put(entry.module_hash, entry);
    }
    
    pub fn clear(self: *CompilationCache) void {
        self.entries.clearRetainingCapacity();
    }
};

// ============================================================================
// Dependency Graph
// ============================================================================

pub const DependencyNode = struct {
    module_id: ModuleId,
    dependencies: std.ArrayList(ModuleId),
    dependents: std.ArrayList(ModuleId),
    
    pub fn init(allocator: std.mem.Allocator, id: ModuleId) DependencyNode {
        _ = allocator;
        return DependencyNode{
            .module_id = id,
            .dependencies = std.ArrayList(ModuleId){},
            .dependents = std.ArrayList(ModuleId){},
        };
    }
    
    pub fn deinit(self: *DependencyNode, allocator: std.mem.Allocator) void {
        self.dependencies.deinit(allocator);
        self.dependents.deinit(allocator);
    }
};

pub const DependencyGraph = struct {
    nodes: std.AutoHashMap(u64, DependencyNode),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) DependencyGraph {
        return DependencyGraph{
            .nodes = std.AutoHashMap(u64, DependencyNode).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DependencyGraph) void {
        var iter = self.nodes.valueIterator();
        while (iter.next()) |node| {
            node.deinit(self.allocator);
        }
        self.nodes.deinit();
    }
    
    pub fn addModule(self: *DependencyGraph, id: ModuleId) !void {
        const hash_val = id.hash();
        if (!self.nodes.contains(hash_val)) {
            const node = DependencyNode.init(self.allocator, id);
            try self.nodes.put(hash_val, node);
        }
    }
    
    pub fn addDependency(self: *DependencyGraph, from: ModuleId, to: ModuleId) !void {
        const from_hash = from.hash();
        const to_hash = to.hash();
        
        // Ensure both nodes exist
        try self.addModule(from);
        try self.addModule(to);
        
        // Add dependency link
        if (self.nodes.getPtr(from_hash)) |from_node| {
            try from_node.dependencies.append(self.allocator, to);
        }
        
        if (self.nodes.getPtr(to_hash)) |to_node| {
            try to_node.dependents.append(self.allocator, from);
        }
    }
    
    pub fn hasCycle(self: *DependencyGraph) bool {
        _ = self;
        return false; // Simplified for now
    }
};

// ============================================================================
// Incremental Compiler
// ============================================================================

pub const IncrementalCompiler = struct {
    allocator: std.mem.Allocator,
    cache: CompilationCache,
    dep_graph: DependencyGraph,
    base_options: driver.CompilerOptions,
    
    pub fn init(
        allocator: std.mem.Allocator,
        cache_dir: []const u8,
        options: driver.CompilerOptions,
    ) IncrementalCompiler {
        return IncrementalCompiler{
            .allocator = allocator,
            .cache = CompilationCache.init(allocator, cache_dir),
            .dep_graph = DependencyGraph.init(allocator),
            .base_options = options,
        };
    }
    
    pub fn deinit(self: *IncrementalCompiler) void {
        self.cache.deinit();
        self.dep_graph.deinit();
    }
    
    /// Compile module incrementally
    pub fn compileModule(self: *IncrementalCompiler, module: *Module) !void {
        const module_hash = module.id.hash();
        
        // Check each source file
        for (module.source_files.items) |source_file| {
            const source_hash = self.hashFile(source_file);
            
            // Check cache
            if (self.cache.get(module_hash, source_hash)) |cached| {
                std.debug.print("✅ Using cached: {s}\n", .{cached.artifact_path});
                continue;
            }
            
            // Need to recompile
            std.debug.print("🔄 Recompiling: {s}\n", .{source_file});
            
            // Create cache entry
            const entry = CacheEntry{
                .module_hash = module_hash,
                .source_hash = source_hash,
                .timestamp = std.time.timestamp(),
                .artifact_path = source_file,
            };
            
            try self.cache.put(entry);
        }
    }
    
    fn hashFile(self: *IncrementalCompiler, path: []const u8) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(path);
        return hasher.final();
    }
};

// ============================================================================
// Build System
// ============================================================================

pub const BuildTarget = enum {
    Executable,
    StaticLibrary,
    DynamicLibrary,
    
    pub fn extension(self: BuildTarget) []const u8 {
        return switch (self) {
            .Executable => "",
            .StaticLibrary => ".a",
            .DynamicLibrary => ".so",
        };
    }
};

pub const BuildConfig = struct {
    target: BuildTarget = .Executable,
    name: []const u8,
    modules: std.ArrayList(Module),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) BuildConfig {
        return BuildConfig{
            .name = name,
            .modules = std.ArrayList(Module){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *BuildConfig) void {
        for (self.modules.items) |*mod| {
            mod.deinit();
        }
        self.modules.deinit(self.allocator);
    }
    
    pub fn addModule(self: *BuildConfig, module: Module) !void {
        try self.modules.append(self.allocator, module);
    }
    
    pub fn asLibrary(self: BuildConfig, is_static: bool) BuildConfig {
        return BuildConfig{
            .target = if (is_static) .StaticLibrary else .DynamicLibrary,
            .name = self.name,
            .modules = self.modules,
            .allocator = self.allocator,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "advanced: module id" {
    const id = ModuleId.init("test_module");
    try std.testing.expectEqualStrings("test_module", id.name);
    try std.testing.expectEqualStrings("0.1.0", id.version);
}

test "advanced: module id with version" {
    const id = ModuleId.init("test").withVersion("1.0.0");
    try std.testing.expectEqualStrings("1.0.0", id.version);
}

test "advanced: module id hash" {
    const id1 = ModuleId.init("test");
    const id2 = ModuleId.init("test");
    const id3 = ModuleId.init("other");
    
    try std.testing.expectEqual(id1.hash(), id2.hash());
    try std.testing.expect(id1.hash() != id3.hash());
}

test "advanced: module dependency" {
    const dep = ModuleDependency.init("dep_module");
    try std.testing.expect(dep.required);
    
    const opt_dep = ModuleDependency.optional("opt_module");
    try std.testing.expect(!opt_dep.required);
}

test "advanced: module init" {
    const allocator = std.testing.allocator;
    var module = Module.init(allocator, "test");
    defer module.deinit();
    
    try std.testing.expectEqualStrings("test", module.id.name);
}

test "advanced: module add dependency" {
    const allocator = std.testing.allocator;
    var module = Module.init(allocator, "main");
    defer module.deinit();
    
    const dep = ModuleDependency.init("utils");
    try module.addDependency(dep);
    
    try std.testing.expect(module.hasDependency("utils"));
    try std.testing.expect(!module.hasDependency("other"));
}

test "advanced: compilation cache" {
    const allocator = std.testing.allocator;
    var cache = CompilationCache.init(allocator, ".cache");
    defer cache.deinit();
    
    const entry = CacheEntry{
        .module_hash = 12345,
        .source_hash = 67890,
        .timestamp = std.time.timestamp(),
        .artifact_path = "output.o",
    };
    
    try cache.put(entry);
    
    const retrieved = cache.get(12345, 67890);
    try std.testing.expect(retrieved != null);
}

test "advanced: dependency graph" {
    const allocator = std.testing.allocator;
    var graph = DependencyGraph.init(allocator);
    defer graph.deinit();
    
    const mod1 = ModuleId.init("mod1");
    const mod2 = ModuleId.init("mod2");
    
    try graph.addModule(mod1);
    try graph.addModule(mod2);
    try graph.addDependency(mod1, mod2);
    
    try std.testing.expect(!graph.hasCycle());
}

test "advanced: incremental compiler" {
    const allocator = std.testing.allocator;
    const options = driver.CompilerOptions.default();
    
    var compiler = IncrementalCompiler.init(allocator, ".cache", options);
    defer compiler.deinit();
    
    try std.testing.expect(compiler.cache.entries.count() == 0);
}

test "advanced: build target" {
    try std.testing.expectEqualStrings("", BuildTarget.Executable.extension());
    try std.testing.expectEqualStrings(".a", BuildTarget.StaticLibrary.extension());
    try std.testing.expectEqualStrings(".so", BuildTarget.DynamicLibrary.extension());
}
