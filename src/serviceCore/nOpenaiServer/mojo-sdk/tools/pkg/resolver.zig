// Mojo Package Manager - Dependency Resolution
// Day 93-94: Basic resolver + Workspace-aware resolution

const std = @import("std");
const manifest = @import("manifest.zig");
const workspace_mod = @import("workspace.zig");
const Allocator = std.mem.Allocator;

// ============================================================================
// Version Constraints
// ============================================================================

pub const VersionConstraint = struct {
    min_version: ?manifest.Version = null,
    max_version: ?manifest.Version = null,
    exact_version: ?manifest.Version = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) VersionConstraint {
        return VersionConstraint{ .allocator = allocator };
    }
    
    pub fn deinit(self: *VersionConstraint) void {
        if (self.min_version) |*v| v.deinit();
        if (self.max_version) |*v| v.deinit();
        if (self.exact_version) |*v| v.deinit();
    }
    
    /// Parse version constraint (e.g., "^1.0.0", ">=1.0.0", "1.2.3")
    pub fn parse(allocator: Allocator, constraint_str: []const u8) !VersionConstraint {
        var result = VersionConstraint.init(allocator);
        
        if (std.mem.startsWith(u8, constraint_str, "^")) {
            // Caret: ^1.2.3 means >=1.2.3 <2.0.0
            const version = try manifest.Version.parse(allocator, constraint_str[1..]);
            result.min_version = version;
            var max = try manifest.Version.parse(allocator, constraint_str[1..]);
            max.major += 1;
            max.minor = 0;
            max.patch = 0;
            result.max_version = max;
        } else if (std.mem.startsWith(u8, constraint_str, ">=")) {
            result.min_version = try manifest.Version.parse(allocator, constraint_str[2..]);
        } else if (std.mem.startsWith(u8, constraint_str, ">")) {
            result.min_version = try manifest.Version.parse(allocator, constraint_str[1..]);
        } else {
            // Exact version
            result.exact_version = try manifest.Version.parse(allocator, constraint_str);
        }
        
        return result;
    }
    
    /// Check if version satisfies constraint
    pub fn satisfies(self: VersionConstraint, version: manifest.Version) bool {
        if (self.exact_version) |exact| {
            return version.compare(exact) == 0;
        }
        
        if (self.min_version) |min| {
            if (version.compare(min) < 0) return false;
        }
        
        if (self.max_version) |max| {
            if (version.compare(max) >= 0) return false;
        }
        
        return true;
    }
};

// ============================================================================
// Dependency Graph
// ============================================================================

pub const DependencyNode = struct {
    name: []const u8,
    version: manifest.Version,
    dependencies: std.ArrayList(*DependencyNode),
    source: manifest.DependencySource,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, version: manifest.Version, source: manifest.DependencySource) !*DependencyNode {
        const node = try allocator.create(DependencyNode);
        node.* = DependencyNode{
            .name = try allocator.dupe(u8, name),
            .version = version,
            .dependencies = std.ArrayList(*DependencyNode){},
            .source = source,
            .allocator = allocator,
        };
        return node;
    }
    
    pub fn deinit(self: *DependencyNode) void {
        self.allocator.free(self.name);
        self.version.deinit();
        
        for (self.dependencies.items) |dep| {
            dep.deinit();
        }
        self.dependencies.deinit(self.allocator);
        
        self.allocator.destroy(self);
    }
};

pub const DependencyGraph = struct {
    roots: std.ArrayList(*DependencyNode),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DependencyGraph {
        return DependencyGraph{
            .roots = std.ArrayList(*DependencyNode){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DependencyGraph) void {
        for (self.roots.items) |root| {
            root.deinit();
        }
        self.roots.deinit(self.allocator);
    }
};

// ============================================================================
// Resolution Result
// ============================================================================

pub const ResolvedDependency = struct {
    name: []const u8,
    version: manifest.Version,
    source: manifest.DependencySource,
    path: ?[]const u8 = null,  // For path dependencies
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, version: manifest.Version, source: manifest.DependencySource) !ResolvedDependency {
        return ResolvedDependency{
            .name = try allocator.dupe(u8, name),
            .version = version,
            .source = source,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *ResolvedDependency) void {
        self.allocator.free(self.name);
        self.version.deinit();
        if (self.path) |p| self.allocator.free(p);
    }
};

pub const Resolution = struct {
    dependencies: std.ArrayList(ResolvedDependency),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) Resolution {
        return Resolution{
            .dependencies = std.ArrayList(ResolvedDependency){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Resolution) void {
        for (self.dependencies.items) |*dep| {
            dep.deinit();
        }
        self.dependencies.deinit(self.allocator);
    }
};

// ============================================================================
// Basic Dependency Resolver (Day 93)
// ============================================================================

pub const DependencyResolver = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DependencyResolver {
        return DependencyResolver{ .allocator = allocator };
    }
    
    /// Build dependency graph from package manifest
    pub fn buildGraph(self: *DependencyResolver, pkg: manifest.PackageManifest) !DependencyGraph {
        var graph = DependencyGraph.init(self.allocator);
        
        // Add package as root
        const root = try DependencyNode.init(
            self.allocator,
            pkg.name,
            pkg.version,
            .Path  // Root is always a path dependency (current package)
        );
        try graph.roots.append(self.allocator, root);
        
        // Add dependencies (simplified - would recursively resolve in full impl)
        for (pkg.dependencies.items) |dep| {
            const dep_version = if (dep.version) |v|
                try manifest.Version.parse(self.allocator, v)
            else
                manifest.Version.init(self.allocator, 0, 0, 0);
            
            const dep_node = try DependencyNode.init(
                self.allocator,
                dep.name,
                dep_version,
                dep.source
            );
            try root.dependencies.append(self.allocator, dep_node);
        }
        
        return graph;
    }
    
    /// Resolve dependencies (basic version selection)
    pub fn resolve(self: *DependencyResolver, graph: *DependencyGraph) !Resolution {
        var resolution = Resolution.init(self.allocator);
        
        // Process all roots and their dependencies
        for (graph.roots.items) |root| {
            try self.resolveNode(root, &resolution);
        }
        
        return resolution;
    }
    
    fn resolveNode(self: *DependencyResolver, node: *DependencyNode, resolution: *Resolution) !void {
        // Add this node to resolution
        const resolved = try ResolvedDependency.init(
            self.allocator,
            node.name,
            node.version,
            node.source
        );
        try resolution.dependencies.append(self.allocator, resolved);
        
        // Recursively resolve dependencies
        for (node.dependencies.items) |dep| {
            try self.resolveNode(dep, resolution);
        }
    }
    
    /// Check for conflicts in resolution
    pub fn checkConflicts(self: *DependencyResolver, resolution: Resolution) !void {
        // Check for same package with different versions
        var seen = std.StringHashMap(manifest.Version).init(self.allocator);
        defer seen.deinit();
        
        for (resolution.dependencies.items) |dep| {
            if (seen.get(dep.name)) |existing_version| {
                if (existing_version.compare(dep.version) != 0) {
                    return error.VersionConflict;
                }
            } else {
                try seen.put(dep.name, dep.version);
            }
        }
    }
};

// ============================================================================
// Workspace-Aware Resolver (Day 94)
// ============================================================================

pub const WorkspaceResolver = struct {
    allocator: Allocator,
    workspace: ?*workspace_mod.Workspace,
    basic_resolver: DependencyResolver,
    
    pub fn init(allocator: Allocator, ws: ?*workspace_mod.Workspace) WorkspaceResolver {
        return WorkspaceResolver{
            .allocator = allocator,
            .workspace = ws,
            .basic_resolver = DependencyResolver.init(allocator),
        };
    }
    
    /// Resolve dependencies across entire workspace
    pub fn resolveWorkspace(self: *WorkspaceResolver) !Resolution {
        if (self.workspace) |ws| {
            var resolution = Resolution.init(self.allocator);
            
            // Resolve each member
            for (ws.members.items) |member| {
                try self.resolveMember(member, &resolution);
            }
            
            // Check for conflicts across workspace
            try self.unifyResolutions(&resolution);
            
            return resolution;
        }
        
        return error.NoWorkspace;
    }
    
    fn resolveMember(self: *WorkspaceResolver, member: workspace_mod.Member, resolution: *Resolution) !void {
        // Build graph for this member
        var graph = try self.basic_resolver.buildGraph(member.manifest);
        defer graph.deinit();
        
        // Resolve member's dependencies
        var member_resolution = try self.basic_resolver.resolve(&graph);
        defer member_resolution.deinit();
        
        // Merge into workspace resolution
        for (member_resolution.dependencies.items) |dep| {
            const resolved = try ResolvedDependency.init(
                self.allocator,
                dep.name,
                dep.version,
                dep.source
            );
            try resolution.dependencies.append(self.allocator, resolved);
        }
    }
    
    /// Resolve path dependencies (workspace members reference each other)
    pub fn resolvePathDeps(self: *WorkspaceResolver, member: workspace_mod.Member) !std.ArrayList(PathDependency) {
        var path_deps = std.ArrayList(PathDependency){};
        
        for (member.manifest.dependencies.items) |dep| {
            if (dep.source == .Path) {
                const path_dep = try PathDependency.init(
                    self.allocator,
                    dep.name,
                    dep.path orelse return error.MissingPath
                );
                try path_deps.append(self.allocator, path_dep);
            }
        }
        
        return path_deps;
    }
    
    /// Unify resolutions across workspace (detect conflicts)
    fn unifyResolutions(self: *WorkspaceResolver, resolution: *Resolution) !void {
        var version_map = std.StringHashMap(manifest.Version).init(self.allocator);
        defer version_map.deinit();
        
        for (resolution.dependencies.items) |dep| {
            if (version_map.get(dep.name)) |existing| {
                // Check if versions are compatible
                if (existing.compare(dep.version) != 0) {
                    std.debug.print("Conflict: {s} needs both v{} and v{}\n", .{
                        dep.name, existing, dep.version
                    });
                    return error.VersionConflict;
                }
            } else {
                try version_map.put(dep.name, dep.version);
            }
        }
    }
};

// ============================================================================
// Path Dependencies
// ============================================================================

pub const PathDependency = struct {
    name: []const u8,
    path: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, path: []const u8) !PathDependency {
        return PathDependency{
            .name = try allocator.dupe(u8, name),
            .path = try allocator.dupe(u8, path),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *PathDependency) void {
        self.allocator.free(self.name);
        self.allocator.free(self.path);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "VersionConstraint: exact version" {
    var constraint = try VersionConstraint.parse(std.testing.allocator, "1.2.3");
    defer constraint.deinit();
    
    var v1 = try manifest.Version.parse(std.testing.allocator, "1.2.3");
    defer v1.deinit();
    
    try std.testing.expect(constraint.satisfies(v1));
}

test "VersionConstraint: caret range" {
    var constraint = try VersionConstraint.parse(std.testing.allocator, "^1.2.0");
    defer constraint.deinit();
    
    var v1 = try manifest.Version.parse(std.testing.allocator, "1.3.0");
    defer v1.deinit();
    try std.testing.expect(constraint.satisfies(v1));
    
    var v2 = try manifest.Version.parse(std.testing.allocator, "2.0.0");
    defer v2.deinit();
    try std.testing.expect(!constraint.satisfies(v2));
}

test "DependencyNode: creation" {
    const version = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    var node = try DependencyNode.init(std.testing.allocator, "test-pkg", version, .Registry);
    defer node.deinit();
    
    try std.testing.expectEqualStrings("test-pkg", node.name);
}

test "DependencyGraph: empty graph" {
    var graph = DependencyGraph.init(std.testing.allocator);
    defer graph.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), graph.roots.items.len);
}

test "DependencyResolver: build graph" {
    var resolver = DependencyResolver.init(std.testing.allocator);
    
    const version = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    var pkg = try manifest.PackageManifest.init(std.testing.allocator, "test-pkg", version);
    defer pkg.deinit();
    
    var graph = try resolver.buildGraph(pkg);
    defer graph.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), graph.roots.items.len);
}

test "DependencyResolver: resolve empty" {
    var resolver = DependencyResolver.init(std.testing.allocator);
    
    const version = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    var pkg = try manifest.PackageManifest.init(std.testing.allocator, "test-pkg", version);
    defer pkg.deinit();
    
    var graph = try resolver.buildGraph(pkg);
    defer graph.deinit();
    
    var resolution = try resolver.resolve(&graph);
    defer resolution.deinit();
    
    try std.testing.expect(resolution.dependencies.items.len >= 1);
}

test "ResolvedDependency: creation" {
    const version = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    var resolved = try ResolvedDependency.init(std.testing.allocator, "test", version, .Registry);
    defer resolved.deinit();
    
    try std.testing.expectEqualStrings("test", resolved.name);
}

test "Resolution: multiple dependencies" {
    var resolution = Resolution.init(std.testing.allocator);
    defer resolution.deinit();
    
    const v1 = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    const dep1 = try ResolvedDependency.init(std.testing.allocator, "dep1", v1, .Registry);
    try resolution.dependencies.append(std.testing.allocator, dep1);
    
    const v2 = try manifest.Version.parse(std.testing.allocator, "2.0.0");
    const dep2 = try ResolvedDependency.init(std.testing.allocator, "dep2", v2, .Registry);
    try resolution.dependencies.append(std.testing.allocator, dep2);
    
    try std.testing.expectEqual(@as(usize, 2), resolution.dependencies.items.len);
}

test "WorkspaceResolver: path dependencies" {
    var resolver = WorkspaceResolver.init(std.testing.allocator, null);
    
    const version = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    var pkg = try manifest.PackageManifest.init(std.testing.allocator, "test-member", version);
    defer pkg.deinit();
    
    // Add path dependency
    var dep = try manifest.Dependency.init(std.testing.allocator, "mojo-sdk", .Path);
    dep.path = try std.testing.allocator.dupe(u8, "../../mojo-sdk");
    try pkg.dependencies.append(std.testing.allocator, dep);
    
    const member = workspace_mod.Member{
        .name = "test-member",
        .path = "services/test",
        .manifest = pkg,
        .allocator = std.testing.allocator,
    };
    
    var path_deps = try resolver.resolvePathDeps(member);
    defer {
        for (path_deps.items) |*pd| {
            pd.deinit();
        }
        path_deps.deinit(std.testing.allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 1), path_deps.items.len);
    try std.testing.expectEqualStrings("mojo-sdk", path_deps.items[0].name);
}

test "PathDependency: creation" {
    var path_dep = try PathDependency.init(std.testing.allocator, "mojo-sdk", "../../mojo-sdk");
    defer path_dep.deinit();
    
    try std.testing.expectEqualStrings("mojo-sdk", path_dep.name);
    try std.testing.expectEqualStrings("../../mojo-sdk", path_dep.path);
}

test "DependencyResolver: conflict detection" {
    var resolver = DependencyResolver.init(std.testing.allocator);
    
    var resolution = Resolution.init(std.testing.allocator);
    defer resolution.deinit();
    
    // Add same package with same version - should be ok
    const v1 = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    const dep1 = try ResolvedDependency.init(std.testing.allocator, "pkg", v1, .Registry);
    try resolution.dependencies.append(std.testing.allocator, dep1);
    
    const v2 = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    const dep2 = try ResolvedDependency.init(std.testing.allocator, "pkg", v2, .Registry);
    try resolution.dependencies.append(std.testing.allocator, dep2);
    
    // Should not error - same version
    try resolver.checkConflicts(resolution);
}

test "WorkspaceResolver: without workspace" {
    var resolver = WorkspaceResolver.init(std.testing.allocator, null);
    
    const result = resolver.resolveWorkspace();
    
    try std.testing.expectError(error.NoWorkspace, result);
}
