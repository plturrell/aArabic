// Mojo Package Manager - Workspace Detection & Resolution
// Day 92: Multi-module workspace support

const std = @import("std");
const manifest = @import("manifest.zig");
const Allocator = std.mem.Allocator;

// ============================================================================
// Workspace Root Detection
// ============================================================================

pub const WorkspaceDetector = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) WorkspaceDetector {
        return WorkspaceDetector{ .allocator = allocator };
    }
    
    /// Find workspace root by walking up directory tree
    /// Returns null if no workspace found (standalone package)
    pub fn findWorkspaceRoot(self: *WorkspaceDetector, start_path: []const u8) !?[]const u8 {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const abs_path = try std.fs.cwd().realpath(start_path, &path_buf);
        
        var current = try self.allocator.dupe(u8, abs_path);
        defer self.allocator.free(current);
        
        // Walk up directories looking for mojo.toml with [workspace]
        while (true) {
            const mojo_toml = try std.fs.path.join(self.allocator, &[_][]const u8{ current, "mojo.toml" });
            defer self.allocator.free(mojo_toml);
            
            // Check if mojo.toml exists
            if (std.fs.cwd().access(mojo_toml, .{})) |_| {
                // Read and check if it's a workspace
                const file = std.fs.cwd().openFile(mojo_toml, .{}) catch continue;
                defer file.close();
                
                const content = file.readToEndAlloc(self.allocator, 1024 * 1024) catch continue;
                defer self.allocator.free(content);
                
                // Check for [workspace] marker
                if (std.mem.indexOf(u8, content, "[workspace]") != null) {
                    return try self.allocator.dupe(u8, current);
                }
            } else |_| {}
            
            // Move to parent directory
            const parent = std.fs.path.dirname(current) orelse break;
            
            // Stop at root
            if (std.mem.eql(u8, parent, current)) break;
            
            const new_current = try self.allocator.dupe(u8, parent);
            self.allocator.free(current);
            current = new_current;
        }
        
        return null;  // No workspace found
    }
    
    /// Detect if current directory is in a workspace
    pub fn isInWorkspace(self: *WorkspaceDetector, path: []const u8) !bool {
        const root = try self.findWorkspaceRoot(path);
        if (root) |r| {
            defer self.allocator.free(r);
            return true;
        }
        return false;
    }
};

// ============================================================================
// Workspace Structure
// ============================================================================

pub const Member = struct {
    name: []const u8,
    path: []const u8,
    manifest: manifest.PackageManifest,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, path: []const u8, pkg_manifest: manifest.PackageManifest) !Member {
        return Member{
            .name = try allocator.dupe(u8, name),
            .path = try allocator.dupe(u8, path),
            .manifest = pkg_manifest,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Member) void {
        self.allocator.free(self.name);
        self.allocator.free(self.path);
        self.manifest.deinit();
    }
};

pub const Workspace = struct {
    root: []const u8,
    manifest: manifest.WorkspaceManifest,
    members: std.ArrayList(Member),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, root: []const u8, ws_manifest: manifest.WorkspaceManifest) !Workspace {
        return Workspace{
            .root = try allocator.dupe(u8, root),
            .manifest = ws_manifest,
            .members = std.ArrayList(Member){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Workspace) void {
        self.allocator.free(self.root);
        self.manifest.deinit();
        
        for (self.members.items) |*member| {
            member.deinit();
        }
        self.members.deinit(self.allocator);
    }
    
    /// Get member by name
    pub fn getMember(self: *Workspace, name: []const u8) ?*Member {
        for (self.members.items) |*member| {
            if (std.mem.eql(u8, member.name, name)) {
                return member;
            }
        }
        return null;
    }
    
    /// Check if path is a workspace member
    pub fn isMember(self: *Workspace, path: []const u8) bool {
        for (self.members.items) |member| {
            if (std.mem.eql(u8, member.path, path)) {
                return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Workspace Loader
// ============================================================================

pub const WorkspaceLoader = struct {
    allocator: Allocator,
    detector: WorkspaceDetector,
    
    pub fn init(allocator: Allocator) WorkspaceLoader {
        return WorkspaceLoader{
            .allocator = allocator,
            .detector = WorkspaceDetector.init(allocator),
        };
    }
    
    /// Load workspace from directory
    pub fn loadWorkspace(self: *WorkspaceLoader, path: []const u8) !Workspace {
        // Find workspace root
        const root = try self.detector.findWorkspaceRoot(path) orelse return error.NotAWorkspace;
        defer self.allocator.free(root);
        
        // Load workspace manifest
        const mojo_toml = try std.fs.path.join(self.allocator, &[_][]const u8{ root, "mojo.toml" });
        defer self.allocator.free(mojo_toml);
        
        var parser = manifest.ManifestParser.init(self.allocator);
        var ws_manifest = try parser.parseFile(mojo_toml);
        
        if (!ws_manifest.isWorkspace()) {
            ws_manifest.deinit();
            return error.NotAWorkspace;
        }
        
        var workspace = try Workspace.init(self.allocator, root, ws_manifest.Workspace);
        errdefer workspace.deinit();
        
        // Load members
        try self.loadMembers(&workspace);
        
        return workspace;
    }
    
    /// Load all workspace members
    fn loadMembers(self: *WorkspaceLoader, workspace: *Workspace) !void {
        var parser = manifest.ManifestParser.init(self.allocator);
        
        for (workspace.manifest.members.items) |ws_member| {
            // Construct full path to member
            const member_path = try std.fs.path.join(
                self.allocator,
                &[_][]const u8{ workspace.root, ws_member.path }
            );
            defer self.allocator.free(member_path);
            
            // Load member's mojo.toml
            const member_toml = try std.fs.path.join(
                self.allocator,
                &[_][]const u8{ member_path, "mojo.toml" }
            );
            defer self.allocator.free(member_toml);
            
            var member_manifest = try parser.parseFile(member_toml);
            
            if (!member_manifest.isPackage()) {
                member_manifest.deinit();
                continue;  // Skip non-package members
            }
            
            // Create member
            const member = try Member.init(
                self.allocator,
                member_manifest.Package.name,
                ws_member.path,
                member_manifest.Package
            );
            
            try workspace.members.append(self.allocator, member);
        }
    }
    
    /// Load package (standalone, not in workspace)
    pub fn loadPackage(self: *WorkspaceLoader, path: []const u8) !manifest.PackageManifest {
        const mojo_toml = try std.fs.path.join(self.allocator, &[_][]const u8{ path, "mojo.toml" });
        defer self.allocator.free(mojo_toml);
        
        var parser = manifest.ManifestParser.init(self.allocator);
        var pkg_manifest = try parser.parseFile(mojo_toml);
        
        if (!pkg_manifest.isPackage()) {
            pkg_manifest.deinit();
            return error.NotAPackage;
        }
        
        return pkg_manifest.Package;
    }
};

// ============================================================================
// Path Resolution
// ============================================================================

pub const PathResolver = struct {
    workspace_root: ?[]const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, workspace_root: ?[]const u8) PathResolver {
        return PathResolver{
            .allocator = allocator,
            .workspace_root = workspace_root,
        };
    }
    
    /// Resolve relative path within workspace
    pub fn resolvePath(self: *PathResolver, relative_path: []const u8) ![]const u8 {
        if (self.workspace_root) |root| {
            return try std.fs.path.join(
                self.allocator,
                &[_][]const u8{ root, relative_path }
            );
        }
        
        // No workspace, return as-is
        return try self.allocator.dupe(u8, relative_path);
    }
    
    /// Resolve member path
    pub fn resolveMemberPath(self: *PathResolver, member_name: []const u8) !?[]const u8 {
        if (self.workspace_root) |root| {
            // Try direct path first
            {
                const full_path = try std.fs.path.join(
                    self.allocator,
                    &[_][]const u8{ root, member_name }
                );
                
                if (std.fs.cwd().access(full_path, .{})) |_| {
                    return full_path;
                } else |_| {
                    self.allocator.free(full_path);
                }
            }
            
            // Try services/ prefix
            {
                const services_path = try std.fmt.allocPrint(self.allocator, "services/{s}", .{member_name});
                defer self.allocator.free(services_path);
                
                const full_path = try std.fs.path.join(
                    self.allocator,
                    &[_][]const u8{ root, services_path }
                );
                
                if (std.fs.cwd().access(full_path, .{})) |_| {
                    return full_path;
                } else |_| {
                    self.allocator.free(full_path);
                }
            }
            
            // Try integrations/ prefix
            {
                const integrations_path = try std.fmt.allocPrint(self.allocator, "integrations/{s}", .{member_name});
                defer self.allocator.free(integrations_path);
                
                const full_path = try std.fs.path.join(
                    self.allocator,
                    &[_][]const u8{ root, integrations_path }
                );
                
                if (std.fs.cwd().access(full_path, .{})) |_| {
                    return full_path;
                } else |_| {
                    self.allocator.free(full_path);
                }
            }
        }
        
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "WorkspaceDetector: not in workspace" {
    var detector = WorkspaceDetector.init(std.testing.allocator);
    
    const root = try detector.findWorkspaceRoot(".");
    
    // Current test directory likely not in a workspace
    if (root) |r| {
        defer std.testing.allocator.free(r);
    }
}

test "Member: creation and cleanup" {
    const version = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    const pkg = try manifest.PackageManifest.init(std.testing.allocator, "test-member", version);
    
    var member = try Member.init(std.testing.allocator, "test-member", "services/test", pkg);
    defer member.deinit();
    
    try std.testing.expectEqualStrings("test-member", member.name);
    try std.testing.expectEqualStrings("services/test", member.path);
}

test "Workspace: member management" {
    const ws_manifest = manifest.WorkspaceManifest.init(std.testing.allocator);
    var workspace = try Workspace.init(std.testing.allocator, "/workspace/root", ws_manifest);
    defer workspace.deinit();
    
    // Add a member
    const version = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    const pkg = try manifest.PackageManifest.init(std.testing.allocator, "test-pkg", version);
    const member = try Member.init(std.testing.allocator, "test-pkg", "mojo-sdk", pkg);
    try workspace.members.append(std.testing.allocator, member);
    
    try std.testing.expectEqual(@as(usize, 1), workspace.members.items.len);
    
    // Test getMember
    const found = workspace.getMember("test-pkg");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("test-pkg", found.?.name);
    
    // Test isMember
    try std.testing.expect(workspace.isMember("mojo-sdk"));
    try std.testing.expect(!workspace.isMember("nonexistent"));
}

test "PathResolver: without workspace" {
    var resolver = PathResolver.init(std.testing.allocator, null);
    
    const resolved = try resolver.resolvePath("some/path");
    defer std.testing.allocator.free(resolved);
    
    try std.testing.expectEqualStrings("some/path", resolved);
}

test "PathResolver: with workspace" {
    var resolver = PathResolver.init(std.testing.allocator, "/workspace/root");
    
    const resolved = try resolver.resolvePath("mojo-sdk");
    defer std.testing.allocator.free(resolved);
    
    try std.testing.expect(std.mem.indexOf(u8, resolved, "/workspace/root") != null);
    try std.testing.expect(std.mem.indexOf(u8, resolved, "mojo-sdk") != null);
}

test "Workspace: empty workspace" {
    const ws_manifest = manifest.WorkspaceManifest.init(std.testing.allocator);
    var workspace = try Workspace.init(std.testing.allocator, "/test/root", ws_manifest);
    defer workspace.deinit();
    
    try std.testing.expectEqualStrings("/test/root", workspace.root);
    try std.testing.expectEqual(@as(usize, 0), workspace.members.items.len);
}

test "WorkspaceDetector: isInWorkspace" {
    var detector = WorkspaceDetector.init(std.testing.allocator);
    
    const in_workspace = try detector.isInWorkspace(".");
    
    // Just verify it doesn't crash - result depends on actual directory structure
    _ = in_workspace;
}

test "PathResolver: resolve member path" {
    var resolver = PathResolver.init(std.testing.allocator, "/workspace");
    
    const resolved = try resolver.resolveMemberPath("test-member");
    
    // Will be null since path doesn't exist, but shouldn't crash
    if (resolved) |r| {
        defer std.testing.allocator.free(r);
    }
}

test "Member: multiple members" {
    const ws_manifest = manifest.WorkspaceManifest.init(std.testing.allocator);
    var workspace = try Workspace.init(std.testing.allocator, "/root", ws_manifest);
    defer workspace.deinit();
    
    // Add multiple members
    const v1 = try manifest.Version.parse(std.testing.allocator, "1.0.0");
    const pkg1 = try manifest.PackageManifest.init(std.testing.allocator, "member1", v1);
    const member1 = try Member.init(std.testing.allocator, "member1", "path1", pkg1);
    try workspace.members.append(std.testing.allocator, member1);
    
    const v2 = try manifest.Version.parse(std.testing.allocator, "2.0.0");
    const pkg2 = try manifest.PackageManifest.init(std.testing.allocator, "member2", v2);
    const member2 = try Member.init(std.testing.allocator, "member2", "path2", pkg2);
    try workspace.members.append(std.testing.allocator, member2);
    
    try std.testing.expectEqual(@as(usize, 2), workspace.members.items.len);
}
