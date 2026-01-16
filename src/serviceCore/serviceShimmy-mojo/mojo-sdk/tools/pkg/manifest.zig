// Mojo Package Manager - Manifest Parser
// Day 91: Core mojo.toml format & parser (hybrid: package + workspace)

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Version (Semantic Versioning)
// ============================================================================

pub const Version = struct {
    major: u32,
    minor: u32,
    patch: u32,
    pre_release: ?[]const u8 = null,
    build: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, major: u32, minor: u32, patch: u32) Version {
        return Version{
            .major = major,
            .minor = minor,
            .patch = patch,
            .allocator = allocator,
        };
    }
    
    pub fn parse(allocator: Allocator, version_str: []const u8) !Version {
        // Parse "1.2.3" or "1.2.3-beta" or "1.2.3+build"
        var parts = std.mem.splitSequence(u8, version_str, ".");
        
        const major_str = parts.next() orelse return error.InvalidVersion;
        const minor_str = parts.next() orelse return error.InvalidVersion;
        var patch_full = parts.next() orelse return error.InvalidVersion;
        
        // Check for pre-release or build metadata
        var pre_release: ?[]const u8 = null;
        var build: ?[]const u8 = null;
        
        if (std.mem.indexOf(u8, patch_full, "-")) |idx| {
            const patch_str = patch_full[0..idx];
            const rest = patch_full[idx + 1 ..];
            
            if (std.mem.indexOf(u8, rest, "+")) |build_idx| {
                pre_release = try allocator.dupe(u8, rest[0..build_idx]);
                build = try allocator.dupe(u8, rest[build_idx + 1 ..]);
            } else {
                pre_release = try allocator.dupe(u8, rest);
            }
            
            patch_full = patch_str;
        } else if (std.mem.indexOf(u8, patch_full, "+")) |idx| {
            const patch_str = patch_full[0..idx];
            build = try allocator.dupe(u8, patch_full[idx + 1 ..]);
            patch_full = patch_str;
        }
        
        const major = try std.fmt.parseInt(u32, major_str, 10);
        const minor = try std.fmt.parseInt(u32, minor_str, 10);
        const patch = try std.fmt.parseInt(u32, patch_full, 10);
        
        return Version{
            .major = major,
            .minor = minor,
            .patch = patch,
            .pre_release = pre_release,
            .build = build,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Version) void {
        if (self.pre_release) |pr| {
            self.allocator.free(pr);
        }
        if (self.build) |b| {
            self.allocator.free(b);
        }
    }
    
    pub fn toString(self: Version, allocator: Allocator) ![]const u8 {
        if (self.pre_release) |pr| {
            if (self.build) |b| {
                return try std.fmt.allocPrint(allocator, "{d}.{d}.{d}-{s}+{s}", 
                    .{ self.major, self.minor, self.patch, pr, b });
            }
            return try std.fmt.allocPrint(allocator, "{d}.{d}.{d}-{s}", 
                .{ self.major, self.minor, self.patch, pr });
        }
        if (self.build) |b| {
            return try std.fmt.allocPrint(allocator, "{d}.{d}.{d}+{s}", 
                .{ self.major, self.minor, self.patch, b });
        }
        return try std.fmt.allocPrint(allocator, "{d}.{d}.{d}", 
            .{ self.major, self.minor, self.patch });
    }
    
    pub fn compare(self: Version, other: Version) i32 {
        if (self.major != other.major) return @as(i32, @intCast(self.major)) - @as(i32, @intCast(other.major));
        if (self.minor != other.minor) return @as(i32, @intCast(self.minor)) - @as(i32, @intCast(other.minor));
        if (self.patch != other.patch) return @as(i32, @intCast(self.patch)) - @as(i32, @intCast(other.patch));
        return 0;
    }
};

// ============================================================================
// Dependency Specification
// ============================================================================

pub const DependencySource = enum {
    Registry,  // From package registry
    Path,      // Local path dependency
    Git,       // Git repository
};

pub const Dependency = struct {
    name: []const u8,
    source: DependencySource,
    version: ?[]const u8 = null,  // For registry
    path: ?[]const u8 = null,      // For path dependencies
    git_url: ?[]const u8 = null,   // For git dependencies
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, source: DependencySource) !Dependency {
        return Dependency{
            .name = try allocator.dupe(u8, name),
            .source = source,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Dependency) void {
        self.allocator.free(self.name);
        if (self.version) |v| self.allocator.free(v);
        if (self.path) |p| self.allocator.free(p);
        if (self.git_url) |g| self.allocator.free(g);
    }
};

// ============================================================================
// Package Manifest
// ============================================================================

pub const PackageManifest = struct {
    name: []const u8,
    version: Version,
    authors: [][]const u8,
    description: ?[]const u8 = null,
    license: ?[]const u8 = null,
    dependencies: std.ArrayList(Dependency),
    dev_dependencies: std.ArrayList(Dependency),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, version: Version) !PackageManifest {
        return PackageManifest{
            .name = try allocator.dupe(u8, name),
            .version = version,
            .authors = &[_][]const u8{},
            .dependencies = std.ArrayList(Dependency){},
            .dev_dependencies = std.ArrayList(Dependency){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *PackageManifest) void {
        self.allocator.free(self.name);
        for (self.authors) |author| {
            self.allocator.free(author);
        }
        self.allocator.free(self.authors);
        if (self.description) |d| self.allocator.free(d);
        if (self.license) |l| self.allocator.free(l);
        
        for (self.dependencies.items) |*dep| {
            dep.deinit();
        }
        self.dependencies.deinit(self.allocator);
        
        for (self.dev_dependencies.items) |*dep| {
            dep.deinit();
        }
        self.dev_dependencies.deinit(self.allocator);
        
        self.version.deinit();
    }
};

// ============================================================================
// Workspace Manifest
// ============================================================================

pub const WorkspaceMember = struct {
    path: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, path: []const u8) !WorkspaceMember {
        return WorkspaceMember{
            .path = try allocator.dupe(u8, path),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *WorkspaceMember) void {
        self.allocator.free(self.path);
    }
};

pub const WorkspaceManifest = struct {
    members: std.ArrayList(WorkspaceMember),
    shared_dependencies: std.ArrayList(Dependency),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) WorkspaceManifest {
        return WorkspaceManifest{
            .members = std.ArrayList(WorkspaceMember){},
            .shared_dependencies = std.ArrayList(Dependency){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *WorkspaceManifest) void {
        for (self.members.items) |*member| {
            member.deinit();
        }
        self.members.deinit(self.allocator);
        
        for (self.shared_dependencies.items) |*dep| {
            dep.deinit();
        }
        self.shared_dependencies.deinit(self.allocator);
    }
};

// ============================================================================
// Unified Manifest (Either Package OR Workspace)
// ============================================================================

pub const ManifestType = enum {
    Package,
    Workspace,
};

pub const Manifest = union(ManifestType) {
    Package: PackageManifest,
    Workspace: WorkspaceManifest,
    
    pub fn deinit(self: *Manifest) void {
        switch (self.*) {
            .Package => |*pkg| pkg.deinit(),
            .Workspace => |*ws| ws.deinit(),
        }
    }
    
    pub fn isWorkspace(self: Manifest) bool {
        return self == .Workspace;
    }
    
    pub fn isPackage(self: Manifest) bool {
        return self == .Package;
    }
};

// ============================================================================
// Manifest Parser (Simple TOML-like parser)
// ============================================================================

pub const ManifestParser = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) ManifestParser {
        return ManifestParser{ .allocator = allocator };
    }
    
    /// Parse mojo.toml from file path
    pub fn parseFile(self: *ManifestParser, path: []const u8) !Manifest {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        
        const content = try file.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(content);
        
        return try self.parseContent(content);
    }
    
    /// Parse mojo.toml content
    pub fn parseContent(self: *ManifestParser, content: []const u8) !Manifest {
        // Simple parser: check for [workspace] or [package]
        if (std.mem.indexOf(u8, content, "[workspace]") != null) {
            return Manifest{ .Workspace = try self.parseWorkspace(content) };
        } else if (std.mem.indexOf(u8, content, "[package]") != null) {
            return Manifest{ .Package = try self.parsePackage(content) };
        }
        
        return error.InvalidManifest;
    }
    
    fn parsePackage(self: *ManifestParser, content: []const u8) !PackageManifest {
        // Extract package name
        const name = try self.extractField(content, "name");
        defer self.allocator.free(name);  // Free original, init() will dupe
        
        // Extract version
        const version_str = try self.extractField(content, "version");
        defer self.allocator.free(version_str);
        
        var version = try Version.parse(self.allocator, version_str);
        errdefer version.deinit();
        
        var manifest = try PackageManifest.init(self.allocator, name, version);
        
        // Extract optional fields
        if (self.extractField(content, "description")) |desc| {
            manifest.description = desc;
        } else |_| {}
        
        if (self.extractField(content, "license")) |lic| {
            manifest.license = lic;
        } else |_| {}
        
        return manifest;
    }
    
    fn parseWorkspace(self: *ManifestParser, content: []const u8) !WorkspaceManifest {
        var manifest = WorkspaceManifest.init(self.allocator);
        
        // Extract members from [workspace] section
        if (std.mem.indexOf(u8, content, "members = [")) |start| {
            const members_start = start + "members = [".len;
            if (std.mem.indexOf(u8, content[members_start..], "]")) |end| {
                const members_content = content[members_start .. members_start + end];
                
                // Parse member paths
                var lines = std.mem.splitSequence(u8, members_content, "\n");
                while (lines.next()) |line| {
                    const trimmed = std.mem.trim(u8, line, " \t\r,");
                    if (trimmed.len == 0) continue;
                    if (trimmed[0] == '"' and trimmed[trimmed.len - 1] == '"') {
                        const path = trimmed[1 .. trimmed.len - 1];
                        const member = try WorkspaceMember.init(self.allocator, path);
                        try manifest.members.append(self.allocator, member);
                    }
                }
            }
        }
        
        return manifest;
    }
    
    fn extractField(self: *ManifestParser, content: []const u8, field: []const u8) ![]const u8 {
        const search = try std.fmt.allocPrint(self.allocator, "{s} = ", .{field});
        defer self.allocator.free(search);
        
        if (std.mem.indexOf(u8, content, search)) |start| {
            const value_start = start + search.len;
            var end = value_start;
            
            // Skip to opening quote
            while (end < content.len and content[end] != '"') : (end += 1) {}
            if (end >= content.len) return error.InvalidFormat;
            
            end += 1;  // Skip opening quote
            const val_start = end;
            
            // Find closing quote
            while (end < content.len and content[end] != '"') : (end += 1) {}
            if (end >= content.len) return error.InvalidFormat;
            
            return try self.allocator.dupe(u8, content[val_start..end]);
        }
        
        return error.FieldNotFound;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Version: parse and compare" {
    var version = try Version.parse(std.testing.allocator, "1.2.3");
    defer version.deinit();
    
    try std.testing.expectEqual(@as(u32, 1), version.major);
    try std.testing.expectEqual(@as(u32, 2), version.minor);
    try std.testing.expectEqual(@as(u32, 3), version.patch);
}

test "Version: pre-release" {
    var version = try Version.parse(std.testing.allocator, "1.0.0-beta");
    defer version.deinit();
    
    try std.testing.expectEqualStrings("beta", version.pre_release.?);
}

test "Version: compare" {
    var v1 = try Version.parse(std.testing.allocator, "1.2.3");
    defer v1.deinit();
    
    var v2 = try Version.parse(std.testing.allocator, "1.2.4");
    defer v2.deinit();
    
    try std.testing.expect(v1.compare(v2) < 0);
    try std.testing.expect(v2.compare(v1) > 0);
}

test "PackageManifest: creation" {
    const version = try Version.parse(std.testing.allocator, "0.1.0");
    var manifest = try PackageManifest.init(std.testing.allocator, "test-pkg", version);
    defer manifest.deinit();
    
    try std.testing.expectEqualStrings("test-pkg", manifest.name);
}

test "WorkspaceManifest: creation" {
    var manifest = WorkspaceManifest.init(std.testing.allocator);
    defer manifest.deinit();
    
    const member = try WorkspaceMember.init(std.testing.allocator, "mojo-sdk");
    try manifest.members.append(std.testing.allocator, member);
    
    try std.testing.expectEqual(@as(usize, 1), manifest.members.items.len);
}

test "ManifestParser: parse package" {
    var parser = ManifestParser.init(std.testing.allocator);
    
    const content =
        \\[package]
        \\name = "mojo-sdk"
        \\version = "0.1.0"
        \\description = "Mojo SDK"
    ;
    
    var manifest = try parser.parseContent(content);
    defer manifest.deinit();
    
    try std.testing.expect(manifest.isPackage());
    try std.testing.expectEqualStrings("mojo-sdk", manifest.Package.name);
}

test "ManifestParser: parse workspace" {
    var parser = ManifestParser.init(std.testing.allocator);
    
    const content =
        \\[workspace]
        \\members = [
        \\    "mojo-sdk",
        \\    "services/llm"
        \\]
    ;
    
    var manifest = try parser.parseContent(content);
    defer manifest.deinit();
    
    try std.testing.expect(manifest.isWorkspace());
    try std.testing.expectEqual(@as(usize, 2), manifest.Workspace.members.items.len);
}

test "Dependency: path dependency" {
    var dep = try Dependency.init(std.testing.allocator, "mojo-sdk", .Path);
    dep.path = try std.testing.allocator.dupe(u8, "../../mojo-sdk");
    defer dep.deinit();
    
    try std.testing.expectEqualStrings("mojo-sdk", dep.name);
    try std.testing.expect(dep.source == .Path);
}

test "Manifest: union type" {
    const version = try Version.parse(std.testing.allocator, "1.0.0");
    const pkg = try PackageManifest.init(std.testing.allocator, "test", version);
    
    var manifest = Manifest{ .Package = pkg };
    defer manifest.deinit();
    
    try std.testing.expect(manifest.isPackage());
    try std.testing.expect(!manifest.isWorkspace());
}
