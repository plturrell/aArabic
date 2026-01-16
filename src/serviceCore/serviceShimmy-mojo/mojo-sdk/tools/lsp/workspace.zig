// Workspace Management
// Day 74: Workspace and project handling for LSP

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Workspace Types
// ============================================================================

/// Workspace Folder
pub const WorkspaceFolder = struct {
    uri: []const u8,
    name: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, uri: []const u8, name: []const u8) !WorkspaceFolder {
        const uri_copy = try allocator.dupe(u8, uri);
        const name_copy = try allocator.dupe(u8, name);
        
        return WorkspaceFolder{
            .uri = uri_copy,
            .name = name_copy,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *WorkspaceFolder) void {
        self.allocator.free(self.uri);
        self.allocator.free(self.name);
    }
    
    /// Get filesystem path from URI
    pub fn getPath(self: WorkspaceFolder, allocator: Allocator) ![]const u8 {
        // Convert file:/// URI to filesystem path
        if (std.mem.startsWith(u8, self.uri, "file://")) {
            const path = self.uri[7..]; // Skip "file://"
            return try allocator.dupe(u8, path);
        }
        return try allocator.dupe(u8, self.uri);
    }
};

/// Project Configuration (mojo.toml)
pub const ProjectConfig = struct {
    name: ?[]const u8 = null,
    version: ?[]const u8 = null,
    dependencies: std.StringHashMap([]const u8),
    source_dirs: std.ArrayList([]const u8),
    build_target: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) ProjectConfig {
        return ProjectConfig{
            .dependencies = std.StringHashMap([]const u8).init(allocator),
            .source_dirs = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *ProjectConfig) void {
        if (self.name) |name| {
            self.allocator.free(name);
        }
        if (self.version) |version| {
            self.allocator.free(version);
        }
        if (self.build_target) |target| {
            self.allocator.free(target);
        }
        
        // Clean up dependencies
        var dep_iter = self.dependencies.iterator();
        while (dep_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.dependencies.deinit();
        
        // Clean up source dirs
        for (self.source_dirs.items) |dir| {
            self.allocator.free(dir);
        }
        self.source_dirs.deinit(self.allocator);
    }
    
    /// Add a dependency
    pub fn addDependency(self: *ProjectConfig, name: []const u8, version: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        const version_copy = try self.allocator.dupe(u8, version);
        try self.dependencies.put(name_copy, version_copy);
    }
    
    /// Add a source directory
    pub fn addSourceDir(self: *ProjectConfig, dir: []const u8) !void {
        const dir_copy = try self.allocator.dupe(u8, dir);
        try self.source_dirs.append(self.allocator, dir_copy);
    }
};

/// File Watch Event Type
pub const FileChangeType = enum(u8) {
    Created = 1,
    Changed = 2,
    Deleted = 3,
};

/// File Watch Event
pub const FileEvent = struct {
    uri: []const u8,
    change_type: FileChangeType,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, uri: []const u8, change_type: FileChangeType) !FileEvent {
        const uri_copy = try allocator.dupe(u8, uri);
        return FileEvent{
            .uri = uri_copy,
            .change_type = change_type,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *FileEvent) void {
        self.allocator.free(self.uri);
    }
};

// ============================================================================
// Workspace
// ============================================================================

pub const Workspace = struct {
    // Workspace folders (multi-root support)
    folders: std.ArrayList(WorkspaceFolder),
    
    // Project configurations per folder
    configs: std.StringHashMap(ProjectConfig),
    
    // File watch events queue
    file_events: std.ArrayList(FileEvent),
    
    // Workspace root (single-root fallback)
    root_uri: ?[]const u8 = null,
    
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) Workspace {
        return Workspace{
            .folders = std.ArrayList(WorkspaceFolder){},
            .configs = std.StringHashMap(ProjectConfig).init(allocator),
            .file_events = std.ArrayList(FileEvent){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Workspace) void {
        // Clean up folders
        for (self.folders.items) |*folder| {
            folder.deinit();
        }
        self.folders.deinit(self.allocator);
        
        // Clean up configs
        var config_iter = self.configs.iterator();
        while (config_iter.next()) |entry| {
            var config = entry.value_ptr;
            config.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.configs.deinit();
        
        // Clean up file events
        for (self.file_events.items) |*event| {
            event.deinit();
        }
        self.file_events.deinit(self.allocator);
        
        if (self.root_uri) |uri| {
            self.allocator.free(uri);
        }
    }
    
    /// Initialize workspace with root URI (single-root)
    pub fn initializeWithRoot(self: *Workspace, root_uri: []const u8) !void {
        self.root_uri = try self.allocator.dupe(u8, root_uri);
        
        // Extract folder name from URI
        const name = self.extractFolderName(root_uri);
        const folder = try WorkspaceFolder.init(self.allocator, root_uri, name);
        try self.folders.append(self.allocator, folder);
        
        // Try to load project config
        try self.loadProjectConfig(root_uri);
    }
    
    /// Add a workspace folder (multi-root)
    pub fn addFolder(self: *Workspace, uri: []const u8, name: []const u8) !void {
        const folder = try WorkspaceFolder.init(self.allocator, uri, name);
        try self.folders.append(self.allocator, folder);
        
        // Try to load project config for this folder
        try self.loadProjectConfig(uri);
    }
    
    /// Remove a workspace folder
    pub fn removeFolder(self: *Workspace, uri: []const u8) !void {
        var i: usize = 0;
        while (i < self.folders.items.len) : (i += 1) {
            if (std.mem.eql(u8, self.folders.items[i].uri, uri)) {
                var folder = self.folders.orderedRemove(i);
                folder.deinit();
                
                // Remove associated config
                if (self.configs.fetchRemove(uri)) |kv| {
                    var config = kv.value;
                    config.deinit();
                    self.allocator.free(kv.key);
                }
                
                return;
            }
        }
        return error.FolderNotFound;
    }
    
    /// Get folder containing a file URI
    pub fn getFolderForFile(self: *Workspace, file_uri: []const u8) ?*WorkspaceFolder {
        for (self.folders.items) |*folder| {
            if (std.mem.startsWith(u8, file_uri, folder.uri)) {
                return folder;
            }
        }
        return null;
    }
    
    /// Load project configuration from mojo.toml
    fn loadProjectConfig(self: *Workspace, folder_uri: []const u8) !void {
        // In a real implementation, this would:
        // 1. Convert URI to filesystem path
        // 2. Look for mojo.toml in the folder
        // 3. Parse the TOML file
        // 4. Populate ProjectConfig
        
        // For now, create a default config
        var config = ProjectConfig.init(self.allocator);
        config.name = try self.allocator.dupe(u8, "default-project");
        config.version = try self.allocator.dupe(u8, "0.1.0");
        
        // Add default source directory
        try config.addSourceDir("src");
        
        const uri_key = try self.allocator.dupe(u8, folder_uri);
        try self.configs.put(uri_key, config);
    }
    
    /// Get project configuration for a folder
    pub fn getConfig(self: *Workspace, folder_uri: []const u8) ?*ProjectConfig {
        return self.configs.getPtr(folder_uri);
    }
    
    /// Add a file watch event
    pub fn addFileEvent(self: *Workspace, uri: []const u8, change_type: FileChangeType) !void {
        const event = try FileEvent.init(self.allocator, uri, change_type);
        try self.file_events.append(self.allocator, event);
    }
    
    /// Process file watch events (returns count, events are still owned by workspace)
    pub fn processFileEvents(self: *Workspace) usize {
        // Return count and clear the queue
        const count = self.file_events.items.len;
        
        // Free event memory
        for (self.file_events.items) |*event| {
            event.deinit();
        }
        
        self.file_events.clearRetainingCapacity();
        return count;
    }
    
    /// Check if a file is in the workspace
    pub fn containsFile(self: *Workspace, file_uri: []const u8) bool {
        return self.getFolderForFile(file_uri) != null;
    }
    
    /// Get workspace folder count
    pub fn getFolderCount(self: *Workspace) usize {
        return self.folders.items.len;
    }
    
    /// Extract folder name from URI
    fn extractFolderName(self: *Workspace, uri: []const u8) []const u8 {
        _ = self;
        
        // Find last '/' or '\' in URI
        var i = uri.len;
        while (i > 0) {
            i -= 1;
            if (uri[i] == '/' or uri[i] == '\\') {
                return uri[i + 1 ..];
            }
        }
        return uri;
    }
};

// ============================================================================
// Configuration Parser (Simple TOML-like)
// ============================================================================

pub const ConfigParser = struct {
    content: []const u8,
    position: usize,
    
    pub fn init(content: []const u8) ConfigParser {
        return ConfigParser{
            .content = content,
            .position = 0,
        };
    }
    
    fn peek(self: *ConfigParser) ?u8 {
        if (self.position >= self.content.len) return null;
        return self.content[self.position];
    }
    
    fn advance(self: *ConfigParser) ?u8 {
        if (self.position >= self.content.len) return null;
        const ch = self.content[self.position];
        self.position += 1;
        return ch;
    }
    
    fn skipWhitespace(self: *ConfigParser) void {
        while (self.peek()) |ch| {
            if (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r') {
                _ = self.advance();
            } else {
                break;
            }
        }
    }
    
    fn skipComment(self: *ConfigParser) void {
        if (self.peek() == '#') {
            while (self.peek()) |ch| {
                _ = self.advance();
                if (ch == '\n') break;
            }
        }
    }
    
    fn readUntil(self: *ConfigParser, delimiter: u8) ?[]const u8 {
        const start = self.position;
        while (self.peek()) |ch| {
            if (ch == delimiter) {
                const result = self.content[start..self.position];
                _ = self.advance(); // Skip delimiter
                return result;
            }
            _ = self.advance();
        }
        return null;
    }
    
    /// Parse a simple key-value configuration
    pub fn parse(self: *ConfigParser, config: *ProjectConfig) !void {
        while (self.position < self.content.len) {
            self.skipWhitespace();
            self.skipComment();
            
            if (self.position >= self.content.len) break;
            
            // Read key
            const key_start = self.position;
            while (self.peek()) |ch| {
                if (ch == '=' or ch == ' ' or ch == '\t') break;
                _ = self.advance();
            }
            
            if (self.position == key_start) {
                _ = self.advance();
                continue;
            }
            
            const key = self.content[key_start..self.position];
            
            self.skipWhitespace();
            
            // Expect '='
            if (self.peek() != '=') continue;
            _ = self.advance();
            
            self.skipWhitespace();
            
            // Read value
            if (self.peek() == '"') {
                _ = self.advance(); // Skip opening quote
                if (self.readUntil('"')) |value| {
                    try self.processKeyValue(config, key, value);
                }
            }
        }
    }
    
    fn processKeyValue(self: *ConfigParser, config: *ProjectConfig, key: []const u8, value: []const u8) !void {
        _ = self;
        
        if (std.mem.eql(u8, key, "name")) {
            config.name = try config.allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, key, "version")) {
            config.version = try config.allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, key, "build_target")) {
            config.build_target = try config.allocator.dupe(u8, value);
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "WorkspaceFolder: creation and path" {
    var folder = try WorkspaceFolder.init(std.testing.allocator, "file:///home/user/project", "project");
    defer folder.deinit();
    
    try std.testing.expectEqualStrings("file:///home/user/project", folder.uri);
    try std.testing.expectEqualStrings("project", folder.name);
    
    const path = try folder.getPath(std.testing.allocator);
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("/home/user/project", path);
}

test "ProjectConfig: dependencies" {
    var config = ProjectConfig.init(std.testing.allocator);
    defer config.deinit();
    
    try config.addDependency("stdlib", "1.0.0");
    try config.addDependency("math", "2.1.0");
    
    try std.testing.expectEqual(@as(usize, 2), config.dependencies.count());
    
    const stdlib_version = config.dependencies.get("stdlib").?;
    try std.testing.expectEqualStrings("1.0.0", stdlib_version);
}

test "ProjectConfig: source directories" {
    var config = ProjectConfig.init(std.testing.allocator);
    defer config.deinit();
    
    try config.addSourceDir("src");
    try config.addSourceDir("lib");
    
    try std.testing.expectEqual(@as(usize, 2), config.source_dirs.items.len);
    try std.testing.expectEqualStrings("src", config.source_dirs.items[0]);
}

test "FileEvent: creation" {
    var event = try FileEvent.init(std.testing.allocator, "file:///test.mojo", .Changed);
    defer event.deinit();
    
    try std.testing.expectEqualStrings("file:///test.mojo", event.uri);
    try std.testing.expectEqual(FileChangeType.Changed, event.change_type);
}

test "Workspace: initialization" {
    var workspace = Workspace.init(std.testing.allocator);
    defer workspace.deinit();
    
    try workspace.initializeWithRoot("file:///home/user/project");
    
    try std.testing.expectEqual(@as(usize, 1), workspace.getFolderCount());
    try std.testing.expect(workspace.root_uri != null);
}

test "Workspace: multi-root folders" {
    var workspace = Workspace.init(std.testing.allocator);
    defer workspace.deinit();
    
    try workspace.addFolder("file:///project1", "project1");
    try workspace.addFolder("file:///project2", "project2");
    
    try std.testing.expectEqual(@as(usize, 2), workspace.getFolderCount());
}

test "Workspace: file containment" {
    var workspace = Workspace.init(std.testing.allocator);
    defer workspace.deinit();
    
    try workspace.addFolder("file:///home/user/project", "project");
    
    try std.testing.expect(workspace.containsFile("file:///home/user/project/src/main.mojo"));
    try std.testing.expect(!workspace.containsFile("file:///other/location/file.mojo"));
}

test "Workspace: file events" {
    var workspace = Workspace.init(std.testing.allocator);
    defer workspace.deinit();
    
    try workspace.addFileEvent("file:///test.mojo", .Created);
    try workspace.addFileEvent("file:///test2.mojo", .Changed);
    
    const count = workspace.processFileEvents();
    try std.testing.expectEqual(@as(usize, 2), count);
    
    // Events should be cleared after processing
    const count2 = workspace.processFileEvents();
    try std.testing.expectEqual(@as(usize, 0), count2);
}

test "ConfigParser: simple key-value" {
    const content =
        \\name = "my-project"
        \\version = "1.0.0"
        \\build_target = "release"
    ;
    
    var parser = ConfigParser.init(content);
    var config = ProjectConfig.init(std.testing.allocator);
    defer config.deinit();
    
    try parser.parse(&config);
    
    try std.testing.expectEqualStrings("my-project", config.name.?);
    try std.testing.expectEqualStrings("1.0.0", config.version.?);
    try std.testing.expectEqualStrings("release", config.build_target.?);
}
