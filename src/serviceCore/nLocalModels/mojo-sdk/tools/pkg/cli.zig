// Mojo Package Manager - CLI Commands
// Day 96: User-facing command-line interface

const std = @import("std");
const manifest = @import("manifest.zig");
const workspace_mod = @import("workspace.zig");
const resolver = @import("resolver.zig");
const zig_bridge = @import("zig_bridge.zig");
const Allocator = std.mem.Allocator;

// ============================================================================
// CLI Context
// ============================================================================

pub const CliContext = struct {
    allocator: Allocator,
    current_dir: []const u8,
    workspace: ?*workspace_mod.Workspace = null,
    
    pub fn init(allocator: Allocator, current_dir: []const u8) CliContext {
        return CliContext{
            .allocator = allocator,
            .current_dir = current_dir,
        };
    }
    
    pub fn deinit(self: *CliContext) void {
        if (self.workspace) |ws| {
            ws.deinit();
            self.allocator.destroy(ws);
        }
    }
};

// ============================================================================
// Init Command - Initialize new package
// ============================================================================

pub const InitCommand = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) InitCommand {
        return InitCommand{ .allocator = allocator };
    }
    
    pub fn execute(self: *InitCommand, ctx: *CliContext, name: []const u8) !void {
        _ = ctx;
        
        std.debug.print("Initializing Mojo package: {s}\n", .{name});
        
        // Create mojo.toml
        const mojo_toml = try std.fmt.allocPrint(self.allocator,
            \\[package]
            \\name = "{s}"
            \\version = "0.1.0"
            \\authors = []
            \\
            \\[dependencies]
            \\
        , .{name});
        defer self.allocator.free(mojo_toml);
        
        const file = try std.fs.cwd().createFile("mojo.toml", .{});
        defer file.close();
        try file.writeAll(mojo_toml);
        
        std.debug.print("✓ Created mojo.toml\n", .{});
        std.debug.print("✓ Package initialized successfully!\n", .{});
    }
};

// ============================================================================
// Install Command - Install dependencies
// ============================================================================

pub const InstallCommand = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) InstallCommand {
        return InstallCommand{ .allocator = allocator };
    }
    
    pub fn execute(self: *InstallCommand, ctx: *CliContext) !void {
        std.debug.print("Installing dependencies...\n", .{});
        
        // Load manifest
        var parser = manifest.ManifestParser.init(self.allocator);
        var pkg_manifest = try parser.parseFile("mojo.toml");
        defer pkg_manifest.deinit();
        
        if (!pkg_manifest.isPackage()) {
            return error.NotAPackage;
        }
        
        const pkg = pkg_manifest.Package;
        
        // Resolve dependencies
        var dep_resolver = resolver.DependencyResolver.init(self.allocator);
        var graph = try dep_resolver.buildGraph(pkg);
        defer graph.deinit();
        
        var resolution = try dep_resolver.resolve(&graph);
        defer resolution.deinit();
        
        std.debug.print("Resolved {d} dependencies\n", .{resolution.dependencies.items.len});
        
        // Generate build.zig.zon
        var generator = zig_bridge.ZigBuildGenerator.init(self.allocator);
        const zon = try generator.generateZonFile(pkg, resolution);
        defer self.allocator.free(zon);
        
        const zon_file = try std.fs.cwd().createFile("build.zig.zon", .{});
        defer zon_file.close();
        try zon_file.writeAll(zon);
        
        std.debug.print("✓ Generated build.zig.zon\n", .{});
        
        // In workspace mode, coordinate across all members
        if (ctx.workspace) |ws| {
            std.debug.print("Workspace mode: coordinating {d} members\n", .{ws.members.items.len});
            
            var ws_resolver = resolver.WorkspaceResolver.init(self.allocator, ws);
            var ws_resolution = try ws_resolver.resolveWorkspace();
            defer ws_resolution.deinit();
            
            std.debug.print("✓ Resolved workspace dependencies\n", .{});
        }
        
        std.debug.print("✓ Dependencies installed successfully!\n", .{});
    }
};

// ============================================================================
// Build Command - Build the package
// ============================================================================

pub const BuildCommand = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) BuildCommand {
        return BuildCommand{ .allocator = allocator };
    }
    
    pub fn execute(self: *BuildCommand, ctx: *CliContext, mode: zig_bridge.BuildMode) !void {
        std.debug.print("Building package...\n", .{});
        
        // Load manifest
        var parser = manifest.ManifestParser.init(self.allocator);
        var pkg_manifest = try parser.parseFile("mojo.toml");
        defer pkg_manifest.deinit();
        
        if (!pkg_manifest.isPackage()) {
            return error.NotAPackage;
        }
        
        const pkg = pkg_manifest.Package;
        std.debug.print("Building {s} v{}\n", .{ pkg.name, pkg.version });
        
        // Generate build.zig if needed
        var config = try zig_bridge.BuildConfig.init(self.allocator, pkg.name, .lib);
        defer config.deinit();
        
        var generator = zig_bridge.ZigBuildGenerator.init(self.allocator);
        const script = try generator.generateBuildScript(pkg, config);
        defer self.allocator.free(script);
        
        const build_file = try std.fs.cwd().createFile("build.zig", .{});
        defer build_file.close();
        try build_file.writeAll(script);
        
        std.debug.print("✓ Generated build.zig\n", .{});
        
        // Execute build
        var executor = zig_bridge.BuildExecutor.init(self.allocator);
        try executor.executeBuild(ctx.current_dir, mode);
        
        std.debug.print("✓ Build completed successfully!\n", .{});
    }
};

// ============================================================================
// Test Command - Run tests
// ============================================================================

pub const TestCommand = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) TestCommand {
        return TestCommand{ .allocator = allocator };
    }
    
    pub fn execute(self: *TestCommand, ctx: *CliContext) !void {
        std.debug.print("Running tests...\n", .{});
        
        var executor = zig_bridge.BuildExecutor.init(self.allocator);
        try executor.executeTest(ctx.current_dir);
        
        std.debug.print("✓ Tests completed!\n", .{});
    }
};

// ============================================================================
// Add Command - Add dependency
// ============================================================================

pub const AddCommand = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) AddCommand {
        return AddCommand{ .allocator = allocator };
    }
    
    pub fn execute(self: *AddCommand, _: *CliContext, dep_name: []const u8, version: []const u8) !void {
        std.debug.print("Adding dependency: {s}@{s}\n", .{ dep_name, version });
        
        // Read existing mojo.toml
        const file = try std.fs.cwd().openFile("mojo.toml", .{});
        defer file.close();
        
        const content = try file.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(content);
        
        // Append dependency (simplified - would use proper TOML editing)
        const new_dep = try std.fmt.allocPrint(self.allocator, "\n{s} = \"{s}\"\n", .{ dep_name, version });
        defer self.allocator.free(new_dep);
        
        const updated = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ content, new_dep });
        defer self.allocator.free(updated);
        
        const out_file = try std.fs.cwd().createFile("mojo.toml", .{});
        defer out_file.close();
        try out_file.writeAll(updated);
        
        std.debug.print("✓ Added {s}@{s} to dependencies\n", .{ dep_name, version });
        std.debug.print("Run 'mojo install' to install the dependency\n", .{});
    }
};

// ============================================================================
// Workspace Command - Manage workspace
// ============================================================================

pub const WorkspaceCommand = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) WorkspaceCommand {
        return WorkspaceCommand{ .allocator = allocator };
    }
    
    pub fn executeNew(self: *WorkspaceCommand, _: *CliContext, name: []const u8) !void {
        std.debug.print("Creating workspace: {s}\n", .{name});
        
        const mojo_toml = try std.fmt.allocPrint(self.allocator,
            \\[workspace]
            \\members = []
            \\
        , .{});
        defer self.allocator.free(mojo_toml);
        
        const file = try std.fs.cwd().createFile("mojo.toml", .{});
        defer file.close();
        try file.writeAll(mojo_toml);
        
        std.debug.print("✓ Created workspace mojo.toml\n", .{});
        std.debug.print("✓ Workspace initialized successfully!\n", .{});
    }
    
    pub fn executeList(_: *WorkspaceCommand, ctx: *CliContext) !void {
        if (ctx.workspace) |ws| {
            std.debug.print("Workspace members ({d}):\n", .{ws.members.items.len});
            for (ws.members.items) |member| {
                std.debug.print("  - {s} ({s})\n", .{ member.name, member.path });
            }
        } else {
            std.debug.print("Not in a workspace\n", .{});
        }
    }
};

// ============================================================================
// CLI Parser
// ============================================================================

pub const CliParser = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CliParser {
        return CliParser{ .allocator = allocator };
    }
    
    pub fn parse(_: *CliParser, args: []const []const u8) !Command {
        if (args.len < 1) {
            return Command{ .Help = {} };
        }
        
        const cmd = args[0];
        
        if (std.mem.eql(u8, cmd, "init")) {
            if (args.len < 2) return error.MissingArgument;
            return Command{ .Init = args[1] };
        } else if (std.mem.eql(u8, cmd, "install")) {
            return Command{ .Install = {} };
        } else if (std.mem.eql(u8, cmd, "build")) {
            const mode: zig_bridge.BuildMode = if (args.len > 1 and std.mem.eql(u8, args[1], "--release"))
                .ReleaseFast
            else
                .Debug;
            return Command{ .Build = mode };
        } else if (std.mem.eql(u8, cmd, "test")) {
            return Command{ .Test = {} };
        } else if (std.mem.eql(u8, cmd, "add")) {
            if (args.len < 3) return error.MissingArgument;
            return Command{ .Add = .{ .name = args[1], .version = args[2] } };
        } else if (std.mem.eql(u8, cmd, "workspace")) {
            if (args.len < 2) return error.MissingArgument;
            if (std.mem.eql(u8, args[1], "new")) {
                if (args.len < 3) return error.MissingArgument;
                return Command{ .WorkspaceNew = args[2] };
            } else if (std.mem.eql(u8, args[1], "list")) {
                return Command{ .WorkspaceList = {} };
            }
        }
        
        return Command{ .Help = {} };
    }
};

pub const Command = union(enum) {
    Init: []const u8,
    Install: void,
    Build: zig_bridge.BuildMode,
    Test: void,
    Add: struct { name: []const u8, version: []const u8 },
    WorkspaceNew: []const u8,
    WorkspaceList: void,
    Help: void,
};

// ============================================================================
// CLI Runner
// ============================================================================

pub const CliRunner = struct {
    allocator: Allocator,
    ctx: CliContext,
    
    pub fn init(allocator: Allocator) CliRunner {
        return CliRunner{
            .allocator = allocator,
            .ctx = CliContext.init(allocator, "."),
        };
    }
    
    pub fn deinit(self: *CliRunner) void {
        self.ctx.deinit();
    }
    
    pub fn run(self: *CliRunner, cmd: Command) !void {
        switch (cmd) {
            .Init => |name| {
                var init_cmd = InitCommand.init(self.allocator);
                try init_cmd.execute(&self.ctx, name);
            },
            .Install => {
                var install_cmd = InstallCommand.init(self.allocator);
                try install_cmd.execute(&self.ctx);
            },
            .Build => |mode| {
                var build_cmd = BuildCommand.init(self.allocator);
                try build_cmd.execute(&self.ctx, mode);
            },
            .Test => {
                var test_cmd = TestCommand.init(self.allocator);
                try test_cmd.execute(&self.ctx);
            },
            .Add => |dep| {
                var add_cmd = AddCommand.init(self.allocator);
                try add_cmd.execute(&self.ctx, dep.name, dep.version);
            },
            .WorkspaceNew => |name| {
                var ws_cmd = WorkspaceCommand.init(self.allocator);
                try ws_cmd.executeNew(&self.ctx, name);
            },
            .WorkspaceList => {
                var ws_cmd = WorkspaceCommand.init(self.allocator);
                try ws_cmd.executeList(&self.ctx);
            },
            .Help => {
                try self.printHelp();
            },
        }
    }
    
    fn printHelp(self: *CliRunner) !void {
        _ = self;
        std.debug.print(
            \\Mojo Package Manager
            \\
            \\USAGE:
            \\    mojo <COMMAND> [OPTIONS]
            \\
            \\COMMANDS:
            \\    init <name>              Initialize a new package
            \\    install                  Install dependencies
            \\    build [--release]        Build the package
            \\    test                     Run tests
            \\    add <name> <version>     Add a dependency
            \\    workspace new <name>     Create a new workspace
            \\    workspace list           List workspace members
            \\    help                     Show this help message
            \\
        , .{});
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CliContext: creation" {
    var ctx = CliContext.init(std.testing.allocator, ".");
    defer ctx.deinit();
    
    try std.testing.expectEqualStrings(".", ctx.current_dir);
    try std.testing.expect(ctx.workspace == null);
}

test "InitCommand: creation" {
    const cmd = InitCommand.init(std.testing.allocator);
    _ = cmd;
}

test "InstallCommand: creation" {
    const cmd = InstallCommand.init(std.testing.allocator);
    _ = cmd;
}

test "BuildCommand: creation" {
    const cmd = BuildCommand.init(std.testing.allocator);
    _ = cmd;
}

test "TestCommand: creation" {
    const cmd = TestCommand.init(std.testing.allocator);
    _ = cmd;
}

test "AddCommand: creation" {
    const cmd = AddCommand.init(std.testing.allocator);
    _ = cmd;
}

test "WorkspaceCommand: creation" {
    const cmd = WorkspaceCommand.init(std.testing.allocator);
    _ = cmd;
}

test "CliParser: parse init" {
    var parser = CliParser.init(std.testing.allocator);
    
    const args = [_][]const u8{ "init", "my-project" };
    const cmd = try parser.parse(&args);
    
    try std.testing.expect(cmd == .Init);
    try std.testing.expectEqualStrings("my-project", cmd.Init);
}

test "CliParser: parse install" {
    var parser = CliParser.init(std.testing.allocator);
    
    const args = [_][]const u8{"install"};
    const cmd = try parser.parse(&args);
    
    try std.testing.expect(cmd == .Install);
}

test "CliParser: parse build" {
    var parser = CliParser.init(std.testing.allocator);
    
    const args = [_][]const u8{"build"};
    const cmd = try parser.parse(&args);
    
    try std.testing.expect(cmd == .Build);
    try std.testing.expect(cmd.Build == .Debug);
}

test "CliParser: parse build release" {
    var parser = CliParser.init(std.testing.allocator);
    
    const args = [_][]const u8{ "build", "--release" };
    const cmd = try parser.parse(&args);
    
    try std.testing.expect(cmd == .Build);
    try std.testing.expect(cmd.Build == .ReleaseFast);
}

test "CliParser: parse add" {
    var parser = CliParser.init(std.testing.allocator);
    
    const args = [_][]const u8{ "add", "http-client", "^1.0.0" };
    const cmd = try parser.parse(&args);
    
    try std.testing.expect(cmd == .Add);
    try std.testing.expectEqualStrings("http-client", cmd.Add.name);
    try std.testing.expectEqualStrings("^1.0.0", cmd.Add.version);
}

test "CliRunner: creation" {
    var runner = CliRunner.init(std.testing.allocator);
    defer runner.deinit();
}
