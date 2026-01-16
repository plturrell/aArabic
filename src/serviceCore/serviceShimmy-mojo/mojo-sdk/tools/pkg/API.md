# Mojo Package Manager - API Documentation

Internal API documentation for developers working on or extending `mojo-pkg`.

## Module Overview

The package manager consists of 5 core modules:

1. **manifest.zig** - TOML parsing and version handling
2. **workspace.zig** - Workspace detection and management  
3. **resolver.zig** - Dependency resolution
4. **zig_bridge.zig** - Zig build system integration
5. **cli.zig** - Command-line interface

---

## 1. manifest.zig

Handles package manifests (`mojo.toml`) and version management.

### Types

#### `Version`

Semantic version representation.

```zig
pub const Version = struct {
    major: u32,
    minor: u32,
    patch: u32,
    pre_release: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, major: u32, minor: u32, patch: u32) Version
    pub fn deinit(self: *Version) void
    pub fn parse(allocator: Allocator, version_str: []const u8) !Version
    pub fn toString(self: Version, allocator: Allocator) ![]const u8
    pub fn compare(self: Version, other: Version) i32
};
```

**Methods:**
- `parse()` - Parse version string (e.g., "1.2.3")
- `toString()` - Convert to string
- `compare()` - Compare versions (-1, 0, 1)

**Example:**
```zig
const version = try Version.parse(allocator, "1.2.3");
defer version.deinit();

const cmp = version.compare(other_version);
// cmp < 0: version is older
// cmp = 0: versions are equal
// cmp > 0: version is newer
```

#### `Dependency`

Represents a package dependency.

```zig
pub const DependencySource = enum {
    Registry,  // From package registry
    Path,      // Local path dependency
    Git,       // Git repository (future)
};

pub const Dependency = struct {
    name: []const u8,
    version: ?[]const u8 = null,
    source: DependencySource,
    path: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, source: DependencySource) !Dependency
    pub fn deinit(self: *Dependency) void
};
```

**Example:**
```zig
// Registry dependency
var dep1 = try Dependency.init(allocator, "http-client", .Registry);
dep1.version = try allocator.dupe(u8, "^1.0.0");

// Path dependency
var dep2 = try Dependency.init(allocator, "mojo-sdk", .Path);
dep2.path = try allocator.dupe(u8, "../../mojo-sdk");
```

#### `PackageManifest`

Package-level manifest.

```zig
pub const PackageManifest = struct {
    name: []const u8,
    version: Version,
    authors: std.ArrayList([]const u8),
    description: ?[]const u8 = null,
    license: ?[]const u8 = null,
    dependencies: std.ArrayList(Dependency),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, version: Version) !PackageManifest
    pub fn deinit(self: *PackageManifest) void
};
```

#### `WorkspaceManifest`

Workspace-level manifest.

```zig
pub const WorkspaceManifest = struct {
    members: std.ArrayList([]const u8),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) WorkspaceManifest
    pub fn deinit(self: *WorkspaceManifest) void
};
```

#### `Manifest`

Union of package and workspace manifests.

```zig
pub const Manifest = union(enum) {
    Package: PackageManifest,
    Workspace: WorkspaceManifest,
    
    pub fn deinit(self: *Manifest) void
    pub fn isPackage(self: Manifest) bool
    pub fn isWorkspace(self: Manifest) bool
};
```

#### `ManifestParser`

TOML parser for manifests.

```zig
pub const ManifestParser = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) ManifestParser
    pub fn parseFile(self: *ManifestParser, path: []const u8) !Manifest
    pub fn parseString(self: *ManifestParser, content: []const u8) !Manifest
};
```

**Example:**
```zig
var parser = ManifestParser.init(allocator);
var manifest = try parser.parseFile("mojo.toml");
defer manifest.deinit();

if (manifest.isPackage()) {
    const pkg = manifest.Package;
    std.debug.print("Package: {s} v{}\n", .{pkg.name, pkg.version});
}
```

---

## 2. workspace.zig

Workspace detection and management.

### Types

#### `Member`

Workspace member package.

```zig
pub const Member = struct {
    name: []const u8,
    path: []const u8,
    manifest: manifest_mod.PackageManifest,
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        name: []const u8,
        path: []const u8,
        pkg_manifest: manifest_mod.PackageManifest
    ) !Member
    pub fn deinit(self: *Member) void
};
```

#### `Workspace`

Workspace with all members.

```zig
pub const Workspace = struct {
    root_path: []const u8,
    manifest: manifest_mod.WorkspaceManifest,
    members: std.ArrayList(Member),
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        root_path: []const u8,
        ws_manifest: manifest_mod.WorkspaceManifest
    ) !Workspace
    pub fn deinit(self: *Workspace) void
    pub fn addMember(self: *Workspace, member: Member) !void
    pub fn findMember(self: *Workspace, name: []const u8) ?*Member
};
```

#### `WorkspaceDetector`

Detects workspace root.

```zig
pub const WorkspaceDetector = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) WorkspaceDetector
    pub fn findWorkspaceRoot(self: *WorkspaceDetector, start_path: []const u8) !?[]const u8
    pub fn isInWorkspace(self: *WorkspaceDetector, current_path: []const u8) !bool
};
```

**Example:**
```zig
var detector = WorkspaceDetector.init(allocator);
if (try detector.findWorkspaceRoot(".")) |root| {
    defer allocator.free(root);
    std.debug.print("Found workspace at: {s}\n", .{root});
}
```

#### `WorkspaceLoader`

Loads workspace with all members.

```zig
pub const WorkspaceLoader = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) WorkspaceLoader
    pub fn loadWorkspace(self: *WorkspaceLoader, root_path: []const u8) !Workspace
    pub fn loadPackage(self: *WorkspaceLoader, path: []const u8) !manifest_mod.PackageManifest
};
```

**Example:**
```zig
var loader = WorkspaceLoader.init(allocator);
var workspace = try loader.loadWorkspace("/path/to/workspace");
defer workspace.deinit();

for (workspace.members.items) |member| {
    std.debug.print("Member: {s} at {s}\n", .{member.name, member.path});
}
```

#### `PathResolver`

Resolves paths within workspace.

```zig
pub const PathResolver = struct {
    allocator: Allocator,
    workspace: ?*Workspace,
    
    pub fn init(allocator: Allocator, ws: ?*Workspace) PathResolver
    pub fn resolvePath(self: *PathResolver, member_name: []const u8, relative_path: []const u8) ![]const u8
    pub fn resolveAbsolute(self: *PathResolver, workspace_root: []const u8, member_path: []const u8) ![]const u8
};
```

---

## 3. resolver.zig

Dependency resolution and conflict detection.

### Types

#### `VersionConstraint`

Version constraint specification.

```zig
pub const VersionConstraint = struct {
    min_version: ?manifest_mod.Version = null,
    max_version: ?manifest_mod.Version = null,
    exact_version: ?manifest_mod.Version = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) VersionConstraint
    pub fn deinit(self: *VersionConstraint) void
    pub fn parse(allocator: Allocator, constraint_str: []const u8) !VersionConstraint
    pub fn satisfies(self: VersionConstraint, version: manifest_mod.Version) bool
};
```

**Supported Formats:**
- `^1.2.3` - Caret: `>=1.2.3 <2.0.0`
- `>=1.0.0` - Minimum version
- `1.2.3` - Exact version

**Example:**
```zig
var constraint = try VersionConstraint.parse(allocator, "^1.2.0");
defer constraint.deinit();

const version = try manifest_mod.Version.parse(allocator, "1.3.0");
defer version.deinit();

if (constraint.satisfies(version)) {
    // Version 1.3.0 satisfies ^1.2.0
}
```

#### `DependencyNode`

Node in dependency graph.

```zig
pub const DependencyNode = struct {
    name: []const u8,
    version: manifest_mod.Version,
    dependencies: std.ArrayList(*DependencyNode),
    source: manifest_mod.DependencySource,
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        name: []const u8,
        version: manifest_mod.Version,
        source: manifest_mod.DependencySource
    ) !*DependencyNode
    pub fn deinit(self: *DependencyNode) void
};
```

#### `DependencyGraph`

Complete dependency graph.

```zig
pub const DependencyGraph = struct {
    roots: std.ArrayList(*DependencyNode),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DependencyGraph
    pub fn deinit(self: *DependencyGraph) void
};
```

#### `ResolvedDependency`

Resolved dependency with version.

```zig
pub const ResolvedDependency = struct {
    name: []const u8,
    version: manifest_mod.Version,
    source: manifest_mod.DependencySource,
    path: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        name: []const u8,
        version: manifest_mod.Version,
        source: manifest_mod.DependencySource
    ) !ResolvedDependency
    pub fn deinit(self: *ResolvedDependency) void
};
```

#### `Resolution`

Complete resolution result.

```zig
pub const Resolution = struct {
    dependencies: std.ArrayList(ResolvedDependency),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) Resolution
    pub fn deinit(self: *Resolution) void
};
```

#### `DependencyResolver`

Basic dependency resolver.

```zig
pub const DependencyResolver = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DependencyResolver
    pub fn buildGraph(self: *DependencyResolver, pkg: manifest_mod.PackageManifest) !DependencyGraph
    pub fn resolve(self: *DependencyResolver, graph: *DependencyGraph) !Resolution
    pub fn checkConflicts(self: *DependencyResolver, resolution: Resolution) !void
};
```

**Example:**
```zig
var resolver = DependencyResolver.init(allocator);

var graph = try resolver.buildGraph(package_manifest);
defer graph.deinit();

var resolution = try resolver.resolve(&graph);
defer resolution.deinit();

try resolver.checkConflicts(resolution);
```

#### `WorkspaceResolver`

Workspace-aware resolver.

```zig
pub const WorkspaceResolver = struct {
    allocator: Allocator,
    workspace: ?*workspace_mod.Workspace,
    basic_resolver: DependencyResolver,
    
    pub fn init(allocator: Allocator, ws: ?*workspace_mod.Workspace) WorkspaceResolver
    pub fn resolveWorkspace(self: *WorkspaceResolver) !Resolution
    pub fn resolvePathDeps(self: *WorkspaceResolver, member: workspace_mod.Member) !std.ArrayList(PathDependency)
};
```

**Example:**
```zig
var ws_resolver = WorkspaceResolver.init(allocator, &workspace);
var resolution = try ws_resolver.resolveWorkspace();
defer resolution.deinit();

// Resolution includes all workspace members
for (resolution.dependencies.items) |dep| {
    std.debug.print("{s} v{}\n", .{dep.name, dep.version});
}
```

---

## 4. zig_bridge.zig

Zig build system integration.

### Types

#### `BuildTarget`

Build artifact type.

```zig
pub const BuildTarget = enum {
    exe,       // Executable
    lib,       // Library
    test_exe,  // Test executable
};
```

#### `BuildMode`

Build optimization mode.

```zig
pub const BuildMode = enum {
    Debug,
    ReleaseSafe,
    ReleaseFast,
    ReleaseSmall,
};
```

#### `BuildConfig`

Build configuration.

```zig
pub const BuildConfig = struct {
    name: []const u8,
    target: BuildTarget,
    mode: BuildMode,
    source_files: std.ArrayList([]const u8),
    dependencies: std.ArrayList([]const u8),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, target: BuildTarget) !BuildConfig
    pub fn deinit(self: *BuildConfig) void
    pub fn addSourceFile(self: *BuildConfig, file_path: []const u8) !void
    pub fn addDependency(self: *BuildConfig, dep_name: []const u8) !void
};
```

#### `ZigBuildGenerator`

Generates Zig build files.

```zig
pub const ZigBuildGenerator = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) ZigBuildGenerator
    pub fn generateBuildScript(
        self: *ZigBuildGenerator,
        pkg: manifest_mod.PackageManifest,
        config: BuildConfig
    ) ![]const u8
    pub fn generateZonFile(
        self: *ZigBuildGenerator,
        pkg: manifest_mod.PackageManifest,
        resolution: resolver_mod.Resolution
    ) ![]const u8
};
```

**Example:**
```zig
var generator = ZigBuildGenerator.init(allocator);

var config = try BuildConfig.init(allocator, "my-app", .exe);
defer config.deinit();
try config.addSourceFile("src/main.zig");

const build_script = try generator.generateBuildScript(package, config);
defer allocator.free(build_script);

// Write to build.zig
const file = try std.fs.cwd().createFile("build.zig", .{});
defer file.close();
try file.writeAll(build_script);
```

#### `WorkspaceBuildCoordinator`

Coordinates workspace builds.

```zig
pub const WorkspaceBuildCoordinator = struct {
    allocator: Allocator,
    workspace: *workspace_mod.Workspace,
    generator: ZigBuildGenerator,
    
    pub fn init(allocator: Allocator, ws: *workspace_mod.Workspace) WorkspaceBuildCoordinator
    pub fn generateWorkspaceBuildScripts(self: *WorkspaceBuildCoordinator) !std.ArrayList(MemberBuildScript)
    pub fn generateUnifiedBuildScript(self: *WorkspaceBuildCoordinator) ![]const u8
};
```

#### `BuildExecutor`

Executes builds.

```zig
pub const BuildExecutor = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) BuildExecutor
    pub fn executeBuild(self: *BuildExecutor, project_path: []const u8, mode: BuildMode) !void
    pub fn executeTest(self: *BuildExecutor, project_path: []const u8) !void
};
```

---

## 5. cli.zig

Command-line interface.

### Types

#### `CliContext`

CLI execution context.

```zig
pub const CliContext = struct {
    allocator: Allocator,
    current_dir: []const u8,
    workspace: ?*workspace_mod.Workspace = null,
    
    pub fn init(allocator: Allocator, current_dir: []const u8) CliContext
    pub fn deinit(self: *CliContext) void
};
```

#### Commands

Each command is a struct with `init()` and `execute()` methods:

- `InitCommand` - Initialize package
- `InstallCommand` - Install dependencies
- `BuildCommand` - Build package
- `TestCommand` - Run tests
- `AddCommand` - Add dependency
- `WorkspaceCommand` - Manage workspace

**Example:**
```zig
var ctx = CliContext.init(allocator, ".");
defer ctx.deinit();

var install_cmd = InstallCommand.init(allocator);
try install_cmd.execute(&ctx);
```

#### `CliParser`

Parses command-line arguments.

```zig
pub const CliParser = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CliParser
    pub fn parse(self: *CliParser, args: []const []const u8) !Command
};

pub const Command = union(enum) {
    Init: []const u8,
    Install: void,
    Build: BuildMode,
    Test: void,
    Add: struct { name: []const u8, version: []const u8 },
    WorkspaceNew: []const u8,
    WorkspaceList: void,
    Help: void,
};
```

#### `CliRunner`

Executes commands.

```zig
pub const CliRunner = struct {
    allocator: Allocator,
    ctx: CliContext,
    
    pub fn init(allocator: Allocator) CliRunner
    pub fn deinit(self: *CliRunner) void
    pub fn run(self: *CliRunner, cmd: Command) !void
};
```

---

## Testing

All modules include comprehensive tests. Run with:

```bash
zig build test
```

### Test Coverage

- **manifest.zig**: 9 tests
- **workspace.zig**: 9 tests
- **resolver.zig**: 12 tests
- **zig_bridge.zig**: 10 tests
- **cli.zig**: 13 tests

**Total: 53/53 tests passing (100%)**

---

## Error Handling

All functions return Zig errors:

```zig
pub const Error = error{
    InvalidVersion,
    NotAPackage,
    NotAWorkspace,
    NoWorkspace,
    VersionConflict,
    MissingPath,
    MissingArgument,
    // ... more errors
};
```

---

## Memory Management

All structs follow Zig's init/deinit pattern:

```zig
var thing = try Thing.init(allocator, ...);
defer thing.deinit();
```

The package manager is designed to be leak-free with proper cleanup.

---

## Extension Points

### Adding New Commands

1. Create command struct in `cli.zig`
2. Implement `init()` and `execute()` methods
3. Add to `Command` union
4. Update parser in `CliParser.parse()`
5. Add handler in `CliRunner.run()`

### Adding New Dependency Sources

1. Add source type to `DependencySource` enum
2. Update `Dependency` struct
3. Handle in resolver
4. Update build.zig.zon generation

### Custom Build Targets

1. Add to `BuildTarget` enum
2. Update `ZigBuildGenerator.generateBuildScript()`
3. Handle in build execution

---

## Performance Considerations

- **Lazy loading**: Manifests loaded on-demand
- **Single-pass resolution**: Efficient graph traversal
- **Memory pooling**: Allocator passed throughout
- **Zero allocations in hot paths**: Where possible

---

## Thread Safety

Current implementation is single-threaded. For parallel operations:

- Use separate allocators per thread
- Avoid shared mutable state
- Consider Arc/Mutex for shared data

---

**For more examples, see EXAMPLES.md**
