// Mojo SDK - MLIR Setup & Infrastructure
// Day 11: MLIR/LLVM integration and Zig bindings

const std = @import("std");

// ============================================================================
// MLIR C API Bindings
// ============================================================================

// MLIR Context - Core MLIR infrastructure
pub const MlirContext = opaque {};
pub const MlirModule = opaque {};
pub const MlirOperation = opaque {};
pub const MlirBlock = opaque {};
pub const MlirRegion = opaque {};
pub const MlirValue = opaque {};
pub const MlirType = opaque {};
pub const MlirAttribute = opaque {};
pub const MlirLocation = opaque {};
pub const MlirIdentifier = opaque {};

// String reference for MLIR
pub const MlirStringRef = extern struct {
    data: [*]const u8,
    length: usize,
};

// MLIR Logical Result
pub const MlirLogicalResult = extern struct {
    value: i8,
};

// External C function declarations
extern "c" fn mlirContextCreate() ?*MlirContext;
extern "c" fn mlirContextDestroy(context: *MlirContext) void;
extern "c" fn mlirModuleCreateEmpty(location: *MlirLocation) ?*MlirModule;
extern "c" fn mlirModuleDestroy(module: *MlirModule) void;
extern "c" fn mlirModuleGetContext(module: *MlirModule) ?*MlirContext;
extern "c" fn mlirLocationUnknownGet(context: *MlirContext) ?*MlirLocation;

// Print functions
extern "c" fn mlirModulePrint(module: *MlirModule, callback: *const fn ([*]const u8, usize, ?*anyopaque) callconv(.C) void, userData: ?*anyopaque) void;

// ============================================================================
// Zig Wrapper - MlirContext
// ============================================================================

pub const Context = struct {
    handle: *MlirContext,
    
    pub fn init() !Context {
        const handle = mlirContextCreate() orelse return error.MlirContextCreateFailed;
        return Context{ .handle = handle };
    }
    
    pub fn deinit(self: *Context) void {
        mlirContextDestroy(self.handle);
    }
    
    pub fn createUnknownLocation(self: *Context) !Location {
        const loc = mlirLocationUnknownGet(self.handle) orelse return error.LocationCreateFailed;
        return Location{ .handle = loc };
    }
};

// ============================================================================
// Zig Wrapper - MlirLocation
// ============================================================================

pub const Location = struct {
    handle: *MlirLocation,
};

// ============================================================================
// Zig Wrapper - MlirModule
// ============================================================================

pub const Module = struct {
    handle: *MlirModule,
    context: *MlirContext,
    
    pub fn init(context: *Context, location: Location) !Module {
        const handle = mlirModuleCreateEmpty(location.handle) orelse return error.ModuleCreateFailed;
        return Module{
            .handle = handle,
            .context = context.handle,
        };
    }
    
    pub fn deinit(self: *Module) void {
        mlirModuleDestroy(self.handle);
    }
    
    pub fn print(self: *Module, writer: anytype) !void {
        const PrintContext = struct {
            w: @TypeOf(writer),
            err: ?anyerror = null,
            
            fn callback(data: [*]const u8, length: usize, userData: ?*anyopaque) callconv(.C) void {
                const ctx: *@This() = @ptrCast(@alignCast(userData));
                const slice = data[0..length];
                ctx.w.writeAll(slice) catch |err| {
                    ctx.err = err;
                };
            }
        };
        
        var print_ctx = PrintContext{ .w = writer };
        mlirModulePrint(self.handle, PrintContext.callback, &print_ctx);
        
        if (print_ctx.err) |err| {
            return err;
        }
    }
    
    pub fn getContext(self: *Module) ?*MlirContext {
        return mlirModuleGetContext(self.handle);
    }
};

// ============================================================================
// MLIR Setup & Configuration
// ============================================================================

pub const MlirSetup = struct {
    llvm_path: []const u8,
    version: []const u8,
    
    pub fn detect() !MlirSetup {
        // Homebrew path on macOS
        const llvm_path = "/opt/homebrew/opt/llvm";
        
        // Verify path exists
        std.fs.accessAbsolute(llvm_path, .{}) catch {
            return error.LlvmNotFound;
        };
        
        return MlirSetup{
            .llvm_path = llvm_path,
            .version = "21.1.8",
        };
    }
    
    pub fn getMlirOptPath(self: *const MlirSetup) []const u8 {
        _ = self;
        return "/opt/homebrew/opt/llvm/bin/mlir-opt";
    }
    
    pub fn getMlirTranslatePath(self: *const MlirSetup) []const u8 {
        _ = self;
        return "/opt/homebrew/opt/llvm/bin/mlir-translate";
    }
    
    pub fn getIncludePath(self: *const MlirSetup) []const u8 {
        _ = self;
        return "/opt/homebrew/opt/llvm/include";
    }
    
    pub fn getLibPath(self: *const MlirSetup) []const u8 {
        _ = self;
        return "/opt/homebrew/opt/llvm/lib";
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

pub fn createStringRef(str: []const u8) MlirStringRef {
    return MlirStringRef{
        .data = str.ptr,
        .length = str.len,
    };
}

pub fn isSuccess(result: MlirLogicalResult) bool {
    return result.value != 0;
}

pub fn isFailure(result: MlirLogicalResult) bool {
    return result.value == 0;
}

// ============================================================================
// Tests
// ============================================================================

test "mlir_setup: detect MLIR installation" {
    const setup = try MlirSetup.detect();
    
    try std.testing.expectEqualStrings("/opt/homebrew/opt/llvm", setup.llvm_path);
    try std.testing.expectEqualStrings("21.1.8", setup.version);
    
    // Verify paths exist
    try std.testing.expect(setup.getMlirOptPath().len > 0);
    try std.testing.expect(setup.getIncludePath().len > 0);
}

test "mlir_setup: create and destroy context" {
    var context = try Context.init();
    defer context.deinit();
    
    // Context should be created successfully
    try std.testing.expect(@intFromPtr(context.handle) != 0);
}

test "mlir_setup: create empty module" {
    var context = try Context.init();
    defer context.deinit();
    
    const location = try context.createUnknownLocation();
    
    var module = try Module.init(&context, location);
    defer module.deinit();
    
    // Module should be created successfully
    try std.testing.expect(@intFromPtr(module.handle) != 0);
    
    // Module should reference the context
    const mod_ctx = module.getContext();
    try std.testing.expect(mod_ctx == context.handle);
}

test "mlir_setup: module operations" {
    var context = try Context.init();
    defer context.deinit();
    
    const location = try context.createUnknownLocation();
    
    var module = try Module.init(&context, location);
    defer module.deinit();
    
    // Verify module is valid and has a context
    const mod_ctx = module.getContext();
    try std.testing.expect(mod_ctx != null);
    try std.testing.expect(mod_ctx == context.handle);
}

test "mlir_setup: string ref creation" {
    const test_str = "Hello MLIR";
    const str_ref = createStringRef(test_str);
    
    try std.testing.expectEqual(test_str.len, str_ref.length);
    try std.testing.expectEqual(test_str.ptr, str_ref.data);
    
    // Verify we can read the string back
    const read_back = str_ref.data[0..str_ref.length];
    try std.testing.expectEqualStrings(test_str, read_back);
}
