// Mojo SDK - Symbol Table
// Day 5: Symbol table for semantic analysis

const std = @import("std");
const ast = @import("ast");

// ============================================================================
// Symbol Types
// ============================================================================

pub const SymbolKind = enum {
    variable,
    function,
    parameter,
    struct_type,
    trait_type,
    field,
};

pub const Symbol = struct {
    name: []const u8,
    kind: SymbolKind,
    type_name: ?[]const u8,
    is_mutable: bool,
    scope_level: usize,
    
    pub fn init(name: []const u8, kind: SymbolKind, type_name: ?[]const u8, is_mutable: bool, scope_level: usize) Symbol {
        return Symbol{
            .name = name,
            .kind = kind,
            .type_name = type_name,
            .is_mutable = is_mutable,
            .scope_level = scope_level,
        };
    }
};

// ============================================================================
// Scope Management
// ============================================================================

pub const Scope = struct {
    symbols: std.StringHashMap(Symbol),
    parent: ?*Scope,
    level: usize,
    
    pub fn init(allocator: std.mem.Allocator, parent: ?*Scope, level: usize) !Scope {
        return Scope{
            .symbols = std.StringHashMap(Symbol).init(allocator),
            .parent = parent,
            .level = level,
        };
    }
    
    pub fn deinit(self: *Scope) void {
        self.symbols.deinit();
    }
    
    pub fn define(self: *Scope, symbol: Symbol) !void {
        try self.symbols.put(symbol.name, symbol);
    }
    
    pub fn lookup(self: *Scope, name: []const u8) ?Symbol {
        // Check current scope
        if (self.symbols.get(name)) |symbol| {
            return symbol;
        }
        
        // Check parent scopes
        if (self.parent) |parent| {
            return parent.lookup(name);
        }
        
        return null;
    }
    
    pub fn lookupLocal(self: *Scope, name: []const u8) ?Symbol {
        return self.symbols.get(name);
    }
};

// ============================================================================
// Symbol Table
// ============================================================================

pub const SymbolTable = struct {
    allocator: std.mem.Allocator,
    global_scope: *Scope,
    current_scope: *Scope,
    scope_stack: std.ArrayList(*Scope),
    
    pub fn init(allocator: std.mem.Allocator) !SymbolTable {
        const global_scope = try allocator.create(Scope);
        global_scope.* = try Scope.init(allocator, null, 0);
        
        // Add built-in types
        try global_scope.define(Symbol.init("Int", .struct_type, null, false, 0));
        try global_scope.define(Symbol.init("Float", .struct_type, null, false, 0));
        try global_scope.define(Symbol.init("Bool", .struct_type, null, false, 0));
        try global_scope.define(Symbol.init("String", .struct_type, null, false, 0));
        
        const scope_stack = try std.ArrayList(*Scope).initCapacity(allocator, 16);
        
        return SymbolTable{
            .allocator = allocator,
            .global_scope = global_scope,
            .current_scope = global_scope,
            .scope_stack = scope_stack,
        };
    }
    
    pub fn deinit(self: *SymbolTable) void {
        // Clean up all scopes in stack
        for (self.scope_stack.items) |scope| {
            scope.deinit();
            self.allocator.destroy(scope);
        }
        self.scope_stack.deinit(self.allocator);
        self.global_scope.deinit();
        self.allocator.destroy(self.global_scope);
    }
    
    pub fn enterScope(self: *SymbolTable) !void {
        const new_level = self.current_scope.level + 1;
        const new_scope = try self.allocator.create(Scope);
        new_scope.* = try Scope.init(self.allocator, self.current_scope, new_level);
        
        try self.scope_stack.append(self.allocator, new_scope);
        self.current_scope = new_scope;
    }
    
    pub fn exitScope(self: *SymbolTable) !void {
        if (self.scope_stack.items.len == 0) {
            return error.ScopeUnderflow;
        }
        
        // Get the scope to clean up
        const old_scope = self.current_scope;
        
        // Pop from stack
        _ = self.scope_stack.pop();
        
        // Set new current scope
        if (self.scope_stack.items.len > 0) {
            self.current_scope = self.scope_stack.items[self.scope_stack.items.len - 1];
        } else {
            self.current_scope = self.global_scope;
        }
        
        // Clean up old scope (unless it's global)
        if (old_scope != self.global_scope) {
            old_scope.deinit();
            self.allocator.destroy(old_scope);
        }
    }
    
    pub fn define(self: *SymbolTable, symbol: Symbol) !void {
        try self.current_scope.define(symbol);
    }
    
    pub fn lookup(self: *SymbolTable, name: []const u8) ?Symbol {
        return self.current_scope.lookup(name);
    }
    
    pub fn lookupLocal(self: *SymbolTable, name: []const u8) ?Symbol {
        return self.current_scope.lookupLocal(name);
    }
    
    pub fn isTypeDefined(self: *SymbolTable, type_name: []const u8) bool {
        if (self.lookup(type_name)) |symbol| {
            return symbol.kind == .struct_type or symbol.kind == .trait_type;
        }
        return false;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "symbol table: basic operations" {
    var table = try SymbolTable.init(std.testing.allocator);
    defer table.deinit();
    
    // Define a variable
    try table.define(Symbol.init("x", .variable, "Int", true, 0));
    
    // Lookup the variable
    const symbol = table.lookup("x");
    try std.testing.expect(symbol != null);
    try std.testing.expectEqualStrings("x", symbol.?.name);
    try std.testing.expectEqual(SymbolKind.variable, symbol.?.kind);
}

test "symbol table: scope management" {
    var table = try SymbolTable.init(std.testing.allocator);
    defer table.deinit();
    
    // Define in global scope
    try table.define(Symbol.init("global_var", .variable, "Int", true, 0));
    
    // Enter new scope
    try table.enterScope();
    try table.define(Symbol.init("local_var", .variable, "Int", true, 1));
    
    // Both should be visible
    try std.testing.expect(table.lookup("global_var") != null);
    try std.testing.expect(table.lookup("local_var") != null);
    
    // Exit scope
    try table.exitScope();
    
    // Only global should be visible
    try std.testing.expect(table.lookup("global_var") != null);
    try std.testing.expect(table.lookup("local_var") == null);
}

test "symbol table: shadowing" {
    var table = try SymbolTable.init(std.testing.allocator);
    defer table.deinit();
    
    // Define in global scope
    try table.define(Symbol.init("x", .variable, "Int", true, 0));
    
    // Enter new scope and shadow
    try table.enterScope();
    try table.define(Symbol.init("x", .variable, "Float", true, 1));
    
    // Should find the inner scope variable
    const symbol = table.lookup("x");
    try std.testing.expect(symbol != null);
    try std.testing.expectEqualStrings("Float", symbol.?.type_name.?);
    
    // Exit scope
    try table.exitScope();
    
    // Should find the global variable again
    const global_symbol = table.lookup("x");
    try std.testing.expect(global_symbol != null);
    try std.testing.expectEqualStrings("Int", global_symbol.?.type_name.?);
}

test "symbol table: built-in types" {
    var table = try SymbolTable.init(std.testing.allocator);
    defer table.deinit();
    
    // Check built-in types
    try std.testing.expect(table.isTypeDefined("Int"));
    try std.testing.expect(table.isTypeDefined("Float"));
    try std.testing.expect(table.isTypeDefined("Bool"));
    try std.testing.expect(table.isTypeDefined("String"));
    try std.testing.expect(!table.isTypeDefined("CustomType"));
}

test "symbol table: function symbols" {
    var table = try SymbolTable.init(std.testing.allocator);
    defer table.deinit();
    
    // Define a function
    try table.define(Symbol.init("add", .function, "Int", false, 0));
    
    // Lookup the function
    const symbol = table.lookup("add");
    try std.testing.expect(symbol != null);
    try std.testing.expectEqual(SymbolKind.function, symbol.?.kind);
}
