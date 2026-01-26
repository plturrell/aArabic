// Mojo SDK - MLIR Dialect Definition
// Day 12: Custom Mojo dialect operations in MLIR

const std = @import("std");
const mlir = @import("mlir_setup");

// ============================================================================
// Mojo Dialect - Custom Operations for Mojo Language
// ============================================================================

/// The Mojo dialect provides MLIR operations specific to the Mojo language.
/// Namespace: "mojo"
/// 
/// Operations:
/// - mojo.fn: Function definition
/// - mojo.call: Function call
/// - mojo.var: Variable declaration
/// - mojo.assign: Variable assignment
/// - mojo.return: Return statement
/// - mojo.struct: Struct type definition
/// - mojo.field_access: Access struct field
/// - mojo.if: Conditional statement
/// - mojo.while: While loop
/// - mojo.const: Compile-time constant
pub const MojoDialect = struct {
    name: []const u8 = "mojo",
    namespace_prefix: []const u8 = "mojo.",
    
    /// Initialize the Mojo dialect
    pub fn init() MojoDialect {
        return MojoDialect{};
    }
    
    /// Get the dialect name
    pub fn getName(self: *const MojoDialect) []const u8 {
        return self.name;
    }
    
    /// Get the namespace prefix for operations
    pub fn getNamespacePrefix(self: *const MojoDialect) []const u8 {
        return self.namespace_prefix;
    }
};

// ============================================================================
// Mojo Type System in MLIR
// ============================================================================

pub const MojoTypeKind = enum {
    // Primitive types
    Int,      // Integer types (i8, i16, i32, i64)
    Float,    // Floating point (f32, f64)
    Bool,     // Boolean
    String,   // String type
    
    // Composite types
    Struct,   // User-defined struct
    Array,    // Array type
    Tuple,    // Tuple type
    
    // Function types
    Function, // Function type
    
    // Special types
    Void,     // No return value
    Unknown,  // Type inference pending
};

pub const MojoType = struct {
    kind: MojoTypeKind,
    bit_width: u32 = 0, // For Int/Float
    name: ?[]const u8 = null, // For Struct/named types
    
    pub fn createInt(bit_width: u32) MojoType {
        return MojoType{
            .kind = .Int,
            .bit_width = bit_width,
        };
    }
    
    pub fn createFloat(bit_width: u32) MojoType {
        return MojoType{
            .kind = .Float,
            .bit_width = bit_width,
        };
    }
    
    pub fn createBool() MojoType {
        return MojoType{
            .kind = .Bool,
            .bit_width = 1,
        };
    }
    
    pub fn createString() MojoType {
        return MojoType{
            .kind = .String,
        };
    }
    
    pub fn createStruct(name: []const u8) MojoType {
        return MojoType{
            .kind = .Struct,
            .name = name,
        };
    }
    
    pub fn createVoid() MojoType {
        return MojoType{
            .kind = .Void,
        };
    }
    
    pub fn isIntegral(self: *const MojoType) bool {
        return self.kind == .Int or self.kind == .Bool;
    }
    
    pub fn isFloatingPoint(self: *const MojoType) bool {
        return self.kind == .Float;
    }
    
    pub fn isNumeric(self: *const MojoType) bool {
        return self.isIntegral() or self.isFloatingPoint();
    }
};

// ============================================================================
// Mojo Operations - Base Operation Structure
// ============================================================================

pub const MojoOpKind = enum {
    // Function operations
    Fn,          // mojo.fn
    Call,        // mojo.call
    Return,      // mojo.return
    
    // Variable operations
    Var,         // mojo.var
    Assign,      // mojo.assign
    Load,        // mojo.load
    
    // Struct operations
    Struct,      // mojo.struct
    FieldAccess, // mojo.field_access
    
    // Control flow
    If,          // mojo.if
    While,       // mojo.while
    
    // Constants
    Const,       // mojo.const
    
    // Arithmetic
    Add,         // mojo.add
    Sub,         // mojo.sub
    Mul,         // mojo.mul
    Div,         // mojo.div
    
    // Comparison
    Eq,          // mojo.eq
    Ne,          // mojo.ne
    Lt,          // mojo.lt
    Le,          // mojo.le
    Gt,          // mojo.gt
    Ge,          // mojo.ge
};

pub const MojoOp = struct {
    kind: MojoOpKind,
    name: []const u8,
    result_type: ?MojoType = null,
    
    pub fn create(kind: MojoOpKind, name: []const u8) MojoOp {
        return MojoOp{
            .kind = kind,
            .name = name,
        };
    }
    
    pub fn getOpName(self: *const MojoOp) []const u8 {
        return self.name;
    }
    
    pub fn hasResult(self: *const MojoOp) bool {
        return self.result_type != null;
    }
};

// ============================================================================
// Specific Operation Definitions
// ============================================================================

/// mojo.fn - Function definition
/// 
/// Example:
///   mojo.fn @my_func(%arg0: i32, %arg1: i32) -> i32 {
///     %0 = mojo.add %arg0, %arg1
///     mojo.return %0
///   }
pub const FnOp = struct {
    base: MojoOp,
    function_name: []const u8,
    parameters: []const MojoType,
    return_type: MojoType,
    
    pub fn create(name: []const u8, params: []const MojoType, ret_type: MojoType) FnOp {
        return FnOp{
            .base = MojoOp.create(.Fn, "mojo.fn"),
            .function_name = name,
            .parameters = params,
            .return_type = ret_type,
        };
    }
};

/// mojo.call - Function call
/// 
/// Example:
///   %result = mojo.call @my_func(%arg0, %arg1) : (i32, i32) -> i32
pub const CallOp = struct {
    base: MojoOp,
    callee: []const u8,
    arguments: usize,
    
    pub fn create(callee: []const u8, num_args: usize) CallOp {
        return CallOp{
            .base = MojoOp.create(.Call, "mojo.call"),
            .callee = callee,
            .arguments = num_args,
        };
    }
};

/// mojo.var - Variable declaration
/// 
/// Example:
///   %var = mojo.var "x" : i32
pub const VarOp = struct {
    base: MojoOp,
    var_name: []const u8,
    var_type: MojoType,
    
    pub fn create(name: []const u8, var_type: MojoType) VarOp {
        return VarOp{
            .base = MojoOp.create(.Var, "mojo.var"),
            .var_name = name,
            .var_type = var_type,
        };
    }
};

/// mojo.assign - Variable assignment
/// 
/// Example:
///   mojo.assign %var, %value : i32
pub const AssignOp = struct {
    base: MojoOp,
    
    pub fn create() AssignOp {
        return AssignOp{
            .base = MojoOp.create(.Assign, "mojo.assign"),
        };
    }
};

/// mojo.return - Return statement
/// 
/// Example:
///   mojo.return %value : i32
pub const ReturnOp = struct {
    base: MojoOp,
    has_value: bool,
    
    pub fn create(has_value: bool) ReturnOp {
        return ReturnOp{
            .base = MojoOp.create(.Return, "mojo.return"),
            .has_value = has_value,
        };
    }
};

/// mojo.const - Compile-time constant
/// 
/// Example:
///   %c = mojo.const 42 : i32
pub const ConstOp = struct {
    base: MojoOp,
    value: i64,
    const_type: MojoType,
    
    pub fn create(value: i64, const_type: MojoType) ConstOp {
        return ConstOp{
            .base = MojoOp.create(.Const, "mojo.const"),
            .value = value,
            .const_type = const_type,
        };
    }
};

/// mojo.add - Addition operation
/// 
/// Example:
///   %result = mojo.add %lhs, %rhs : i32
pub const AddOp = struct {
    base: MojoOp,
    
    pub fn create() AddOp {
        return AddOp{
            .base = MojoOp.create(.Add, "mojo.add"),
        };
    }
};

// ============================================================================
// Operation Builder - Creates MLIR operations
// ============================================================================

pub const MojoOpBuilder = struct {
    dialect: MojoDialect,
    
    pub fn init() MojoOpBuilder {
        return MojoOpBuilder{
            .dialect = MojoDialect.init(),
        };
    }
    
    /// Build a function operation
    pub fn buildFn(self: *MojoOpBuilder, name: []const u8, params: []const MojoType, ret_type: MojoType) FnOp {
        _ = self;
        return FnOp.create(name, params, ret_type);
    }
    
    /// Build a call operation
    pub fn buildCall(self: *MojoOpBuilder, callee: []const u8, num_args: usize) CallOp {
        _ = self;
        return CallOp.create(callee, num_args);
    }
    
    /// Build a variable declaration
    pub fn buildVar(self: *MojoOpBuilder, name: []const u8, var_type: MojoType) VarOp {
        _ = self;
        return VarOp.create(name, var_type);
    }
    
    /// Build a constant
    pub fn buildConst(self: *MojoOpBuilder, value: i64, const_type: MojoType) ConstOp {
        _ = self;
        return ConstOp.create(value, const_type);
    }
    
    /// Build an addition
    pub fn buildAdd(self: *MojoOpBuilder) AddOp {
        _ = self;
        return AddOp.create();
    }
    
    /// Build a return
    pub fn buildReturn(self: *MojoOpBuilder, has_value: bool) ReturnOp {
        _ = self;
        return ReturnOp.create(has_value);
    }
};

// ============================================================================
// Operation Verification
// ============================================================================

pub const VerificationResult = struct {
    success: bool,
    error_message: ?[]const u8 = null,
    
    pub fn ok() VerificationResult {
        return VerificationResult{ .success = true };
    }
    
    pub fn err(message: []const u8) VerificationResult {
        return VerificationResult{
            .success = false,
            .error_message = message,
        };
    }
};

pub const MojoOpVerifier = struct {
    /// Verify a function operation
    pub fn verifyFn(op: *const FnOp) VerificationResult {
        if (op.function_name.len == 0) {
            return VerificationResult.err("Function name cannot be empty");
        }
        return VerificationResult.ok();
    }
    
    /// Verify a call operation
    pub fn verifyCall(op: *const CallOp) VerificationResult {
        if (op.callee.len == 0) {
            return VerificationResult.err("Callee name cannot be empty");
        }
        return VerificationResult.ok();
    }
    
    /// Verify a variable declaration
    pub fn verifyVar(op: *const VarOp) VerificationResult {
        if (op.var_name.len == 0) {
            return VerificationResult.err("Variable name cannot be empty");
        }
        return VerificationResult.ok();
    }
    
    /// Verify a constant
    pub fn verifyConst(op: *const ConstOp) VerificationResult {
        if (!op.const_type.isNumeric()) {
            return VerificationResult.err("Constant must have numeric type");
        }
        return VerificationResult.ok();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "mojo_dialect: create dialect" {
    const dialect = MojoDialect.init();
    
    try std.testing.expectEqualStrings("mojo", dialect.getName());
    try std.testing.expectEqualStrings("mojo.", dialect.getNamespacePrefix());
}

test "mojo_dialect: create types" {
    const int32 = MojoType.createInt(32);
    try std.testing.expectEqual(MojoTypeKind.Int, int32.kind);
    try std.testing.expectEqual(@as(u32, 32), int32.bit_width);
    try std.testing.expect(int32.isIntegral());
    try std.testing.expect(int32.isNumeric());
    
    const float64 = MojoType.createFloat(64);
    try std.testing.expectEqual(MojoTypeKind.Float, float64.kind);
    try std.testing.expect(float64.isFloatingPoint());
    try std.testing.expect(float64.isNumeric());
    
    const bool_type = MojoType.createBool();
    try std.testing.expectEqual(MojoTypeKind.Bool, bool_type.kind);
    try std.testing.expect(bool_type.isIntegral());
}

test "mojo_dialect: create function operation" {
    var builder = MojoOpBuilder.init();
    
    const params = [_]MojoType{
        MojoType.createInt(32),
        MojoType.createInt(32),
    };
    const ret_type = MojoType.createInt(32);
    
    const fn_op = builder.buildFn("add", params[0..], ret_type);
    
    try std.testing.expectEqualStrings("add", fn_op.function_name);
    try std.testing.expectEqual(@as(usize, 2), fn_op.parameters.len);
    try std.testing.expectEqual(MojoTypeKind.Int, fn_op.return_type.kind);
}

test "mojo_dialect: create operations" {
    var builder = MojoOpBuilder.init();
    
    // Create a call
    const call_op = builder.buildCall("my_func", 2);
    try std.testing.expectEqualStrings("my_func", call_op.callee);
    try std.testing.expectEqual(@as(usize, 2), call_op.arguments);
    
    // Create a variable
    const var_op = builder.buildVar("x", MojoType.createInt(32));
    try std.testing.expectEqualStrings("x", var_op.var_name);
    try std.testing.expectEqual(MojoTypeKind.Int, var_op.var_type.kind);
    
    // Create a constant
    const const_op = builder.buildConst(42, MojoType.createInt(32));
    try std.testing.expectEqual(@as(i64, 42), const_op.value);
    
    // Create arithmetic
    const add_op = builder.buildAdd();
    try std.testing.expectEqual(MojoOpKind.Add, add_op.base.kind);
}

test "mojo_dialect: operation verification" {
    var builder = MojoOpBuilder.init();
    
    // Valid function
    const params = [_]MojoType{MojoType.createInt(32)};
    const fn_op = builder.buildFn("valid_func", params[0..], MojoType.createInt(32));
    const fn_result = MojoOpVerifier.verifyFn(&fn_op);
    try std.testing.expect(fn_result.success);
    
    // Valid constant
    const const_op = builder.buildConst(10, MojoType.createInt(32));
    const const_result = MojoOpVerifier.verifyConst(&const_op);
    try std.testing.expect(const_result.success);
    
    // Valid call
    const call_op = builder.buildCall("func", 1);
    const call_result = MojoOpVerifier.verifyCall(&call_op);
    try std.testing.expect(call_result.success);
}
