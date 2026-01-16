# Mojo FFI (Foreign Function Interface) Module
# Day 46 - C function calls and struct marshalling
#
# This module provides the interface for calling C functions from Mojo,
# loading dynamic libraries, and marshalling data between Mojo and C.

# =============================================================================
# C Type Definitions
# =============================================================================

struct CType:
    """Represents C type information for FFI."""

    # Type kind enumeration
    alias VOID = 0
    alias CHAR = 1
    alias SCHAR = 2
    alias UCHAR = 3
    alias SHORT = 4
    alias USHORT = 5
    alias INT = 6
    alias UINT = 7
    alias LONG = 8
    alias ULONG = 9
    alias LONGLONG = 10
    alias ULONGLONG = 11
    alias FLOAT = 12
    alias DOUBLE = 13
    alias POINTER = 14
    alias STRUCT = 15
    alias ARRAY = 16
    alias FUNCTION = 17

    var kind: Int
    var size: Int
    var alignment: Int
    var name: String

    fn __init__(inout self, kind: Int, size: Int, alignment: Int, name: String):
        self.kind = kind
        self.size = size
        self.alignment = alignment
        self.name = name

    @staticmethod
    fn void() -> CType:
        return CType(CType.VOID, 0, 1, "void")

    @staticmethod
    fn char() -> CType:
        return CType(CType.CHAR, 1, 1, "char")

    @staticmethod
    fn int8() -> CType:
        return CType(CType.SCHAR, 1, 1, "int8_t")

    @staticmethod
    fn uint8() -> CType:
        return CType(CType.UCHAR, 1, 1, "uint8_t")

    @staticmethod
    fn int16() -> CType:
        return CType(CType.SHORT, 2, 2, "int16_t")

    @staticmethod
    fn uint16() -> CType:
        return CType(CType.USHORT, 2, 2, "uint16_t")

    @staticmethod
    fn int32() -> CType:
        return CType(CType.INT, 4, 4, "int32_t")

    @staticmethod
    fn uint32() -> CType:
        return CType(CType.UINT, 4, 4, "uint32_t")

    @staticmethod
    fn int64() -> CType:
        return CType(CType.LONGLONG, 8, 8, "int64_t")

    @staticmethod
    fn uint64() -> CType:
        return CType(CType.ULONGLONG, 8, 8, "uint64_t")

    @staticmethod
    fn float32() -> CType:
        return CType(CType.FLOAT, 4, 4, "float")

    @staticmethod
    fn float64() -> CType:
        return CType(CType.DOUBLE, 8, 8, "double")

    @staticmethod
    fn pointer() -> CType:
        return CType(CType.POINTER, 8, 8, "void*")  # Assuming 64-bit

    @staticmethod
    fn c_int() -> CType:
        """Platform-specific C int."""
        return CType(CType.INT, 4, 4, "int")

    @staticmethod
    fn c_long() -> CType:
        """Platform-specific C long."""
        # On most 64-bit Unix: 8 bytes, on Windows: 4 bytes
        return CType(CType.LONG, 8, 8, "long")

    @staticmethod
    fn c_size_t() -> CType:
        """Platform-specific size_t."""
        return CType(CType.ULONG, 8, 8, "size_t")

    fn is_integer(self) -> Bool:
        return self.kind >= CType.CHAR and self.kind <= CType.ULONGLONG

    fn is_floating(self) -> Bool:
        return self.kind == CType.FLOAT or self.kind == CType.DOUBLE

    fn is_pointer(self) -> Bool:
        return self.kind == CType.POINTER

    fn __str__(self) -> String:
        return self.name


# =============================================================================
# C Value Representation
# =============================================================================

struct CValue:
    """A value that can be passed to/from C functions."""

    var _type: CType
    var _data: UnsafePointer[UInt8]
    var _owns_data: Bool

    fn __init__(inout self, type: CType):
        self._type = type
        self._data = UnsafePointer[UInt8].alloc(type.size)
        self._owns_data = True

    fn __init__(inout self, value: Int):
        self._type = CType.int64()
        self._data = UnsafePointer[UInt8].alloc(8)
        self._owns_data = True
        var ptr = self._data.bitcast[Int64]()
        ptr.store(value)

    fn __init__(inout self, value: Int32):
        self._type = CType.int32()
        self._data = UnsafePointer[UInt8].alloc(4)
        self._owns_data = True
        var ptr = self._data.bitcast[Int32]()
        ptr.store(value)

    fn __init__(inout self, value: Float64):
        self._type = CType.float64()
        self._data = UnsafePointer[UInt8].alloc(8)
        self._owns_data = True
        var ptr = self._data.bitcast[Float64]()
        ptr.store(value)

    fn __init__(inout self, value: Float32):
        self._type = CType.float32()
        self._data = UnsafePointer[UInt8].alloc(4)
        self._owns_data = True
        var ptr = self._data.bitcast[Float32]()
        ptr.store(value)

    fn __init__(inout self, value: Bool):
        self._type = CType.int32()
        self._data = UnsafePointer[UInt8].alloc(4)
        self._owns_data = True
        var ptr = self._data.bitcast[Int32]()
        ptr.store(1 if value else 0)

    fn __del__(owned self):
        if self._owns_data:
            self._data.free()

    fn as_int(self) -> Int:
        """Get value as Int."""
        if self._type.size == 8:
            return int(self._data.bitcast[Int64]().load())
        elif self._type.size == 4:
            return int(self._data.bitcast[Int32]().load())
        elif self._type.size == 2:
            return int(self._data.bitcast[Int16]().load())
        else:
            return int(self._data.bitcast[Int8]().load())

    fn as_float(self) -> Float64:
        """Get value as Float64."""
        if self._type.kind == CType.DOUBLE:
            return self._data.bitcast[Float64]().load()
        else:
            return Float64(self._data.bitcast[Float32]().load())

    fn as_bool(self) -> Bool:
        """Get value as Bool."""
        return self.as_int() != 0

    fn as_pointer(self) -> UnsafePointer[UInt8]:
        """Get value as pointer."""
        return self._data.bitcast[UnsafePointer[UInt8]]().load()

    fn data_ptr(self) -> UnsafePointer[UInt8]:
        """Get raw data pointer."""
        return self._data


# =============================================================================
# C String Utilities
# =============================================================================

struct CString:
    """Wrapper for null-terminated C strings."""

    var _data: UnsafePointer[Int8]
    var _len: Int
    var _owns_data: Bool

    fn __init__(inout self, s: String):
        """Create a C string from a Mojo String."""
        self._len = len(s)
        self._data = UnsafePointer[Int8].alloc(self._len + 1)
        self._owns_data = True

        # Copy string data
        for i in range(self._len):
            self._data.store(i, Int8(ord(s[i])))
        # Null terminator
        self._data.store(self._len, 0)

    fn __init__(inout self, ptr: UnsafePointer[Int8], owned: Bool = False):
        """Create from an existing C string pointer."""
        self._data = ptr
        self._owns_data = owned

        # Calculate length
        self._len = 0
        while ptr.load(self._len) != 0:
            self._len += 1

    fn __del__(owned self):
        if self._owns_data:
            self._data.free()

    fn __len__(self) -> Int:
        return self._len

    fn c_str(self) -> UnsafePointer[Int8]:
        """Get the null-terminated C string pointer."""
        return self._data

    fn to_string(self) -> String:
        """Convert to a Mojo String."""
        var result = String()
        for i in range(self._len):
            result += chr(int(self._data.load(i)))
        return result

    @staticmethod
    fn from_ptr(ptr: UnsafePointer[Int8]) -> String:
        """Convert a C string pointer to Mojo String without ownership."""
        var cstr = CString(ptr, owned=False)
        return cstr.to_string()


# =============================================================================
# Function Signature
# =============================================================================

struct FunctionSignature:
    """Describes a C function's signature."""

    var return_type: CType
    var arg_types: List[CType]
    var is_variadic: Bool

    fn __init__(inout self, return_type: CType):
        self.return_type = return_type
        self.arg_types = List[CType]()
        self.is_variadic = False

    fn __init__(inout self, return_type: CType, *args: CType):
        self.return_type = return_type
        self.arg_types = List[CType]()
        self.is_variadic = False
        for arg in args:
            self.arg_types.append(arg[])

    fn add_arg(inout self, arg_type: CType):
        self.arg_types.append(arg_type)

    fn set_variadic(inout self):
        self.is_variadic = True

    fn arg_count(self) -> Int:
        return len(self.arg_types)

    fn __str__(self) -> String:
        var result = self.return_type.name + " ("
        for i in range(len(self.arg_types)):
            if i > 0:
                result += ", "
            result += self.arg_types[i].name
        if self.is_variadic:
            result += ", ..."
        result += ")"
        return result


# =============================================================================
# External Function
# =============================================================================

struct ExternalFunction:
    """Wrapper for calling external C functions."""

    var _name: String
    var _address: UnsafePointer[UInt8]
    var _signature: FunctionSignature

    fn __init__(inout self, name: String, address: UnsafePointer[UInt8], signature: FunctionSignature):
        self._name = name
        self._address = address
        self._signature = signature

    fn name(self) -> String:
        return self._name

    fn address(self) -> UnsafePointer[UInt8]:
        return self._address

    fn signature(self) -> FunctionSignature:
        return self._signature

    fn call(self) -> CValue:
        """Call with no arguments."""
        # This would use the runtime FFI bridge
        # For now, return a placeholder
        return CValue(self._signature.return_type)

    fn call(self, arg0: CValue) -> CValue:
        """Call with one argument."""
        return CValue(self._signature.return_type)

    fn call(self, arg0: CValue, arg1: CValue) -> CValue:
        """Call with two arguments."""
        return CValue(self._signature.return_type)

    fn call(self, arg0: CValue, arg1: CValue, arg2: CValue) -> CValue:
        """Call with three arguments."""
        return CValue(self._signature.return_type)

    fn call_with_args(self, args: List[CValue]) -> CValue:
        """Call with arbitrary number of arguments."""
        # Validate argument count
        if not self._signature.is_variadic:
            if len(args) != self._signature.arg_count():
                # Argument count mismatch
                return CValue(CType.void())

        return CValue(self._signature.return_type)


# =============================================================================
# Dynamic Library
# =============================================================================

struct DynamicLibrary:
    """Wrapper for dynamically loaded libraries."""

    var _handle: UnsafePointer[UInt8]
    var _path: String
    var _is_open: Bool

    fn __init__(inout self):
        self._handle = UnsafePointer[UInt8]()
        self._path = ""
        self._is_open = False

    fn __init__(inout self, path: String) raises:
        """Open a dynamic library."""
        self._path = path
        self._handle = UnsafePointer[UInt8]()
        self._is_open = False
        self.open(path)

    fn __del__(owned self):
        if self._is_open:
            self.close()

    fn open(inout self, path: String) raises:
        """Open a dynamic library by path."""
        # This would call the runtime FFI to load the library
        # mojo_ffi_load_library(path)
        self._path = path
        # Placeholder - actual implementation uses runtime
        self._is_open = True

    fn close(inout self):
        """Close the library."""
        if self._is_open:
            # mojo_ffi_close_library(self._handle)
            self._is_open = False

    fn is_open(self) -> Bool:
        return self._is_open

    fn path(self) -> String:
        return self._path

    fn get_symbol(self, name: String) -> UnsafePointer[UInt8]:
        """Get a symbol (function or variable) from the library."""
        if not self._is_open:
            return UnsafePointer[UInt8]()

        # mojo_ffi_get_symbol(self._handle, name)
        return UnsafePointer[UInt8]()

    fn get_function(self, name: String, signature: FunctionSignature) -> ExternalFunction:
        """Get a function from the library."""
        var address = self.get_symbol(name)
        return ExternalFunction(name, address, signature)

    @staticmethod
    fn load(name: String) raises -> DynamicLibrary:
        """Load a library with platform-specific naming."""
        # Try different extensions based on platform
        var lib = DynamicLibrary()

        # Try lib{name}.dylib (macOS)
        try:
            lib.open("lib" + name + ".dylib")
            return lib
        except:
            pass

        # Try lib{name}.so (Linux)
        try:
            lib.open("lib" + name + ".so")
            return lib
        except:
            pass

        # Try {name}.dll (Windows)
        try:
            lib.open(name + ".dll")
            return lib
        except:
            pass

        # Try exact name
        lib.open(name)
        return lib


# =============================================================================
# C Struct Definition
# =============================================================================

struct CStructField:
    """A field in a C struct."""

    var name: String
    var type: CType
    var offset: Int

    fn __init__(inout self, name: String, type: CType, offset: Int):
        self.name = name
        self.type = type
        self.offset = offset


struct CStructDef:
    """Definition of a C struct for marshalling."""

    var name: String
    var fields: List[CStructField]
    var size: Int
    var alignment: Int

    fn __init__(inout self, name: String):
        self.name = name
        self.fields = List[CStructField]()
        self.size = 0
        self.alignment = 1

    fn add_field(inout self, name: String, type: CType):
        """Add a field with automatic offset calculation."""
        # Align offset to field's alignment
        var offset = self.size
        if offset % type.alignment != 0:
            offset += type.alignment - (offset % type.alignment)

        self.fields.append(CStructField(name, type, offset))
        self.size = offset + type.size

        # Update struct alignment
        if type.alignment > self.alignment:
            self.alignment = type.alignment

    fn finalize(inout self):
        """Finalize struct with padding."""
        # Add trailing padding for array alignment
        if self.size % self.alignment != 0:
            self.size += self.alignment - (self.size % self.alignment)

    fn field_count(self) -> Int:
        return len(self.fields)

    fn get_field(self, name: String) -> CStructField:
        """Get field by name."""
        for i in range(len(self.fields)):
            if self.fields[i].name == name:
                return self.fields[i]
        # Return dummy field if not found
        return CStructField("", CType.void(), 0)

    fn get_field_offset(self, name: String) -> Int:
        """Get field offset by name."""
        return self.get_field(name).offset


# =============================================================================
# C Struct Instance
# =============================================================================

struct CStruct:
    """Instance of a C struct."""

    var _def: CStructDef
    var _data: UnsafePointer[UInt8]
    var _owns_data: Bool

    fn __init__(inout self, definition: CStructDef):
        self._def = definition
        self._data = UnsafePointer[UInt8].alloc(definition.size)
        self._owns_data = True
        # Zero-initialize
        for i in range(definition.size):
            self._data.store(i, 0)

    fn __init__(inout self, definition: CStructDef, data: UnsafePointer[UInt8]):
        """Wrap existing memory as a struct."""
        self._def = definition
        self._data = data
        self._owns_data = False

    fn __del__(owned self):
        if self._owns_data:
            self._data.free()

    fn data_ptr(self) -> UnsafePointer[UInt8]:
        return self._data

    fn size(self) -> Int:
        return self._def.size

    fn set_int(inout self, field_name: String, value: Int):
        """Set an integer field."""
        var field = self._def.get_field(field_name)
        var ptr = self._data.offset(field.offset)

        if field.type.size == 8:
            ptr.bitcast[Int64]().store(value)
        elif field.type.size == 4:
            ptr.bitcast[Int32]().store(Int32(value))
        elif field.type.size == 2:
            ptr.bitcast[Int16]().store(Int16(value))
        else:
            ptr.bitcast[Int8]().store(Int8(value))

    fn get_int(self, field_name: String) -> Int:
        """Get an integer field."""
        var field = self._def.get_field(field_name)
        var ptr = self._data.offset(field.offset)

        if field.type.size == 8:
            return int(ptr.bitcast[Int64]().load())
        elif field.type.size == 4:
            return int(ptr.bitcast[Int32]().load())
        elif field.type.size == 2:
            return int(ptr.bitcast[Int16]().load())
        else:
            return int(ptr.bitcast[Int8]().load())

    fn set_float(inout self, field_name: String, value: Float64):
        """Set a floating-point field."""
        var field = self._def.get_field(field_name)
        var ptr = self._data.offset(field.offset)

        if field.type.kind == CType.DOUBLE:
            ptr.bitcast[Float64]().store(value)
        else:
            ptr.bitcast[Float32]().store(Float32(value))

    fn get_float(self, field_name: String) -> Float64:
        """Get a floating-point field."""
        var field = self._def.get_field(field_name)
        var ptr = self._data.offset(field.offset)

        if field.type.kind == CType.DOUBLE:
            return ptr.bitcast[Float64]().load()
        else:
            return Float64(ptr.bitcast[Float32]().load())

    fn set_pointer(inout self, field_name: String, value: UnsafePointer[UInt8]):
        """Set a pointer field."""
        var field = self._def.get_field(field_name)
        var ptr = self._data.offset(field.offset)
        ptr.bitcast[UnsafePointer[UInt8]]().store(value)

    fn get_pointer(self, field_name: String) -> UnsafePointer[UInt8]:
        """Get a pointer field."""
        var field = self._def.get_field(field_name)
        var ptr = self._data.offset(field.offset)
        return ptr.bitcast[UnsafePointer[UInt8]]().load()


# =============================================================================
# Callback Support
# =============================================================================

struct CallbackHandle:
    """Handle for a registered callback."""

    var _id: Int
    var _name: String
    var _signature: FunctionSignature

    fn __init__(inout self, id: Int, name: String, signature: FunctionSignature):
        self._id = id
        self._name = name
        self._signature = signature

    fn id(self) -> Int:
        return self._id

    fn name(self) -> String:
        return self._name


struct CallbackRegistry:
    """Registry for Mojo callbacks that can be called from C."""

    var _next_id: Int
    var _callbacks: List[CallbackHandle]

    fn __init__(inout self):
        self._next_id = 0
        self._callbacks = List[CallbackHandle]()

    fn register(inout self, name: String, signature: FunctionSignature) -> CallbackHandle:
        """Register a callback."""
        var handle = CallbackHandle(self._next_id, name, signature)
        self._callbacks.append(handle)
        self._next_id += 1
        return handle

    fn unregister(inout self, handle: CallbackHandle):
        """Unregister a callback."""
        for i in range(len(self._callbacks)):
            if self._callbacks[i]._id == handle._id:
                _ = self._callbacks.pop(i)
                return

    fn count(self) -> Int:
        return len(self._callbacks)


# =============================================================================
# Utility Functions
# =============================================================================

fn c_sizeof[T: AnyType]() -> Int:
    """Get size of a type in C ABI."""
    # This would be implemented with compiler intrinsics
    return 0


fn c_alignof[T: AnyType]() -> Int:
    """Get alignment of a type in C ABI."""
    return 0


fn null_ptr() -> UnsafePointer[UInt8]:
    """Get a null pointer."""
    return UnsafePointer[UInt8]()


fn is_null(ptr: UnsafePointer[UInt8]) -> Bool:
    """Check if a pointer is null."""
    return int(ptr) == 0


# =============================================================================
# Common C Types (Platform-Specific)
# =============================================================================

struct Platform:
    """Platform information for FFI."""

    alias UNKNOWN = 0
    alias MACOS = 1
    alias LINUX = 2
    alias WINDOWS = 3

    @staticmethod
    fn current() -> Int:
        """Get current platform."""
        # This would use compiler intrinsics
        return Platform.MACOS  # Placeholder

    @staticmethod
    fn is_64bit() -> Bool:
        """Check if running on 64-bit platform."""
        return True

    @staticmethod
    fn pointer_size() -> Int:
        """Get pointer size in bytes."""
        return 8 if Platform.is_64bit() else 4

    @staticmethod
    fn long_size() -> Int:
        """Get C long size (platform-specific)."""
        if Platform.current() == Platform.WINDOWS:
            return 4  # Windows LLP64
        else:
            return 8  # Unix LP64


# =============================================================================
# Error Handling
# =============================================================================

struct FFIError:
    """FFI operation error."""

    alias NONE = 0
    alias LIBRARY_NOT_FOUND = 1
    alias SYMBOL_NOT_FOUND = 2
    alias INVALID_SIGNATURE = 3
    alias TYPE_MISMATCH = 4
    alias NULL_POINTER = 5

    var code: Int
    var message: String

    fn __init__(inout self, code: Int, message: String):
        self.code = code
        self.message = message

    fn __init__(inout self):
        self.code = FFIError.NONE
        self.message = ""

    fn is_error(self) -> Bool:
        return self.code != FFIError.NONE

    fn __str__(self) -> String:
        return "FFIError(" + str(self.code) + "): " + self.message


# Global error state
var _last_ffi_error = FFIError()

fn get_last_error() -> FFIError:
    """Get the last FFI error."""
    return _last_ffi_error

fn clear_error():
    """Clear the last FFI error."""
    _last_ffi_error = FFIError()

fn set_error(code: Int, message: String):
    """Set an FFI error."""
    _last_ffi_error = FFIError(code, message)


# =============================================================================
# Tests
# =============================================================================

fn test_ctype():
    """Test CType functionality."""
    var int32 = CType.int32()
    assert_true(int32.size == 4, "int32 size should be 4")
    assert_true(int32.alignment == 4, "int32 alignment should be 4")
    assert_true(int32.is_integer(), "int32 should be integer")

    var float64 = CType.float64()
    assert_true(float64.size == 8, "float64 size should be 8")
    assert_true(float64.is_floating(), "float64 should be floating")

    var ptr = CType.pointer()
    assert_true(ptr.is_pointer(), "pointer should be pointer type")

    print("test_ctype: PASSED")


fn test_cvalue():
    """Test CValue functionality."""
    var int_val = CValue(42)
    assert_true(int_val.as_int() == 42, "CValue int should be 42")

    var float_val = CValue(Float64(3.14))
    var diff = float_val.as_float() - 3.14
    assert_true(diff < 0.001 and diff > -0.001, "CValue float should be ~3.14")

    var bool_val = CValue(True)
    assert_true(bool_val.as_bool() == True, "CValue bool should be True")

    print("test_cvalue: PASSED")


fn test_cstring():
    """Test CString functionality."""
    var cs = CString("Hello, FFI!")
    assert_true(len(cs) == 11, "CString length should be 11")

    var back = cs.to_string()
    assert_true(back == "Hello, FFI!", "Round-trip should preserve string")

    print("test_cstring: PASSED")


fn test_struct_def():
    """Test CStruct definition."""
    var def = CStructDef("Point")
    def.add_field("x", CType.float64())
    def.add_field("y", CType.float64())
    def.finalize()

    assert_true(def.size == 16, "Point struct should be 16 bytes")
    assert_true(def.field_count() == 2, "Point should have 2 fields")
    assert_true(def.get_field_offset("x") == 0, "x offset should be 0")
    assert_true(def.get_field_offset("y") == 8, "y offset should be 8")

    print("test_struct_def: PASSED")


fn test_struct_instance():
    """Test CStruct instance."""
    var def = CStructDef("Point")
    def.add_field("x", CType.float64())
    def.add_field("y", CType.float64())
    def.finalize()

    var point = CStruct(def)
    point.set_float("x", 1.5)
    point.set_float("y", 2.5)

    var x_diff = point.get_float("x") - 1.5
    var y_diff = point.get_float("y") - 2.5
    assert_true(x_diff < 0.001 and x_diff > -0.001, "x should be ~1.5")
    assert_true(y_diff < 0.001 and y_diff > -0.001, "y should be ~2.5")

    print("test_struct_instance: PASSED")


fn test_function_signature():
    """Test FunctionSignature."""
    var sig = FunctionSignature(CType.int32())
    sig.add_arg(CType.pointer())
    sig.add_arg(CType.c_size_t())

    assert_true(sig.arg_count() == 2, "Signature should have 2 args")

    print("test_function_signature: PASSED")


fn test_callback_registry():
    """Test CallbackRegistry."""
    var registry = CallbackRegistry()
    var sig = FunctionSignature(CType.void())
    sig.add_arg(CType.int32())

    var handle = registry.register("my_callback", sig)
    assert_true(registry.count() == 1, "Registry should have 1 callback")
    assert_true(handle.name() == "my_callback", "Handle name should match")

    registry.unregister(handle)
    assert_true(registry.count() == 0, "Registry should be empty")

    print("test_callback_registry: PASSED")


fn assert_true(condition: Bool, message: String):
    """Simple assertion helper."""
    if not condition:
        print("ASSERTION FAILED: " + message)


fn run_all_tests():
    """Run all FFI tests."""
    print("=== FFI Module Tests ===")
    test_ctype()
    test_cvalue()
    test_cstring()
    test_struct_def()
    test_struct_instance()
    test_function_signature()
    test_callback_registry()
    print("=== All Tests Passed ===")
