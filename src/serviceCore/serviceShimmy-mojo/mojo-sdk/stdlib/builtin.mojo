# Mojo Standard Library - Builtin Types
# Core foundational types and operations

# ============================================================================
# Core Type Definitions
# ============================================================================

# Note: In a real Mojo implementation, these would be implemented at the
# compiler level. This is a conceptual representation of the builtin module.

struct Int:
    """
    Signed integer type (platform-dependent size).
    Typically 64-bit on 64-bit systems, 32-bit on 32-bit systems.
    """
    var value: __mlir_type.index
    
    fn __init__(inout self):
        """Initialize to 0."""
        self.value = 0
    
    fn __init__(inout self, value: __mlir_type.index):
        """Initialize from a value."""
        self.value = value
    
    fn __init__(inout self, value: Float64):
        """Initialize from float (truncates)."""
        self.value = __mlir_type.index(value)
    
    fn __init__(inout self, value: Bool):
        """Initialize from bool (0 or 1)."""
        self.value = 1 if value else 0
    
    fn __init__(inout self, value: String):
        """Parse from string."""
        # Simple integer parsing
        self.value = parse_int(value)
    
    # Arithmetic operators
    fn __add__(self, other: Int) -> Int:
        """Addition."""
        return Int(self.value + other.value)
    
    fn __sub__(self, other: Int) -> Int:
        """Subtraction."""
        return Int(self.value - other.value)
    
    fn __mul__(self, other: Int) -> Int:
        """Multiplication."""
        return Int(self.value * other.value)
    
    fn __truediv__(self, other: Int) -> Float64:
        """True division (returns float)."""
        return Float64(self.value) / Float64(other.value)
    
    fn __floordiv__(self, other: Int) -> Int:
        """Floor division."""
        return Int(self.value // other.value)
    
    fn __mod__(self, other: Int) -> Int:
        """Modulo."""
        return Int(self.value % other.value)
    
    fn __pow__(self, other: Int) -> Int:
        """Power."""
        var result = 1
        for _ in range(other.value):
            result *= self.value
        return Int(result)
    
    # Unary operators
    fn __neg__(self) -> Int:
        """Negation."""
        return Int(-self.value)
    
    fn __pos__(self) -> Int:
        """Positive (identity)."""
        return self
    
    fn __abs__(self) -> Int:
        """Absolute value."""
        return Int(abs(self.value))
    
    # Comparison operators
    fn __eq__(self, other: Int) -> Bool:
        """Equality."""
        return Bool(self.value == other.value)
    
    fn __ne__(self, other: Int) -> Bool:
        """Inequality."""
        return Bool(self.value != other.value)
    
    fn __lt__(self, other: Int) -> Bool:
        """Less than."""
        return Bool(self.value < other.value)
    
    fn __le__(self, other: Int) -> Bool:
        """Less than or equal."""
        return Bool(self.value <= other.value)
    
    fn __gt__(self, other: Int) -> Bool:
        """Greater than."""
        return Bool(self.value > other.value)
    
    fn __ge__(self, other: Int) -> Bool:
        """Greater than or equal."""
        return Bool(self.value >= other.value)
    
    # Bitwise operators
    fn __and__(self, other: Int) -> Int:
        """Bitwise AND."""
        return Int(self.value & other.value)
    
    fn __or__(self, other: Int) -> Int:
        """Bitwise OR."""
        return Int(self.value | other.value)
    
    fn __xor__(self, other: Int) -> Int:
        """Bitwise XOR."""
        return Int(self.value ^ other.value)
    
    fn __invert__(self) -> Int:
        """Bitwise NOT."""
        return Int(~self.value)
    
    fn __lshift__(self, other: Int) -> Int:
        """Left shift."""
        return Int(self.value << other.value)
    
    fn __rshift__(self, other: Int) -> Int:
        """Right shift."""
        return Int(self.value >> other.value)
    
    # Type conversions
    fn to_float(self) -> Float64:
        """Convert to float."""
        return Float64(self.value)
    
    fn to_bool(self) -> Bool:
        """Convert to bool (False if 0, True otherwise)."""
        return Bool(self.value != 0)
    
    fn to_string(self) -> String:
        """Convert to string."""
        return str(self.value)
    
    fn __str__(self) -> String:
        """String representation."""
        return self.to_string()
    
    fn __repr__(self) -> String:
        """Detailed representation."""
        return "Int(" + self.to_string() + ")"


# ============================================================================
# Float64 - 64-bit floating point
# ============================================================================

struct Float64:
    """
    64-bit floating point number (IEEE 754 double precision).
    """
    var value: __mlir_type.f64
    
    fn __init__(inout self):
        """Initialize to 0.0."""
        self.value = 0.0
    
    fn __init__(inout self, value: __mlir_type.f64):
        """Initialize from a value."""
        self.value = value
    
    fn __init__(inout self, value: Int):
        """Initialize from int."""
        self.value = __mlir_type.f64(value.value)
    
    fn __init__(inout self, value: Bool):
        """Initialize from bool."""
        self.value = 1.0 if value else 0.0
    
    fn __init__(inout self, value: String):
        """Parse from string."""
        self.value = parse_float(value)
    
    # Arithmetic operators
    fn __add__(self, other: Float64) -> Float64:
        """Addition."""
        return Float64(self.value + other.value)
    
    fn __sub__(self, other: Float64) -> Float64:
        """Subtraction."""
        return Float64(self.value - other.value)
    
    fn __mul__(self, other: Float64) -> Float64:
        """Multiplication."""
        return Float64(self.value * other.value)
    
    fn __truediv__(self, other: Float64) -> Float64:
        """Division."""
        return Float64(self.value / other.value)
    
    fn __mod__(self, other: Float64) -> Float64:
        """Modulo."""
        return Float64(self.value % other.value)
    
    fn __pow__(self, other: Float64) -> Float64:
        """Power."""
        return Float64(pow(self.value, other.value))
    
    # Unary operators
    fn __neg__(self) -> Float64:
        """Negation."""
        return Float64(-self.value)
    
    fn __pos__(self) -> Float64:
        """Positive (identity)."""
        return self
    
    fn __abs__(self) -> Float64:
        """Absolute value."""
        return Float64(abs(self.value))
    
    # Comparison operators
    fn __eq__(self, other: Float64) -> Bool:
        """Equality."""
        return Bool(self.value == other.value)
    
    fn __ne__(self, other: Float64) -> Bool:
        """Inequality."""
        return Bool(self.value != other.value)
    
    fn __lt__(self, other: Float64) -> Bool:
        """Less than."""
        return Bool(self.value < other.value)
    
    fn __le__(self, other: Float64) -> Bool:
        """Less than or equal."""
        return Bool(self.value <= other.value)
    
    fn __gt__(self, other: Float64) -> Bool:
        """Greater than."""
        return Bool(self.value > other.value)
    
    fn __ge__(self, other: Float64) -> Bool:
        """Greater than or equal."""
        return Bool(self.value >= other.value)
    
    # Type conversions
    fn to_int(self) -> Int:
        """Convert to int (truncates)."""
        return Int(__mlir_type.index(self.value))
    
    fn to_bool(self) -> Bool:
        """Convert to bool (False if 0.0, True otherwise)."""
        return Bool(self.value != 0.0)
    
    fn to_string(self) -> String:
        """Convert to string."""
        return str(self.value)
    
    fn __str__(self) -> String:
        """String representation."""
        return self.to_string()
    
    fn __repr__(self) -> String:
        """Detailed representation."""
        return "Float64(" + self.to_string() + ")"
    
    # Math methods
    fn floor(self) -> Float64:
        """Round down to nearest integer."""
        return Float64(__mlir_op.`math.floor`(self.value))
    
    fn ceil(self) -> Float64:
        """Round up to nearest integer."""
        return Float64(__mlir_op.`math.ceil`(self.value))
    
    fn round(self) -> Float64:
        """Round to nearest integer."""
        return Float64(__mlir_op.`math.round`(self.value))
    
    fn is_nan(self) -> Bool:
        """Check if value is NaN."""
        return Bool(self.value != self.value)
    
    fn is_inf(self) -> Bool:
        """Check if value is infinite."""
        return Bool(abs(self.value) == __mlir_attr.`#float.infinity : f64`)


# ============================================================================
# Bool - Boolean type
# ============================================================================

struct Bool:
    """
    Boolean type (True or False).
    """
    var value: __mlir_type.i1
    
    fn __init__(inout self):
        """Initialize to False."""
        self.value = False
    
    fn __init__(inout self, value: __mlir_type.i1):
        """Initialize from a value."""
        self.value = value
    
    fn __init__(inout self, value: Int):
        """Initialize from int (0 is False, non-zero is True)."""
        self.value = value.value != 0
    
    fn __init__(inout self, value: Float64):
        """Initialize from float (0.0 is False, non-zero is True)."""
        self.value = value.value != 0.0
    
    fn __init__(inout self, value: String):
        """Parse from string."""
        self.value = value == "True" or value == "true" or value == "1"
    
    # Logical operators
    fn __and__(self, other: Bool) -> Bool:
        """Logical AND."""
        return Bool(self.value and other.value)
    
    fn __or__(self, other: Bool) -> Bool:
        """Logical OR."""
        return Bool(self.value or other.value)
    
    fn __xor__(self, other: Bool) -> Bool:
        """Logical XOR."""
        return Bool(self.value != other.value)
    
    fn __invert__(self) -> Bool:
        """Logical NOT."""
        return Bool(not self.value)
    
    # Comparison operators
    fn __eq__(self, other: Bool) -> Bool:
        """Equality."""
        return Bool(self.value == other.value)
    
    fn __ne__(self, other: Bool) -> Bool:
        """Inequality."""
        return Bool(self.value != other.value)
    
    # Type conversions
    fn to_int(self) -> Int:
        """Convert to int (0 or 1)."""
        return Int(1 if self.value else 0)
    
    fn to_float(self) -> Float64:
        """Convert to float (0.0 or 1.0)."""
        return Float64(1.0 if self.value else 0.0)
    
    fn to_string(self) -> String:
        """Convert to string."""
        return "True" if self.value else "False"
    
    fn __str__(self) -> String:
        """String representation."""
        return self.to_string()
    
    fn __repr__(self) -> String:
        """Detailed representation."""
        return "Bool(" + self.to_string() + ")"


# ============================================================================
# String - String type
# ============================================================================

struct String:
    """
    Immutable string type (UTF-8 encoded).
    """
    var data: Pointer[UInt8]
    var length: Int
    
    fn __init__(inout self):
        """Initialize empty string."""
        self.length = 0
        self.data = Pointer[UInt8].alloc(1)
        self.data[0] = 0  # Null terminator
    
    fn __init__(inout self, value: StringLiteral):
        """Initialize from string literal."""
        self.length = len(value)
        self.data = Pointer[UInt8].alloc(self.length + 1)
        # Copy data
        for i in range(self.length):
            self.data[i] = value[i]
        self.data[self.length] = 0  # Null terminator
    
    fn __init__(inout self, other: String):
        """Copy constructor."""
        self.length = other.length
        self.data = Pointer[UInt8].alloc(self.length + 1)
        for i in range(self.length):
            self.data[i] = other.data[i]
        self.data[self.length] = 0
    
    fn __del__(owned self):
        """Destructor."""
        if self.data:
            self.data.free()
    
    # String operations
    fn len(self) -> Int:
        """Get string length."""
        return self.length
    
    fn __getitem__(self, index: Int) -> String:
        """Get character at index."""
        if index < 0 or index >= self.length:
            return String()  # Return empty string
        
        var result = String()
        result.length = 1
        result.data = Pointer[UInt8].alloc(2)
        result.data[0] = self.data[index]
        result.data[1] = 0
        return result
    
    fn __add__(self, other: String) -> String:
        """Concatenation."""
        var result = String()
        result.length = self.length + other.length
        result.data = Pointer[UInt8].alloc(result.length + 1)
        
        # Copy self
        for i in range(self.length):
            result.data[i] = self.data[i]
        
        # Copy other
        for i in range(other.length):
            result.data[self.length + i] = other.data[i]
        
        result.data[result.length] = 0
        return result
    
    fn __eq__(self, other: String) -> Bool:
        """Equality."""
        if self.length != other.length:
            return Bool(False)
        
        for i in range(self.length):
            if self.data[i] != other.data[i]:
                return Bool(False)
        
        return Bool(True)
    
    fn __ne__(self, other: Bool) -> Bool:
        """Inequality."""
        return not self.__eq__(other)
    
    fn __str__(self) -> String:
        """String representation (returns self)."""
        return self
    
    fn __repr__(self) -> String:
        """Detailed representation."""
        return "String(\"" + self + "\")"
    
    # String methods
    fn upper(self) -> String:
        """Convert to uppercase."""
        var result = String(self)
        for i in range(result.length):
            let c = result.data[i]
            if c >= ord('a') and c <= ord('z'):
                result.data[i] = c - 32
        return result
    
    fn lower(self) -> String:
        """Convert to lowercase."""
        var result = String(self)
        for i in range(result.length):
            let c = result.data[i]
            if c >= ord('A') and c <= ord('Z'):
                result.data[i] = c + 32
        return result
    
    fn starts_with(self, prefix: String) -> Bool:
        """Check if string starts with prefix."""
        if prefix.length > self.length:
            return Bool(False)
        
        for i in range(prefix.length):
            if self.data[i] != prefix.data[i]:
                return Bool(False)
        
        return Bool(True)
    
    fn ends_with(self, suffix: String) -> Bool:
        """Check if string ends with suffix."""
        if suffix.length > self.length:
            return Bool(False)
        
        let offset = self.length - suffix.length
        for i in range(suffix.length):
            if self.data[offset + i] != suffix.data[i]:
                return Bool(False)
        
        return Bool(True)


# ============================================================================
# Helper Functions
# ============================================================================

fn parse_int(s: String) -> Int:
    """Parse integer from string."""
    var result = 0
    var negative = False
    var start = 0
    
    if s.length > 0 and s[0] == "-":
        negative = True
        start = 1
    
    for i in range(start, s.length):
        let digit = int(s.data[i]) - ord('0')
        if digit >= 0 and digit <= 9:
            result = result * 10 + digit
    
    return -result if negative else result

fn parse_float(s: String) -> Float64:
    """Parse float from string."""
    # Simplified implementation
    var result = 0.0
    var negative = False
    var decimal_places = 0
    var after_decimal = False
    var start = 0
    
    if s.length > 0 and s[0] == "-":
        negative = True
        start = 1
    
    for i in range(start, s.length):
        let c = s.data[i]
        if c == ord('.'):
            after_decimal = True
        else:
            let digit = int(c) - ord('0')
            if digit >= 0 and digit <= 9:
                if after_decimal:
                    decimal_places += 1
                    result = result + Float64(digit) / pow(10.0, Float64(decimal_places))
                else:
                    result = result * 10.0 + Float64(digit)
    
    return -result if negative else result

fn ord(c: String) -> Int:
    """Get ASCII/Unicode code point of character."""
    if c.length == 0:
        return Int(0)
    return Int(c.data[0])

fn chr(code: Int) -> String:
    """Convert ASCII/Unicode code point to character."""
    var result = String()
    result.length = 1
    result.data = Pointer[UInt8].alloc(2)
    result.data[0] = UInt8(code.value)
    result.data[1] = 0
    return result

fn str[T](value: T) -> String:
    """Convert any value to string."""
    return value.__str__()

fn int(value: String) -> Int:
    """Convert string to int."""
    return Int(value)

fn float(value: String) -> Float64:
    """Convert string to float."""
    return Float64(value)

fn bool(value: String) -> Bool:
    """Convert string to bool."""
    return Bool(value)
