"""
Mojo Service Framework - JSON Utilities
Days 105-109: JSON Parsing and Building for HTTP APIs

Lightweight JSON utilities for API request/response handling.
Focused on common API patterns without full JSON spec compliance.
"""

from collections import Dict, List

# ============================================================================
# JSON Value Types
# ============================================================================

@value
struct JsonType:
    """JSON value type enumeration."""
    var _value: Int

    alias NULL = JsonType(0)
    alias BOOL = JsonType(1)
    alias NUMBER = JsonType(2)
    alias STRING = JsonType(3)
    alias ARRAY = JsonType(4)
    alias OBJECT = JsonType(5)

    fn __eq__(self, other: JsonType) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: JsonType) -> Bool:
        return self._value != other._value

# ============================================================================
# JSON Value
# ============================================================================

@value
struct JsonValue:
    """
    Represents a JSON value.
    Lightweight implementation for API use cases.
    """
    var type: JsonType
    var string_value: String
    var number_value: Float64
    var bool_value: Bool
    var array_values: List[JsonValue]
    var object_keys: List[String]
    var object_values: List[JsonValue]

    fn __init__(inout self):
        """Create null value."""
        self.type = JsonType.NULL
        self.string_value = ""
        self.number_value = 0.0
        self.bool_value = False
        self.array_values = List[JsonValue]()
        self.object_keys = List[String]()
        self.object_values = List[JsonValue]()

    @staticmethod
    fn null() -> JsonValue:
        """Create null value."""
        return JsonValue()

    @staticmethod
    fn bool_val(value: Bool) -> JsonValue:
        """Create boolean value."""
        var v = JsonValue()
        v.type = JsonType.BOOL
        v.bool_value = value
        return v

    @staticmethod
    fn number(value: Float64) -> JsonValue:
        """Create number value."""
        var v = JsonValue()
        v.type = JsonType.NUMBER
        v.number_value = value
        return v

    @staticmethod
    fn number(value: Int) -> JsonValue:
        """Create number value from int."""
        var v = JsonValue()
        v.type = JsonType.NUMBER
        v.number_value = Float64(value)
        return v

    @staticmethod
    fn string(value: String) -> JsonValue:
        """Create string value."""
        var v = JsonValue()
        v.type = JsonType.STRING
        v.string_value = value
        return v

    @staticmethod
    fn array() -> JsonValue:
        """Create empty array."""
        var v = JsonValue()
        v.type = JsonType.ARRAY
        return v

    @staticmethod
    fn object() -> JsonValue:
        """Create empty object."""
        var v = JsonValue()
        v.type = JsonType.OBJECT
        return v

    # Type checking

    fn is_null(self) -> Bool:
        return self.type == JsonType.NULL

    fn is_bool(self) -> Bool:
        return self.type == JsonType.BOOL

    fn is_number(self) -> Bool:
        return self.type == JsonType.NUMBER

    fn is_string(self) -> Bool:
        return self.type == JsonType.STRING

    fn is_array(self) -> Bool:
        return self.type == JsonType.ARRAY

    fn is_object(self) -> Bool:
        return self.type == JsonType.OBJECT

    # Value access

    fn as_bool(self) -> Bool:
        """Get boolean value."""
        return self.bool_value

    fn as_number(self) -> Float64:
        """Get number value."""
        return self.number_value

    fn as_int(self) -> Int:
        """Get number as integer."""
        return Int(self.number_value)

    fn as_string(self) -> String:
        """Get string value."""
        return self.string_value

    # Array operations

    fn len(self) -> Int:
        """Get array length or object key count."""
        if self.type == JsonType.ARRAY:
            return len(self.array_values)
        elif self.type == JsonType.OBJECT:
            return len(self.object_keys)
        return 0

    fn get(self, index: Int) -> JsonValue:
        """Get array element by index."""
        if self.type == JsonType.ARRAY and index < len(self.array_values):
            return self.array_values[index]
        return JsonValue.null()

    fn push(inout self, value: JsonValue):
        """Add element to array."""
        if self.type == JsonType.ARRAY:
            self.array_values.append(value)

    # Object operations

    fn get(self, key: String) -> JsonValue:
        """Get object value by key."""
        if self.type == JsonType.OBJECT:
            for i in range(len(self.object_keys)):
                if self.object_keys[i] == key:
                    return self.object_values[i]
        return JsonValue.null()

    fn set(inout self, key: String, value: JsonValue):
        """Set object value by key."""
        if self.type == JsonType.OBJECT:
            # Check if key exists
            for i in range(len(self.object_keys)):
                if self.object_keys[i] == key:
                    self.object_values[i] = value
                    return
            # Add new key
            self.object_keys.append(key)
            self.object_values.append(value)

    fn has(self, key: String) -> Bool:
        """Check if object has key."""
        if self.type == JsonType.OBJECT:
            for i in range(len(self.object_keys)):
                if self.object_keys[i] == key:
                    return True
        return False

    # Serialization

    fn to_string(self) -> String:
        """Convert to JSON string."""
        if self.type == JsonType.NULL:
            return "null"
        elif self.type == JsonType.BOOL:
            return "true" if self.bool_value else "false"
        elif self.type == JsonType.NUMBER:
            # Simple number formatting
            var n = self.number_value
            if n == Float64(Int(n)):
                return String(Int(n))
            return String(n)
        elif self.type == JsonType.STRING:
            return '"' + _escape_string(self.string_value) + '"'
        elif self.type == JsonType.ARRAY:
            var result = String("[")
            for i in range(len(self.array_values)):
                if i > 0:
                    result += ","
                result += self.array_values[i].to_string()
            result += "]"
            return result
        elif self.type == JsonType.OBJECT:
            var result = String("{")
            for i in range(len(self.object_keys)):
                if i > 0:
                    result += ","
                result += '"' + _escape_string(self.object_keys[i]) + '":'
                result += self.object_values[i].to_string()
            result += "}"
            return result
        return "null"

    fn to_pretty_string(self, indent: Int = 0) -> String:
        """Convert to pretty-printed JSON string."""
        var spaces = "  " * indent

        if self.type == JsonType.NULL:
            return "null"
        elif self.type == JsonType.BOOL:
            return "true" if self.bool_value else "false"
        elif self.type == JsonType.NUMBER:
            var n = self.number_value
            if n == Float64(Int(n)):
                return String(Int(n))
            return String(n)
        elif self.type == JsonType.STRING:
            return '"' + _escape_string(self.string_value) + '"'
        elif self.type == JsonType.ARRAY:
            if len(self.array_values) == 0:
                return "[]"
            var result = String("[\n")
            for i in range(len(self.array_values)):
                if i > 0:
                    result += ",\n"
                result += "  " * (indent + 1)
                result += self.array_values[i].to_pretty_string(indent + 1)
            result += "\n" + spaces + "]"
            return result
        elif self.type == JsonType.OBJECT:
            if len(self.object_keys) == 0:
                return "{}"
            var result = String("{\n")
            for i in range(len(self.object_keys)):
                if i > 0:
                    result += ",\n"
                result += "  " * (indent + 1)
                result += '"' + _escape_string(self.object_keys[i]) + '": '
                result += self.object_values[i].to_pretty_string(indent + 1)
            result += "\n" + spaces + "}"
            return result
        return "null"

# ============================================================================
# JSON Parser
# ============================================================================

struct JsonParser:
    """
    Simple JSON parser for API payloads.
    """
    var source: String
    var pos: Int
    var error: String

    fn __init__(inout self, source: String):
        self.source = source
        self.pos = 0
        self.error = ""

    fn parse(inout self) -> JsonValue:
        """Parse JSON string into value."""
        self._skip_whitespace()
        if self.pos >= len(self.source):
            return JsonValue.null()
        return self._parse_value()

    fn has_error(self) -> Bool:
        """Check if parsing had errors."""
        return len(self.error) > 0

    fn _parse_value(inout self) -> JsonValue:
        """Parse any JSON value."""
        self._skip_whitespace()

        if self.pos >= len(self.source):
            self.error = "Unexpected end of input"
            return JsonValue.null()

        var c = self.source[self.pos]

        if c == 'n':
            return self._parse_null()
        elif c == 't' or c == 'f':
            return self._parse_bool()
        elif c == '"':
            return self._parse_string()
        elif c == '[':
            return self._parse_array()
        elif c == '{':
            return self._parse_object()
        elif c == '-' or (c >= '0' and c <= '9'):
            return self._parse_number()
        else:
            self.error = "Unexpected character: " + String(c)
            return JsonValue.null()

    fn _parse_null(inout self) -> JsonValue:
        """Parse null literal."""
        if self._match("null"):
            return JsonValue.null()
        self.error = "Expected 'null'"
        return JsonValue.null()

    fn _parse_bool(inout self) -> JsonValue:
        """Parse boolean literal."""
        if self._match("true"):
            return JsonValue.bool_val(True)
        elif self._match("false"):
            return JsonValue.bool_val(False)
        self.error = "Expected boolean"
        return JsonValue.null()

    fn _parse_string(inout self) -> JsonValue:
        """Parse string value."""
        if self.source[self.pos] != '"':
            self.error = "Expected '\"'"
            return JsonValue.null()

        self.pos += 1
        var start = self.pos
        var result = String("")

        while self.pos < len(self.source) and self.source[self.pos] != '"':
            if self.source[self.pos] == '\\' and self.pos + 1 < len(self.source):
                # Handle escape
                self.pos += 1
                var esc = self.source[self.pos]
                if esc == 'n':
                    result += "\n"
                elif esc == 't':
                    result += "\t"
                elif esc == 'r':
                    result += "\r"
                elif esc == '\\':
                    result += "\\"
                elif esc == '"':
                    result += '"'
                else:
                    result += String(esc)
            else:
                result += String(self.source[self.pos])
            self.pos += 1

        if self.pos >= len(self.source):
            self.error = "Unterminated string"
            return JsonValue.null()

        self.pos += 1  # Skip closing quote
        return JsonValue.string(result)

    fn _parse_number(inout self) -> JsonValue:
        """Parse number value."""
        var start = self.pos
        var has_decimal = False

        # Optional minus
        if self.source[self.pos] == '-':
            self.pos += 1

        # Integer part
        while self.pos < len(self.source):
            var c = self.source[self.pos]
            if c >= '0' and c <= '9':
                self.pos += 1
            elif c == '.' and not has_decimal:
                has_decimal = True
                self.pos += 1
            elif c == 'e' or c == 'E':
                self.pos += 1
                if self.pos < len(self.source) and (self.source[self.pos] == '+' or self.source[self.pos] == '-'):
                    self.pos += 1
            else:
                break

        var num_str = self.source[start:self.pos]
        try:
            var value = Float64(atof(num_str))
            return JsonValue.number(value)
        except:
            self.error = "Invalid number: " + num_str
            return JsonValue.null()

    fn _parse_array(inout self) -> JsonValue:
        """Parse array value."""
        if self.source[self.pos] != '[':
            self.error = "Expected '['"
            return JsonValue.null()

        self.pos += 1
        var arr = JsonValue.array()

        self._skip_whitespace()
        if self.pos < len(self.source) and self.source[self.pos] == ']':
            self.pos += 1
            return arr

        while True:
            self._skip_whitespace()
            var value = self._parse_value()
            if self.has_error():
                return JsonValue.null()
            arr.push(value)

            self._skip_whitespace()
            if self.pos >= len(self.source):
                self.error = "Unterminated array"
                return JsonValue.null()

            if self.source[self.pos] == ']':
                self.pos += 1
                break
            elif self.source[self.pos] == ',':
                self.pos += 1
            else:
                self.error = "Expected ',' or ']'"
                return JsonValue.null()

        return arr

    fn _parse_object(inout self) -> JsonValue:
        """Parse object value."""
        if self.source[self.pos] != '{':
            self.error = "Expected '{'"
            return JsonValue.null()

        self.pos += 1
        var obj = JsonValue.object()

        self._skip_whitespace()
        if self.pos < len(self.source) and self.source[self.pos] == '}':
            self.pos += 1
            return obj

        while True:
            self._skip_whitespace()

            # Parse key
            var key_val = self._parse_string()
            if self.has_error():
                return JsonValue.null()
            var key = key_val.as_string()

            self._skip_whitespace()
            if self.pos >= len(self.source) or self.source[self.pos] != ':':
                self.error = "Expected ':'"
                return JsonValue.null()
            self.pos += 1

            # Parse value
            self._skip_whitespace()
            var value = self._parse_value()
            if self.has_error():
                return JsonValue.null()

            obj.set(key, value)

            self._skip_whitespace()
            if self.pos >= len(self.source):
                self.error = "Unterminated object"
                return JsonValue.null()

            if self.source[self.pos] == '}':
                self.pos += 1
                break
            elif self.source[self.pos] == ',':
                self.pos += 1
            else:
                self.error = "Expected ',' or '}'"
                return JsonValue.null()

        return obj

    fn _skip_whitespace(inout self):
        """Skip whitespace characters."""
        while self.pos < len(self.source):
            var c = self.source[self.pos]
            if c == ' ' or c == '\t' or c == '\n' or c == '\r':
                self.pos += 1
            else:
                break

    fn _match(inout self, expected: String) -> Bool:
        """Try to match expected string."""
        if self.pos + len(expected) > len(self.source):
            return False
        for i in range(len(expected)):
            if self.source[self.pos + i] != expected[i]:
                return False
        self.pos += len(expected)
        return True

# ============================================================================
# JSON Builder (Fluent API)
# ============================================================================

struct JsonBuilder:
    """
    Fluent API for building JSON objects and arrays.
    """
    var value: JsonValue

    fn __init__(inout self):
        self.value = JsonValue.object()

    @staticmethod
    fn object() -> JsonBuilder:
        """Start building an object."""
        var builder = JsonBuilder()
        builder.value = JsonValue.object()
        return builder

    @staticmethod
    fn array() -> JsonBuilder:
        """Start building an array."""
        var builder = JsonBuilder()
        builder.value = JsonValue.array()
        return builder

    fn set(inout self, key: String, value: String) -> JsonBuilder:
        """Add string field."""
        self.value.set(key, JsonValue.string(value))
        return self

    fn set(inout self, key: String, value: Int) -> JsonBuilder:
        """Add integer field."""
        self.value.set(key, JsonValue.number(value))
        return self

    fn set(inout self, key: String, value: Float64) -> JsonBuilder:
        """Add float field."""
        self.value.set(key, JsonValue.number(value))
        return self

    fn set(inout self, key: String, value: Bool) -> JsonBuilder:
        """Add boolean field."""
        self.value.set(key, JsonValue.bool_val(value))
        return self

    fn set(inout self, key: String, value: JsonValue) -> JsonBuilder:
        """Add JSON value field."""
        self.value.set(key, value)
        return self

    fn set_null(inout self, key: String) -> JsonBuilder:
        """Add null field."""
        self.value.set(key, JsonValue.null())
        return self

    fn push(inout self, value: String) -> JsonBuilder:
        """Add string to array."""
        self.value.push(JsonValue.string(value))
        return self

    fn push(inout self, value: Int) -> JsonBuilder:
        """Add integer to array."""
        self.value.push(JsonValue.number(value))
        return self

    fn push(inout self, value: Float64) -> JsonBuilder:
        """Add float to array."""
        self.value.push(JsonValue.number(value))
        return self

    fn push(inout self, value: Bool) -> JsonBuilder:
        """Add boolean to array."""
        self.value.push(JsonValue.bool_val(value))
        return self

    fn push(inout self, value: JsonValue) -> JsonBuilder:
        """Add JSON value to array."""
        self.value.push(value)
        return self

    fn build(self) -> JsonValue:
        """Get the built JSON value."""
        return self.value

    fn to_string(self) -> String:
        """Convert to JSON string."""
        return self.value.to_string()

    fn to_pretty_string(self) -> String:
        """Convert to pretty JSON string."""
        return self.value.to_pretty_string()

# ============================================================================
# Helper Functions
# ============================================================================

fn _escape_string(s: String) -> String:
    """Escape string for JSON."""
    var result = String("")
    for i in range(len(s)):
        var c = s[i]
        if c == '"':
            result += '\\"'
        elif c == '\\':
            result += "\\\\"
        elif c == '\n':
            result += "\\n"
        elif c == '\r':
            result += "\\r"
        elif c == '\t':
            result += "\\t"
        else:
            result += String(c)
    return result

fn parse_json(source: String) -> JsonValue:
    """Parse JSON string into value."""
    var parser = JsonParser(source)
    return parser.parse()

fn atof(s: String) -> Float64:
    """Convert string to float (simplified)."""
    # Simplified implementation
    var result: Float64 = 0.0
    var decimal_place: Float64 = 0.0
    var negative = False
    var i = 0

    # Handle sign
    if len(s) > 0 and s[0] == '-':
        negative = True
        i = 1
    elif len(s) > 0 and s[0] == '+':
        i = 1

    # Parse integer part
    while i < len(s) and s[i] >= '0' and s[i] <= '9':
        result = result * 10 + Float64(ord(s[i]) - ord('0'))
        i += 1

    # Parse decimal part
    if i < len(s) and s[i] == '.':
        i += 1
        decimal_place = 0.1
        while i < len(s) and s[i] >= '0' and s[i] <= '9':
            result += Float64(ord(s[i]) - ord('0')) * decimal_place
            decimal_place *= 0.1
            i += 1

    return -result if negative else result

# ============================================================================
# API Response Builders
# ============================================================================

fn json_ok(data: JsonValue) -> String:
    """Build success response."""
    return data.to_string()

fn json_error(message: String, code: String = "error") -> String:
    """Build error response."""
    var builder = JsonBuilder.object()
    _ = builder.set("error", code)
    _ = builder.set("message", message)
    return builder.to_string()

fn json_list(items: List[JsonValue], total: Int = -1) -> String:
    """Build list response with optional total count."""
    var builder = JsonBuilder.object()
    var arr = JsonValue.array()
    for i in range(len(items)):
        arr.push(items[i])
    _ = builder.set("data", arr)
    if total >= 0:
        _ = builder.set("total", total)
    return builder.to_string()

fn json_page(items: List[JsonValue], page: Int, per_page: Int, total: Int) -> String:
    """Build paginated list response."""
    var builder = JsonBuilder.object()
    var arr = JsonValue.array()
    for i in range(len(items)):
        arr.push(items[i])
    _ = builder.set("data", arr)
    _ = builder.set("page", page)
    _ = builder.set("per_page", per_page)
    _ = builder.set("total", total)
    var total_pages = (total + per_page - 1) // per_page
    _ = builder.set("total_pages", total_pages)
    return builder.to_string()
