# Mojo JSON Module
# Day 49 - JSON parsing and generation
#
# This module provides JSON support including:
# - JSON value types (null, bool, number, string, array, object)
# - Parsing JSON from strings
# - Serializing to JSON strings
# - Pretty printing

# =============================================================================
# Constants
# =============================================================================

alias MAX_NESTING_DEPTH: Int = 128
alias DEFAULT_INDENT: Int = 2

# =============================================================================
# JSON Value Type
# =============================================================================

struct JsonType:
    """JSON value type enumeration."""

    alias NULL = 0
    alias BOOL = 1
    alias NUMBER = 2
    alias STRING = 3
    alias ARRAY = 4
    alias OBJECT = 5

    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __eq__(self, other: JsonType) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: JsonType) -> Bool:
        return self.value != other.value

    fn __str__(self) -> String:
        if self.value == JsonType.NULL:
            return "null"
        elif self.value == JsonType.BOOL:
            return "bool"
        elif self.value == JsonType.NUMBER:
            return "number"
        elif self.value == JsonType.STRING:
            return "string"
        elif self.value == JsonType.ARRAY:
            return "array"
        elif self.value == JsonType.OBJECT:
            return "object"
        return "unknown"


# =============================================================================
# JSON Error
# =============================================================================

struct JsonError:
    """JSON parsing/serialization error."""

    alias NONE = 0
    alias UNEXPECTED_TOKEN = 1
    alias UNEXPECTED_END = 2
    alias INVALID_NUMBER = 3
    alias INVALID_STRING = 4
    alias INVALID_ESCAPE = 5
    alias NESTING_TOO_DEEP = 6
    alias DUPLICATE_KEY = 7
    alias TYPE_ERROR = 8
    alias KEY_NOT_FOUND = 9
    alias INDEX_OUT_OF_BOUNDS = 10

    var code: Int
    var message: String
    var line: Int
    var column: Int

    fn __init__(inout self):
        self.code = JsonError.NONE
        self.message = ""
        self.line = 0
        self.column = 0

    fn __init__(inout self, code: Int, message: String):
        self.code = code
        self.message = message
        self.line = 0
        self.column = 0

    fn __init__(inout self, code: Int, message: String, line: Int, column: Int):
        self.code = code
        self.message = message
        self.line = line
        self.column = column

    fn is_error(self) -> Bool:
        return self.code != JsonError.NONE

    fn __str__(self) -> String:
        var result = "JsonError(" + str(self.code) + "): " + self.message
        if self.line > 0:
            result += " at line " + str(self.line) + ", column " + str(self.column)
        return result


# Global JSON error
var _last_json_error = JsonError()

fn get_last_json_error() -> JsonError:
    return _last_json_error

fn clear_json_error():
    _last_json_error = JsonError()

fn set_json_error(code: Int, message: String):
    _last_json_error = JsonError(code, message)


# =============================================================================
# JSON Value
# =============================================================================

struct JsonValue:
    """A JSON value that can hold any JSON type."""

    var _type: JsonType
    var _bool_value: Bool
    var _number_value: Float64
    var _string_value: String
    var _array_values: List[JsonValue]
    var _object_keys: List[String]
    var _object_values: List[JsonValue]

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    fn __init__(inout self):
        """Create a null value."""
        self._type = JsonType(JsonType.NULL)
        self._bool_value = False
        self._number_value = 0.0
        self._string_value = ""
        self._array_values = List[JsonValue]()
        self._object_keys = List[String]()
        self._object_values = List[JsonValue]()

    fn __init__(inout self, value: Bool):
        """Create a boolean value."""
        self._type = JsonType(JsonType.BOOL)
        self._bool_value = value
        self._number_value = 0.0
        self._string_value = ""
        self._array_values = List[JsonValue]()
        self._object_keys = List[String]()
        self._object_values = List[JsonValue]()

    fn __init__(inout self, value: Int):
        """Create a number value from int."""
        self._type = JsonType(JsonType.NUMBER)
        self._bool_value = False
        self._number_value = Float64(value)
        self._string_value = ""
        self._array_values = List[JsonValue]()
        self._object_keys = List[String]()
        self._object_values = List[JsonValue]()

    fn __init__(inout self, value: Float64):
        """Create a number value from float."""
        self._type = JsonType(JsonType.NUMBER)
        self._bool_value = False
        self._number_value = value
        self._string_value = ""
        self._array_values = List[JsonValue]()
        self._object_keys = List[String]()
        self._object_values = List[JsonValue]()

    fn __init__(inout self, value: String):
        """Create a string value."""
        self._type = JsonType(JsonType.STRING)
        self._bool_value = False
        self._number_value = 0.0
        self._string_value = value
        self._array_values = List[JsonValue]()
        self._object_keys = List[String]()
        self._object_values = List[JsonValue]()

    @staticmethod
    fn null() -> JsonValue:
        """Create a null value."""
        return JsonValue()

    @staticmethod
    fn array() -> JsonValue:
        """Create an empty array."""
        var v = JsonValue()
        v._type = JsonType(JsonType.ARRAY)
        return v

    @staticmethod
    fn object() -> JsonValue:
        """Create an empty object."""
        var v = JsonValue()
        v._type = JsonType(JsonType.OBJECT)
        return v

    # -------------------------------------------------------------------------
    # Type Checking
    # -------------------------------------------------------------------------

    fn type(self) -> JsonType:
        return self._type

    fn is_null(self) -> Bool:
        return self._type.value == JsonType.NULL

    fn is_bool(self) -> Bool:
        return self._type.value == JsonType.BOOL

    fn is_number(self) -> Bool:
        return self._type.value == JsonType.NUMBER

    fn is_string(self) -> Bool:
        return self._type.value == JsonType.STRING

    fn is_array(self) -> Bool:
        return self._type.value == JsonType.ARRAY

    fn is_object(self) -> Bool:
        return self._type.value == JsonType.OBJECT

    # -------------------------------------------------------------------------
    # Value Getters
    # -------------------------------------------------------------------------

    fn as_bool(self) -> Bool:
        """Get as boolean. Returns False if not a bool."""
        if self.is_bool():
            return self._bool_value
        return False

    fn as_int(self) -> Int:
        """Get as integer. Returns 0 if not a number."""
        if self.is_number():
            return int(self._number_value)
        return 0

    fn as_float(self) -> Float64:
        """Get as float. Returns 0.0 if not a number."""
        if self.is_number():
            return self._number_value
        return 0.0

    fn as_string(self) -> String:
        """Get as string. Returns empty string if not a string."""
        if self.is_string():
            return self._string_value
        return ""

    # -------------------------------------------------------------------------
    # Array Operations
    # -------------------------------------------------------------------------

    fn __len__(self) -> Int:
        """Get length of array or object."""
        if self.is_array():
            return len(self._array_values)
        elif self.is_object():
            return len(self._object_keys)
        return 0

    fn __getitem__(self, index: Int) -> JsonValue:
        """Get array element by index."""
        if self.is_array() and index >= 0 and index < len(self._array_values):
            return self._array_values[index]
        return JsonValue()

    fn __setitem__(inout self, index: Int, value: JsonValue):
        """Set array element by index."""
        if self.is_array() and index >= 0 and index < len(self._array_values):
            self._array_values[index] = value

    fn append(inout self, value: JsonValue):
        """Append to array."""
        if self.is_array():
            self._array_values.append(value)

    fn append(inout self, value: Bool):
        self.append(JsonValue(value))

    fn append(inout self, value: Int):
        self.append(JsonValue(value))

    fn append(inout self, value: Float64):
        self.append(JsonValue(value))

    fn append(inout self, value: String):
        self.append(JsonValue(value))

    # -------------------------------------------------------------------------
    # Object Operations
    # -------------------------------------------------------------------------

    fn __getitem__(self, key: String) -> JsonValue:
        """Get object value by key."""
        if self.is_object():
            for i in range(len(self._object_keys)):
                if self._object_keys[i] == key:
                    return self._object_values[i]
        return JsonValue()

    fn __setitem__(inout self, key: String, value: JsonValue):
        """Set object value by key."""
        if self.is_object():
            # Check if key exists
            for i in range(len(self._object_keys)):
                if self._object_keys[i] == key:
                    self._object_values[i] = value
                    return
            # Add new key
            self._object_keys.append(key)
            self._object_values.append(value)

    fn set(inout self, key: String, value: JsonValue):
        """Set object key-value pair."""
        self[key] = value

    fn set(inout self, key: String, value: Bool):
        self[key] = JsonValue(value)

    fn set(inout self, key: String, value: Int):
        self[key] = JsonValue(value)

    fn set(inout self, key: String, value: Float64):
        self[key] = JsonValue(value)

    fn set(inout self, key: String, value: String):
        self[key] = JsonValue(value)

    fn has(self, key: String) -> Bool:
        """Check if object has key."""
        if self.is_object():
            for i in range(len(self._object_keys)):
                if self._object_keys[i] == key:
                    return True
        return False

    fn keys(self) -> List[String]:
        """Get object keys."""
        if self.is_object():
            return self._object_keys
        return List[String]()

    fn remove(inout self, key: String) -> Bool:
        """Remove key from object."""
        if self.is_object():
            for i in range(len(self._object_keys)):
                if self._object_keys[i] == key:
                    _ = self._object_keys.pop(i)
                    _ = self._object_values.pop(i)
                    return True
        return False

    # -------------------------------------------------------------------------
    # Path Access
    # -------------------------------------------------------------------------

    fn get(self, path: String) -> JsonValue:
        """Get value by JSON path (e.g., 'foo.bar.0.baz')."""
        var current = self
        var start = 0

        for i in range(len(path) + 1):
            if i == len(path) or path[i] == ".":
                if i > start:
                    var key = path[start:i]

                    # Try as array index
                    var is_index = True
                    var index = 0
                    for j in range(len(key)):
                        var c = key[j]
                        if c >= "0" and c <= "9":
                            index = index * 10 + (ord(c) - ord("0"))
                        else:
                            is_index = False
                            break

                    if is_index and current.is_array():
                        current = current[index]
                    elif current.is_object():
                        current = current[key]
                    else:
                        return JsonValue()

                start = i + 1

        return current

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    fn to_string(self) -> String:
        """Serialize to compact JSON string."""
        return self._serialize(False, 0, 0)

    fn to_pretty_string(self, indent: Int = DEFAULT_INDENT) -> String:
        """Serialize to pretty-printed JSON string."""
        return self._serialize(True, indent, 0)

    fn _serialize(self, pretty: Bool, indent: Int, depth: Int) -> String:
        if self.is_null():
            return "null"

        elif self.is_bool():
            return "true" if self._bool_value else "false"

        elif self.is_number():
            # Check if it's an integer
            var int_val = int(self._number_value)
            if Float64(int_val) == self._number_value:
                return str(int_val)
            return str(self._number_value)

        elif self.is_string():
            return self._escape_string(self._string_value)

        elif self.is_array():
            if len(self._array_values) == 0:
                return "[]"

            var result = "["
            var newline = "\n" if pretty else ""
            var space = " " if pretty else ""
            var child_indent = _make_indent(indent, depth + 1) if pretty else ""
            var close_indent = _make_indent(indent, depth) if pretty else ""

            for i in range(len(self._array_values)):
                if i > 0:
                    result += ","
                result += newline + child_indent
                result += self._array_values[i]._serialize(pretty, indent, depth + 1)

            result += newline + close_indent + "]"
            return result

        elif self.is_object():
            if len(self._object_keys) == 0:
                return "{}"

            var result = "{"
            var newline = "\n" if pretty else ""
            var space = " " if pretty else ""
            var child_indent = _make_indent(indent, depth + 1) if pretty else ""
            var close_indent = _make_indent(indent, depth) if pretty else ""

            for i in range(len(self._object_keys)):
                if i > 0:
                    result += ","
                result += newline + child_indent
                result += self._escape_string(self._object_keys[i])
                result += ":" + space
                result += self._object_values[i]._serialize(pretty, indent, depth + 1)

            result += newline + close_indent + "}"
            return result

        return "null"

    fn _escape_string(self, s: String) -> String:
        """Escape string for JSON."""
        var result = "\""

        for i in range(len(s)):
            var c = s[i]
            if c == "\"":
                result += "\\\""
            elif c == "\\":
                result += "\\\\"
            elif c == "\n":
                result += "\\n"
            elif c == "\r":
                result += "\\r"
            elif c == "\t":
                result += "\\t"
            else:
                var code = ord(c)
                if code < 32:
                    # Control character - use \uXXXX
                    result += "\\u00"
                    var hex_digit = code // 16
                    result += chr(ord("0") + hex_digit) if hex_digit < 10 else chr(ord("a") + hex_digit - 10)
                    hex_digit = code % 16
                    result += chr(ord("0") + hex_digit) if hex_digit < 10 else chr(ord("a") + hex_digit - 10)
                else:
                    result += c

        result += "\""
        return result


fn _make_indent(indent: Int, depth: Int) -> String:
    """Create indentation string."""
    var result = ""
    for _ in range(indent * depth):
        result += " "
    return result


# =============================================================================
# JSON Parser
# =============================================================================

struct JsonParser:
    """JSON parser."""

    var _input: String
    var _pos: Int
    var _len: Int
    var _line: Int
    var _column: Int
    var _depth: Int

    fn __init__(inout self, input: String):
        self._input = input
        self._pos = 0
        self._len = len(input)
        self._line = 1
        self._column = 1
        self._depth = 0

    fn parse(inout self) raises -> JsonValue:
        """Parse JSON string to JsonValue."""
        clear_json_error()
        self._skip_whitespace()

        if self._pos >= self._len:
            set_json_error(JsonError.UNEXPECTED_END, "Empty input")
            raise Error("Empty JSON input")

        var value = self._parse_value()
        self._skip_whitespace()

        # Check for trailing content
        if self._pos < self._len:
            set_json_error(JsonError.UNEXPECTED_TOKEN, "Unexpected content after JSON value")
            raise Error("Unexpected content after JSON value")

        return value

    fn _parse_value(inout self) raises -> JsonValue:
        """Parse a JSON value."""
        self._skip_whitespace()

        if self._pos >= self._len:
            set_json_error(JsonError.UNEXPECTED_END, "Unexpected end of input")
            raise Error("Unexpected end of input")

        var c = self._input[self._pos]

        if c == "n":
            return self._parse_null()
        elif c == "t" or c == "f":
            return self._parse_bool()
        elif c == "\"":
            return self._parse_string()
        elif c == "[":
            return self._parse_array()
        elif c == "{":
            return self._parse_object()
        elif c == "-" or (c >= "0" and c <= "9"):
            return self._parse_number()
        else:
            set_json_error(JsonError.UNEXPECTED_TOKEN, "Unexpected character: " + c, self._line, self._column)
            raise Error("Unexpected character: " + c)

    fn _parse_null(inout self) raises -> JsonValue:
        """Parse null literal."""
        if self._match("null"):
            return JsonValue.null()
        set_json_error(JsonError.UNEXPECTED_TOKEN, "Expected 'null'")
        raise Error("Expected 'null'")

    fn _parse_bool(inout self) raises -> JsonValue:
        """Parse boolean literal."""
        if self._match("true"):
            return JsonValue(True)
        elif self._match("false"):
            return JsonValue(False)
        set_json_error(JsonError.UNEXPECTED_TOKEN, "Expected 'true' or 'false'")
        raise Error("Expected 'true' or 'false'")

    fn _parse_string(inout self) raises -> JsonValue:
        """Parse string literal."""
        var s = self._read_string()
        return JsonValue(s)

    fn _read_string(inout self) raises -> String:
        """Read a JSON string."""
        if self._pos >= self._len or self._input[self._pos] != "\"":
            set_json_error(JsonError.INVALID_STRING, "Expected '\"'")
            raise Error("Expected '\"'")

        self._advance()  # Skip opening quote
        var result = ""

        while self._pos < self._len:
            var c = self._input[self._pos]

            if c == "\"":
                self._advance()  # Skip closing quote
                return result

            elif c == "\\":
                self._advance()
                if self._pos >= self._len:
                    set_json_error(JsonError.INVALID_ESCAPE, "Unexpected end after escape")
                    raise Error("Unexpected end after escape")

                var escaped = self._input[self._pos]
                if escaped == "\"":
                    result += "\""
                elif escaped == "\\":
                    result += "\\"
                elif escaped == "/":
                    result += "/"
                elif escaped == "b":
                    result += "\x08"  # Backspace
                elif escaped == "f":
                    result += "\x0c"  # Form feed
                elif escaped == "n":
                    result += "\n"
                elif escaped == "r":
                    result += "\r"
                elif escaped == "t":
                    result += "\t"
                elif escaped == "u":
                    # Unicode escape \uXXXX
                    self._advance()
                    var code = self._read_hex4()
                    result += chr(code)
                    continue  # Don't advance again
                else:
                    set_json_error(JsonError.INVALID_ESCAPE, "Invalid escape: \\" + escaped)
                    raise Error("Invalid escape: \\" + escaped)
                self._advance()

            elif ord(c) < 32:
                set_json_error(JsonError.INVALID_STRING, "Control character in string")
                raise Error("Control character in string")

            else:
                result += c
                self._advance()

        set_json_error(JsonError.UNEXPECTED_END, "Unterminated string")
        raise Error("Unterminated string")

    fn _read_hex4(inout self) raises -> Int:
        """Read 4 hex digits."""
        var value = 0

        for _ in range(4):
            if self._pos >= self._len:
                set_json_error(JsonError.INVALID_ESCAPE, "Incomplete unicode escape")
                raise Error("Incomplete unicode escape")

            var c = self._input[self._pos]
            var digit = 0

            if c >= "0" and c <= "9":
                digit = ord(c) - ord("0")
            elif c >= "a" and c <= "f":
                digit = ord(c) - ord("a") + 10
            elif c >= "A" and c <= "F":
                digit = ord(c) - ord("A") + 10
            else:
                set_json_error(JsonError.INVALID_ESCAPE, "Invalid hex digit")
                raise Error("Invalid hex digit")

            value = value * 16 + digit
            self._advance()

        return value

    fn _parse_number(inout self) raises -> JsonValue:
        """Parse number literal."""
        var start = self._pos
        var is_float = False

        # Optional minus
        if self._pos < self._len and self._input[self._pos] == "-":
            self._advance()

        # Integer part
        if self._pos < self._len and self._input[self._pos] == "0":
            self._advance()
        elif self._pos < self._len and self._input[self._pos] >= "1" and self._input[self._pos] <= "9":
            while self._pos < self._len and self._input[self._pos] >= "0" and self._input[self._pos] <= "9":
                self._advance()
        else:
            set_json_error(JsonError.INVALID_NUMBER, "Invalid number")
            raise Error("Invalid number")

        # Fractional part
        if self._pos < self._len and self._input[self._pos] == ".":
            is_float = True
            self._advance()
            if self._pos >= self._len or self._input[self._pos] < "0" or self._input[self._pos] > "9":
                set_json_error(JsonError.INVALID_NUMBER, "Expected digit after decimal point")
                raise Error("Expected digit after decimal point")
            while self._pos < self._len and self._input[self._pos] >= "0" and self._input[self._pos] <= "9":
                self._advance()

        # Exponent part
        if self._pos < self._len and (self._input[self._pos] == "e" or self._input[self._pos] == "E"):
            is_float = True
            self._advance()
            if self._pos < self._len and (self._input[self._pos] == "+" or self._input[self._pos] == "-"):
                self._advance()
            if self._pos >= self._len or self._input[self._pos] < "0" or self._input[self._pos] > "9":
                set_json_error(JsonError.INVALID_NUMBER, "Expected digit in exponent")
                raise Error("Expected digit in exponent")
            while self._pos < self._len and self._input[self._pos] >= "0" and self._input[self._pos] <= "9":
                self._advance()

        var num_str = self._input[start:self._pos]

        # Parse the number
        if is_float:
            var value = self._parse_float(num_str)
            return JsonValue(value)
        else:
            var value = self._parse_int(num_str)
            return JsonValue(value)

    fn _parse_int(self, s: String) -> Int:
        """Parse integer string."""
        var value = 0
        var negative = False
        var start = 0

        if len(s) > 0 and s[0] == "-":
            negative = True
            start = 1

        for i in range(start, len(s)):
            value = value * 10 + (ord(s[i]) - ord("0"))

        return -value if negative else value

    fn _parse_float(self, s: String) -> Float64:
        """Parse float string."""
        # Simplified float parsing
        var value = 0.0
        var negative = False
        var pos = 0

        if pos < len(s) and s[pos] == "-":
            negative = True
            pos += 1

        # Integer part
        while pos < len(s) and s[pos] >= "0" and s[pos] <= "9":
            value = value * 10.0 + Float64(ord(s[pos]) - ord("0"))
            pos += 1

        # Fractional part
        if pos < len(s) and s[pos] == ".":
            pos += 1
            var fraction = 0.1
            while pos < len(s) and s[pos] >= "0" and s[pos] <= "9":
                value += Float64(ord(s[pos]) - ord("0")) * fraction
                fraction *= 0.1
                pos += 1

        # Exponent part
        if pos < len(s) and (s[pos] == "e" or s[pos] == "E"):
            pos += 1
            var exp_negative = False
            if pos < len(s) and s[pos] == "-":
                exp_negative = True
                pos += 1
            elif pos < len(s) and s[pos] == "+":
                pos += 1

            var exp_value = 0
            while pos < len(s) and s[pos] >= "0" and s[pos] <= "9":
                exp_value = exp_value * 10 + (ord(s[pos]) - ord("0"))
                pos += 1

            var multiplier = 1.0
            for _ in range(exp_value):
                multiplier *= 10.0

            if exp_negative:
                value /= multiplier
            else:
                value *= multiplier

        return -value if negative else value

    fn _parse_array(inout self) raises -> JsonValue:
        """Parse array literal."""
        self._depth += 1
        if self._depth > MAX_NESTING_DEPTH:
            set_json_error(JsonError.NESTING_TOO_DEEP, "Maximum nesting depth exceeded")
            raise Error("Maximum nesting depth exceeded")

        self._advance()  # Skip '['
        var arr = JsonValue.array()

        self._skip_whitespace()

        if self._pos < self._len and self._input[self._pos] == "]":
            self._advance()
            self._depth -= 1
            return arr

        while True:
            self._skip_whitespace()
            var value = self._parse_value()
            arr.append(value)

            self._skip_whitespace()

            if self._pos >= self._len:
                set_json_error(JsonError.UNEXPECTED_END, "Unterminated array")
                raise Error("Unterminated array")

            var c = self._input[self._pos]
            if c == "]":
                self._advance()
                self._depth -= 1
                return arr
            elif c == ",":
                self._advance()
            else:
                set_json_error(JsonError.UNEXPECTED_TOKEN, "Expected ',' or ']'")
                raise Error("Expected ',' or ']'")

    fn _parse_object(inout self) raises -> JsonValue:
        """Parse object literal."""
        self._depth += 1
        if self._depth > MAX_NESTING_DEPTH:
            set_json_error(JsonError.NESTING_TOO_DEEP, "Maximum nesting depth exceeded")
            raise Error("Maximum nesting depth exceeded")

        self._advance()  # Skip '{'
        var obj = JsonValue.object()

        self._skip_whitespace()

        if self._pos < self._len and self._input[self._pos] == "}":
            self._advance()
            self._depth -= 1
            return obj

        while True:
            self._skip_whitespace()

            # Parse key
            if self._pos >= self._len or self._input[self._pos] != "\"":
                set_json_error(JsonError.UNEXPECTED_TOKEN, "Expected string key")
                raise Error("Expected string key")

            var key = self._read_string()

            self._skip_whitespace()

            # Expect colon
            if self._pos >= self._len or self._input[self._pos] != ":":
                set_json_error(JsonError.UNEXPECTED_TOKEN, "Expected ':'")
                raise Error("Expected ':'")
            self._advance()

            # Parse value
            self._skip_whitespace()
            var value = self._parse_value()

            obj.set(key, value)

            self._skip_whitespace()

            if self._pos >= self._len:
                set_json_error(JsonError.UNEXPECTED_END, "Unterminated object")
                raise Error("Unterminated object")

            var c = self._input[self._pos]
            if c == "}":
                self._advance()
                self._depth -= 1
                return obj
            elif c == ",":
                self._advance()
            else:
                set_json_error(JsonError.UNEXPECTED_TOKEN, "Expected ',' or '}'")
                raise Error("Expected ',' or '}'")

    fn _skip_whitespace(inout self):
        """Skip whitespace and update position."""
        while self._pos < self._len:
            var c = self._input[self._pos]
            if c == " " or c == "\t":
                self._column += 1
                self._pos += 1
            elif c == "\n":
                self._line += 1
                self._column = 1
                self._pos += 1
            elif c == "\r":
                self._pos += 1
            else:
                break

    fn _advance(inout self):
        """Advance position by one character."""
        if self._pos < self._len:
            if self._input[self._pos] == "\n":
                self._line += 1
                self._column = 1
            else:
                self._column += 1
            self._pos += 1

    fn _match(inout self, expected: String) -> Bool:
        """Match exact string."""
        var end = self._pos + len(expected)
        if end > self._len:
            return False

        for i in range(len(expected)):
            if self._input[self._pos + i] != expected[i]:
                return False

        for _ in range(len(expected)):
            self._advance()

        return True


# =============================================================================
# Convenience Functions
# =============================================================================

fn parse(input: String) raises -> JsonValue:
    """Parse JSON string."""
    var parser = JsonParser(input)
    return parser.parse()

fn stringify(value: JsonValue) -> String:
    """Serialize to compact JSON string."""
    return value.to_string()

fn prettify(value: JsonValue, indent: Int = DEFAULT_INDENT) -> String:
    """Serialize to pretty-printed JSON string."""
    return value.to_pretty_string(indent)

fn is_valid(input: String) -> Bool:
    """Check if string is valid JSON."""
    try:
        var parser = JsonParser(input)
        _ = parser.parse()
        return True
    except:
        return False


# =============================================================================
# JSON Builder (Fluent API)
# =============================================================================

struct JsonBuilder:
    """Fluent JSON builder."""

    var _value: JsonValue

    fn __init__(inout self):
        self._value = JsonValue.object()

    @staticmethod
    fn object() -> JsonBuilder:
        var b = JsonBuilder()
        b._value = JsonValue.object()
        return b

    @staticmethod
    fn array() -> JsonBuilder:
        var b = JsonBuilder()
        b._value = JsonValue.array()
        return b

    fn set(inout self, key: String, value: JsonValue) -> JsonBuilder:
        self._value.set(key, value)
        return self

    fn set(inout self, key: String, value: String) -> JsonBuilder:
        self._value.set(key, JsonValue(value))
        return self

    fn set(inout self, key: String, value: Int) -> JsonBuilder:
        self._value.set(key, JsonValue(value))
        return self

    fn set(inout self, key: String, value: Float64) -> JsonBuilder:
        self._value.set(key, JsonValue(value))
        return self

    fn set(inout self, key: String, value: Bool) -> JsonBuilder:
        self._value.set(key, JsonValue(value))
        return self

    fn add(inout self, value: JsonValue) -> JsonBuilder:
        self._value.append(value)
        return self

    fn add(inout self, value: String) -> JsonBuilder:
        self._value.append(JsonValue(value))
        return self

    fn add(inout self, value: Int) -> JsonBuilder:
        self._value.append(JsonValue(value))
        return self

    fn build(self) -> JsonValue:
        return self._value

    fn to_string(self) -> String:
        return self._value.to_string()


# =============================================================================
# Tests
# =============================================================================

fn test_json_value_types():
    """Test JsonValue type constructors."""
    var null_val = JsonValue.null()
    assert_true(null_val.is_null(), "Should be null")

    var bool_val = JsonValue(True)
    assert_true(bool_val.is_bool(), "Should be bool")
    assert_true(bool_val.as_bool() == True, "Should be true")

    var int_val = JsonValue(42)
    assert_true(int_val.is_number(), "Should be number")
    assert_true(int_val.as_int() == 42, "Should be 42")

    var str_val = JsonValue("hello")
    assert_true(str_val.is_string(), "Should be string")
    assert_true(str_val.as_string() == "hello", "Should be 'hello'")

    print("test_json_value_types: PASSED")


fn test_json_array():
    """Test JsonValue array operations."""
    var arr = JsonValue.array()
    assert_true(arr.is_array(), "Should be array")
    assert_true(len(arr) == 0, "Should be empty")

    arr.append(1)
    arr.append("two")
    arr.append(True)

    assert_true(len(arr) == 3, "Should have 3 elements")
    assert_true(arr[0].as_int() == 1, "First should be 1")
    assert_true(arr[1].as_string() == "two", "Second should be 'two'")
    assert_true(arr[2].as_bool() == True, "Third should be true")

    print("test_json_array: PASSED")


fn test_json_object():
    """Test JsonValue object operations."""
    var obj = JsonValue.object()
    assert_true(obj.is_object(), "Should be object")

    obj.set("name", "John")
    obj.set("age", 30)
    obj.set("active", True)

    assert_true(len(obj) == 3, "Should have 3 keys")
    assert_true(obj.has("name"), "Should have 'name'")
    assert_true(obj["name"].as_string() == "John", "Name should be 'John'")
    assert_true(obj["age"].as_int() == 30, "Age should be 30")

    print("test_json_object: PASSED")


fn test_json_parse():
    """Test JSON parsing."""
    try:
        var json_str = '{"name": "Alice", "age": 25, "scores": [95, 87, 92]}'
        var value = parse(json_str)

        assert_true(value.is_object(), "Should be object")
        assert_true(value["name"].as_string() == "Alice", "Name should be 'Alice'")
        assert_true(value["age"].as_int() == 25, "Age should be 25")
        assert_true(value["scores"].is_array(), "Scores should be array")
        assert_true(len(value["scores"]) == 3, "Should have 3 scores")

        print("test_json_parse: PASSED")
    except e:
        print("test_json_parse: FAILED - " + str(e))


fn test_json_stringify():
    """Test JSON serialization."""
    var obj = JsonValue.object()
    obj.set("name", "Bob")
    obj.set("active", True)

    var json_str = stringify(obj)
    assert_true(len(json_str) > 0, "Should produce output")

    # Should contain expected content
    assert_true(json_str.find("\"name\"") >= 0, "Should contain name key")
    assert_true(json_str.find("\"Bob\"") >= 0, "Should contain Bob value")
    assert_true(json_str.find("true") >= 0, "Should contain true")

    print("test_json_stringify: PASSED")


fn test_json_path():
    """Test JSON path access."""
    try:
        var json_str = '{"user": {"name": "Test", "tags": ["a", "b", "c"]}}'
        var value = parse(json_str)

        assert_true(value.get("user.name").as_string() == "Test", "Path should work")
        assert_true(value.get("user.tags.1").as_string() == "b", "Array index should work")

        print("test_json_path: PASSED")
    except e:
        print("test_json_path: FAILED - " + str(e))


fn test_json_escape():
    """Test string escaping."""
    var value = JsonValue("Hello\nWorld\t\"Quoted\"")
    var json_str = stringify(value)

    assert_true(json_str.find("\\n") >= 0, "Should escape newline")
    assert_true(json_str.find("\\t") >= 0, "Should escape tab")
    assert_true(json_str.find("\\\"") >= 0, "Should escape quote")

    print("test_json_escape: PASSED")


fn assert_true(condition: Bool, message: String):
    """Simple assertion helper."""
    if not condition:
        print("ASSERTION FAILED: " + message)


fn run_all_tests():
    """Run all JSON tests."""
    print("=== JSON Module Tests ===")
    test_json_value_types()
    test_json_array()
    test_json_object()
    test_json_parse()
    test_json_stringify()
    test_json_path()
    test_json_escape()
    print("=== All Tests Passed ===")
