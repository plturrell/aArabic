"""
String utility wrappers for Mojo - FFI to Zig string library.
Provides UTF-8 handling, string building, and text processing.
"""

from .ffi import NExtractFFI
from sys.ffi import external_call

# ========== String Builder Wrapper ==========

struct StringBuilderHandle:
    """Opaque handle to Zig StringBuilder."""
    var ptr: UnsafePointer[NoneType]
    
    fn __init__(inout self):
        self.ptr = UnsafePointer[NoneType]()

struct StringBuilderWrapper:
    """RAII wrapper for Zig StringBuilder."""
    var handle: StringBuilderHandle
    var valid: Bool
    
    fn __init__(inout self):
        let ptr = external_call["nExtract_StringBuilder_create", UnsafePointer[NoneType]]()
        self.handle = StringBuilderHandle()
        self.handle.ptr = ptr
        self.valid = ptr != UnsafePointer[NoneType]()
    
    fn __del__(owned self):
        if self.valid:
            external_call["nExtract_StringBuilder_destroy", NoneType](self.handle.ptr)
    
    fn append(inout self, text: String) -> Bool:
        """Append text to builder."""
        if not self.valid:
            return False
        return external_call["nExtract_StringBuilder_append", Bool](
            self.handle.ptr,
            text.unsafe_ptr(),
            len(text)
        )
    
    fn to_string(self) -> String:
        """Get current string content."""
        if not self.valid:
            return ""
        
        var out_len: Int = 0
        let ptr = external_call["nExtract_StringBuilder_toSlice", UnsafePointer[UInt8]](
            self.handle.ptr,
            UnsafePointer.address_of(out_len)
        )
        
        if ptr == UnsafePointer[UInt8]():
            return ""
        
        # Copy to String
        var result = String()
        for i in range(out_len):
            result += String(chr(int(ptr[i])))
        return result

# ========== UTF-8 Validation ==========

fn validate_utf8(text: String) -> Bool:
    """Validate UTF-8 encoding of string."""
    return external_call["nExtract_validateUtf8", Bool](
        text.unsafe_ptr(),
        len(text)
    )

# ========== Case Conversion ==========

fn to_upper(text: String) -> String:
    """Convert string to uppercase (ASCII)."""
    var out_len: Int = 0
    let ptr = external_call["nExtract_toUpper", UnsafePointer[UInt8]](
        text.unsafe_ptr(),
        len(text),
        UnsafePointer.address_of(out_len)
    )
    
    if ptr == UnsafePointer[UInt8]():
        return text
    
    # Copy to String
    var result = String()
    for i in range(out_len):
        result += String(chr(int(ptr[i])))
    
    # Free allocated memory
    external_call["nExtract_freeString", NoneType](ptr, out_len)
    
    return result

fn to_lower(text: String) -> String:
    """Convert string to lowercase (ASCII)."""
    var out_len: Int = 0
    let ptr = external_call["nExtract_toLower", UnsafePointer[UInt8]](
        text.unsafe_ptr(),
        len(text),
        UnsafePointer.address_of(out_len)
    )
    
    if ptr == UnsafePointer[UInt8]():
        return text
    
    # Copy to String
    var result = String()
    for i in range(out_len):
        result += String(chr(int(ptr[i])))
    
    # Free allocated memory
    external_call["nExtract_freeString", NoneType](ptr, out_len)
    
    return result

# ========== High-Level String Utilities ==========

fn trim(text: String) -> String:
    """Trim whitespace from both ends."""
    var start = 0
    var end = len(text)
    
    # Trim start
    while start < end and text[start].isspace():
        start += 1
    
    # Trim end
    while end > start and text[end - 1].isspace():
        end -= 1
    
    return text[start:end]

fn split(text: String, delimiter: String) -> List[String]:
    """Split string by delimiter."""
    var result = List[String]()
    var start = 0
    
    for i in range(len(text)):
        if text[i:i+len(delimiter)] == delimiter:
            result.append(text[start:i])
            start = i + len(delimiter)
    
    # Add remaining part
    if start < len(text):
        result.append(text[start:])
    
    return result

fn join(strings: List[String], delimiter: String) -> String:
    """Join strings with delimiter."""
    if len(strings) == 0:
        return ""
    
    var builder = StringBuilderWrapper()
    for i in range(len(strings)):
        _ = builder.append(strings[i])
        if i < len(strings) - 1:
            _ = builder.append(delimiter)
    
    return builder.to_string()

fn replace_all(text: String, pattern: String, replacement: String) -> String:
    """Replace all occurrences of pattern."""
    if len(pattern) == 0:
        return text
    
    var builder = StringBuilderWrapper()
    var i = 0
    
    while i < len(text):
        if i + len(pattern) <= len(text) and text[i:i+len(pattern)] == pattern:
            _ = builder.append(replacement)
            i += len(pattern)
        else:
            _ = builder.append(text[i:i+1])
            i += 1
    
    return builder.to_string()

fn starts_with(text: String, prefix: String) -> Bool:
    """Check if string starts with prefix."""
    if len(prefix) > len(text):
        return False
    return text[:len(prefix)] == prefix

fn ends_with(text: String, suffix: String) -> Bool:
    """Check if string ends with suffix."""
    if len(suffix) > len(text):
        return False
    return text[len(text)-len(suffix):] == suffix

fn count_occurrences(text: String, pattern: String) -> Int:
    """Count occurrences of pattern in text."""
    if len(pattern) == 0:
        return 0
    
    var result = 0
    var i = 0
    
    while i + len(pattern) <= len(text):
        if text[i:i+len(pattern)] == pattern:
            result += 1
            i += len(pattern)
        else:
            i += 1
    
    return result
