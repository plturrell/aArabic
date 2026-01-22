"""
Generic JSON Parser for Mojo SDK

Zero Python dependencies - pure Mojo + Zig backend.
Follows toon_integration.mojo pattern.
"""

from sys.ffi import DLHandle
from memory import UnsafePointer


# ============================================================================
# JSON Parser
# ============================================================================

struct JsonParser:
    """
    Generic JSON parser using Zig std.json backend.
    
    Zero Python dependencies - production-ready parser for:
    - Configuration files
    - API responses
    - Data serialization
    - Schema definitions
    
    Usage:
        var parser = JsonParser()
        var json = parser.parse_file("config.json")
        
        # Or parse string
        var data = parser.parse_string('{"key": "value"}')
    
    Pattern: Follows zig_bolt_shimmy, zig_toon architecture
    """
    var lib: DLHandle
    var parse_file_fn: fn(UnsafePointer[UInt8]) -> UnsafePointer[UInt8]
    var parse_string_fn: fn(UnsafePointer[UInt8], Int) -> UnsafePointer[UInt8]
    var validate_fn: fn(UnsafePointer[UInt8], Int) -> Bool
    var get_value_fn: fn(UnsafePointer[UInt8], Int, UnsafePointer[UInt8]) -> UnsafePointer[UInt8]
    var get_keys_fn: fn(UnsafePointer[UInt8], Int) -> UnsafePointer[UInt8]
    var get_key_count_fn: fn(UnsafePointer[UInt8], Int) -> Int
    var has_key_fn: fn(UnsafePointer[UInt8], Int, UnsafePointer[UInt8]) -> Bool
    var get_array_length_fn: fn(UnsafePointer[UInt8], Int) -> Int
    var get_array_item_fn: fn(UnsafePointer[UInt8], Int, Int) -> UnsafePointer[UInt8]
    var get_nested_value_fn: fn(UnsafePointer[UInt8], Int, UnsafePointer[UInt8]) -> UnsafePointer[UInt8]
    var free_fn: fn(UnsafePointer[UInt8]) -> None
    var test_fn: fn() -> UnsafePointer[UInt8]
    var enabled: Bool
    var verbose: Bool
    
    fn __init__(
        inout self,
        lib_path: String = "./libzig_json.dylib",
        enabled: Bool = True,
        verbose: Bool = False
    ):
        """
        Initialize JSON parser.
        
        Args:
            lib_path: Path to libzig_json.dylib
            enabled: Enable JSON parsing
            verbose: Debug output
        """
        self.enabled = enabled
        self.verbose = verbose
        
        if not enabled:
            if verbose:
                print("[JsonParser] Disabled - parser not loaded")
            return
        
        try:
            # Load Zig JSON library
            self.lib = DLHandle(lib_path)
            
            # Get parse_file function
            self.parse_file_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8]) -> UnsafePointer[UInt8]
            ]("zig_json_parse_file")
            
            # Get parse_string function
            self.parse_string_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8], Int) -> UnsafePointer[UInt8]
            ]("zig_json_parse_string")
            
            # Get validate function
            self.validate_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8], Int) -> Bool
            ]("zig_json_validate")
            
            # Get get_value function
            self.get_value_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8], Int, UnsafePointer[UInt8]) -> UnsafePointer[UInt8]
            ]("zig_json_get_value")
            
            # Get free function
            self.free_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8]) -> None
            ]("zig_json_free")
            
            # Get dict/array operation functions
            self.get_keys_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8], Int) -> UnsafePointer[UInt8]
            ]("zig_json_get_keys")
            
            self.get_key_count_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8], Int) -> Int
            ]("zig_json_get_key_count")
            
            self.has_key_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8], Int, UnsafePointer[UInt8]) -> Bool
            ]("zig_json_has_key")
            
            self.get_array_length_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8], Int) -> Int
            ]("zig_json_get_array_length")
            
            self.get_array_item_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8], Int, Int) -> UnsafePointer[UInt8]
            ]("zig_json_get_array_item")
            
            self.get_nested_value_fn = self.lib.get_function[
                fn(UnsafePointer[UInt8], Int, UnsafePointer[UInt8]) -> UnsafePointer[UInt8]
            ]("zig_json_get_nested_value")
            
            # Get test function
            self.test_fn = self.lib.get_function[
                fn() -> UnsafePointer[UInt8]
            ]("zig_json_test")
            
            if verbose:
                var test_result_ptr = self.test_fn()
                var test_result = String(test_result_ptr)
                print(f"[JsonParser] {test_result}")
                print(f"[JsonParser] Loaded from {lib_path}")
                print(f"[JsonParser] Dict/Array operations available!")
                
        except e:
            print(f"[JsonParser] Error loading library: {e}")
            print(f"[JsonParser] Falling back to disabled mode")
            self.enabled = False
    
    fn parse_file(self, path: String) raises -> String:
        """
        Parse JSON file using Zig backend (zero Python!).
        
        Args:
            path: Path to JSON file
            
        Returns:
            Minified JSON string (validated)
            
        Raises:
            Error if file not found or invalid JSON
            
        Example:
            var parser = JsonParser()
            var json = parser.parse_file("config/schema.json")
        """
        if not self.enabled:
            raise Error("JSON parser not initialized")
        
        var result_ptr = self.parse_file_fn(path.unsafe_ptr())
        var result = String(result_ptr)
        
        if result.startswith("ERROR:"):
            raise Error(result)
        
        if self.verbose:
            print(f"[JsonParser] Parsed file: {path}")
            var size = len(result)
            print(f"[JsonParser] Result size: {size} bytes")
        
        return result
    
    fn parse_string(self, json: String) raises -> String:
        """
        Parse JSON string using Zig backend (zero Python!).
        
        Args:
            json: JSON string to parse
            
        Returns:
            Minified JSON string (validated)
            
        Raises:
            Error if invalid JSON syntax
            
        Example:
            var parser = JsonParser()
            var json = parser.parse_string('{"key": "value"}')
        """
        if not self.enabled:
            raise Error("JSON parser not initialized")
        
        var result_ptr = self.parse_string_fn(
            json.unsafe_ptr(),
            len(json)
        )
        var result = String(result_ptr)
        
        if result.startswith("ERROR:"):
            raise Error(result)
        
        if self.verbose:
            print(f"[JsonParser] Parsed string ({len(json)} bytes)")
        
        return result
    
    fn validate(self, json: String) -> Bool:
        """
        Validate JSON syntax without full parsing.
        
        Fast validation for checking JSON structure.
        
        Args:
            json: JSON string to validate
            
        Returns:
            True if valid JSON, False otherwise
            
        Example:
            var parser = JsonParser()
            if parser.validate('{"key": "value"}'):
                print("Valid JSON!")
        """
        if not self.enabled:
            return False
        
        return self.validate_fn(json.unsafe_ptr(), len(json))
    
    fn get_value(self, json: String, key: String) raises -> String:
        """
        Get value from JSON object by key.
        
        Args:
            json: JSON object string
            key: Key to lookup
            
        Returns:
            JSON value as string
            
        Raises:
            Error if key not found or not an object
            
        Example:
            var parser = JsonParser()
            var json = '{"name": "test", "value": 42}'
            var name = parser.get_value(json, "name")  # Returns: "test"
        """
        if not self.enabled:
            raise Error("JSON parser not initialized")
        
        var result_ptr = self.get_value_fn(
            json.unsafe_ptr(),
            len(json),
            key.unsafe_ptr()
        )
        var result = String(result_ptr)
        
        if result.startswith("ERROR:"):
            raise Error(result)
        
        return result
    
    fn get_keys(self, json: String) raises -> String:
        """
        Get all keys from JSON object.
        
        Returns comma-separated list of keys.
        
        Example:
            var parser = JsonParser()
            var json = '{"a": 1, "b": 2}'
            var keys = parser.get_keys(json)  # Returns: "a,b"
        """
        if not self.enabled:
            raise Error("JSON parser not initialized")
        
        var result_ptr = self.get_keys_fn(json.unsafe_ptr(), len(json))
        var result = String(result_ptr)
        
        if result.startswith("ERROR:"):
            raise Error(result)
        
        return result
    
    fn get_key_count(self, json: String) -> Int:
        """Get number of keys in JSON object"""
        if not self.enabled:
            return 0
        return self.get_key_count_fn(json.unsafe_ptr(), len(json))
    
    fn has_key(self, json: String, key: String) -> Bool:
        """Check if JSON object has key"""
        if not self.enabled:
            return False
        return self.has_key_fn(json.unsafe_ptr(), len(json), key.unsafe_ptr())
    
    fn get_array_length(self, json: String) -> Int:
        """Get length of JSON array"""
        if not self.enabled:
            return 0
        return self.get_array_length_fn(json.unsafe_ptr(), len(json))
    
    fn get_array_item(self, json: String, index: Int) raises -> String:
        """Get item from JSON array at index"""
        if not self.enabled:
            raise Error("JSON parser not initialized")
        
        var result_ptr = self.get_array_item_fn(json.unsafe_ptr(), len(json), index)
        var result = String(result_ptr)
        
        if result.startswith("ERROR:"):
            raise Error(result)
        
        return result
    
    fn get_nested_value(self, json: String, path: String) raises -> String:
        """
        Get nested value by path.
        
        Path uses dot notation (e.g., "graphs.supply_chain.nodes")
        
        Example:
            var parser = JsonParser()
            var json = '{"graphs": {"supply_chain": {"nodes": {...}}}}'
            var nodes = parser.get_nested_value(json, "graphs.supply_chain.nodes")
        """
        if not self.enabled:
            raise Error("JSON parser not initialized")
        
        var result_ptr = self.get_nested_value_fn(
            json.unsafe_ptr(),
            len(json),
            path.unsafe_ptr()
        )
        var result = String(result_ptr)
        
        if result.startswith("ERROR:"):
            raise Error(result)
        
        return result


# ============================================================================
# Factory Function
# ============================================================================

fn create_json_parser(
    lib_path: String = "./libzig_json.dylib",
    enabled: Bool = True,
    verbose: Bool = False
) -> JsonParser:
    """
    Create JSON parser instance.
    
    This is the recommended way to create a parser.
    
    Args:
        lib_path: Path to libzig_json.dylib
        enabled: Enable parsing
        verbose: Debug output
        
    Returns:
        Configured JsonParser
        
    Example:
        var parser = create_json_parser(verbose=True)
        var json = parser.parse_file("config.json")
    """
    return JsonParser(lib_path, enabled, verbose)


# ============================================================================
# Export for Zig/C Integration
# ============================================================================

@export
fn json_parse_file_export(path_ptr: UnsafePointer[UInt8]) -> UnsafePointer[UInt8]:
    """
    C ABI function for JSON file parsing.
    
    Args:
        path_ptr: File path string pointer
        
    Returns:
        JSON string pointer (or error string)
    """
    var path = String(path_ptr)
    
    var parser = create_json_parser(verbose=False)
    
    try:
        var result = parser.parse_file(path)
        return result.unsafe_ptr()
    except e:
        var error = String("ERROR: ") + str(e)
        return error.unsafe_ptr()


@export
fn json_validate_export(
    json_ptr: UnsafePointer[UInt8],
    json_len: Int
) -> Bool:
    """
    C ABI function for JSON validation.
    
    Args:
        json_ptr: JSON string pointer
        json_len: JSON string length
        
    Returns:
        True if valid JSON
    """
    var json = String(json_ptr, json_len)
    var parser = create_json_parser(verbose=False)
    return parser.validate(json)
