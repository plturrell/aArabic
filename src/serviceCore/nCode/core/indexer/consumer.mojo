"""
SCIP file consumer for nCode.

This module provides a Mojo interface to consume SCIP index files
produced by language-specific indexers (scip-typescript, scip-python, etc.)
via Zig FFI.
"""

from collections import List
from memory import UnsafePointer, alloc
from sys.ffi import OwnedDLHandle

from ..scip import (
    Document, Occurrence, SymbolInformation, Kind, SymbolRole
)


# ============================================================================
# Library Path Constants
# ============================================================================

alias LIB_PATH_MAC = "./libzig_scip_reader.dylib"
alias LIB_PATH_LINUX = "./libzig_scip_reader.so"
alias LIB_PATH_FALLBACK_MAC = "./zig-out/lib/libzig_scip_reader.dylib"
alias LIB_PATH_FALLBACK_LINUX = "./zig-out/lib/libzig_scip_reader.so"


# ============================================================================
# FFI Result Types
# ============================================================================

@value
struct ScipLocation:
    """Location in source code returned from FFI."""
    var file_path: String
    var start_line: Int32
    var start_char: Int32
    var end_line: Int32
    var end_char: Int32

    fn __init__(inout self):
        self.file_path = ""
        self.start_line = 0
        self.start_char = 0
        self.end_line = 0
        self.end_char = 0


@value
struct ScipHoverInfo:
    """Hover information returned from FFI."""
    var documentation: String
    var symbol: String
    var kind: Int32

    fn __init__(inout self):
        self.documentation = ""
        self.symbol = ""
        self.kind = 0


# ============================================================================
# Helper Functions
# ============================================================================

fn _string_to_c_ptr(value: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Convert a Mojo String to a null-terminated C string pointer."""
    var bytes = value.as_bytes()
    var ptr = alloc[UInt8](len(bytes) + 1)
    for i in range(len(bytes)):
        ptr.store(i, bytes[i])
    ptr.store(len(bytes), 0)
    return ptr


fn _c_ptr_to_string(ptr: UnsafePointer[UInt8, ImmutExternalOrigin]) raises -> String:
    """Convert a null-terminated C string pointer to a Mojo String."""
    if not ptr:
        return ""
    var length = 0
    while ptr.load(length) != 0:
        length += 1
    var bytes = List[UInt8]()
    for i in range(length):
        bytes.append(ptr.load(i))
    return String(from_utf8=bytes)


fn _load_library() raises -> OwnedDLHandle:
    """Load the SCIP reader Zig library."""
    try:
        return OwnedDLHandle(LIB_PATH_MAC)
    except:
        pass
    try:
        return OwnedDLHandle(LIB_PATH_LINUX)
    except:
        pass
    try:
        return OwnedDLHandle(LIB_PATH_FALLBACK_MAC)
    except:
        pass
    try:
        return OwnedDLHandle(LIB_PATH_FALLBACK_LINUX)
    except:
        pass
    raise Error("Could not load SCIP reader library. Build with: zig build-lib scip_reader.zig -dynamic -OReleaseFast")


# ============================================================================
# CString type alias for FFI
# ============================================================================

alias CString = UnsafePointer[UInt8, MutExternalOrigin]
alias CStringImmut = UnsafePointer[UInt8, ImmutExternalOrigin]


# ============================================================================
# ScipConsumer - Main Consumer Struct
# ============================================================================

struct ScipConsumer:
    """
    SCIP file consumer that wraps the Zig FFI for reading SCIP indexes.
    
    Usage:
        var consumer = ScipConsumer()
        consumer.load("index.scip")
        var docs = consumer.get_documents()
        var def_loc = consumer.find_definition("path/to/file.py", 10, 5)
        consumer.close()
    """
    var _handle: OwnedDLHandle
    var _loaded: Bool
    var _path: String

    # FFI function pointers
    var _scip_load_index: fn(CString) -> Int32
    var _scip_free_index: fn() -> None
    var _scip_get_document_count: fn() -> Int32
    var _scip_get_document_path: fn(Int32) -> CStringImmut
    var _scip_get_document_language: fn(Int32) -> CStringImmut
    var _scip_get_symbol_at: fn(CString, Int32, Int32) -> CStringImmut
    var _scip_free_symbol: fn(CStringImmut) -> None
    var _scip_get_project_root: fn() -> CStringImmut
    var _scip_get_tool_name: fn() -> CStringImmut
    var _scip_get_tool_version: fn() -> CStringImmut

    fn __init__(inout self) raises:
        """Initialize the SCIP consumer and load the Zig library."""
        self._handle = _load_library()
        self._loaded = False
        self._path = ""
        
        # Load FFI function pointers
        self._scip_load_index = self._handle.get_function[fn(CString) -> Int32]("scip_load_index")
        self._scip_free_index = self._handle.get_function[fn() -> None]("scip_free_index")
        self._scip_get_document_count = self._handle.get_function[fn() -> Int32]("scip_get_document_count")
        self._scip_get_document_path = self._handle.get_function[fn(Int32) -> CStringImmut]("scip_get_document_path")
        self._scip_get_document_language = self._handle.get_function[fn(Int32) -> CStringImmut]("scip_get_document_language")
        self._scip_get_symbol_at = self._handle.get_function[fn(CString, Int32, Int32) -> CStringImmut]("scip_get_symbol_at")
        self._scip_free_symbol = self._handle.get_function[fn(CStringImmut) -> None]("scip_free_symbol")
        self._scip_get_project_root = self._handle.get_function[fn() -> CStringImmut]("scip_get_project_root")
        self._scip_get_tool_name = self._handle.get_function[fn() -> CStringImmut]("scip_get_tool_name")
        self._scip_get_tool_version = self._handle.get_function[fn() -> CStringImmut]("scip_get_tool_version")

    fn __del__(owned self):
        """Clean up resources when the consumer is destroyed."""
        if self._loaded:
            self._scip_free_index()

    fn load(inout self, path: String) raises -> Bool:
        """
        Load a SCIP index file.

        Args:
            path: Path to the SCIP index file (e.g., "index.scip")

        Returns:
            True if loading succeeded, False otherwise
        """
        var path_ptr = _string_to_c_ptr(path)
        var result = self._scip_load_index(path_ptr)

        if result == 0:
            self._loaded = True
            self._path = path
            return True
        else:
            self._loaded = False
            return False

    fn close(inout self):
        """Close the loaded index and free resources."""
        if self._loaded:
            self._scip_free_index()
            self._loaded = False
            self._path = ""

    fn is_loaded(self) -> Bool:
        """Check if an index is currently loaded."""
        return self._loaded

    fn get_documents(self) raises -> List[String]:
        """
        Get a list of all indexed document paths.

        Returns:
            List of relative file paths in the index
        """
        var docs = List[String]()
        if not self._loaded:
            return docs

        var count = self._scip_get_document_count()
        for i in range(count):
            var path_ptr = self._scip_get_document_path(Int32(i))
            if path_ptr:
                var path = _c_ptr_to_string(path_ptr)
                docs.append(path)
                self._scip_free_symbol(path_ptr)

        return docs

    fn get_document_count(self) -> Int:
        """Get the number of documents in the loaded index."""
        if not self._loaded:
            return 0
        return int(self._scip_get_document_count())

    fn get_project_root(self) raises -> String:
        """Get the project root from the index metadata."""
        if not self._loaded:
            return ""
        var ptr = self._scip_get_project_root()
        if not ptr:
            return ""
        var result = _c_ptr_to_string(ptr)
        self._scip_free_symbol(ptr)
        return result

    fn get_tool_info(self) raises -> Tuple[String, String]:
        """
        Get the tool name and version from metadata.

        Returns:
            Tuple of (tool_name, tool_version)
        """
        if not self._loaded:
            return ("", "")

        var name_ptr = self._scip_get_tool_name()
        var version_ptr = self._scip_get_tool_version()

        var name = "" if not name_ptr else _c_ptr_to_string(name_ptr)
        var version = "" if not version_ptr else _c_ptr_to_string(version_ptr)

        if name_ptr:
            self._scip_free_symbol(name_ptr)
        if version_ptr:
            self._scip_free_symbol(version_ptr)

        return (name, version)

    fn find_definition(self, file: String, line: Int, char: Int) raises -> ScipLocation:
        """
        Find the definition of the symbol at the given position.

        Args:
            file: Relative path to the source file
            line: Line number (0-based)
            char: Character offset (0-based)

        Returns:
            ScipLocation with the definition location, or empty if not found
        """
        var result = ScipLocation()
        if not self._loaded:
            return result

        # First get the symbol at the position
        var file_ptr = _string_to_c_ptr(file)
        var symbol_ptr = self._scip_get_symbol_at(file_ptr, Int32(line), Int32(char))

        if not symbol_ptr:
            return result

        # Get the find_definition function and call it
        var scip_find_definition = self._handle.get_function[
            fn(CString) -> UnsafePointer[UInt8, MutExternalOrigin]
        ]("scip_find_definition")

        # Convert symbol to CString for the call
        var symbol_str = _c_ptr_to_string(symbol_ptr)
        var symbol_cstr = _string_to_c_ptr(symbol_str)

        var loc_ptr = scip_find_definition(symbol_cstr)

        if loc_ptr:
            # Parse the location struct from the pointer
            # The struct layout is: file_path (ptr), start_line, start_char, end_line, end_char
            var file_path_ptr = loc_ptr.bitcast[CStringImmut]().load()
            if file_path_ptr:
                result.file_path = _c_ptr_to_string(file_path_ptr)

            # Read the integer fields (offset by pointer size)
            var int_ptr = loc_ptr.offset(8).bitcast[Int32]()
            result.start_line = int_ptr.load(0)
            result.start_char = int_ptr.load(1)
            result.end_line = int_ptr.load(2)
            result.end_char = int_ptr.load(3)

            # Free the location
            var scip_free_location = self._handle.get_function[
                fn(UnsafePointer[UInt8, MutExternalOrigin]) -> None
            ]("scip_free_location")
            scip_free_location(loc_ptr)

        self._scip_free_symbol(symbol_ptr)
        return result

    fn find_references(self, symbol: String) raises -> List[ScipLocation]:
        """
        Find all references to a symbol.

        Args:
            symbol: The symbol identifier string

        Returns:
            List of ScipLocation for each reference
        """
        var results = List[ScipLocation]()
        if not self._loaded:
            return results

        var scip_find_references = self._handle.get_function[
            fn(CString) -> UnsafePointer[UInt8, MutExternalOrigin]
        ]("scip_find_references")

        var symbol_ptr = _string_to_c_ptr(symbol)
        var list_ptr = scip_find_references(symbol_ptr)

        if not list_ptr:
            return results

        # Parse the LocationList struct: locations (ptr), count (int)
        var locs_ptr_ptr = list_ptr.bitcast[UnsafePointer[UInt8, MutExternalOrigin]]()
        var locs_ptr = locs_ptr_ptr.load()
        var count = list_ptr.offset(8).bitcast[Int32]().load()

        # Each location is 24 bytes (ptr + 4 ints)
        for i in range(int(count)):
            var loc = ScipLocation()
            var loc_base = locs_ptr.offset(i * 24)

            var file_ptr = loc_base.bitcast[CStringImmut]().load()
            if file_ptr:
                loc.file_path = _c_ptr_to_string(file_ptr)

            var int_base = loc_base.offset(8).bitcast[Int32]()
            loc.start_line = int_base.load(0)
            loc.start_char = int_base.load(1)
            loc.end_line = int_base.load(2)
            loc.end_char = int_base.load(3)

            results.append(loc)

        # Free the list
        var scip_free_locations = self._handle.get_function[
            fn(UnsafePointer[UInt8, MutExternalOrigin]) -> None
        ]("scip_free_locations")
        scip_free_locations(list_ptr)

        return results

    fn get_hover(self, file: String, line: Int, char: Int) raises -> ScipHoverInfo:
        """
        Get hover information (documentation, symbol kind) at a position.

        Args:
            file: Relative path to the source file
            line: Line number (0-based)
            char: Character offset (0-based)

        Returns:
            ScipHoverInfo with documentation and symbol info
        """
        var result = ScipHoverInfo()
        if not self._loaded:
            return result

        var scip_get_hover = self._handle.get_function[
            fn(CString, Int32, Int32) -> UnsafePointer[UInt8, MutExternalOrigin]
        ]("scip_get_hover")

        var file_ptr = _string_to_c_ptr(file)
        var hover_ptr = scip_get_hover(file_ptr, Int32(line), Int32(char))

        if not hover_ptr:
            return result

        # Parse HoverResult struct: documentation (ptr), symbol (ptr), kind (int)
        var doc_ptr = hover_ptr.bitcast[CStringImmut]().load()
        var sym_ptr = hover_ptr.offset(8).bitcast[CStringImmut]().load()
        var kind = hover_ptr.offset(16).bitcast[Int32]().load()

        if doc_ptr:
            result.documentation = _c_ptr_to_string(doc_ptr)
        if sym_ptr:
            result.symbol = _c_ptr_to_string(sym_ptr)
        result.kind = kind

        # Free the hover result
        var scip_free_hover = self._handle.get_function[
            fn(UnsafePointer[UInt8, MutExternalOrigin]) -> None
        ]("scip_free_hover")
        scip_free_hover(hover_ptr)

        return result

    fn get_symbol_at(self, file: String, line: Int, char: Int) raises -> String:
        """
        Get the symbol identifier at a specific position.

        Args:
            file: Relative path to the source file
            line: Line number (0-based)
            char: Character offset (0-based)

        Returns:
            Symbol identifier string, or empty if no symbol at position
        """
        if not self._loaded:
            return ""

        var file_ptr = _string_to_c_ptr(file)
        var symbol_ptr = self._scip_get_symbol_at(file_ptr, Int32(line), Int32(char))

        if not symbol_ptr:
            return ""

        var result = _c_ptr_to_string(symbol_ptr)
        self._scip_free_symbol(symbol_ptr)
        return result

