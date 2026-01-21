"""
FFI bindings for nExtract Zig library.

This module provides low-level FFI bindings to the Zig implementation.
Auto-generated bindings with manual memory management wrappers.
"""

from sys.ffi import external_call, DLHandle
from sys import sizeof
from memory import UnsafePointer
from pathlib import Path

# ========== Type Aliases ==========

alias CDoclingDocument = UnsafePointer[NoneType]
alias CElement = UnsafePointer[NoneType]
alias CPage = UnsafePointer[NoneType]

# ========== Element Types ==========

@value
struct ElementType:
    """Element type enumeration matching Zig definition."""
    var value: Int32
    
    alias Text = ElementType(0)
    alias Heading = ElementType(1)
    alias Paragraph = ElementType(2)
    alias Table = ElementType(3)
    alias Image = ElementType(4)
    alias Code = ElementType(5)
    alias Formula = ElementType(6)
    alias List = ElementType(7)
    alias ListItem = ElementType(8)
    alias BlockQuote = ElementType(9)
    alias HorizontalRule = ElementType(10)
    alias PageBreak = ElementType(11)
    
    fn __init__(inout self, value: Int32):
        self.value = value
    
    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value
    
    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value
    
    fn to_string(self) -> String:
        """Convert element type to string representation."""
        if self == Self.Text:
            return "text"
        elif self == Self.Heading:
            return "heading"
        elif self == Self.Paragraph:
            return "paragraph"
        elif self == Self.Table:
            return "table"
        elif self == Self.Image:
            return "image"
        elif self == Self.Code:
            return "code"
        elif self == Self.Formula:
            return "formula"
        elif self == Self.List:
            return "list"
        elif self == Self.ListItem:
            return "list_item"
        elif self == Self.BlockQuote:
            return "blockquote"
        elif self == Self.HorizontalRule:
            return "horizontal_rule"
        else:  # PageBreak
            return "page_break"

# ========== Geometric Types (C-compatible structs) ==========

@value
@register_passable("trivial")
struct CPoint:
    """C-compatible point structure."""
    var x: Float32
    var y: Float32
    
    fn __init__(inout self, x: Float32, y: Float32):
        self.x = x
        self.y = y

@value
@register_passable("trivial")
struct CSize:
    """C-compatible size structure."""
    var width: Float32
    var height: Float32
    
    fn __init__(inout self, width: Float32, height: Float32):
        self.width = width
        self.height = height

@value
@register_passable("trivial")
struct CBoundingBox:
    """C-compatible bounding box structure."""
    var x: Float32
    var y: Float32
    var width: Float32
    var height: Float32
    
    fn __init__(inout self, x: Float32, y: Float32, width: Float32, height: Float32):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    fn contains(self, point: CPoint) -> Bool:
        """Check if point is inside the bounding box."""
        return (point.x >= self.x and 
                point.x <= self.x + self.width and
                point.y >= self.y and 
                point.y <= self.y + self.height)
    
    fn intersects(self, other: Self) -> Bool:
        """Check if two bounding boxes intersect."""
        return not (self.x + self.width < other.x or
                   other.x + other.width < self.x or
                   self.y + self.height < other.y or
                   other.y + other.height < self.y)
    
    fn area(self) -> Float32:
        """Calculate area of bounding box."""
        return self.width * self.height

# ========== FFI Function Declarations ==========

struct NExtractFFI:
    """Low-level FFI interface to nExtract Zig library."""
    
    var _lib: DLHandle
    
    fn __init__(inout self) raises:
        """Initialize FFI by loading the shared library."""
        # Try different possible library names/paths
        var lib_paths = List[String]()
        lib_paths.append("libnextract.dylib")  # macOS
        lib_paths.append("libnextract.so")     # Linux
        lib_paths.append("nextract.dll")       # Windows
        lib_paths.append("./zig-out/lib/libnextract.dylib")
        lib_paths.append("./zig-out/lib/libnextract.so")
        
        var loaded = False
        for i in range(len(lib_paths)):
            try:
                self._lib = DLHandle(lib_paths[i])
                loaded = True
                break
            except:
                pass
        
        if not loaded:
            raise Error("Failed to load nExtract library. Build with: zig build")
    
    fn __del__(owned self):
        """Clean up library handle."""
        # DLHandle cleanup is automatic
        pass
    
    # ========== Document Management ==========
    
    fn document_create(self) -> CDoclingDocument:
        """Create a new document instance."""
        return external_call["nExtract_Document_create", CDoclingDocument](self._lib)
    
    fn document_destroy(self, doc: CDoclingDocument):
        """Destroy a document instance."""
        _ = external_call["nExtract_Document_destroy", NoneType](self._lib, doc)
    
    # ========== Element Management ==========
    
    fn element_create(self, element_type: ElementType) -> CElement:
        """Create a new element instance."""
        return external_call["nExtract_Element_create", CElement](
            self._lib, element_type.value
        )
    
    fn element_destroy(self, element: CElement):
        """Destroy an element instance."""
        _ = external_call["nExtract_Element_destroy", NoneType](self._lib, element)

# ========== Managed Wrappers ==========

struct CDoclingDocumentWrapper:
    """RAII wrapper for C document pointer."""
    var _handle: CDoclingDocument
    var _ffi: NExtractFFI
    var _valid: Bool
    
    fn __init__(inout self, ffi: NExtractFFI) raises:
        """Create a new document wrapper."""
        self._ffi = ffi
        self._handle = self._ffi.document_create()
        self._valid = True
        
        if not self._handle:
            raise Error("Failed to create document")
    
    fn __del__(owned self):
        """Automatically destroy document on cleanup."""
        if self._valid and self._handle:
            self._ffi.document_destroy(self._handle)
            self._valid = False
    
    fn handle(self) -> CDoclingDocument:
        """Get raw handle (for FFI calls)."""
        return self._handle
    
    fn invalidate(inout self):
        """Mark handle as invalid (ownership transferred)."""
        self._valid = False

struct CElementWrapper:
    """RAII wrapper for C element pointer."""
    var _handle: CElement
    var _ffi: NExtractFFI
    var _valid: Bool
    
    fn __init__(inout self, ffi: NExtractFFI, element_type: ElementType) raises:
        """Create a new element wrapper."""
        self._ffi = ffi
        self._handle = self._ffi.element_create(element_type)
        self._valid = True
        
        if not self._handle:
            raise Error("Failed to create element")
    
    fn __del__(owned self):
        """Automatically destroy element on cleanup."""
        if self._valid and self._handle:
            self._ffi.element_destroy(self._handle)
            self._valid = False
    
    fn handle(self) -> CElement:
        """Get raw handle (for FFI calls)."""
        return self._handle
    
    fn invalidate(inout self):
        """Mark handle as invalid (ownership transferred)."""
        self._valid = False

# ========== Utility Functions ==========

fn test_ffi_availability() -> Bool:
    """Test if FFI library is available and working."""
    try:
        var ffi = NExtractFFI()
        var doc = ffi.document_create()
        if doc:
            ffi.document_destroy(doc)
            return True
        return False
    except:
        return False

fn get_library_version() -> String:
    """Get nExtract library version (placeholder)."""
    return "1.0.0-dev"
