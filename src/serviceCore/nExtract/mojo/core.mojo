"""
High-level Mojo API for nExtract document processing.

This module provides ergonomic, type-safe wrappers around the FFI layer,
with proper memory management and Pythonic interfaces.
"""

from .ffi import (
    NExtractFFI,
    CDoclingDocumentWrapper,
    CElementWrapper,
    ElementType,
    CPoint,
    CSize,
    CBoundingBox,
)
from collections import List, Dict, Optional

# ========== Error Types ==========

struct DocumentError(Error):
    """Base error type for document operations."""
    var message: String
    
    fn __init__(inout self, message: String):
        self.message = message
    
    fn __str__(self) -> String:
        return "DocumentError: " + self.message

struct ParsingError(DocumentError):
    """Error during document parsing."""
    pass

struct ConversionError(DocumentError):
    """Error during document conversion."""
    pass

# ========== High-Level Geometric Types ==========

@value
struct Point:
    """High-level point with floating-point coordinates."""
    var x: Float32
    var y: Float32
    
    fn __init__(inout self, x: Float32, y: Float32):
        self.x = x
        self.y = y
    
    fn to_c(self) -> CPoint:
        """Convert to C-compatible representation."""
        return CPoint(self.x, self.y)
    
    @staticmethod
    fn from_c(c_point: CPoint) -> Point:
        """Create from C representation."""
        return Point(c_point.x, c_point.y)
    
    fn distance(self, other: Point) -> Float32:
        """Calculate Euclidean distance to another point."""
        var dx = self.x - other.x
        var dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)
    
    fn __str__(self) -> String:
        return "Point(" + String(self.x) + ", " + String(self.y) + ")"

@value
struct Size:
    """High-level size with width and height."""
    var width: Float32
    var height: Float32
    
    fn __init__(inout self, width: Float32, height: Float32):
        self.width = width
        self.height = height
    
    fn to_c(self) -> CSize:
        """Convert to C-compatible representation."""
        return CSize(self.width, self.height)
    
    @staticmethod
    fn from_c(c_size: CSize) -> Size:
        """Create from C representation."""
        return Size(c_size.width, c_size.height)
    
    fn area(self) -> Float32:
        """Calculate area."""
        return self.width * self.height
    
    fn __str__(self) -> String:
        return "Size(" + String(self.width) + ", " + String(self.height) + ")"

@value
struct BoundingBox:
    """High-level bounding box representing a rectangular region."""
    var x: Float32
    var y: Float32
    var width: Float32
    var height: Float32
    
    fn __init__(inout self, x: Float32, y: Float32, width: Float32, height: Float32):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    fn to_c(self) -> CBoundingBox:
        """Convert to C-compatible representation."""
        return CBoundingBox(self.x, self.y, self.width, self.height)
    
    @staticmethod
    fn from_c(c_bbox: CBoundingBox) -> BoundingBox:
        """Create from C representation."""
        return BoundingBox(c_bbox.x, c_bbox.y, c_bbox.width, c_bbox.height)
    
    fn contains(self, point: Point) -> Bool:
        """Check if point is inside the bounding box."""
        return self.to_c().contains(point.to_c())
    
    fn intersects(self, other: BoundingBox) -> Bool:
        """Check if two bounding boxes intersect."""
        return self.to_c().intersects(other.to_c())
    
    fn area(self) -> Float32:
        """Calculate area of bounding box."""
        return self.width * self.height
    
    fn center(self) -> Point:
        """Get center point of bounding box."""
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)
    
    fn __str__(self) -> String:
        return ("BoundingBox(" + String(self.x) + ", " + String(self.y) + 
                ", " + String(self.width) + ", " + String(self.height) + ")")

# ========== Document Element Properties ==========

struct ElementProperties:
    """Properties for document elements (formatting, style, etc)."""
    var font_family: Optional[String]
    var font_size: Optional[Float32]
    var is_bold: Bool
    var is_italic: Bool
    var is_underline: Bool
    var color: Optional[UInt32]  # RGB color
    var heading_level: Optional[Int]  # 1-6 for headings
    
    fn __init__(inout self):
        self.font_family = None
        self.font_size = None
        self.is_bold = False
        self.is_italic = False
        self.is_underline = False
        self.color = None
        self.heading_level = None
    
    fn with_font(inout self, family: String, size: Float32) -> Self:
        """Set font properties."""
        self.font_family = family
        self.font_size = size
        return self
    
    fn with_style(inout self, bold: Bool = False, italic: Bool = False, 
                  underline: Bool = False) -> Self:
        """Set text style."""
        self.is_bold = bold
        self.is_italic = italic
        self.is_underline = underline
        return self
    
    fn with_color(inout self, rgb: UInt32) -> Self:
        """Set color (RGB format)."""
        self.color = rgb
        return self
    
    fn with_heading_level(inout self, level: Int) -> Self:
        """Set heading level (1-6)."""
        if level >= 1 and level <= 6:
            self.heading_level = level
        return self

# ========== Document Element ==========

struct Element:
    """High-level document element with content and properties."""
    var element_type: ElementType
    var bbox: BoundingBox
    var content: String
    var properties: ElementProperties
    var page_number: Int
    
    fn __init__(inout self, element_type: ElementType):
        self.element_type = element_type
        self.bbox = BoundingBox(0, 0, 0, 0)
        self.content = ""
        self.properties = ElementProperties()
        self.page_number = 0
    
    fn with_content(inout self, content: String) -> Self:
        """Set element content."""
        self.content = content
        return self
    
    fn with_bbox(inout self, bbox: BoundingBox) -> Self:
        """Set bounding box."""
        self.bbox = bbox
        return self
    
    fn with_page(inout self, page_num: Int) -> Self:
        """Set page number."""
        self.page_number = page_num
        return self
    
    fn is_heading(self) -> Bool:
        """Check if element is a heading."""
        return self.element_type == ElementType.Heading
    
    fn is_text(self) -> Bool:
        """Check if element is text or paragraph."""
        return (self.element_type == ElementType.Text or 
                self.element_type == ElementType.Paragraph)
    
    fn __str__(self) -> String:
        return (self.element_type.to_string() + ": " + 
                self.content[:50] + "..." if len(self.content) > 50 else self.content)

# ========== Document Page ==========

struct Page:
    """Document page with metadata and elements."""
    var page_number: Int
    var width: Float32
    var height: Float32
    var rotation: Int  # 0, 90, 180, 270
    var elements: List[Element]
    
    fn __init__(inout self, page_number: Int, width: Float32, height: Float32):
        self.page_number = page_number
        self.width = width
        self.height = height
        self.rotation = 0
        self.elements = List[Element]()
    
    fn add_element(inout self, element: Element) raises:
        """Add an element to the page."""
        self.elements.append(element)
    
    fn element_count(self) -> Int:
        """Get number of elements on page."""
        return len(self.elements)
    
    fn size(self) -> Size:
        """Get page size."""
        return Size(self.width, self.height)
    
    fn __str__(self) -> String:
        return ("Page " + String(self.page_number) + 
                " (" + String(self.width) + "x" + String(self.height) + 
                ") - " + String(len(self.elements)) + " elements")

# ========== Document Metadata ==========

struct Metadata:
    """Document metadata (title, author, dates, etc)."""
    var title: Optional[String]
    var author: Optional[String]
    var subject: Optional[String]
    var keywords: Optional[String]
    var creator: Optional[String]
    var producer: Optional[String]
    var creation_date: Optional[Int64]  # Unix timestamp
    var modification_date: Optional[Int64]
    
    fn __init__(inout self):
        self.title = None
        self.author = None
        self.subject = None
        self.keywords = None
        self.creator = None
        self.producer = None
        self.creation_date = None
        self.modification_date = None
    
    fn with_title(inout self, title: String) -> Self:
        """Set document title."""
        self.title = title
        return self
    
    fn with_author(inout self, author: String) -> Self:
        """Set document author."""
        self.author = author
        return self
    
    fn __str__(self) -> String:
        var result = "Metadata("
        if self.title:
            result += "title=" + self.title.value() + ", "
        if self.author:
            result += "author=" + self.author.value() + ", "
        result += ")"
        return result

# ========== Main Document Class ==========

struct DoclingDocument:
    """
    High-level document wrapper providing ergonomic API for document processing.
    
    This is the main interface for working with documents in nExtract.
    It manages the underlying Zig implementation via FFI and provides
    memory-safe, Pythonic access to document structure and content.
    """
    var _ffi: NExtractFFI
    var _handle: CDoclingDocumentWrapper
    var pages: List[Page]
    var metadata: Metadata
    var elements: List[Element]  # Flattened list of all elements
    
    fn __init__(inout self) raises:
        """Create a new empty document."""
        self._ffi = NExtractFFI()
        self._handle = CDoclingDocumentWrapper(self._ffi)
        self.pages = List[Page]()
        self.metadata = Metadata()
        self.elements = List[Element]()
    
    fn add_page(inout self, page: Page) raises:
        """Add a page to the document."""
        self.pages.append(page)
        
        # Add page elements to flattened element list
        for i in range(len(page.elements)):
            self.elements.append(page.elements[i])
    
    fn page_count(self) -> Int:
        """Get number of pages in document."""
        return len(self.pages)
    
    fn element_count(self) -> Int:
        """Get total number of elements across all pages."""
        return len(self.elements)
    
    fn get_page(self, page_number: Int) raises -> Page:
        """Get page by number (1-indexed)."""
        if page_number < 1 or page_number > len(self.pages):
            raise DocumentError("Page number out of range")
        return self.pages[page_number - 1]
    
    fn get_text(self) -> String:
        """Extract all text from document."""
        var result = String("")
        for i in range(len(self.elements)):
            var elem = self.elements[i]
            if elem.is_text():
                result += elem.content + "\n"
        return result
    
    fn get_headings(self) -> List[Element]:
        """Get all heading elements from document."""
        var headings = List[Element]()
        for i in range(len(self.elements)):
            var elem = self.elements[i]
            if elem.is_heading():
                headings.append(elem)
        return headings
    
    fn __str__(self) -> String:
        return ("DoclingDocument(" + String(len(self.pages)) + " pages, " + 
                String(len(self.elements)) + " elements)")

# ========== Result Type ==========

@value
struct Result[T: AnyType, E: Error]:
    """Result type for operations that may fail."""
    var _value: Optional[T]
    var _error: Optional[E]
    
    fn __init__(inout self, value: T):
        self._value = value
        self._error = None
    
    fn __init__(inout self, error: E):
        self._value = None
        self._error = error
    
    fn is_ok(self) -> Bool:
        """Check if result contains a value."""
        return self._value is not None
    
    fn is_err(self) -> Bool:
        """Check if result contains an error."""
        return self._error is not None
    
    fn unwrap(self) raises -> T:
        """Get the value or raise the error."""
        if self._error:
            raise self._error.value()
        if self._value:
            return self._value.value()
        raise Error("Result is neither Ok nor Err")
    
    fn unwrap_or(self, default: T) -> T:
        """Get the value or return default."""
        if self._value:
            return self._value.value()
        return default
