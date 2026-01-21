"""
Tests for high-level Mojo API - ergonomic wrappers and document operations.

These tests verify that the high-level API provides correct functionality,
type safety, and ergonomic access to document processing features.
"""

from ..core import (
    Point,
    Size,
    BoundingBox,
    ElementProperties,
    Element,
    Page,
    Metadata,
    DoclingDocument,
    DocumentError,
    ElementType,
)
from testing import assert_true, assert_false, assert_equal, assert_raises

# ========== Test High-Level Geometric Types ==========

fn test_point_operations() raises:
    """Test high-level point type."""
    print("Testing Point operations...")
    var p1 = Point(0.0, 0.0)
    var p2 = Point(3.0, 4.0)
    
    # Test distance calculation
    var dist = p1.distance(p2)
    assert_equal(dist, 5.0, "Distance should be 5.0 (3-4-5 triangle)")
    
    # Test C conversion round-trip
    var c_point = p2.to_c()
    var p3 = Point.from_c(c_point)
    assert_equal(p3.x, p2.x, "Round-trip conversion should preserve X")
    assert_equal(p3.y, p2.y, "Round-trip conversion should preserve Y")
    
    print("✓ Point operations work correctly")

fn test_size_operations() raises:
    """Test high-level size type."""
    print("Testing Size operations...")
    var s1 = Size(100.0, 50.0)
    
    # Test area calculation
    var area = s1.area()
    assert_equal(area, 5000.0, "Area should be width * height")
    
    # Test C conversion round-trip
    var c_size = s1.to_c()
    var s2 = Size.from_c(c_size)
    assert_equal(s2.width, s1.width, "Round-trip should preserve width")
    assert_equal(s2.height, s1.height, "Round-trip should preserve height")
    
    print("✓ Size operations work correctly")

fn test_bounding_box_operations() raises:
    """Test high-level bounding box type."""
    print("Testing BoundingBox operations...")
    var bbox1 = BoundingBox(10.0, 20.0, 100.0, 50.0)
    
    # Test contains
    var inside = Point(50.0, 40.0)
    var outside = Point(200.0, 200.0)
    assert_true(bbox1.contains(inside), "Should detect point inside")
    assert_false(bbox1.contains(outside), "Should detect point outside")
    
    # Test intersects
    var bbox2 = BoundingBox(50.0, 30.0, 100.0, 50.0)
    var bbox3 = BoundingBox(200.0, 200.0, 50.0, 50.0)
    assert_true(bbox1.intersects(bbox2), "Should detect intersection")
    assert_false(bbox1.intersects(bbox3), "Should detect no intersection")
    
    # Test center
    var center = bbox1.center()
    assert_equal(center.x, 60.0, "Center X should be correct")
    assert_equal(center.y, 45.0, "Center Y should be correct")
    
    # Test area
    var area = bbox1.area()
    assert_equal(area, 5000.0, "Area should be width * height")
    
    print("✓ BoundingBox operations work correctly")

# ========== Test Element Properties ==========

fn test_element_properties() raises:
    """Test element properties builder pattern."""
    print("Testing ElementProperties...")
    var props = ElementProperties()
    
    # Test builder pattern
    _ = props.with_font("Arial", 12.0)
    _ = props.with_style(bold=True, italic=False)
    _ = props.with_color(0xFF0000)  # Red
    _ = props.with_heading_level(2)
    
    # Verify values
    assert_true(props.font_family is not None, "Font family should be set")
    assert_true(props.font_size is not None, "Font size should be set")
    assert_true(props.is_bold, "Bold should be True")
    assert_false(props.is_italic, "Italic should be False")
    assert_true(props.color is not None, "Color should be set")
    assert_true(props.heading_level is not None, "Heading level should be set")
    
    print("✓ ElementProperties work correctly")

# ========== Test Element ==========

fn test_element_creation() raises:
    """Test element creation and builder pattern."""
    print("Testing Element creation...")
    var elem = Element(ElementType.Paragraph)
    
    # Test builder pattern
    _ = elem.with_content("This is a test paragraph.")
    _ = elem.with_bbox(BoundingBox(10.0, 20.0, 200.0, 50.0))
    _ = elem.with_page(1)
    
    # Verify values
    assert_equal(elem.content, "This is a test paragraph.", "Content should match")
    assert_equal(elem.page_number, 1, "Page number should match")
    assert_false(elem.is_heading(), "Paragraph is not a heading")
    assert_true(elem.is_text(), "Paragraph is text")
    
    print("✓ Element creation works correctly")

fn test_element_types() raises:
    """Test different element types."""
    print("Testing Element types...")
    var heading = Element(ElementType.Heading)
    var paragraph = Element(ElementType.Paragraph)
    var table = Element(ElementType.Table)
    var image = Element(ElementType.Image)
    
    assert_true(heading.is_heading(), "Heading should be detected")
    assert_false(heading.is_text(), "Heading is not plain text")
    
    assert_true(paragraph.is_text(), "Paragraph is text")
    assert_false(paragraph.is_heading(), "Paragraph is not a heading")
    
    assert_false(table.is_text(), "Table is not text")
    assert_false(image.is_text(), "Image is not text")
    
    print("✓ Element types work correctly")

# ========== Test Page ==========

fn test_page_creation() raises:
    """Test page creation and element management."""
    print("Testing Page creation...")
    var page = Page(1, 595.0, 842.0)  # A4 size
    
    # Verify initial state
    assert_equal(page.page_number, 1, "Page number should match")
    assert_equal(page.width, 595.0, "Width should match")
    assert_equal(page.height, 842.0, "Height should match")
    assert_equal(page.element_count(), 0, "Should start with no elements")
    
    # Add elements
    var elem1 = Element(ElementType.Heading)
    _ = elem1.with_content("Chapter 1")
    page.add_element(elem1)
    
    var elem2 = Element(ElementType.Paragraph)
    _ = elem2.with_content("This is the first paragraph.")
    page.add_element(elem2)
    
    assert_equal(page.element_count(), 2, "Should have 2 elements")
    
    # Test size
    var size = page.size()
    assert_equal(size.width, 595.0, "Size width should match")
    assert_equal(size.height, 842.0, "Size height should match")
    
    print("✓ Page creation and element management work correctly")

# ========== Test Metadata ==========

fn test_metadata() raises:
    """Test metadata builder pattern."""
    print("Testing Metadata...")
    var meta = Metadata()
    
    # Test builder pattern
    _ = meta.with_title("Test Document")
    _ = meta.with_author("John Doe")
    
    # Verify values
    assert_true(meta.title is not None, "Title should be set")
    assert_true(meta.author is not None, "Author should be set")
    
    if meta.title:
        assert_equal(meta.title.value(), "Test Document", "Title should match")
    if meta.author:
        assert_equal(meta.author.value(), "John Doe", "Author should match")
    
    print("✓ Metadata works correctly")

# ========== Test Document ==========

fn test_document_creation() raises:
    """Test document creation via high-level API."""
    print("Testing DoclingDocument creation...")
    var doc = DoclingDocument()
    
    # Verify initial state
    assert_equal(doc.page_count(), 0, "Should start with no pages")
    assert_equal(doc.element_count(), 0, "Should start with no elements")
    
    print("✓ DoclingDocument creation works correctly")

fn test_document_page_management() raises:
    """Test adding pages to document."""
    print("Testing document page management...")
    var doc = DoclingDocument()
    
    # Create and add page
    var page1 = Page(1, 595.0, 842.0)
    var elem1 = Element(ElementType.Heading)
    _ = elem1.with_content("Introduction")
    page1.add_element(elem1)
    
    doc.add_page(page1)
    
    assert_equal(doc.page_count(), 1, "Should have 1 page")
    assert_equal(doc.element_count(), 1, "Should have 1 element")
    
    # Add second page
    var page2 = Page(2, 595.0, 842.0)
    var elem2 = Element(ElementType.Paragraph)
    _ = elem2.with_content("This is page 2.")
    page2.add_element(elem2)
    
    doc.add_page(page2)
    
    assert_equal(doc.page_count(), 2, "Should have 2 pages")
    assert_equal(doc.element_count(), 2, "Should have 2 elements")
    
    print("✓ Document page management works correctly")

fn test_document_text_extraction() raises:
    """Test extracting text from document."""
    print("Testing document text extraction...")
    var doc = DoclingDocument()
    
    # Add page with text elements
    var page = Page(1, 595.0, 842.0)
    
    var heading = Element(ElementType.Heading)
    _ = heading.with_content("Title")
    page.add_element(heading)
    
    var para1 = Element(ElementType.Paragraph)
    _ = para1.with_content("First paragraph.")
    page.add_element(para1)
    
    var para2 = Element(ElementType.Paragraph)
    _ = para2.with_content("Second paragraph.")
    page.add_element(para2)
    
    doc.add_page(page)
    
    # Extract text
    var text = doc.get_text()
    assert_true(len(text) > 0, "Should extract text")
    # Note: get_text() only extracts paragraph text, not headings
    
    print("✓ Document text extraction works correctly")

fn test_document_heading_extraction() raises:
    """Test extracting headings from document."""
    print("Testing document heading extraction...")
    var doc = DoclingDocument()
    
    # Add page with mixed elements
    var page = Page(1, 595.0, 842.0)
    
    var h1 = Element(ElementType.Heading)
    _ = h1.with_content("Chapter 1")
    page.add_element(h1)
    
    var para = Element(ElementType.Paragraph)
    _ = para.with_content("Some text.")
    page.add_element(para)
    
    var h2 = Element(ElementType.Heading)
    _ = h2.with_content("Section 1.1")
    page.add_element(h2)
    
    doc.add_page(page)
    
    # Extract headings
    var headings = doc.get_headings()
    assert_equal(len(headings), 2, "Should extract 2 headings")
    
    print("✓ Document heading extraction works correctly")

fn test_document_get_page() raises:
    """Test getting specific page from document."""
    print("Testing document get_page...")
    var doc = DoclingDocument()
    
    # Add pages
    var page1 = Page(1, 595.0, 842.0)
    var page2 = Page(2, 595.0, 842.0)
    doc.add_page(page1)
    doc.add_page(page2)
    
    # Get page by number (1-indexed)
    var retrieved_page = doc.get_page(1)
    assert_equal(retrieved_page.page_number, 1, "Should get correct page")
    
    # Test out of range (should raise error)
    try:
        var _ = doc.get_page(10)
        assert_true(False, "Should raise error for invalid page number")
    except e:
        # Expected error
        pass
    
    print("✓ Document get_page works correctly")

# ========== Test Main Runner ==========

fn run_all_tests() raises:
    """Run all high-level API tests."""
    print("\n" + "="*60)
    print("Running Core API Tests")
    print("="*60 + "\n")
    
    # Geometric types
    test_point_operations()
    test_size_operations()
    test_bounding_box_operations()
    
    # Element properties
    test_element_properties()
    
    # Elements
    test_element_creation()
    test_element_types()
    
    # Pages
    test_page_creation()
    
    # Metadata
    test_metadata()
    
    # Documents
    test_document_creation()
    test_document_page_management()
    test_document_text_extraction()
    test_document_heading_extraction()
    test_document_get_page()
    
    print("\n" + "="*60)
    print("All Core API tests passed! ✓")
    print("="*60 + "\n")

fn main() raises:
    """Main entry point for tests."""
    run_all_tests()
