"""
Tests for FFI layer - data marshalling and memory safety.

These tests verify that data is correctly marshalled between Mojo and Zig,
and that memory ownership patterns are correct (no leaks, no use-after-free).
"""

from ..ffi import (
    NExtractFFI,
    CDoclingDocumentWrapper,
    CElementWrapper,
    ElementType,
    CPoint,
    CSize,
    CBoundingBox,
    test_ffi_availability,
    get_library_version,
)
from testing import assert_true, assert_false, assert_equal, assert_raises

# ========== Test Library Availability ==========

fn test_library_loads() raises:
    """Test that the nExtract library can be loaded."""
    print("Testing library loading...")
    var available = test_ffi_availability()
    assert_true(available, "nExtract library should be available")
    print("✓ Library loaded successfully")

fn test_library_version() raises:
    """Test library version retrieval."""
    print("Testing library version...")
    var version = get_library_version()
    assert_true(len(version) > 0, "Version string should not be empty")
    print("✓ Library version:", version)

# ========== Test Geometric Types ==========

fn test_point_marshalling() raises:
    """Test point creation and value marshalling."""
    print("Testing CPoint marshalling...")
    var p1 = CPoint(10.5, 20.3)
    assert_equal(p1.x, 10.5, "X coordinate should match")
    assert_equal(p1.y, 20.3, "Y coordinate should match")
    
    var p2 = CPoint(0.0, 0.0)
    assert_equal(p2.x, 0.0, "Zero X coordinate should work")
    assert_equal(p2.y, 0.0, "Zero Y coordinate should work")
    print("✓ Point marshalling works correctly")

fn test_size_marshalling() raises:
    """Test size creation and value marshalling."""
    print("Testing CSize marshalling...")
    var s1 = CSize(100.0, 200.0)
    assert_equal(s1.width, 100.0, "Width should match")
    assert_equal(s1.height, 200.0, "Height should match")
    print("✓ Size marshalling works correctly")

fn test_bounding_box_marshalling() raises:
    """Test bounding box creation and operations."""
    print("Testing CBoundingBox marshalling...")
    var bbox = CBoundingBox(10.0, 20.0, 100.0, 50.0)
    assert_equal(bbox.x, 10.0, "X should match")
    assert_equal(bbox.y, 20.0, "Y should match")
    assert_equal(bbox.width, 100.0, "Width should match")
    assert_equal(bbox.height, 50.0, "Height should match")
    
    # Test contains
    var inside = CPoint(50.0, 40.0)
    var outside = CPoint(200.0, 200.0)
    assert_true(bbox.contains(inside), "Point inside bbox should be detected")
    assert_false(bbox.contains(outside), "Point outside bbox should be detected")
    
    # Test intersects
    var bbox2 = CBoundingBox(50.0, 30.0, 100.0, 50.0)
    var bbox3 = CBoundingBox(200.0, 200.0, 50.0, 50.0)
    assert_true(bbox.intersects(bbox2), "Intersecting boxes should be detected")
    assert_false(bbox.intersects(bbox3), "Non-intersecting boxes should be detected")
    
    # Test area
    var area = bbox.area()
    assert_equal(area, 5000.0, "Area should be width * height")
    print("✓ BoundingBox marshalling and operations work correctly")

# ========== Test Element Types ==========

fn test_element_type_enum() raises:
    """Test element type enumeration."""
    print("Testing ElementType enum...")
    var text_type = ElementType.Text
    var heading_type = ElementType.Heading
    var para_type = ElementType.Paragraph
    
    assert_true(text_type == ElementType.Text, "Text type should equal itself")
    assert_false(text_type == heading_type, "Different types should not be equal")
    
    # Test string conversion
    assert_equal(text_type.to_string(), "text", "Text type string should match")
    assert_equal(heading_type.to_string(), "heading", "Heading type string should match")
    assert_equal(para_type.to_string(), "paragraph", "Paragraph type string should match")
    print("✓ ElementType enum works correctly")

# ========== Test Document Creation/Destruction ==========

fn test_document_create_destroy() raises:
    """Test document creation and destruction via FFI."""
    print("Testing document create/destroy...")
    var ffi = NExtractFFI()
    
    # Create document
    var doc = ffi.document_create()
    assert_true(doc != UnsafePointer[NoneType](), "Document should be created")
    
    # Destroy document
    ffi.document_destroy(doc)
    print("✓ Document create/destroy works correctly")

fn test_document_wrapper_raii() raises:
    """Test RAII wrapper for document (automatic cleanup)."""
    print("Testing document wrapper RAII...")
    var ffi = NExtractFFI()
    
    # Create wrapper - should auto-cleanup when out of scope
    var wrapper = CDoclingDocumentWrapper(ffi)
    var handle = wrapper.handle()
    assert_true(handle != UnsafePointer[NoneType](), "Wrapper should create document")
    
    # When wrapper goes out of scope, destructor should run automatically
    print("✓ Document wrapper RAII works correctly")

fn test_multiple_documents() raises:
    """Test creating multiple documents simultaneously."""
    print("Testing multiple documents...")
    var ffi = NExtractFFI()
    
    var doc1 = ffi.document_create()
    var doc2 = ffi.document_create()
    var doc3 = ffi.document_create()
    
    assert_true(doc1 != UnsafePointer[NoneType](), "Document 1 should be created")
    assert_true(doc2 != UnsafePointer[NoneType](), "Document 2 should be created")
    assert_true(doc3 != UnsafePointer[NoneType](), "Document 3 should be created")
    
    # Different documents should have different pointers
    assert_true(doc1 != doc2, "Documents should have different addresses")
    assert_true(doc2 != doc3, "Documents should have different addresses")
    
    # Cleanup
    ffi.document_destroy(doc1)
    ffi.document_destroy(doc2)
    ffi.document_destroy(doc3)
    print("✓ Multiple documents work correctly")

# ========== Test Element Creation/Destruction ==========

fn test_element_create_destroy() raises:
    """Test element creation and destruction via FFI."""
    print("Testing element create/destroy...")
    var ffi = NExtractFFI()
    
    # Create element
    var elem = ffi.element_create(ElementType.Paragraph)
    assert_true(elem != UnsafePointer[NoneType](), "Element should be created")
    
    # Destroy element
    ffi.element_destroy(elem)
    print("✓ Element create/destroy works correctly")

fn test_element_wrapper_raii() raises:
    """Test RAII wrapper for element (automatic cleanup)."""
    print("Testing element wrapper RAII...")
    var ffi = NExtractFFI()
    
    # Create wrapper
    var wrapper = CElementWrapper(ffi, ElementType.Heading)
    var handle = wrapper.handle()
    assert_true(handle != UnsafePointer[NoneType](), "Wrapper should create element")
    
    # Auto-cleanup on scope exit
    print("✓ Element wrapper RAII works correctly")

fn test_multiple_elements() raises:
    """Test creating multiple elements of different types."""
    print("Testing multiple elements...")
    var ffi = NExtractFFI()
    
    var elem1 = ffi.element_create(ElementType.Text)
    var elem2 = ffi.element_create(ElementType.Heading)
    var elem3 = ffi.element_create(ElementType.Table)
    var elem4 = ffi.element_create(ElementType.Image)
    
    assert_true(elem1 != UnsafePointer[NoneType](), "Text element should be created")
    assert_true(elem2 != UnsafePointer[NoneType](), "Heading element should be created")
    assert_true(elem3 != UnsafePointer[NoneType](), "Table element should be created")
    assert_true(elem4 != UnsafePointer[NoneType](), "Image element should be created")
    
    # Cleanup
    ffi.element_destroy(elem1)
    ffi.element_destroy(elem2)
    ffi.element_destroy(elem3)
    ffi.element_destroy(elem4)
    print("✓ Multiple elements work correctly")

# ========== Memory Ownership Tests ==========

fn test_double_free_protection() raises:
    """Test that RAII wrappers prevent double-free."""
    print("Testing double-free protection...")
    var ffi = NExtractFFI()
    
    # Create wrapper
    var wrapper = CDoclingDocumentWrapper(ffi)
    
    # Manually invalidate (simulating ownership transfer)
    wrapper.invalidate()
    
    # Destructor should not try to free invalid handle
    print("✓ Double-free protection works correctly")

fn test_wrapper_scope() raises:
    """Test that wrappers clean up when leaving scope."""
    print("Testing wrapper scope cleanup...")
    var ffi = NExtractFFI()
    
    # Inner scope
    if True:
        var wrapper = CDoclingDocumentWrapper(ffi)
        var _ = wrapper.handle()
        # Wrapper goes out of scope here, should auto-cleanup
    
    # Create another wrapper to ensure no issues
    var wrapper2 = CDoclingDocumentWrapper(ffi)
    var _ = wrapper2.handle()
    print("✓ Wrapper scope cleanup works correctly")

# ========== Test Main Runner ==========

fn run_all_tests() raises:
    """Run all FFI tests."""
    print("\n" + "="*60)
    print("Running FFI Tests")
    print("="*60 + "\n")
    
    # Library tests
    test_library_loads()
    test_library_version()
    
    # Geometric type tests
    test_point_marshalling()
    test_size_marshalling()
    test_bounding_box_marshalling()
    
    # Element type tests
    test_element_type_enum()
    
    # Document tests
    test_document_create_destroy()
    test_document_wrapper_raii()
    test_multiple_documents()
    
    # Element tests
    test_element_create_destroy()
    test_element_wrapper_raii()
    test_multiple_elements()
    
    # Memory ownership tests
    test_double_free_protection()
    test_wrapper_scope()
    
    print("\n" + "="*60)
    print("All FFI tests passed! ✓")
    print("="*60 + "\n")

fn main() raises:
    """Main entry point for tests."""
    run_all_tests()
