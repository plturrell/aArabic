# Test Suite for List[T]
# Day 29: Collections - List Tests

from testing import assert_equal, assert_true, assert_false
from collections import List

fn test_list_initialization():
    """Test list initialization and basic properties."""
    var list = List[Int]()
    
    assert_equal(list.len(), 0, "Empty list should have size 0")
    assert_true(list.is_empty(), "Empty list should report is_empty")
    assert_equal(list.get_capacity(), 8, "Default capacity should be 8")
    
    print("✓ test_list_initialization passed")

fn test_list_append():
    """Test appending elements to list."""
    var list = List[Int]()
    
    list.append(1)
    assert_equal(list.len(), 1, "List should have 1 element")
    assert_equal(list[0], 1, "First element should be 1")
    
    list.append(2)
    list.append(3)
    assert_equal(list.len(), 3, "List should have 3 elements")
    assert_equal(list[2], 3, "Third element should be 3")
    
    print("✓ test_list_append passed")

fn test_list_insert():
    """Test inserting elements at specific positions."""
    var list = List[Int]()
    
    list.append(1)
    list.append(3)
    list.insert(1, 2)  # Insert 2 between 1 and 3
    
    assert_equal(list.len(), 3, "List should have 3 elements")
    assert_equal(list[0], 1, "First element should be 1")
    assert_equal(list[1], 2, "Second element should be 2")
    assert_equal(list[2], 3, "Third element should be 3")
    
    print("✓ test_list_insert passed")

fn test_list_remove():
    """Test removing elements from list."""
    var list = List[Int]()
    
    list.append(1)
    list.append(2)
    list.append(3)
    
    let removed = list.remove(1)  # Remove middle element
    
    assert_equal(removed, 2, "Removed element should be 2")
    assert_equal(list.len(), 2, "List should have 2 elements")
    assert_equal(list[0], 1, "First element should be 1")
    assert_equal(list[1], 3, "Second element should be 3")
    
    print("✓ test_list_remove passed")

fn test_list_pop():
    """Test popping elements from end of list."""
    var list = List[Int]()
    
    list.append(1)
    list.append(2)
    list.append(3)
    
    let popped = list.pop()
    
    assert_equal(popped, 3, "Popped element should be 3")
    assert_equal(list.len(), 2, "List should have 2 elements")
    assert_equal(list.back(), 2, "Last element should now be 2")
    
    print("✓ test_list_pop passed")

fn test_list_indexing():
    """Test element access via indexing."""
    var list = List[Int]()
    
    list.append(10)
    list.append(20)
    list.append(30)
    
    assert_equal(list[0], 10, "Element at index 0 should be 10")
    assert_equal(list[1], 20, "Element at index 1 should be 20")
    assert_equal(list[2], 30, "Element at index 2 should be 30")
    
    list[1] = 25  # Modify element
    assert_equal(list[1], 25, "Modified element should be 25")
    
    print("✓ test_list_indexing passed")

fn test_list_contains():
    """Test checking if list contains an element."""
    var list = List[Int]()
    
    list.append(1)
    list.append(2)
    list.append(3)
    
    assert_true(list.contains(2), "List should contain 2")
    assert_false(list.contains(5), "List should not contain 5")
    
    print("✓ test_list_contains passed")

fn test_list_find():
    """Test finding index of elements."""
    var list = List[Int]()
    
    list.append(10)
    list.append(20)
    list.append(30)
    list.append(20)  # Duplicate
    
    assert_equal(list.find(20), 1, "First occurrence of 20 at index 1")
    assert_equal(list.find(30), 2, "30 should be at index 2")
    assert_equal(list.find(99), -1, "99 should not be found")
    
    print("✓ test_list_find passed")

fn test_list_count():
    """Test counting occurrences of elements."""
    var list = List[Int]()
    
    list.append(1)
    list.append(2)
    list.append(1)
    list.append(3)
    list.append(1)
    
    assert_equal(list.count(1), 3, "1 appears 3 times")
    assert_equal(list.count(2), 1, "2 appears 1 time")
    assert_equal(list.count(99), 0, "99 appears 0 times")
    
    print("✓ test_list_count passed")

fn test_list_clear():
    """Test clearing all elements from list."""
    var list = List[Int]()
    
    list.append(1)
    list.append(2)
    list.append(3)
    
    list.clear()
    
    assert_equal(list.len(), 0, "Cleared list should have size 0")
    assert_true(list.is_empty(), "Cleared list should be empty")
    
    print("✓ test_list_clear passed")

fn test_list_extend():
    """Test extending one list with another."""
    var list1 = List[Int]()
    var list2 = List[Int]()
    
    list1.append(1)
    list1.append(2)
    
    list2.append(3)
    list2.append(4)
    
    list1.extend(list2)
    
    assert_equal(list1.len(), 4, "Extended list should have 4 elements")
    assert_equal(list1[0], 1, "Element 0 should be 1")
    assert_equal(list1[3], 4, "Element 3 should be 4")
    
    print("✓ test_list_extend passed")

fn test_list_reverse():
    """Test reversing list in-place."""
    var list = List[Int]()
    
    list.append(1)
    list.append(2)
    list.append(3)
    list.append(4)
    
    list.reverse()
    
    assert_equal(list[0], 4, "First element should be 4 after reverse")
    assert_equal(list[1], 3, "Second element should be 3 after reverse")
    assert_equal(list[2], 2, "Third element should be 2 after reverse")
    assert_equal(list[3], 1, "Fourth element should be 1 after reverse")
    
    print("✓ test_list_reverse passed")

fn test_list_equality():
    """Test list equality comparison."""
    var list1 = List[Int]()
    var list2 = List[Int]()
    var list3 = List[Int]()
    
    list1.append(1)
    list1.append(2)
    
    list2.append(1)
    list2.append(2)
    
    list3.append(1)
    list3.append(3)
    
    assert_true(list1 == list2, "Lists with same elements should be equal")
    assert_false(list1 == list3, "Lists with different elements should not be equal")
    assert_true(list1 != list3, "Lists with different elements should be not equal")
    
    print("✓ test_list_equality passed")

fn test_list_concatenation():
    """Test list concatenation with + operator."""
    var list1 = List[Int]()
    var list2 = List[Int]()
    
    list1.append(1)
    list1.append(2)
    
    list2.append(3)
    list2.append(4)
    
    var list3 = list1 + list2
    
    assert_equal(list3.len(), 4, "Concatenated list should have 4 elements")
    assert_equal(list3[0], 1, "Element 0 should be 1")
    assert_equal(list3[3], 4, "Element 3 should be 4")
    
    # Original lists should be unchanged
    assert_equal(list1.len(), 2, "Original list1 should still have 2 elements")
    assert_equal(list2.len(), 2, "Original list2 should still have 2 elements")
    
    print("✓ test_list_concatenation passed")

fn test_list_growth():
    """Test that list grows automatically when capacity is exceeded."""
    var list = List[Int]()
    
    let initial_capacity = list.get_capacity()
    
    # Add more elements than initial capacity
    for i in range(20):
        list.append(i)
    
    assert_equal(list.len(), 20, "List should have 20 elements")
    assert_true(list.get_capacity() > initial_capacity, "Capacity should have grown")
    
    # Verify all elements are correct
    for i in range(20):
        assert_equal(list[i], i, "Element should match index")
    
    print("✓ test_list_growth passed")

fn test_list_copy_constructor():
    """Test copying a list."""
    var list1 = List[Int]()
    
    list1.append(1)
    list1.append(2)
    list1.append(3)
    
    var list2 = List[Int](list1)
    
    # Check that copy has same elements
    assert_equal(list2.len(), 3, "Copy should have 3 elements")
    assert_equal(list2[0], 1, "Copy element 0 should be 1")
    assert_equal(list2[2], 3, "Copy element 2 should be 3")
    
    # Modify copy shouldn't affect original
    list2[0] = 99
    assert_equal(list1[0], 1, "Original should still have 1 at index 0")
    assert_equal(list2[0], 99, "Copy should have 99 at index 0")
    
    print("✓ test_list_copy_constructor passed")

fn test_list_slice():
    """Test slicing a list."""
    var list = List[Int]()
    
    for i in range(10):
        list.append(i)
    
    var slice = list.slice(2, 5)
    
    assert_equal(slice.len(), 3, "Slice should have 3 elements")
    assert_equal(slice[0], 2, "Slice element 0 should be 2")
    assert_equal(slice[1], 3, "Slice element 1 should be 3")
    assert_equal(slice[2], 4, "Slice element 2 should be 4")
    
    print("✓ test_list_slice passed")

# ============================================================================
# Main Test Runner
# ============================================================================

fn main():
    """Run all list tests."""
    print("=" * 70)
    print("Running List[T] Test Suite - Day 29")
    print("=" * 70)
    print()
    
    test_list_initialization()
    test_list_append()
    test_list_insert()
    test_list_remove()
    test_list_pop()
    test_list_indexing()
    test_list_contains()
    test_list_find()
    test_list_count()
    test_list_clear()
    test_list_extend()
    test_list_reverse()
    test_list_equality()
    test_list_concatenation()
    test_list_growth()
    test_list_copy_constructor()
    test_list_slice()
    
    print()
    print("=" * 70)
    print("✅ All 17 tests passed!")
    print("=" * 70)
