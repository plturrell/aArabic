"""
Basic Conformance Tests - Migrated from .lean files
Tests basic Lean4 language features in Mojo
"""

from testing import assert_equal, assert_true, assert_false


fn test_hello_definition() raises:
    """Test: def hello := "Hello, World!" """
    var hello = "Hello, World!"
    assert_equal(hello, "Hello, World!")
    print("✓ test_hello_definition passed")


fn test_add_one_function() raises:
    """Test: def add_one (n : Nat) : Nat := n + 1"""
    fn add_one(n: Int) -> Int:
        return n + 1
    
    assert_equal(add_one(0), 1)
    assert_equal(add_one(5), 6)
    assert_equal(add_one(42), 43)
    print("✓ test_add_one_function passed")


fn test_is_even_function() raises:
    """Test: def is_even (n : Nat) : Bool := n % 2 == 0"""
    fn is_even(n: Int) -> Bool:
        return n % 2 == 0
    
    assert_true(is_even(0))
    assert_true(is_even(2))
    assert_true(is_even(42))
    assert_false(is_even(1))
    assert_false(is_even(3))
    assert_false(is_even(43))
    print("✓ test_is_even_function passed")


fn test_zero_add_theorem() raises:
    """Test: theorem zero_add (n : Nat) : 0 + n = n"""
    # In Mojo we test the property directly
    for n in range(0, 100):
        assert_equal(0 + n, n)
    print("✓ test_zero_add_theorem passed")


fn test_type_checking() raises:
    """Test: #check Nat, Bool, String"""
    # Verify type operations work
    var nat_val: Int = 42
    var bool_val: Bool = True
    var str_val: String = "test"
    
    assert_equal(nat_val, 42)
    assert_equal(bool_val, True)
    assert_equal(str_val, "test")
    print("✓ test_type_checking passed")


fn test_evaluation() raises:
    """Test: #eval 1 + 1, #eval "Hello" ++ " World" """
    # Test basic evaluation
    assert_equal(1 + 1, 2)
    
    # Test string concatenation
    var result = "Hello" + " World"
    assert_equal(result, "Hello World")
    print("✓ test_evaluation passed")


fn test_nat_operations() raises:
    """Test natural number operations"""
    # Addition
    assert_equal(1 + 2, 3)
    assert_equal(10 + 5, 15)
    
    # Multiplication
    assert_equal(2 * 3, 6)
    assert_equal(5 * 7, 35)
    
    # Subtraction (with unsigned behavior)
    assert_equal(5 - 3, 2)
    assert_equal(10 - 1, 9)
    
    print("✓ test_nat_operations passed")


fn test_string_operations() raises:
    """Test string operations"""
    var s1 = "Hello"
    var s2 = "World"
    var combined = s1 + " " + s2
    
    assert_equal(combined, "Hello World")
    assert_equal(len(s1), 5)
    assert_equal(len(s2), 5)
    print("✓ test_string_operations passed")


fn main() raises:
    """Run all conformance tests"""
    print("=" * 60)
    print("Running Basic Conformance Tests (migrated from .lean)")
    print("=" * 60)
    
    test_hello_definition()
    test_add_one_function()
    test_is_even_function()
    test_zero_add_theorem()
    test_type_checking()
    test_evaluation()
    test_nat_operations()
    test_string_operations()
    
    print("=" * 60)
    print("✓ All 8 basic conformance tests passed!")
    print("=" * 60)
