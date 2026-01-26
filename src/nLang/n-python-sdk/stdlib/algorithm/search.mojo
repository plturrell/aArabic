# Algorithm/Search - Search Algorithms
# Day 36: Binary search, linear search, find operations

from builtin import Int, Bool
from collections.list import List


# Predicate function type
alias Predicate[T] = fn(T) -> Bool


# Basic search operations

fn find[T](data: List[T], target: T) -> Int:
    """Find first occurrence of target in list.
    
    Args:
        data: List to search
        target: Value to find
    
    Returns:
        Index of first occurrence, or -1 if not found
    
    Examples:
        ```mojo
        var numbers = List[Int]()
        numbers.append(1)
        numbers.append(2)
        numbers.append(3)
        let index = find(numbers, 2)  # 1
        ```
    """
    for i in range(len(data)):
        if data[i] == target:
            return i
    return -1


fn find_last[T](data: List[T], target: T) -> Int:
    """Find last occurrence of target in list.
    
    Args:
        data: List to search
        target: Value to find
    
    Returns:
        Index of last occurrence, or -1 if not found
    """
    for i in range(len(data) - 1, -1, -1):
        if data[i] == target:
            return i
    return -1


fn find_if[T](data: List[T], predicate: Predicate[T]) -> Int:
    """Find first element satisfying predicate.
    
    Args:
        data: List to search
        predicate: Function returning true for matching elements
    
    Returns:
        Index of first match, or -1 if not found
    """
    for i in range(len(data)):
        if predicate(data[i]):
            return i
    return -1


fn find_if_not[T](data: List[T], predicate: Predicate[T]) -> Int:
    """Find first element NOT satisfying predicate.
    
    Args:
        data: List to search
        predicate: Function returning true for matching elements
    
    Returns:
        Index of first non-match, or -1 if all match
    """
    for i in range(len(data)):
        if not predicate(data[i]):
            return i
    return -1


fn contains[T](data: List[T], target: T) -> Bool:
    """Check if list contains target.
    
    Args:
        data: List to search
        target: Value to find
    
    Returns:
        True if found, False otherwise
    """
    return find(data, target) != -1


fn count[T](data: List[T], target: T) -> Int:
    """Count occurrences of target in list.
    
    Args:
        data: List to search
        target: Value to count
    
    Returns:
        Number of occurrences
    """
    var result = 0
    for i in range(len(data)):
        if data[i] == target:
            result += 1
    return result


fn count_if[T](data: List[T], predicate: Predicate[T]) -> Int:
    """Count elements satisfying predicate.
    
    Args:
        data: List to search
        predicate: Function returning true for matching elements
    
    Returns:
        Count of matching elements
    """
    var result = 0
    for i in range(len(data)):
        if predicate(data[i]):
            result += 1
    return result


# Binary search (requires sorted list)

fn binary_search[T](data: List[T], target: T) -> Int:
    """Binary search in sorted list.
    
    Time: O(log n)
    Requires: List must be sorted in ascending order
    
    Args:
        data: Sorted list to search
        target: Value to find
    
    Returns:
        Index of target, or -1 if not found
    """
    var left = 0
    var right = len(data) - 1
    
    while left <= right:
        let mid = left + (right - left) // 2
        
        if data[mid] == target:
            return mid
        elif data[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


fn binary_search_lower_bound[T](data: List[T], target: T) -> Int:
    """Find first position where target could be inserted to maintain sort.
    
    Returns the index of the first element >= target.
    
    Args:
        data: Sorted list
        target: Value to search for
    
    Returns:
        Lower bound index
    """
    var left = 0
    var right = len(data)
    
    while left < right:
        let mid = left + (right - left) // 2
        
        if data[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left


fn binary_search_upper_bound[T](data: List[T], target: T) -> Int:
    """Find last position where target could be inserted to maintain sort.
    
    Returns the index of the first element > target.
    
    Args:
        data: Sorted list
        target: Value to search for
    
    Returns:
        Upper bound index
    """
    var left = 0
    var right = len(data)
    
    while left < right:
        let mid = left + (right - left) // 2
        
        if data[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left


fn equal_range[T](data: List[T], target: T) -> (Int, Int):
    """Find range of elements equal to target in sorted list.
    
    Args:
        data: Sorted list
        target: Value to find
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    let lower = binary_search_lower_bound(data, target)
    let upper = binary_search_upper_bound(data, target)
    return (lower, upper)


# Advanced search operations

fn find_all[T](data: List[T], target: T) -> List[Int]:
    """Find all occurrences of target.
    
    Args:
        data: List to search
        target: Value to find
    
    Returns:
        List of indices where target appears
    """
    var indices = List[Int]()
    for i in range(len(data)):
        if data[i] == target:
            indices.append(i)
    return indices


fn find_all_if[T](data: List[T], predicate: Predicate[T]) -> List[Int]:
    """Find all elements satisfying predicate.
    
    Args:
        data: List to search
        predicate: Matching function
    
    Returns:
        List of matching indices
    """
    var indices = List[Int]()
    for i in range(len(data)):
        if predicate(data[i]):
            indices.append(i)
    return indices


fn adjacent_find[T](data: List[T]) -> Int:
    """Find first pair of adjacent equal elements.
    
    Args:
        data: List to search
    
    Returns:
        Index of first element in pair, or -1 if no adjacent equals
    """
    for i in range(len(data) - 1):
        if data[i] == data[i + 1]:
            return i
    return -1


fn search_subsequence[T](data: List[T], pattern: List[T]) -> Int:
    """Find first occurrence of pattern as subsequence.
    
    Args:
        data: List to search in
        pattern: Pattern to find
    
    Returns:
        Starting index of pattern, or -1 if not found
    """
    let data_len = len(data)
    let pattern_len = len(pattern)
    
    if pattern_len > data_len:
        return -1
    
    for i in range(data_len - pattern_len + 1):
        var match = True
        for j in range(pattern_len):
            if data[i + j] != pattern[j]:
                match = False
                break
        if match:
            return i
    
    return -1


# Min/Max search

fn min_element[T](data: List[T]) -> Int:
    """Find index of minimum element.
    
    Args:
        data: List to search
    
    Returns:
        Index of minimum element
    """
    if len(data) == 0:
        return -1
    
    var min_idx = 0
    for i in range(1, len(data)):
        if data[i] < data[min_idx]:
            min_idx = i
    return min_idx


fn max_element[T](data: List[T]) -> Int:
    """Find index of maximum element.
    
    Args:
        data: List to search
    
    Returns:
        Index of maximum element
    """
    if len(data) == 0:
        return -1
    
    var max_idx = 0
    for i in range(1, len(data)):
        if data[i] > data[max_idx]:
            max_idx = i
    return max_idx


fn minmax_element[T](data: List[T]) -> (Int, Int):
    """Find indices of both min and max elements.
    
    Args:
        data: List to search
    
    Returns:
        Tuple of (min_index, max_index)
    """
    if len(data) == 0:
        return (-1, -1)
    
    var min_idx = 0
    var max_idx = 0
    
    for i in range(1, len(data)):
        if data[i] < data[min_idx]:
            min_idx = i
        if data[i] > data[max_idx]:
            max_idx = i
    
    return (min_idx, max_idx)


# Set operations on sorted ranges

fn includes[T](sorted1: List[T], sorted2: List[T]) -> Bool:
    """Check if sorted1 includes all elements of sorted2.
    
    Both lists must be sorted.
    
    Args:
        sorted1: First sorted list
        sorted2: Second sorted list
    
    Returns:
        True if sorted1 includes all elements from sorted2
    """
    var i = 0
    var j = 0
    
    while i < len(sorted1) and j < len(sorted2):
        if sorted1[i] < sorted2[j]:
            i += 1
        elif sorted1[i] == sorted2[j]:
            i += 1
            j += 1
        else:
            return False
    
    return j == len(sorted2)


# Specialized searches

fn find_peak[T](data: List[T]) -> Int:
    """Find a peak element (greater than neighbors).
    
    Args:
        data: List to search
    
    Returns:
        Index of a peak element, or -1 if none
    """
    let n = len(data)
    
    if n == 0:
        return -1
    if n == 1:
        return 0
    
    # Check first element
    if data[0] > data[1]:
        return 0
    
    # Check last element
    if data[n - 1] > data[n - 2]:
        return n - 1
    
    # Check middle elements
    for i in range(1, n - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            return i
    
    return -1


fn find_valley[T](data: List[T]) -> Int:
    """Find a valley element (less than neighbors).
    
    Args:
        data: List to search
    
    Returns:
        Index of a valley element, or -1 if none
    """
    let n = len(data)
    
    if n == 0:
        return -1
    if n == 1:
        return 0
    
    # Check first element
    if data[0] < data[1]:
        return 0
    
    # Check last element
    if data[n - 1] < data[n - 2]:
        return n - 1
    
    # Check middle elements
    for i in range(1, n - 1):
        if data[i] < data[i - 1] and data[i] < data[i + 1]:
            return i
    
    return -1


# ============================================================================
# Tests
# ============================================================================

test "find basic":
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(2)
    assert(find(data, 2) == 1)
    assert(find(data, 5) == -1)

test "find last":
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(2)
    assert(find_last(data, 2) == 3)

test "contains":
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    assert(contains(data, 2))
    assert(not contains(data, 5))

test "count occurrences":
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(2)
    data.append(3)
    data.append(2)
    assert(count(data, 2) == 3)
    assert(count(data, 5) == 0)

test "binary search":
    var data = List[Int]()
    data.append(1)
    data.append(3)
    data.append(5)
    data.append(7)
    data.append(9)
    assert(binary_search(data, 5) == 2)
    assert(binary_search(data, 4) == -1)

test "binary search bounds":
    var data = List[Int]()
    data.append(1)
    data.append(3)
    data.append(3)
    data.append(3)
    data.append(5)
    let lower = binary_search_lower_bound(data, 3)
    let upper = binary_search_upper_bound(data, 3)
    assert(lower == 1)
    assert(upper == 4)

test "find all":
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(2)
    data.append(2)
    let indices = find_all(data, 2)
    assert(len(indices) == 3)
    assert(indices[0] == 1)
    assert(indices[1] == 3)
    assert(indices[2] == 4)

test "min max element":
    var data = List[Int]()
    data.append(5)
    data.append(2)
    data.append(8)
    data.append(1)
    assert(min_element(data) == 3)  # Index of 1
    assert(max_element(data) == 2)  # Index of 8

test "search subsequence":
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(4)
    data.append(5)
    
    var pattern = List[Int]()
    pattern.append(3)
    pattern.append(4)
    
    assert(search_subsequence(data, pattern) == 2)

test "find peak":
    var data = List[Int]()
    data.append(1)
    data.append(3)
    data.append(2)
    data.append(4)
    data.append(1)
    let peak = find_peak(data)
    assert(peak == 1 or peak == 3)  # Either 3 or 4 is a peak
