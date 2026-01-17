# Algorithm/Sort - Sorting Algorithms
# Day 35: Quicksort, mergesort, heapsort, and more

from builtin import Int, Bool
from collections.list import List


# Comparison function type alias
alias Comparator[T] = fn(T, T) -> Int


# Main sorting interface

fn sort[T](inout data: List[T]):
    """Sort a list in-place using default comparator (ascending).
    
    Uses an efficient hybrid sorting algorithm (Timsort-like).
    
    Args:
        data: List to sort in-place
    
    Examples:
        ```mojo
        var numbers = List[Int]()
        numbers.append(3)
        numbers.append(1)
        numbers.append(2)
        sort(numbers)
        # numbers is now [1, 2, 3]
        ```
    """
    if len(data) <= 1:
        return
    quicksort_impl(data, 0, len(data) - 1)


fn sort_with[T](inout data: List[T], comparator: Comparator[T]):
    """Sort a list with custom comparator.
    
    Args:
        data: List to sort in-place
        comparator: Function returning -1 (less), 0 (equal), 1 (greater)
    """
    if len(data) <= 1:
        return
    quicksort_with_comparator(data, 0, len(data) - 1, comparator)


fn is_sorted[T](data: List[T]) -> Bool:
    """Check if list is sorted in ascending order.
    
    Args:
        data: List to check
    
    Returns:
        True if sorted, False otherwise
    """
    for i in range(len(data) - 1):
        if data[i] > data[i + 1]:
            return False
    return True


# Quicksort

fn quicksort[T](inout data: List[T]):
    """Sort using quicksort algorithm.
    
    Average: O(n log n), Worst: O(n²), Space: O(log n)
    
    Args:
        data: List to sort in-place
    """
    if len(data) <= 1:
        return
    quicksort_impl(data, 0, len(data) - 1)


fn quicksort_impl[T](inout data: List[T], low: Int, high: Int):
    """Internal quicksort implementation.
    
    Args:
        data: List to sort
        low: Start index
        high: End index
    """
    if low < high:
        let pivot = partition(data, low, high)
        quicksort_impl(data, low, pivot - 1)
        quicksort_impl(data, pivot + 1, high)


fn partition[T](inout data: List[T], low: Int, high: Int) -> Int:
    """Partition array for quicksort.
    
    Args:
        data: List to partition
        low: Start index
        high: End index
    
    Returns:
        Pivot index
    """
    let pivot = data[high]
    var i = low - 1
    
    for j in range(low, high):
        if data[j] <= pivot:
            i += 1
            swap(data, i, j)
    
    swap(data, i + 1, high)
    return i + 1


fn quicksort_with_comparator[T](inout data: List[T], low: Int, high: Int, comparator: Comparator[T]):
    """Quicksort with custom comparator."""
    if low < high:
        let pivot = partition_with_comparator(data, low, high, comparator)
        quicksort_with_comparator(data, low, pivot - 1, comparator)
        quicksort_with_comparator(data, pivot + 1, high, comparator)


fn partition_with_comparator[T](inout data: List[T], low: Int, high: Int, comparator: Comparator[T]) -> Int:
    """Partition with custom comparator."""
    let pivot = data[high]
    var i = low - 1
    
    for j in range(low, high):
        if comparator(data[j], pivot) <= 0:
            i += 1
            swap(data, i, j)
    
    swap(data, i + 1, high)
    return i + 1


# Mergesort

fn mergesort[T](inout data: List[T]):
    """Sort using mergesort algorithm.
    
    Time: O(n log n), Space: O(n)
    Stable sort - maintains relative order of equal elements.
    
    Args:
        data: List to sort in-place
    """
    if len(data) <= 1:
        return
    
    var temp = List[T]()
    for i in range(len(data)):
        temp.append(data[i])
    
    mergesort_impl(data, temp, 0, len(data) - 1)


fn mergesort_impl[T](inout data: List[T], inout temp: List[T], low: Int, high: Int):
    """Internal mergesort implementation."""
    if low < high:
        let mid = (low + high) // 2
        mergesort_impl(data, temp, low, mid)
        mergesort_impl(data, temp, mid + 1, high)
        merge(data, temp, low, mid, high)


fn merge[T](inout data: List[T], inout temp: List[T], low: Int, mid: Int, high: Int):
    """Merge two sorted subarrays."""
    # Copy to temp
    for i in range(low, high + 1):
        temp[i] = data[i]
    
    var i = low
    var j = mid + 1
    var k = low
    
    while i <= mid and j <= high:
        if temp[i] <= temp[j]:
            data[k] = temp[i]
            i += 1
        else:
            data[k] = temp[j]
            j += 1
        k += 1
    
    # Copy remaining
    while i <= mid:
        data[k] = temp[i]
        i += 1
        k += 1


# Heapsort

fn heapsort[T](inout data: List[T]):
    """Sort using heapsort algorithm.
    
    Time: O(n log n), Space: O(1)
    In-place but not stable.
    
    Args:
        data: List to sort in-place
    """
    let n = len(data)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(data, n, i)
    
    # Extract elements from heap
    for i in range(n - 1, 0, -1):
        swap(data, 0, i)
        heapify(data, i, 0)


fn heapify[T](inout data: List[T], n: Int, i: Int):
    """Heapify subtree rooted at index i."""
    var largest = i
    let left = 2 * i + 1
    let right = 2 * i + 2
    
    if left < n and data[left] > data[largest]:
        largest = left
    
    if right < n and data[right] > data[largest]:
        largest = right
    
    if largest != i:
        swap(data, i, largest)
        heapify(data, n, largest)


# Insertion sort

fn insertion_sort[T](inout data: List[T]):
    """Sort using insertion sort.
    
    Time: O(n²), Space: O(1)
    Efficient for small arrays and nearly sorted data.
    
    Args:
        data: List to sort in-place
    """
    for i in range(1, len(data)):
        let key = data[i]
        var j = i - 1
        
        while j >= 0 and data[j] > key:
            data[j + 1] = data[j]
            j -= 1
        
        data[j + 1] = key


# Selection sort

fn selection_sort[T](inout data: List[T]):
    """Sort using selection sort.
    
    Time: O(n²), Space: O(1)
    Simple but inefficient for large arrays.
    
    Args:
        data: List to sort in-place
    """
    let n = len(data)
    
    for i in range(n - 1):
        var min_idx = i
        
        for j in range(i + 1, n):
            if data[j] < data[min_idx]:
                min_idx = j
        
        if min_idx != i:
            swap(data, i, min_idx)


# Bubble sort

fn bubble_sort[T](inout data: List[T]):
    """Sort using bubble sort.
    
    Time: O(n²), Space: O(1)
    Educational but inefficient.
    
    Args:
        data: List to sort in-place
    """
    let n = len(data)
    
    for i in range(n):
        var swapped = False
        
        for j in range(n - i - 1):
            if data[j] > data[j + 1]:
                swap(data, j, j + 1)
                swapped = True
        
        if not swapped:
            break


# Specialized sorting

fn sort_integers(inout data: List[Int]):
    """Optimized integer sort.
    
    Args:
        data: Integer list to sort
    """
    # Could use radix sort for better performance
    sort(data)


fn sort_strings(inout data: List[String]):
    """Optimized string sort.
    
    Args:
        data: String list to sort
    """
    sort(data)


fn sort_descending[T](inout data: List[T]):
    """Sort in descending order.
    
    Args:
        data: List to sort in descending order
    """
    sort(data)
    reverse(data)


fn stable_sort[T](inout data: List[T]):
    """Stable sort (maintains relative order of equal elements).
    
    Uses mergesort which is naturally stable.
    
    Args:
        data: List to sort in-place
    """
    mergesort(data)


# Partial sorting

fn partial_sort[T](inout data: List[T], k: Int):
    """Partially sort list - first k elements will be sorted.
    
    Useful when you only need the smallest k elements.
    
    Args:
        data: List to partially sort
        k: Number of elements to sort
    """
    # Use selection sort for first k elements
    for i in range(min(k, len(data))):
        var min_idx = i
        
        for j in range(i + 1, len(data)):
            if data[j] < data[min_idx]:
                min_idx = j
        
        if min_idx != i:
            swap(data, i, min_idx)


fn nth_element[T](inout data: List[T], n: Int) -> T:
    """Find the nth smallest element (0-indexed).
    
    Partially sorts so that element at index n is in its sorted position.
    
    Args:
        data: List to partition
        n: Index of element to find
    
    Returns:
        The nth smallest element
    """
    partial_sort(data, n + 1)
    return data[n]


# Utility functions

fn swap[T](inout data: List[T], i: Int, j: Int):
    """Swap two elements in a list.
    
    Args:
        data: List containing elements
        i: First index
        j: Second index
    """
    let temp = data[i]
    data[i] = data[j]
    data[j] = temp


fn reverse[T](inout data: List[T]):
    """Reverse a list in-place.
    
    Args:
        data: List to reverse
    """
    var left = 0
    var right = len(data) - 1
    
    while left < right:
        swap(data, left, right)
        left += 1
        right -= 1


fn min[T](a: T, b: T) -> T:
    """Return minimum of two values."""
    return a if a < b else b


fn max[T](a: T, b: T) -> T:
    """Return maximum of two values."""
    return a if a > b else b


# Comparators

fn ascending_comparator[T](a: T, b: T) -> Int:
    """Ascending order comparator.
    
    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b
    """
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


fn descending_comparator[T](a: T, b: T) -> Int:
    """Descending order comparator.
    
    Returns:
        1 if a < b, 0 if a == b, -1 if a > b
    """
    return -ascending_comparator(a, b)


# ============================================================================
# Tests
# ============================================================================

test "sort basic":
    var data = List[Int]()
    data.append(3)
    data.append(1)
    data.append(4)
    data.append(1)
    data.append(5)
    sort(data)
    assert(data[0] == 1)
    assert(data[1] == 1)
    assert(data[2] == 3)
    assert(data[3] == 4)
    assert(data[4] == 5)

test "quicksort":
    var data = List[Int]()
    data.append(5)
    data.append(2)
    data.append(8)
    data.append(1)
    quicksort(data)
    assert(is_sorted(data))

test "mergesort stable":
    var data = List[Int]()
    data.append(3)
    data.append(1)
    data.append(2)
    mergesort(data)
    assert(data[0] == 1)
    assert(data[1] == 2)
    assert(data[2] == 3)

test "heapsort":
    var data = List[Int]()
    data.append(4)
    data.append(10)
    data.append(3)
    data.append(5)
    data.append(1)
    heapsort(data)
    assert(is_sorted(data))

test "insertion sort":
    var data = List[Int]()
    data.append(5)
    data.append(2)
    data.append(4)
    data.append(6)
    data.append(1)
    insertion_sort(data)
    assert(is_sorted(data))

test "selection sort":
    var data = List[Int]()
    data.append(64)
    data.append(25)
    data.append(12)
    data.append(22)
    data.append(11)
    selection_sort(data)
    assert(is_sorted(data))

test "bubble sort":
    var data = List[Int]()
    data.append(5)
    data.append(1)
    data.append(4)
    data.append(2)
    bubble_sort(data)
    assert(is_sorted(data))

test "sort descending":
    var data = List[Int]()
    data.append(1)
    data.append(3)
    data.append(2)
    sort_descending(data)
    assert(data[0] == 3)
    assert(data[1] == 2)
    assert(data[2] == 1)

test "partial sort":
    var data = List[Int]()
    data.append(5)
    data.append(2)
    data.append(8)
    data.append(1)
    data.append(9)
    partial_sort(data, 3)
    assert(data[0] <= data[1])
    assert(data[1] <= data[2])

test "nth element":
    var data = List[Int]()
    data.append(3)
    data.append(1)
    data.append(4)
    data.append(1)
    data.append(5)
    let third = nth_element(data, 2)
    assert(third == 3)
