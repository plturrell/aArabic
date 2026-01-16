# Algorithm/Functional - Functional Programming Utilities
# Day 37: map, filter, reduce, zip, and functional operations

from builtin import Int, Bool
from collections.list import List


# Function type aliases
alias UnaryFunction[T, R] = fn(T) -> R
alias BinaryFunction[T1, T2, R] = fn(T1, T2) -> R
alias Predicate[T] = fn(T) -> Bool
alias Reducer[T, R] = fn(R, T) -> R


# Map operations

fn map[T, R](data: List[T], func: UnaryFunction[T, R]) -> List[R]:
    """Apply function to each element.
    
    Args:
        data: Input list
        func: Function to apply to each element
    
    Returns:
        New list with transformed elements
    
    Examples:
        ```mojo
        fn double(x: Int) -> Int:
            return x * 2
        
        var numbers = List[Int]()
        numbers.append(1)
        numbers.append(2)
        numbers.append(3)
        let doubled = map(numbers, double)  # [2, 4, 6]
        ```
    """
    var result = List[R]()
    for i in range(len(data)):
        result.append(func(data[i]))
    return result


fn map2[T1, T2, R](data1: List[T1], data2: List[T2], func: BinaryFunction[T1, T2, R]) -> List[R]:
    """Apply binary function to pairs of elements.
    
    Args:
        data1: First input list
        data2: Second input list
        func: Binary function to apply
    
    Returns:
        New list with results
    """
    let n = min(len(data1), len(data2))
    var result = List[R]()
    
    for i in range(n):
        result.append(func(data1[i], data2[i]))
    
    return result


# Filter operations

fn filter[T](data: List[T], predicate: Predicate[T]) -> List[T]:
    """Filter elements matching predicate.
    
    Args:
        data: Input list
        predicate: Function returning true for elements to keep
    
    Returns:
        New list with matching elements
    
    Examples:
        ```mojo
        fn is_even(x: Int) -> Bool:
            return x % 2 == 0
        
        var numbers = List[Int]()
        numbers.append(1)
        numbers.append(2)
        numbers.append(3)
        numbers.append(4)
        let evens = filter(numbers, is_even)  # [2, 4]
        ```
    """
    var result = List[T]()
    for i in range(len(data)):
        if predicate(data[i]):
            result.append(data[i])
    return result


fn filter_not[T](data: List[T], predicate: Predicate[T]) -> List[T]:
    """Filter elements NOT matching predicate.
    
    Args:
        data: Input list
        predicate: Function returning true for elements to exclude
    
    Returns:
        New list with non-matching elements
    """
    var result = List[T]()
    for i in range(len(data)):
        if not predicate(data[i]):
            result.append(data[i])
    return result


fn partition[T](data: List[T], predicate: Predicate[T]) -> (List[T], List[T]):
    """Partition list into matching and non-matching.
    
    Args:
        data: Input list
        predicate: Predicate function
    
    Returns:
        Tuple of (matching, non_matching)
    """
    var matching = List[T]()
    var non_matching = List[T]()
    
    for i in range(len(data)):
        if predicate(data[i]):
            matching.append(data[i])
        else:
            non_matching.append(data[i])
    
    return (matching, non_matching)


# Reduce operations

fn reduce[T, R](data: List[T], initial: R, func: Reducer[T, R]) -> R:
    """Reduce list to single value using accumulator function.
    
    Args:
        data: Input list
        initial: Initial accumulator value
        func: Reducer function (accumulator, element) -> new_accumulator
    
    Returns:
        Final accumulated value
    
    Examples:
        ```mojo
        fn add(acc: Int, x: Int) -> Int:
            return acc + x
        
        var numbers = List[Int]()
        numbers.append(1)
        numbers.append(2)
        numbers.append(3)
        let sum = reduce(numbers, 0, add)  # 6
        ```
    """
    var acc = initial
    for i in range(len(data)):
        acc = func(acc, data[i])
    return acc


fn fold_left[T, R](data: List[T], initial: R, func: Reducer[T, R]) -> R:
    """Fold from left (same as reduce).
    
    Args:
        data: Input list
        initial: Initial value
        func: Reducer function
    
    Returns:
        Accumulated result
    """
    return reduce(data, initial, func)


fn fold_right[T, R](data: List[T], initial: R, func: Reducer[T, R]) -> R:
    """Fold from right (reverse order).
    
    Args:
        data: Input list
        initial: Initial value
        func: Reducer function
    
    Returns:
        Accumulated result
    """
    var acc = initial
    for i in range(len(data) - 1, -1, -1):
        acc = func(acc, data[i])
    return acc


# Zip operations

fn zip[T1, T2](list1: List[T1], list2: List[T2]) -> List[(T1, T2)]:
    """Zip two lists into list of pairs.
    
    Args:
        list1: First list
        list2: Second list
    
    Returns:
        List of tuples pairing elements
    """
    let n = min(len(list1), len(list2))
    var result = List[(T1, T2)]()
    
    for i in range(n):
        result.append((list1[i], list2[i]))
    
    return result


fn zip3[T1, T2, T3](list1: List[T1], list2: List[T2], list3: List[T3]) -> List[(T1, T2, T3)]:
    """Zip three lists into list of triples.
    
    Args:
        list1: First list
        list2: Second list
        list3: Third list
    
    Returns:
        List of tuples with three elements each
    """
    let n = min(min(len(list1), len(list2)), len(list3))
    var result = List[(T1, T2, T3)]()
    
    for i in range(n):
        result.append((list1[i], list2[i], list3[i]))
    
    return result


fn unzip[T1, T2](pairs: List[(T1, T2)]) -> (List[T1], List[T2]):
    """Unzip list of pairs into two lists.
    
    Args:
        pairs: List of tuples
    
    Returns:
        Tuple of (first_elements, second_elements)
    """
    var list1 = List[T1]()
    var list2 = List[T2]()
    
    for i in range(len(pairs)):
        let (first, second) = pairs[i]
        list1.append(first)
        list2.append(second)
    
    return (list1, list2)


# Enumeration

fn enumerate[T](data: List[T]) -> List[(Int, T)]:
    """Create list of (index, value) pairs.
    
    Args:
        data: Input list
    
    Returns:
        List of (index, element) tuples
    """
    var result = List[(Int, T)]()
    for i in range(len(data)):
        result.append((i, data[i]))
    return result


# Scanning

fn scan[T, R](data: List[T], initial: R, func: Reducer[T, R]) -> List[R]:
    """Like reduce but returns all intermediate results.
    
    Args:
        data: Input list
        initial: Initial value
        func: Reducer function
    
    Returns:
        List of all accumulator values
    """
    var result = List[R]()
    var acc = initial
    result.append(acc)
    
    for i in range(len(data)):
        acc = func(acc, data[i])
        result.append(acc)
    
    return result


# Composition

fn take[T](data: List[T], n: Int) -> List[T]:
    """Take first n elements.
    
    Args:
        data: Input list
        n: Number of elements to take
    
    Returns:
        New list with first n elements
    """
    let count = min(n, len(data))
    var result = List[T]()
    
    for i in range(count):
        result.append(data[i])
    
    return result


fn drop[T](data: List[T], n: Int) -> List[T]:
    """Drop first n elements.
    
    Args:
        data: Input list
        n: Number of elements to drop
    
    Returns:
        New list without first n elements
    """
    var result = List[T]()
    let start = min(n, len(data))
    
    for i in range(start, len(data)):
        result.append(data[i])
    
    return result


fn take_while[T](data: List[T], predicate: Predicate[T]) -> List[T]:
    """Take elements while predicate is true.
    
    Args:
        data: Input list
        predicate: Condition function
    
    Returns:
        Elements from start while predicate holds
    """
    var result = List[T]()
    
    for i in range(len(data)):
        if not predicate(data[i]):
            break
        result.append(data[i])
    
    return result


fn drop_while[T](data: List[T], predicate: Predicate[T]) -> List[T]:
    """Drop elements while predicate is true.
    
    Args:
        data: Input list
        predicate: Condition function
    
    Returns:
        Elements after predicate becomes false
    """
    var start = 0
    
    for i in range(len(data)):
        if not predicate(data[i]):
            start = i
            break
    
    var result = List[T]()
    for i in range(start, len(data)):
        result.append(data[i])
    
    return result


# Chunking

fn chunk[T](data: List[T], size: Int) -> List[List[T]]:
    """Split list into chunks of given size.
    
    Args:
        data: Input list
        size: Chunk size
    
    Returns:
        List of chunks
    """
    var result = List[List[T]]()
    var current = List[T]()
    
    for i in range(len(data)):
        current.append(data[i])
        
        if len(current) == size:
            result.append(current)
            current = List[T]()
    
    if len(current) > 0:
        result.append(current)
    
    return result


fn flatten[T](nested: List[List[T]]) -> List[T]:
    """Flatten list of lists into single list.
    
    Args:
        nested: List of lists
    
    Returns:
        Flattened list
    """
    var result = List[T]()
    
    for i in range(len(nested)):
        let sublist = nested[i]
        for j in range(len(sublist)):
            result.append(sublist[j])
    
    return result


# Utility

fn min(a: Int, b: Int) -> Int:
    """Return minimum of two integers."""
    return a if a < b else b


# Aggregation

fn all_of[T](data: List[T], predicate: Predicate[T]) -> Bool:
    """Check if all elements satisfy predicate.
    
    Args:
        data: Input list
        predicate: Condition to check
    
    Returns:
        True if all elements match
    """
    for i in range(len(data)):
        if not predicate(data[i]):
            return False
    return True


fn any_of[T](data: List[T], predicate: Predicate[T]) -> Bool:
    """Check if any element satisfies predicate.
    
    Args:
        data: Input list
        predicate: Condition to check
    
    Returns:
        True if at least one element matches
    """
    for i in range(len(data)):
        if predicate(data[i]):
            return True
    return False


fn none_of[T](data: List[T], predicate: Predicate[T]) -> Bool:
    """Check if no elements satisfy predicate.
    
    Args:
        data: Input list
        predicate: Condition to check
    
    Returns:
        True if no elements match
    """
    return not any_of(data, predicate)


# ============================================================================
# Tests
# ============================================================================

test "map transformation":
    fn double(x: Int) -> Int:
        return x * 2
    
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    let result = map(data, double)
    assert(result[0] == 2)
    assert(result[1] == 4)
    assert(result[2] == 6)

test "filter predicate":
    fn is_even(x: Int) -> Bool:
        return x % 2 == 0
    
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(4)
    let result = filter(data, is_even)
    assert(len(result) == 2)
    assert(result[0] == 2)
    assert(result[1] == 4)

test "reduce sum":
    fn add(acc: Int, x: Int) -> Int:
        return acc + x
    
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(4)
    let sum = reduce(data, 0, add)
    assert(sum == 10)

test "zip pairs":
    var list1 = List[Int]()
    list1.append(1)
    list1.append(2)
    list1.append(3)
    
    var list2 = List[Int]()
    list2.append(10)
    list2.append(20)
    list2.append(30)
    
    let pairs = zip(list1, list2)
    assert(len(pairs) == 3)
    let (a, b) = pairs[0]
    assert(a == 1 and b == 10)

test "take elements":
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(4)
    data.append(5)
    let result = take(data, 3)
    assert(len(result) == 3)
    assert(result[2] == 3)

test "drop elements":
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(4)
    let result = drop(data, 2)
    assert(len(result) == 2)
    assert(result[0] == 3)

test "chunk list":
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(4)
    data.append(5)
    let chunks = chunk(data, 2)
    assert(len(chunks) == 3)
    assert(len(chunks[0]) == 2)
    assert(len(chunks[2]) == 1)

test "all_of predicate":
    fn is_positive(x: Int) -> Bool:
        return x > 0
    
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    assert(all_of(data, is_positive))

test "any_of predicate":
    fn is_even(x: Int) -> Bool:
        return x % 2 == 0
    
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    assert(any_of(data, is_even))

test "partition list":
    fn is_even(x: Int) -> Bool:
        return x % 2 == 0
    
    var data = List[Int]()
    data.append(1)
    data.append(2)
    data.append(3)
    data.append(4)
    let (evens, odds) = partition(data, is_even)
    assert(len(evens) == 2)
    assert(len(odds) == 2)
