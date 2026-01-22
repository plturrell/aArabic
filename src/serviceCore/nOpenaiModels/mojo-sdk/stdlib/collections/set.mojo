# Set[T] - Hash Set Collection
# Day 30: Unordered collection of unique elements with O(1) average operations

from builtin import Int, Bool, String


struct Set[T]:
    """Generic hash set storing unique elements.
    
    A Set is an unordered collection of unique values with fast membership testing,
    insertion, and deletion operations. Uses hash-based storage for O(1) average
    time complexity.
    
    Type Parameters:
        T: The type of elements stored in the set. Must be hashable and comparable.
    
    Examples:
        ```mojo
        var numbers = Set[Int]()
        numbers.add(1)
        numbers.add(2)
        numbers.add(1)  # Duplicate ignored
        
        if numbers.contains(2):
            print("Found 2")
        
        numbers.remove(1)
        print(numbers.size())  # 1
        ```
    """
    
    var buckets: List[List[T]]
    var count: Int
    var capacity: Int
    var load_factor: Float64
    
    fn __init__(inout self, capacity: Int = 16):
        """Initialize an empty set with specified initial capacity.
        
        Args:
            capacity: Initial number of buckets (default: 16)
        """
        self.capacity = capacity
        self.count = 0
        self.load_factor = 0.75
        self.buckets = List[List[T]]()
        
        # Initialize buckets
        for _ in range(capacity):
            self.buckets.append(List[T]())
    
    fn __init__(inout self, elements: List[T]):
        """Initialize a set from a list of elements.
        
        Args:
            elements: List of elements to add to the set
        """
        self.__init__(len(elements) * 2)
        for element in elements:
            self.add(element)
    
    fn add(inout self, element: T) -> Bool:
        """Add an element to the set.
        
        Args:
            element: The element to add
        
        Returns:
            True if element was added, False if it already existed
        """
        # Check if we need to resize
        if Float64(self.count) / Float64(self.capacity) >= self.load_factor:
            self._resize()
        
        let bucket_index = self._hash(element) % self.capacity
        var bucket = self.buckets[bucket_index]
        
        # Check if element already exists
        for i in range(len(bucket)):
            if bucket[i] == element:
                return False  # Already exists
        
        # Add new element
        bucket.append(element)
        self.buckets[bucket_index] = bucket
        self.count += 1
        return True
    
    fn remove(inout self, element: T) -> Bool:
        """Remove an element from the set.
        
        Args:
            element: The element to remove
        
        Returns:
            True if element was removed, False if not found
        """
        let bucket_index = self._hash(element) % self.capacity
        var bucket = self.buckets[bucket_index]
        
        for i in range(len(bucket)):
            if bucket[i] == element:
                bucket.pop(i)
                self.buckets[bucket_index] = bucket
                self.count -= 1
                return True
        
        return False
    
    fn contains(self, element: T) -> Bool:
        """Check if an element exists in the set.
        
        Args:
            element: The element to check for
        
        Returns:
            True if element exists, False otherwise
        """
        let bucket_index = self._hash(element) % self.capacity
        let bucket = self.buckets[bucket_index]
        
        for i in range(len(bucket)):
            if bucket[i] == element:
                return True
        
        return False
    
    fn size(self) -> Int:
        """Return the number of elements in the set.
        
        Returns:
            Number of unique elements
        """
        return self.count
    
    fn is_empty(self) -> Bool:
        """Check if the set is empty.
        
        Returns:
            True if set is empty, False otherwise
        """
        return self.count == 0
    
    fn clear(inout self):
        """Remove all elements from the set."""
        self.count = 0
        self.buckets = List[List[T]]()
        for _ in range(self.capacity):
            self.buckets.append(List[T]())
    
    fn to_list(self) -> List[T]:
        """Convert the set to a list.
        
        Returns:
            List containing all elements in arbitrary order
        """
        var result = List[T]()
        for bucket in self.buckets:
            for element in bucket:
                result.append(element)
        return result
    
    # Set operations
    
    fn union(self, other: Set[T]) -> Set[T]:
        """Return a new set with elements from both sets.
        
        Args:
            other: Another set to union with
        
        Returns:
            New set containing all unique elements from both sets
        """
        var result = Set[T](self.capacity)
        
        # Add all elements from self
        for element in self.to_list():
            result.add(element)
        
        # Add all elements from other
        for element in other.to_list():
            result.add(element)
        
        return result
    
    fn intersection(self, other: Set[T]) -> Set[T]:
        """Return a new set with elements common to both sets.
        
        Args:
            other: Another set to intersect with
        
        Returns:
            New set containing only elements present in both sets
        """
        var result = Set[T]()
        
        for element in self.to_list():
            if other.contains(element):
                result.add(element)
        
        return result
    
    fn difference(self, other: Set[T]) -> Set[T]:
        """Return a new set with elements in self but not in other.
        
        Args:
            other: Set to subtract from self
        
        Returns:
            New set with elements in self but not in other
        """
        var result = Set[T]()
        
        for element in self.to_list():
            if not other.contains(element):
                result.add(element)
        
        return result
    
    fn symmetric_difference(self, other: Set[T]) -> Set[T]:
        """Return a new set with elements in either set but not both.
        
        Args:
            other: Another set to compare with
        
        Returns:
            New set with elements in either set but not in both
        """
        return self.union(other).difference(self.intersection(other))
    
    fn is_subset(self, other: Set[T]) -> Bool:
        """Check if all elements in self are in other.
        
        Args:
            other: Set to check against
        
        Returns:
            True if self is a subset of other, False otherwise
        """
        for element in self.to_list():
            if not other.contains(element):
                return False
        return True
    
    fn is_superset(self, other: Set[T]) -> Bool:
        """Check if all elements in other are in self.
        
        Args:
            other: Set to check against
        
        Returns:
            True if self is a superset of other, False otherwise
        """
        return other.is_subset(self)
    
    fn is_disjoint(self, other: Set[T]) -> Bool:
        """Check if sets have no elements in common.
        
        Args:
            other: Set to check against
        
        Returns:
            True if sets have no common elements, False otherwise
        """
        return self.intersection(other).is_empty()
    
    # In-place operations
    
    fn update(inout self, other: Set[T]):
        """Add all elements from other set to this set.
        
        Args:
            other: Set whose elements to add
        """
        for element in other.to_list():
            self.add(element)
    
    fn intersection_update(inout self, other: Set[T]):
        """Keep only elements found in both sets.
        
        Args:
            other: Set to intersect with
        """
        let intersection = self.intersection(other)
        self.clear()
        self.update(intersection)
    
    fn difference_update(inout self, other: Set[T]):
        """Remove all elements found in other set.
        
        Args:
            other: Set whose elements to remove
        """
        for element in other.to_list():
            self.remove(element)
    
    fn symmetric_difference_update(inout self, other: Set[T]):
        """Keep only elements found in either set but not both.
        
        Args:
            other: Set to compare with
        """
        let sym_diff = self.symmetric_difference(other)
        self.clear()
        self.update(sym_diff)
    
    # Operators
    
    fn __eq__(self, other: Set[T]) -> Bool:
        """Check if two sets are equal.
        
        Args:
            other: Set to compare with
        
        Returns:
            True if sets contain same elements, False otherwise
        """
        if self.size() != other.size():
            return False
        
        return self.is_subset(other)
    
    fn __ne__(self, other: Set[T]) -> Bool:
        """Check if two sets are not equal.
        
        Args:
            other: Set to compare with
        
        Returns:
            True if sets differ, False if equal
        """
        return not self.__eq__(other)
    
    fn __len__(self) -> Int:
        """Return the number of elements in the set.
        
        Returns:
            Number of elements
        """
        return self.size()
    
    fn __contains__(self, element: T) -> Bool:
        """Check if element is in the set (for 'in' operator).
        
        Args:
            element: Element to check for
        
        Returns:
            True if element exists, False otherwise
        """
        return self.contains(element)
    
    fn __str__(self) -> String:
        """Return string representation of the set.
        
        Returns:
            String representation in format: {1, 2, 3}
        """
        if self.is_empty():
            return "{}"
        
        var result = "{"
        let elements = self.to_list()
        for i in range(len(elements)):
            result += str(elements[i])
            if i < len(elements) - 1:
                result += ", "
        result += "}"
        return result
    
    fn __repr__(self) -> String:
        """Return detailed representation of the set.
        
        Returns:
            Representation showing type and contents
        """
        return "Set[" + T.__name__ + "]" + self.__str__()
    
    # Internal methods
    
    fn _hash(self, element: T) -> Int:
        """Compute hash value for an element.
        
        Args:
            element: Element to hash
        
        Returns:
            Hash value
        """
        # Simple hash for demonstration
        # Real implementation would use proper hash function
        return hash(element)
    
    fn _resize(inout self):
        """Resize the hash table when load factor is exceeded."""
        let old_buckets = self.buckets
        self.capacity *= 2
        self.count = 0
        self.buckets = List[List[T]]()
        
        # Initialize new buckets
        for _ in range(self.capacity):
            self.buckets.append(List[T]())
        
        # Rehash all elements
        for bucket in old_buckets:
            for element in bucket:
                self.add(element)


# Specialized implementations

struct IntSet:
    """Optimized set for integers."""
    
    var set: Set[Int]
    
    fn __init__(inout self):
        self.set = Set[Int]()
    
    fn add(inout self, value: Int) -> Bool:
        return self.set.add(value)
    
    fn remove(inout self, value: Int) -> Bool:
        return self.set.remove(value)
    
    fn contains(self, value: Int) -> Bool:
        return self.set.contains(value)
    
    fn size(self) -> Int:
        return self.set.size()
    
    fn min(self) -> Int:
        """Return the minimum element in the set."""
        let elements = self.set.to_list()
        var minimum = elements[0]
        for element in elements:
            if element < minimum:
                minimum = element
        return minimum
    
    fn max(self) -> Int:
        """Return the maximum element in the set."""
        let elements = self.set.to_list()
        var maximum = elements[0]
        for element in elements:
            if element > maximum:
                maximum = element
        return maximum


struct StringSet:
    """Optimized set for strings."""
    
    var set: Set[String]
    
    fn __init__(inout self):
        self.set = Set[String]()
    
    fn add(inout self, value: String) -> Bool:
        return self.set.add(value)
    
    fn remove(inout self, value: String) -> Bool:
        return self.set.remove(value)
    
    fn contains(self, value: String) -> Bool:
        return self.set.contains(value)
    
    fn size(self) -> Int:
        return self.set.size()
    
    fn sorted(self) -> List[String]:
        """Return sorted list of strings."""
        var elements = self.set.to_list()
        # Simple bubble sort for demonstration
        for i in range(len(elements)):
            for j in range(len(elements) - i - 1):
                if elements[j] > elements[j + 1]:
                    let temp = elements[j]
                    elements[j] = elements[j + 1]
                    elements[j + 1] = temp
        return elements


# Utility functions

fn set_from_list[T](elements: List[T]) -> Set[T]:
    """Create a set from a list of elements.
    
    Args:
        elements: List of elements
    
    Returns:
        New set containing unique elements from list
    """
    return Set[T](elements)


fn set_union[T](set1: Set[T], set2: Set[T]) -> Set[T]:
    """Return union of two sets.
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        New set with all elements from both sets
    """
    return set1.union(set2)


fn set_intersection[T](set1: Set[T], set2: Set[T]) -> Set[T]:
    """Return intersection of two sets.
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        New set with elements common to both sets
    """
    return set1.intersection(set2)
