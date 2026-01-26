# Mojo Standard Library - List[T]
# Day 29: Dynamic Array Collection

from memory import Pointer, UnsafePointer
from builtin import Int, Bool, String

# List[T] - Generic dynamic array type
struct List[T]:
    """
    A dynamic array that grows automatically as elements are added.
    
    Generic over element type T.
    Provides O(1) append, O(n) insert/remove operations.
    """
    
    var data: Pointer[T]
    var size: Int
    var capacity: Int
    
    # ========================================================================
    # Constructors
    # ========================================================================
    
    fn __init__(inout self):
        """Initialize an empty list with default capacity."""
        self.size = 0
        self.capacity = 8
        self.data = Pointer[T].alloc(self.capacity)
    
    fn __init__(inout self, capacity: Int):
        """Initialize an empty list with specified capacity."""
        self.size = 0
        self.capacity = capacity
        self.data = Pointer[T].alloc(capacity)
    
    fn __init__(inout self, other: List[T]):
        """Copy constructor - creates a deep copy of another list."""
        self.size = other.size
        self.capacity = other.capacity
        self.data = Pointer[T].alloc(self.capacity)
        
        # Copy elements
        for i in range(self.size):
            self.data[i] = other.data[i]
    
    fn __del__(owned self):
        """Destructor - frees allocated memory."""
        if self.data:
            self.data.free()
    
    # ========================================================================
    # Core Properties
    # ========================================================================
    
    fn len(self) -> Int:
        """Return the number of elements in the list."""
        return self.size
    
    fn is_empty(self) -> Bool:
        """Return True if the list is empty."""
        return self.size == 0
    
    fn get_capacity(self) -> Int:
        """Return the current capacity of the list."""
        return self.capacity
    
    # ========================================================================
    # Element Access
    # ========================================================================
    
    fn __getitem__(self, index: Int) raises -> T:
        """Get element at index. Raises IndexError if index out of bounds."""
        if index < 0 or index >= self.size:
            raise Error("IndexError: List index out of bounds")
        return self.data[index]

    fn __setitem__(inout self, index: Int, value: T) raises:
        """Set element at index. Raises IndexError if index out of bounds."""
        if index < 0 or index >= self.size:
            raise Error("IndexError: List index out of bounds")
        self.data[index] = value
    
    fn get(self, index: Int) raises -> T:
        """Get element at index with bounds checking."""
        return self.__getitem__(index)

    fn set(inout self, index: Int, value: T) raises:
        """Set element at index with bounds checking."""
        self.__setitem__(index, value)
    
    fn front(self) -> T:
        """Get the first element. Undefined if list is empty."""
        return self.data[0]
    
    fn back(self) -> T:
        """Get the last element. Undefined if list is empty."""
        return self.data[self.size - 1]
    
    # ========================================================================
    # Modifiers
    # ========================================================================
    
    fn append(inout self, value: T):
        """Add an element to the end of the list."""
        if self.size >= self.capacity:
            self._grow()
        
        self.data[self.size] = value
        self.size += 1
    
    fn push_back(inout self, value: T):
        """Alias for append - adds element to end."""
        self.append(value)
    
    fn insert(inout self, index: Int, value: T) raises:
        """Insert an element at the specified index."""
        if index < 0 or index > self.size:
            raise Error("IndexError: Insert index out of bounds")

        if self.size >= self.capacity:
            self._grow()

        # Shift elements right
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i - 1]

        self.data[index] = value
        self.size += 1

    fn remove(inout self, index: Int) raises -> T:
        """Remove and return element at index."""
        if index < 0 or index >= self.size:
            raise Error("IndexError: Remove index out of bounds")

        let removed = self.data[index]

        # Shift elements left
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]

        self.size -= 1
        return removed

    fn pop(inout self) raises -> T:
        """Remove and return the last element."""
        if self.size == 0:
            raise Error("IndexError: Pop from empty list")

        self.size -= 1
        return self.data[self.size]
    
    fn pop_back(inout self) raises -> T:
        """Alias for pop - removes and returns last element."""
        return self.pop()
    
    fn clear(inout self):
        """Remove all elements from the list."""
        self.size = 0
    
    fn resize(inout self, new_size: Int):
        """Resize the list to contain new_size elements."""
        if new_size > self.capacity:
            self._reserve(new_size)
        self.size = new_size
    
    fn reserve(inout self, new_capacity: Int):
        """Reserve capacity for at least new_capacity elements."""
        if new_capacity > self.capacity:
            self._reserve(new_capacity)
    
    # ========================================================================
    # Search Operations
    # ========================================================================
    
    fn contains(self, value: T) -> Bool:
        """Return True if the list contains the value."""
        return self.find(value) != -1
    
    fn find(self, value: T) -> Int:
        """Return index of first occurrence of value, or -1 if not found."""
        for i in range(self.size):
            if self.data[i] == value:
                return i
        return -1
    
    fn count(self, value: T) -> Int:
        """Return the number of occurrences of value."""
        var count = 0
        for i in range(self.size):
            if self.data[i] == value:
                count += 1
        return count
    
    # ========================================================================
    # List Operations
    # ========================================================================
    
    fn extend(inout self, other: List[T]):
        """Extend this list by appending elements from another list."""
        self.reserve(self.size + other.size)
        
        for i in range(other.size):
            self.append(other.data[i])
    
    fn reverse(inout self):
        """Reverse the elements in the list in-place."""
        var left = 0
        var right = self.size - 1
        
        while left < right:
            # Swap elements
            let temp = self.data[left]
            self.data[left] = self.data[right]
            self.data[right] = temp
            
            left += 1
            right -= 1
    
    fn sort(inout self):
        """Sort the list in ascending order (quicksort)."""
        if self.size <= 1:
            return
        
        self._quicksort(0, self.size - 1)
    
    # ========================================================================
    # Slicing (simplified - returns new list)
    # ========================================================================
    
    fn slice(self, start: Int, end: Int) -> List[T]:
        """Return a new list containing elements from start to end."""
        var result = List[T]()
        
        let actual_start = max(0, start)
        let actual_end = min(self.size, end)
        
        for i in range(actual_start, actual_end):
            result.append(self.data[i])
        
        return result
    
    # ========================================================================
    # Iteration Support
    # ========================================================================
    
    fn __iter__(self) -> ListIterator[T]:
        """Return an iterator over the list elements."""
        return ListIterator[T](self)
    
    # ========================================================================
    # String Representation
    # ========================================================================
    
    fn __str__(self) -> String:
        """Return string representation of the list."""
        var result = "["
        
        for i in range(self.size):
            if i > 0:
                result += ", "
            result += str(self.data[i])
        
        result += "]"
        return result
    
    fn __repr__(self) -> String:
        """Return detailed string representation."""
        return "List(size=" + str(self.size) + ", capacity=" + str(self.capacity) + ")"
    
    # ========================================================================
    # Operators
    # ========================================================================
    
    fn __eq__(self, other: List[T]) -> Bool:
        """Check equality with another list."""
        if self.size != other.size:
            return False
        
        for i in range(self.size):
            if self.data[i] != other.data[i]:
                return False
        
        return True
    
    fn __ne__(self, other: List[T]) -> Bool:
        """Check inequality with another list."""
        return not self.__eq__(other)
    
    fn __add__(self, other: List[T]) -> List[T]:
        """Concatenate two lists and return a new list."""
        var result = List[T](self.size + other.size)
        
        # Copy elements from self
        for i in range(self.size):
            result.append(self.data[i])
        
        # Copy elements from other
        for i in range(other.size):
            result.append(other.data[i])
        
        return result
    
    # ========================================================================
    # Internal Helper Methods
    # ========================================================================
    
    fn _grow(inout self):
        """Double the capacity when full."""
        let new_capacity = self.capacity * 2
        self._reserve(new_capacity)
    
    fn _reserve(inout self, new_capacity: Int):
        """Reserve exactly new_capacity space."""
        let new_data = Pointer[T].alloc(new_capacity)
        
        # Copy existing elements
        for i in range(self.size):
            new_data[i] = self.data[i]
        
        # Free old memory
        self.data.free()
        
        # Update pointers
        self.data = new_data
        self.capacity = new_capacity
    
    fn _quicksort(inout self, low: Int, high: Int):
        """Internal quicksort implementation."""
        if low < high:
            let pivot_index = self._partition(low, high)
            self._quicksort(low, pivot_index - 1)
            self._quicksort(pivot_index + 1, high)
    
    fn _partition(inout self, low: Int, high: Int) -> Int:
        """Partition for quicksort."""
        let pivot = self.data[high]
        var i = low - 1
        
        for j in range(low, high):
            if self.data[j] <= pivot:
                i += 1
                # Swap
                let temp = self.data[i]
                self.data[i] = self.data[j]
                self.data[j] = temp
        
        # Swap pivot
        let temp = self.data[i + 1]
        self.data[i + 1] = self.data[high]
        self.data[high] = temp
        
        return i + 1


# ============================================================================
# ListIterator - Iterator support for List[T]
# ============================================================================

struct ListIterator[T]:
    """Iterator for List[T]."""
    
    var list_ref: List[T]
    var index: Int
    
    fn __init__(inout self, list: List[T]):
        self.list_ref = list
        self.index = 0
    
    fn __next__(inout self) -> T:
        """Get next element."""
        let value = self.list_ref.data[self.index]
        self.index += 1
        return value
    
    fn has_next(self) -> Bool:
        """Check if there are more elements."""
        return self.index < self.list_ref.size


# ============================================================================
# Helper Functions
# ============================================================================

fn make_list[T]() -> List[T]:
    """Create an empty list of type T."""
    return List[T]()

fn make_list[T](capacity: Int) -> List[T]:
    """Create an empty list with specified capacity."""
    return List[T](capacity)

fn list_from_array[T](arr: Pointer[T], size: Int) -> List[T]:
    """Create a list from an array."""
    var result = List[T](size)
    for i in range(size):
        result.append(arr[i])
    return result
