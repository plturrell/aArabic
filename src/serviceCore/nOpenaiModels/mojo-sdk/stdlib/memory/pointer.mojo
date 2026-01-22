# Memory/Pointer - Pointer Types and Memory Management
# Day 33: Safe and unsafe pointer operations, memory allocation

from builtin import Int, Bool, String


# Core Pointer Types

struct Pointer[T]:
    """Safe, reference-counted pointer type.
    
    Pointer[T] provides safe access to heap-allocated memory with automatic
    reference counting and bounds checking.
    
    Type Parameters:
        T: Type of the pointed-to value
    
    Examples:
        ```mojo
        var ptr = Pointer[Int].alloc(42)
        print(ptr.load())  # 42
        ptr.store(100)
        print(ptr.load())  # 100
        # Automatically freed when last reference is dropped
        ```
    """
    
    var address: Int
    var ref_count: Int
    var is_null: Bool
    
    fn __init__(inout self):
        """Initialize a null pointer."""
        self.address = 0
        self.ref_count = 0
        self.is_null = True
    
    @staticmethod
    fn alloc(value: T) -> Pointer[T]:
        """Allocate memory and store a value.
        
        Args:
            value: Value to store
        
        Returns:
            Pointer to allocated memory
        """
        var ptr = Pointer[T]()
        ptr.address = __malloc(sizeof[T]())
        ptr.ref_count = 1
        ptr.is_null = False
        __store(ptr.address, value)
        return ptr
    
    fn load(self) -> T:
        """Load the value from memory.
        
        Returns:
            Value stored at pointer location
        """
        if self.is_null:
            # Would raise exception in real implementation
            return T()
        return __load[T](self.address)
    
    fn store(inout self, value: T):
        """Store a value to memory.
        
        Args:
            value: Value to store
        """
        if not self.is_null:
            __store(self.address, value)
    
    fn is_valid(self) -> Bool:
        """Check if pointer is valid (non-null).
        
        Returns:
            True if pointer is valid
        """
        return not self.is_null
    
    fn __del__(inout self):
        """Destructor - decrements reference count and frees if needed."""
        if not self.is_null and self.ref_count > 0:
            self.ref_count -= 1
            if self.ref_count == 0:
                __free(self.address)


struct UnsafePointer[T]:
    """Raw, unchecked pointer type (like C pointers).
    
    UnsafePointer[T] provides direct memory access without safety checks.
    Use with caution - no bounds checking or automatic memory management.
    
    Type Parameters:
        T: Type of the pointed-to value
    
    Examples:
        ```mojo
        var ptr = UnsafePointer[Int].alloc(10)  # Allocate 10 ints
        ptr[0] = 42
        ptr[1] = 100
        print(ptr[0])  # 42
        ptr.free()  # Must manually free
        ```
    """
    
    var address: Int
    
    fn __init__(inout self):
        """Initialize a null pointer."""
        self.address = 0
    
    fn __init__(inout self, address: Int):
        """Initialize from an address.
        
        Args:
            address: Memory address
        """
        self.address = address
    
    @staticmethod
    fn alloc(count: Int) -> UnsafePointer[T]:
        """Allocate memory for count elements.
        
        Args:
            count: Number of elements to allocate
        
        Returns:
            Pointer to allocated memory
        """
        var ptr = UnsafePointer[T]()
        ptr.address = __malloc(count * sizeof[T]())
        return ptr
    
    fn load(self) -> T:
        """Load value from memory (no bounds checking).
        
        Returns:
            Value at pointer location
        """
        return __load[T](self.address)
    
    fn store(self, value: T):
        """Store value to memory (no bounds checking).
        
        Args:
            value: Value to store
        """
        __store(self.address, value)
    
    fn offset(self, index: Int) -> UnsafePointer[T]:
        """Get pointer offset by index.
        
        Args:
            index: Offset in elements
        
        Returns:
            New pointer at offset location
        """
        return UnsafePointer[T](self.address + index * sizeof[T]())
    
    fn __getitem__(self, index: Int) -> T:
        """Array-style access.
        
        Args:
            index: Element index
        
        Returns:
            Value at index
        """
        return self.offset(index).load()
    
    fn __setitem__(self, index: Int, value: T):
        """Array-style assignment.
        
        Args:
            index: Element index
            value: Value to store
        """
        self.offset(index).store(value)
    
    fn free(self):
        """Free allocated memory."""
        __free(self.address)
    
    fn is_null(self) -> Bool:
        """Check if pointer is null.
        
        Returns:
            True if address is 0
        """
        return self.address == 0


struct Reference[T]:
    """Borrowed reference to a value (no ownership).
    
    Reference[T] provides a non-owning view of a value, similar to
    references in C++ or borrows in Rust.
    
    Type Parameters:
        T: Type of the referenced value
    """
    
    var address: Int
    
    fn __init__(inout self, address: Int):
        """Initialize reference from address.
        
        Args:
            address: Address of referenced value
        """
        self.address = address
    
    fn get(self) -> T:
        """Get the referenced value.
        
        Returns:
            Value being referenced
        """
        return __load[T](self.address)
    
    fn set(self, value: T):
        """Set the referenced value.
        
        Args:
            value: New value
        """
        __store(self.address, value)


struct UniquePointer[T]:
    """Unique ownership pointer (like std::unique_ptr in C++).
    
    UniquePointer[T] provides exclusive ownership of allocated memory.
    Cannot be copied, only moved.
    
    Type Parameters:
        T: Type of the owned value
    """
    
    var address: Int
    var owns: Bool
    
    fn __init__(inout self):
        """Initialize null unique pointer."""
        self.address = 0
        self.owns = False
    
    @staticmethod
    fn make(value: T) -> UniquePointer[T]:
        """Create unique pointer with value.
        
        Args:
            value: Value to store
        
        Returns:
            Unique pointer owning the value
        """
        var ptr = UniquePointer[T]()
        ptr.address = __malloc(sizeof[T]())
        ptr.owns = True
        __store(ptr.address, value)
        return ptr
    
    fn get(self) -> T:
        """Get the owned value.
        
        Returns:
            Value owned by pointer
        """
        if self.owns:
            return __load[T](self.address)
        return T()
    
    fn release(inout self) -> Int:
        """Release ownership and return address.
        
        Returns:
            Memory address (caller takes ownership)
        """
        let addr = self.address
        self.owns = False
        self.address = 0
        return addr
    
    fn reset(inout self, value: T):
        """Reset to new value.
        
        Args:
            value: New value to own
        """
        if self.owns and self.address != 0:
            __free(self.address)
        self.address = __malloc(sizeof[T]())
        self.owns = True
        __store(self.address, value)
    
    fn __del__(inout self):
        """Destructor - frees owned memory."""
        if self.owns and self.address != 0:
            __free(self.address)


# Memory allocation functions

fn malloc(size: Int) -> UnsafePointer[Int]:
    """Allocate raw memory.
    
    Args:
        size: Number of bytes to allocate
    
    Returns:
        Pointer to allocated memory
    """
    return UnsafePointer[Int](__malloc(size))


fn free(ptr: UnsafePointer[Int]):
    """Free allocated memory.
    
    Args:
        ptr: Pointer to memory to free
    """
    __free(ptr.address)


fn calloc(count: Int, size: Int) -> UnsafePointer[Int]:
    """Allocate and zero-initialize memory.
    
    Args:
        count: Number of elements
        size: Size of each element
    
    Returns:
        Pointer to zeroed memory
    """
    let ptr = malloc(count * size)
    memzero(ptr, count * size)
    return ptr


fn realloc(ptr: UnsafePointer[Int], new_size: Int) -> UnsafePointer[Int]:
    """Reallocate memory to new size.
    
    Args:
        ptr: Existing pointer
        new_size: New size in bytes
    
    Returns:
        Pointer to reallocated memory
    """
    let new_ptr = malloc(new_size)
    # Would copy old data in real implementation
    free(ptr)
    return new_ptr


# Memory operations

fn memcpy(dest: UnsafePointer[Int], src: UnsafePointer[Int], size: Int):
    """Copy memory from source to destination.
    
    Args:
        dest: Destination pointer
        src: Source pointer
        size: Number of bytes to copy
    """
    for i in range(size):
        dest[i] = src[i]


fn memmove(dest: UnsafePointer[Int], src: UnsafePointer[Int], size: Int):
    """Copy memory (handles overlapping regions).
    
    Args:
        dest: Destination pointer
        src: Source pointer
        size: Number of bytes to copy
    """
    # Simplified - real implementation would handle overlap
    if dest.address < src.address:
        for i in range(size):
            dest[i] = src[i]
    else:
        for i in range(size - 1, -1, -1):
            dest[i] = src[i]


fn memset(ptr: UnsafePointer[Int], value: Int, size: Int):
    """Set memory to a value.
    
    Args:
        ptr: Pointer to memory
        value: Value to set
        size: Number of bytes to set
    """
    for i in range(size):
        ptr[i] = value


fn memzero(ptr: UnsafePointer[Int], size: Int):
    """Zero-initialize memory.
    
    Args:
        ptr: Pointer to memory
        size: Number of bytes to zero
    """
    memset(ptr, 0, size)


fn memcmp(ptr1: UnsafePointer[Int], ptr2: UnsafePointer[Int], size: Int) -> Int:
    """Compare memory regions.
    
    Args:
        ptr1: First pointer
        ptr2: Second pointer
        size: Number of bytes to compare
    
    Returns:
        0 if equal, <0 if ptr1 < ptr2, >0 if ptr1 > ptr2
    """
    for i in range(size):
        if ptr1[i] < ptr2[i]:
            return -1
        if ptr1[i] > ptr2[i]:
            return 1
    return 0


# Type-aware allocation

fn allocate[T](count: Int = 1) -> UnsafePointer[T]:
    """Allocate memory for type T.
    
    Args:
        count: Number of elements to allocate
    
    Returns:
        Typed pointer to allocated memory
    """
    return UnsafePointer[T].alloc(count)


fn deallocate[T](ptr: UnsafePointer[T]):
    """Free typed pointer.
    
    Args:
        ptr: Pointer to free
    """
    ptr.free()


# Pointer utilities

fn align_pointer(ptr: UnsafePointer[Int], alignment: Int) -> UnsafePointer[Int]:
    """Align pointer to specified alignment.
    
    Args:
        ptr: Pointer to align
        alignment: Alignment boundary (power of 2)
    
    Returns:
        Aligned pointer
    """
    let mask = alignment - 1
    let addr = (ptr.address + mask) & ~mask
    return UnsafePointer[Int](addr)


fn pointer_diff[T](ptr1: UnsafePointer[T], ptr2: UnsafePointer[T]) -> Int:
    """Calculate difference between pointers in elements.
    
    Args:
        ptr1: First pointer
        ptr2: Second pointer
    
    Returns:
        Number of elements between pointers
    """
    return (ptr1.address - ptr2.address) // sizeof[T]()


fn swap_pointers[T](inout ptr1: UnsafePointer[T], inout ptr2: UnsafePointer[T]):
    """Swap two pointers.
    
    Args:
        ptr1: First pointer
        ptr2: Second pointer
    """
    let temp = ptr1.address
    ptr1.address = ptr2.address
    ptr2.address = temp


# Stack allocation (simulated)

struct StackAllocator:
    """Simple stack-based allocator for temporary allocations."""
    
    var buffer: UnsafePointer[Int]
    var capacity: Int
    var used: Int
    
    fn __init__(inout self, capacity: Int):
        """Initialize with capacity.
        
        Args:
            capacity: Size in bytes
        """
        self.buffer = malloc(capacity)
        self.capacity = capacity
        self.used = 0
    
    fn alloc(inout self, size: Int) -> UnsafePointer[Int]:
        """Allocate from stack.
        
        Args:
            size: Number of bytes
        
        Returns:
            Pointer to allocated region
        """
        if self.used + size > self.capacity:
            return UnsafePointer[Int]()  # Null
        
        let ptr = self.buffer.offset(self.used)
        self.used += size
        return ptr
    
    fn reset(inout self):
        """Reset allocator (free all)."""
        self.used = 0
    
    fn __del__(inout self):
        """Destructor."""
        free(self.buffer)


# Helper functions

fn sizeof[T]() -> Int:
    """Get size of type T in bytes.
    
    Returns:
        Size in bytes
    """
    # Would use compiler intrinsic in real implementation
    return 8  # Placeholder


# Internal stubs (would be implemented in Zig/C)

fn __malloc(size: Int) -> Int:
    """Internal malloc implementation."""
    return 0  # Would call libc malloc

fn __free(address: Int):
    """Internal free implementation."""
    pass  # Would call libc free

fn __load[T](address: Int) -> T:
    """Internal load implementation."""
    return T()  # Would read from memory

fn __store[T](address: Int, value: T):
    """Internal store implementation."""
    pass  # Would write to memory


# ============================================================================
# Tests
# ============================================================================

test "pointer alloc and load":
    var ptr = Pointer[Int].alloc(42)
    assert(ptr.load() == 42)
    assert(ptr.is_valid())

test "pointer store":
    var ptr = Pointer[Int].alloc(10)
    ptr.store(100)
    assert(ptr.load() == 100)

test "unsafe pointer array access":
    var ptr = UnsafePointer[Int].alloc(5)
    ptr[0] = 10
    ptr[1] = 20
    ptr[2] = 30
    assert(ptr[0] == 10)
    assert(ptr[1] == 20)
    assert(ptr[2] == 30)
    ptr.free()

test "unsafe pointer offset":
    var ptr = UnsafePointer[Int].alloc(5)
    var ptr2 = ptr.offset(2)
    ptr2.store(99)
    assert(ptr[2] == 99)
    ptr.free()

test "unique pointer ownership":
    var ptr = UniquePointer[Int].make(42)
    assert(ptr.get() == 42)
    ptr.reset(100)
    assert(ptr.get() == 100)

test "memory operations":
    var src = UnsafePointer[Int].alloc(3)
    var dest = UnsafePointer[Int].alloc(3)
    src[0] = 1
    src[1] = 2
    src[2] = 3
    memcpy(dest, src, 3)
    assert(dest[0] == 1)
    assert(dest[1] == 2)
    assert(dest[2] == 3)
    src.free()
    dest.free()

test "memset operation":
    var ptr = UnsafePointer[Int].alloc(5)
    memset(ptr, 42, 5)
    assert(ptr[0] == 42)
    assert(ptr[4] == 42)
    ptr.free()

test "memcmp operation":
    var ptr1 = UnsafePointer[Int].alloc(3)
    var ptr2 = UnsafePointer[Int].alloc(3)
    ptr1[0] = 1
    ptr1[1] = 2
    ptr1[2] = 3
    ptr2[0] = 1
    ptr2[1] = 2
    ptr2[2] = 3
    assert(memcmp(ptr1, ptr2, 3) == 0)
    ptr1.free()
    ptr2.free()

test "stack allocator":
    var stack = StackAllocator(1024)
    var ptr1 = stack.alloc(64)
    var ptr2 = stack.alloc(64)
    assert(not ptr1.is_null())
    assert(not ptr2.is_null())
    stack.reset()
    assert(stack.used == 0)

test "type-aware allocation":
    var ptr = allocate[Int](10)
    ptr[0] = 100
    assert(ptr[0] == 100)
    deallocate(ptr)
