# Mojo Standard Library - Dict[K, V]
# Day 30: Hash Table / Dictionary Collection

from memory import Pointer, UnsafePointer
from builtin import Int, Bool, String, Float64
from collections import List

# ============================================================================
# Hash Function
# ============================================================================

fn hash[T](value: T) -> Int:
    """
    Compute hash value for any type T.
    This is a simple hash function - in production, would be more sophisticated.
    """
    # Simple hash based on memory representation
    # In real implementation, would have specialized versions for different types
    var result = 0
    let ptr = UnsafePointer[T].address_of(value)
    let bytes = ptr.bitcast[UInt8]()
    
    for i in range(sizeof[T]()):
        result = (result * 31 + int(bytes[i])) % 2147483647
    
    return result

fn hash_int(value: Int) -> Int:
    """Hash function for integers."""
    return value % 2147483647

fn hash_string(value: String) -> Int:
    """Hash function for strings (djb2 algorithm)."""
    var hash_val = 5381
    
    for i in range(len(value)):
        hash_val = ((hash_val << 5) + hash_val) + int(value[i])
    
    return hash_val % 2147483647


# ============================================================================
# Key-Value Pair
# ============================================================================

struct KeyValuePair[K, V]:
    """
    A key-value pair used internally by Dict.
    """
    var key: K
    var value: V
    var is_occupied: Bool
    var is_deleted: Bool  # For tombstone deletion
    
    fn __init__(inout self):
        """Initialize empty pair."""
        self.is_occupied = False
        self.is_deleted = False
    
    fn __init__(inout self, key: K, value: V):
        """Initialize with key and value."""
        self.key = key
        self.value = value
        self.is_occupied = True
        self.is_deleted = False
    
    fn is_available(self) -> Bool:
        """Check if this slot is available for insertion."""
        return not self.is_occupied or self.is_deleted


# ============================================================================
# Dict[K, V] - Hash Table
# ============================================================================

struct Dict[K, V]:
    """
    A hash table that maps keys to values.
    
    Generic over key type K and value type V.
    Uses open addressing with linear probing for collision resolution.
    Provides average O(1) operations for get, set, and delete.
    """
    
    var buckets: Pointer[KeyValuePair[K, V]]
    var size: Int  # Number of elements
    var capacity: Int  # Number of buckets
    var load_factor_threshold: Float64
    
    # ========================================================================
    # Constructors
    # ========================================================================
    
    fn __init__(inout self):
        """Initialize an empty dictionary with default capacity."""
        self.capacity = 16
        self.size = 0
        self.load_factor_threshold = 0.75
        self.buckets = Pointer[KeyValuePair[K, V]].alloc(self.capacity)
        
        # Initialize all buckets as empty
        for i in range(self.capacity):
            self.buckets[i] = KeyValuePair[K, V]()
    
    fn __init__(inout self, capacity: Int):
        """Initialize an empty dictionary with specified capacity."""
        self.capacity = capacity
        self.size = 0
        self.load_factor_threshold = 0.75
        self.buckets = Pointer[KeyValuePair[K, V]].alloc(self.capacity)
        
        for i in range(self.capacity):
            self.buckets[i] = KeyValuePair[K, V]()
    
    fn __del__(owned self):
        """Destructor - frees allocated memory."""
        if self.buckets:
            self.buckets.free()
    
    # ========================================================================
    # Core Properties
    # ========================================================================
    
    fn len(self) -> Int:
        """Return the number of key-value pairs in the dictionary."""
        return self.size
    
    fn is_empty(self) -> Bool:
        """Return True if the dictionary is empty."""
        return self.size == 0
    
    fn load_factor(self) -> Float64:
        """Return the current load factor (size / capacity)."""
        return Float64(self.size) / Float64(self.capacity)
    
    # ========================================================================
    # Hash & Probing
    # ========================================================================
    
    fn _hash(self, key: K) -> Int:
        """Compute hash for a key."""
        return hash(key) % self.capacity
    
    fn _probe(self, hash_val: Int, attempt: Int) -> Int:
        """Linear probing function."""
        return (hash_val + attempt) % self.capacity
    
    fn _find_slot(self, key: K) -> Int:
        """
        Find the slot for a key.
        Returns index if found, or -1 if not found.
        """
        let hash_val = self._hash(key)
        
        for attempt in range(self.capacity):
            let index = self._probe(hash_val, attempt)
            let pair = self.buckets[index]
            
            if not pair.is_occupied:
                return -1  # Not found
            
            if pair.is_occupied and not pair.is_deleted and pair.key == key:
                return index  # Found
        
        return -1  # Not found after checking all slots
    
    fn _find_insert_slot(self, key: K) -> Int:
        """
        Find a slot for inserting a key.
        Returns index of an available slot.
        """
        let hash_val = self._hash(key)
        var first_deleted = -1
        
        for attempt in range(self.capacity):
            let index = self._probe(hash_val, attempt)
            let pair = self.buckets[index]
            
            # Track first deleted slot
            if pair.is_deleted and first_deleted == -1:
                first_deleted = index
            
            # Found the key - update existing
            if pair.is_occupied and not pair.is_deleted and pair.key == key:
                return index
            
            # Found empty slot
            if not pair.is_occupied:
                return first_deleted if first_deleted != -1 else index
        
        # Use first deleted slot if found
        return first_deleted if first_deleted != -1 else 0
    
    # ========================================================================
    # Element Access
    # ========================================================================
    
    fn __getitem__(self, key: K) raises -> V:
        """Get value for key. Raises KeyError if not found."""
        let index = self._find_slot(key)

        if index == -1:
            raise Error("KeyError: Key not found in dictionary")

        return self.buckets[index].value
    
    fn __setitem__(inout self, key: K, value: V):
        """Set value for key."""
        self.set(key, value)
    
    fn get(self, key: K) raises -> V:
        """Get value for key. Raises KeyError if not found."""
        return self.__getitem__(key)
    
    fn get_or_default(self, key: K, default: V) -> V:
        """Get value for key, or return default if not found."""
        let index = self._find_slot(key)
        
        if index == -1:
            return default
        
        return self.buckets[index].value
    
    # ========================================================================
    # Modifiers
    # ========================================================================
    
    fn set(inout self, key: K, value: V):
        """Set or update a key-value pair."""
        # Check if we need to resize
        if self.load_factor() >= self.load_factor_threshold:
            self._resize()
        
        let index = self._find_insert_slot(key)
        let existing = self.buckets[index]
        
        # If updating existing key
        if existing.is_occupied and not existing.is_deleted and existing.key == key:
            self.buckets[index].value = value
        else:
            # Inserting new key
            self.buckets[index] = KeyValuePair[K, V](key, value)
            self.size += 1
    
    fn insert(inout self, key: K, value: V):
        """Alias for set - insert or update key-value pair."""
        self.set(key, value)
    
    fn remove(inout self, key: K) raises -> V:
        """Remove and return value for key. Raises KeyError if not found."""
        let index = self._find_slot(key)

        if index == -1:
            raise Error("KeyError: Key not found in dictionary")

        let value = self.buckets[index].value
        self.buckets[index].is_deleted = True
        self.buckets[index].is_occupied = True  # Keep as occupied but deleted
        self.size -= 1

        return value
    
    fn delete(inout self, key: K):
        """Delete a key-value pair."""
        _ = self.remove(key)
    
    fn clear(inout self):
        """Remove all key-value pairs."""
        for i in range(self.capacity):
            self.buckets[i] = KeyValuePair[K, V]()
        self.size = 0
    
    fn pop(inout self, key: K) raises -> V:
        """Remove and return value for key (alias for remove). Raises KeyError if not found."""
        return self.remove(key)
    
    fn pop_or_default(inout self, key: K, default: V) -> V:
        """Remove and return value for key, or return default if not found."""
        let index = self._find_slot(key)
        
        if index == -1:
            return default
        
        return self.remove(key)
    
    # ========================================================================
    # Membership Testing
    # ========================================================================
    
    fn contains(self, key: K) -> Bool:
        """Check if dictionary contains a key."""
        return self._find_slot(key) != -1
    
    fn has_key(self, key: K) -> Bool:
        """Alias for contains."""
        return self.contains(key)
    
    fn __contains__(self, key: K) -> Bool:
        """Support for 'in' operator."""
        return self.contains(key)
    
    # ========================================================================
    # Views (Keys, Values, Items)
    # ========================================================================
    
    fn keys(self) -> List[K]:
        """Return a list of all keys."""
        var result = List[K]()
        
        for i in range(self.capacity):
            let pair = self.buckets[i]
            if pair.is_occupied and not pair.is_deleted:
                result.append(pair.key)
        
        return result
    
    fn values(self) -> List[V]:
        """Return a list of all values."""
        var result = List[V]()
        
        for i in range(self.capacity):
            let pair = self.buckets[i]
            if pair.is_occupied and not pair.is_deleted:
                result.append(pair.value)
        
        return result
    
    fn items(self) -> List[KeyValuePair[K, V]]:
        """Return a list of all key-value pairs."""
        var result = List[KeyValuePair[K, V]]()
        
        for i in range(self.capacity):
            let pair = self.buckets[i]
            if pair.is_occupied and not pair.is_deleted:
                result.append(pair)
        
        return result
    
    # ========================================================================
    # Dictionary Operations
    # ========================================================================
    
    fn update(inout self, other: Dict[K, V]):
        """Update this dictionary with key-value pairs from another."""
        let other_items = other.items()
        
        for i in range(other_items.len()):
            let pair = other_items[i]
            self.set(pair.key, pair.value)
    
    fn copy(self) -> Dict[K, V]:
        """Return a shallow copy of the dictionary."""
        var result = Dict[K, V](self.capacity)
        
        for i in range(self.capacity):
            let pair = self.buckets[i]
            if pair.is_occupied and not pair.is_deleted:
                result.set(pair.key, pair.value)
        
        return result
    
    # ========================================================================
    # String Representation
    # ========================================================================
    
    fn __str__(self) -> String:
        """Return string representation of the dictionary."""
        var result = "{"
        var first = True
        
        for i in range(self.capacity):
            let pair = self.buckets[i]
            if pair.is_occupied and not pair.is_deleted:
                if not first:
                    result += ", "
                first = False
                result += str(pair.key) + ": " + str(pair.value)
        
        result += "}"
        return result
    
    fn __repr__(self) -> String:
        """Return detailed string representation."""
        return "Dict(size=" + str(self.size) + ", capacity=" + str(self.capacity) + ", load=" + str(self.load_factor()) + ")"
    
    # ========================================================================
    # Operators
    # ========================================================================
    
    fn __eq__(self, other: Dict[K, V]) -> Bool:
        """Check equality with another dictionary."""
        if self.size != other.size:
            return False
        
        # Check that all key-value pairs match
        for i in range(self.capacity):
            let pair = self.buckets[i]
            if pair.is_occupied and not pair.is_deleted:
                if not other.contains(pair.key):
                    return False
                if other.get(pair.key) != pair.value:
                    return False
        
        return True
    
    fn __ne__(self, other: Dict[K, V]) -> Bool:
        """Check inequality with another dictionary."""
        return not self.__eq__(other)
    
    # ========================================================================
    # Internal Helper Methods
    # ========================================================================
    
    fn _resize(inout self):
        """Resize the hash table when load factor threshold is exceeded."""
        let old_capacity = self.capacity
        let old_buckets = self.buckets
        
        # Double the capacity
        self.capacity = old_capacity * 2
        self.buckets = Pointer[KeyValuePair[K, V]].alloc(self.capacity)
        self.size = 0
        
        # Initialize new buckets
        for i in range(self.capacity):
            self.buckets[i] = KeyValuePair[K, V]()
        
        # Rehash all existing key-value pairs
        for i in range(old_capacity):
            let pair = old_buckets[i]
            if pair.is_occupied and not pair.is_deleted:
                self.set(pair.key, pair.value)
        
        # Free old buckets
        old_buckets.free()
    
    fn _debug_print(self):
        """Print internal state for debugging."""
        print("Dict Debug:")
        print("  Size:", self.size)
        print("  Capacity:", self.capacity)
        print("  Load Factor:", self.load_factor())
        print("  Buckets:")
        
        for i in range(self.capacity):
            let pair = self.buckets[i]
            if pair.is_occupied:
                let status = "DELETED" if pair.is_deleted else "ACTIVE"
                print("    [" + str(i) + "] " + status + ": " + str(pair.key) + " -> " + str(pair.value))
            else:
                print("    [" + str(i) + "] EMPTY")


# ============================================================================
# DictIterator - Iterator support for Dict[K, V]
# ============================================================================

struct DictIterator[K, V]:
    """Iterator for Dict[K, V] keys."""
    
    var dict_ref: Dict[K, V]
    var current_index: Int
    
    fn __init__(inout self, dict: Dict[K, V]):
        self.dict_ref = dict
        self.current_index = 0
        self._advance_to_next_valid()
    
    fn _advance_to_next_valid(inout self):
        """Advance to the next occupied, non-deleted slot."""
        while self.current_index < self.dict_ref.capacity:
            let pair = self.dict_ref.buckets[self.current_index]
            if pair.is_occupied and not pair.is_deleted:
                break
            self.current_index += 1
    
    fn __next__(inout self) -> K:
        """Get next key."""
        let key = self.dict_ref.buckets[self.current_index].key
        self.current_index += 1
        self._advance_to_next_valid()
        return key
    
    fn has_next(self) -> Bool:
        """Check if there are more keys."""
        return self.current_index < self.dict_ref.capacity


# ============================================================================
# Helper Functions
# ============================================================================

fn make_dict[K, V]() -> Dict[K, V]:
    """Create an empty dictionary."""
    return Dict[K, V]()

fn make_dict[K, V](capacity: Int) -> Dict[K, V]:
    """Create an empty dictionary with specified capacity."""
    return Dict[K, V](capacity)

fn dict_from_lists[K, V](keys: List[K], values: List[V]) -> Dict[K, V]:
    """Create a dictionary from parallel lists of keys and values."""
    var result = Dict[K, V]()
    
    let count = min(keys.len(), values.len())
    for i in range(count):
        result.set(keys[i], values[i])
    
    return result
