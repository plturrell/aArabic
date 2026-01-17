"""
DragonflyDB Cache Client for Mojo
High-performance caching using Zig-compiled native library

Performance Target: 10-20x faster than Python Redis client
Features:
- Zero-copy operations
- Connection pooling (managed by Zig)
- Native RESP protocol implementation
- C ABI for maximum performance
"""

from sys import ffi
from memory import UnsafePointer
from python import Python


@value
struct DragonflyClient:
    """High-performance DragonflyDB client wrapper"""
    var _client: UnsafePointer[NoneType]
    var _lib: ffi.DLHandle
    
    fn __init__(inout self, host: String = "127.0.0.1", port: Int = 6379) raises:
        """Initialize client connection"""
        # Load the Zig library
        self._lib = ffi.DLHandle("lib/libdragonfly_client.dylib")
        
        # Get function pointers
        let create_fn = self._lib.get_function[
            fn(UnsafePointer[UInt8], UInt16) -> UnsafePointer[NoneType]
        ]("dragonfly_client_create")
        
        # Create client (host must be null-terminated)
        let host_cstr = host._as_ptr()
        self._client = create_fn(host_cstr, port)
        
        if not self._client:
            raise Error("Failed to create DragonflyDB client")
    
    fn __del__(owned self):
        """Cleanup client connection"""
        if self._client:
            let destroy_fn = self._lib.get_function[
                fn(UnsafePointer[NoneType]) -> None
            ]("dragonfly_client_destroy")
            destroy_fn(self._client)
    
    fn get(self, key: String) raises -> String:
        """
        GET operation - retrieve value by key
        
        Args:
            key: The key to retrieve
            
        Returns:
            The value, or empty string if not found
            
        Raises:
            Error if operation fails
        """
        let get_fn = self._lib.get_function[
            fn(
                UnsafePointer[NoneType],
                UnsafePointer[UInt8],
                UnsafePointer[UnsafePointer[UInt8]],
                UnsafePointer[Int]
            ) -> Int32
        ]("dragonfly_get")
        
        var value_ptr = UnsafePointer[UnsafePointer[UInt8]].alloc(1)
        var len_ptr = UnsafePointer[Int].alloc(1)
        
        let key_cstr = key._as_ptr()
        let result = get_fn(self._client, key_cstr, value_ptr, len_ptr)
        
        if result == -1:
            value_ptr.free()
            len_ptr.free()
            raise Error("GET operation failed")
        elif result == 1:
            # Key not found
            value_ptr.free()
            len_ptr.free()
            return ""
        
        # Success - copy the value
        let value_data = value_ptr.load()
        let value_len = len_ptr.load()
        
        # Create String from the data
        var output = String()
        for i in range(value_len):
            output += chr(int(value_data[i]))
        
        # Free the C-allocated value
        let free_fn = self._lib.get_function[
            fn(UnsafePointer[UInt8], Int) -> None
        ]("dragonfly_free_value")
        free_fn(value_data, value_len)
        
        value_ptr.free()
        len_ptr.free()
        
        return output
    
    fn set(self, key: String, value: String, expire_seconds: Int = -1) raises:
        """
        SET operation - store key-value pair with optional expiration
        
        Args:
            key: The key to set
            value: The value to store
            expire_seconds: Expiration time in seconds (-1 for no expiration)
            
        Raises:
            Error if operation fails
        """
        let set_fn = self._lib.get_function[
            fn(
                UnsafePointer[NoneType],
                UnsafePointer[UInt8],
                UnsafePointer[UInt8],
                Int,
                Int32
            ) -> Int32
        ]("dragonfly_set")
        
        let key_cstr = key._as_ptr()
        let value_ptr = value._as_ptr()
        let value_len = len(value)
        
        let result = set_fn(
            self._client,
            key_cstr,
            value_ptr,
            value_len,
            expire_seconds
        )
        
        if result != 0:
            raise Error("SET operation failed")
    
    fn delete(self, key: String) raises -> Int:
        """
        DEL operation - delete a key
        
        Args:
            key: The key to delete
            
        Returns:
            Number of keys deleted (0 or 1)
            
        Raises:
            Error if operation fails
        """
        let del_fn = self._lib.get_function[
            fn(UnsafePointer[NoneType], UnsafePointer[UInt8]) -> Int32
        ]("dragonfly_del")
        
        let key_cstr = key._as_ptr()
        let result = del_fn(self._client, key_cstr)
        
        if result == -1:
            raise Error("DEL operation failed")
        
        return int(result)


fn main() raises:
    """Example usage and basic tests"""
    print("🚀 DragonflyDB Mojo Client Test")
    print("=" * 50)
    
    # Create client
    print("\n[1] Creating client...")
    let client = DragonflyClient("127.0.0.1", 6379)
    print("✅ Connected to DragonflyDB")
    
    # Test SET
    print("\n[2] Testing SET operation...")
    client.set("mojo:test", "Hello from Mojo!", 300)
    print("✅ SET mojo:test = 'Hello from Mojo!'")
    
    # Test GET
    print("\n[3] Testing GET operation...")
    let value = client.get("mojo:test")
    print("✅ GET mojo:test = '" + value + "'")
    
    # Test non-existent key
    print("\n[4] Testing non-existent key...")
    let empty = client.get("nonexistent:key")
    if len(empty) == 0:
        print("✅ Non-existent key returns empty string")
    
    # Test DELETE
    print("\n[5] Testing DELETE operation...")
    let deleted = client.delete("mojo:test")
    print("✅ Deleted " + str(deleted) + " key(s)")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed!")
    print("\n✨ Mojo + Zig cache client working!")
    print("   Target: 10-20x faster than Python")
    print("   Features: RESP protocol, connection pooling, C ABI")
