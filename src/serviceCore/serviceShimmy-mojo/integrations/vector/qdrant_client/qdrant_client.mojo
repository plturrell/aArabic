"""
Mojo wrapper for the Zig Qdrant client (C ABI).
Returns vectors and payload JSON as Mojo-owned data.
"""

from collections import List
from memory import UnsafePointer, alloc
from sys.ffi import OwnedDLHandle


fn copy_string_to_external_memory(ref str: String) -> UnsafePointer[UInt8, ImmutExternalOrigin]:
    """Copy a Mojo String to externally-allocated memory with correct origin."""
    var length = len(str)
    # Allocate buffer with extra byte for null terminator
    var buffer = alloc[UInt8](length + 1)
    
    # Copy string data
    var bytes = str.as_bytes()
    for i in range(length):
        buffer[i] = bytes[i]
    
    # Add null terminator
    buffer[length] = 0
    
    # Cast to immutable external pointer via address
    return UnsafePointer[UInt8, ImmutExternalOrigin](buffer.address)


@fieldwise_init
struct CSearchResult(Copyable, Movable):
    var id: UnsafePointer[UInt8, ImmutExternalOrigin]
    var score: Float32
    var payload_json: UnsafePointer[UInt8, ImmutExternalOrigin]
    var vector_len: Int
    var vector: UnsafePointer[Float32, ImmutExternalOrigin]


@fieldwise_init
struct QdrantResult(Movable):
    var id: String
    var score: Float32
    var payload_json: String
    var vector: List[Float32]


fn string_len(ptr: UnsafePointer[UInt8, ImmutExternalOrigin]) -> Int:
    var i: Int = 0
    while ptr[i] != 0:
        i += 1
    return i


fn cstr_to_string(ptr: UnsafePointer[UInt8, ImmutExternalOrigin]) -> String:
    var length = string_len(ptr)
    if length == 0:
        return ""
    var bytes = List[UInt8]()
    for i in range(length):
        bytes.append(ptr[i])
    return String(bytes)


fn copy_vector(ptr: UnsafePointer[Float32, ImmutExternalOrigin], length: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(length):
        out.append(ptr[i])
    return out^


struct QdrantClient(Movable):
    var _client: UnsafePointer[NoneType, MutExternalOrigin]
    var _lib: OwnedDLHandle

    fn __init__[
        origin: Origin
    ](out self, ref [origin] host: String, port: Int) raises:
        self._lib = OwnedDLHandle("lib/libqdrant_client.dylib")

        var create_fn = self._lib.get_function[
            fn(UnsafePointer[UInt8, ImmutExternalOrigin], UInt16) -> UnsafePointer[NoneType, MutExternalOrigin]
        ]("qdrant_client_create")

        # Copy host string to external memory
        var host_ptr = copy_string_to_external_memory(host)
        
        self._client = create_fn(host_ptr, UInt16(port))
        
        # Free the buffer after use
        var mutable_ptr = UnsafePointer[UInt8, MutExternalOrigin](host_ptr.address)
        mutable_ptr.free()
        
        if not self._client:
            raise Error("Failed to create Qdrant client")
    
    # Convenience constructor with default host/port
    fn __init__(out self) raises:
        var default_host = String("127.0.0.1")
        var default_port = 6333
        
        self._lib = OwnedDLHandle("lib/libqdrant_client.dylib")

        var create_fn = self._lib.get_function[
            fn(UnsafePointer[UInt8, ImmutExternalOrigin], UInt16) -> UnsafePointer[NoneType, MutExternalOrigin]
        ]("qdrant_client_create")

        # Copy host string to external memory
        var host_ptr = copy_string_to_external_memory(default_host)
        
        self._client = create_fn(host_ptr, UInt16(default_port))
        
        # Free the buffer after use
        var mutable_ptr = UnsafePointer[UInt8, MutExternalOrigin](host_ptr.address)
        mutable_ptr.free()
        
        if not self._client:
            raise Error("Failed to create Qdrant client")

    fn deinit(self):
        if self._client:
            var destroy_fn = self._lib.get_function[
                fn(UnsafePointer[NoneType, MutExternalOrigin]) -> None
            ]("qdrant_client_destroy")
            destroy_fn(self._client)

    fn search[
        origin: Origin
    ](
        self,
        ref [origin] collection: String,
        query: List[Float32],
        limit: Int = 10,
        include_vectors: Bool = True
    ) raises -> List[QdrantResult]:
        var search_fn = self._lib.get_function[
            fn(
                UnsafePointer[NoneType, MutExternalOrigin],
                UnsafePointer[UInt8, ImmutExternalOrigin],
                UnsafePointer[Float32, ImmutExternalOrigin],
                Int,
                UInt32,
                UnsafePointer[UnsafePointer[CSearchResult, ImmutExternalOrigin], MutExternalOrigin],
                UnsafePointer[Int, MutExternalOrigin]
            ) -> Int32
        ]("qdrant_search")

        var free_fn = self._lib.get_function[
            fn(UnsafePointer[CSearchResult, ImmutExternalOrigin], Int) -> None
        ]("qdrant_free_results")

        var query_len = len(query)
        if query_len == 0:
            raise Error("Query vector is empty")

        var query_ptr = alloc[Float32](query_len)
        for i in range(query_len):
            query_ptr[i] = query[i]

        var results_ptr = alloc[UnsafePointer[CSearchResult, ImmutExternalOrigin]](1)
        var count_ptr = alloc[Int](1)

        # Copy collection string to external memory
        var collection_ptr = copy_string_to_external_memory(collection)
        
        var rc = search_fn(
            self._client,
            collection_ptr,
            query_ptr,
            query_len,
            UInt32(limit),
            results_ptr,
            count_ptr
        )
        
        # Free the collection buffer
        var mutable_collection_ptr = UnsafePointer[UInt8, MutExternalOrigin](collection_ptr.address)
        mutable_collection_ptr.free()
        
        query_ptr.free()

        if rc != 0:
            results_ptr.free()
            count_ptr.free()
            raise Error("Qdrant search failed")

        var results = results_ptr[0]
        var count = count_ptr[0]

        var output = List[QdrantResult]()
        for i in range(count):
            var item = results[i]
            var payload = cstr_to_string(item.payload_json)
            var vector = List[Float32]()
            if include_vectors and item.vector_len > 0:
                vector = copy_vector(item.vector, item.vector_len)

            output.append(QdrantResult(
                id=cstr_to_string(item.id),
                score=item.score,
                payload_json=payload,
                vector=vector^
            ))

        free_fn(results, count)
        results_ptr.free()
        count_ptr.free()

        return output^
