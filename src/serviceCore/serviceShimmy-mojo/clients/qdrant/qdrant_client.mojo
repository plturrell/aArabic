"""
Mojo wrapper for the Zig Qdrant client (C ABI).
Returns vectors and payload JSON as Mojo-owned data.
"""

from sys import ffi
from memory import UnsafePointer, alloc


@value
struct CSearchResult:
    var id: UnsafePointer[UInt8]
    var score: Float32
    var payload_json: UnsafePointer[UInt8]
    var vector_len: Int
    var vector: UnsafePointer[Float32]


@value
struct QdrantResult:
    var id: String
    var score: Float32
    var payload_json: String
    var vector: List[Float32]


fn string_len(ptr: UnsafePointer[UInt8]) -> Int:
    var i: Int = 0
    while ptr.load(i) != 0:
        i += 1
    return i


fn cstr_to_string(ptr: UnsafePointer[UInt8]) -> String:
    let length = string_len(ptr)
    if length == 0:
        return ""
    var bytes = List[UInt8]()
    for i in range(length):
        bytes.append(ptr.load(i))
    return String(bytes)


fn copy_vector(ptr: UnsafePointer[Float32], length: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(length):
        out.append(ptr.load(i))
    return out


@value
struct QdrantClient:
    var _client: UnsafePointer[NoneType]
    var _lib: ffi.DLHandle

    fn __init__(inout self, host: String = "127.0.0.1", port: Int = 6333) raises:
        self._lib = ffi.DLHandle("lib/libqdrant_client.dylib")

        let create_fn = self._lib.get_function[
            fn(UnsafePointer[UInt8], UInt16) -> UnsafePointer[NoneType]
        ]("qdrant_client_create")

        self._client = create_fn(host._as_ptr(), UInt16(port))
        if not self._client:
            raise Error("Failed to create Qdrant client")

    fn __del__(owned self):
        if self._client:
            let destroy_fn = self._lib.get_function[
                fn(UnsafePointer[NoneType]) -> None
            ]("qdrant_client_destroy")
            destroy_fn(self._client)

    fn search(
        self,
        collection: String,
        query: List[Float32],
        limit: Int = 10,
        include_vectors: Bool = True
    ) raises -> List[QdrantResult]:
        let search_fn = self._lib.get_function[
            fn(
                UnsafePointer[NoneType],
                UnsafePointer[UInt8],
                UnsafePointer[Float32],
                Int,
                UInt32,
                UnsafePointer[UnsafePointer[CSearchResult]],
                UnsafePointer[Int]
            ) -> Int32
        ]("qdrant_search")

        let free_fn = self._lib.get_function[
            fn(UnsafePointer[CSearchResult], Int) -> None
        ]("qdrant_free_results")

        let query_len = len(query)
        if query_len == 0:
            raise Error("Query vector is empty")

        var query_ptr = alloc[Float32](query_len)
        for i in range(query_len):
            query_ptr.store(i, query[i])

        var results_ptr = UnsafePointer[UnsafePointer[CSearchResult]].alloc(1)
        var count_ptr = UnsafePointer[Int].alloc(1)

        let rc = search_fn(
            self._client,
            collection._as_ptr(),
            query_ptr,
            query_len,
            UInt32(limit),
            results_ptr,
            count_ptr
        )

        query_ptr.free()

        if rc != 0:
            results_ptr.free()
            count_ptr.free()
            raise Error("Qdrant search failed")

        let results = results_ptr.load()
        let count = count_ptr.load()

        var output = List[QdrantResult]()
        for i in range(count):
            let item = results[i]
            let payload = cstr_to_string(item.payload_json)
            var vector = List[Float32]()
            if include_vectors and item.vector_len > 0:
                vector = copy_vector(item.vector, item.vector_len)

            output.append(QdrantResult(
                id=cstr_to_string(item.id),
                score=item.score,
                payload_json=payload,
                vector=vector
            ))

        free_fn(results, count)
        results_ptr.free()
        count_ptr.free()

        return output
