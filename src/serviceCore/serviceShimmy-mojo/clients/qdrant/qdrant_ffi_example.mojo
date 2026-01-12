"""
Minimal Mojo FFI snippet for qdrant_client.zig (C ABI).
Copies vectors/payloads into Mojo-owned memory before freeing C results.
"""

from sys import ffi
from memory import UnsafePointer


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


fn search_vectors(
    lib: ffi.DLHandle,
    client: UnsafePointer[NoneType],
    collection: String,
    query: UnsafePointer[Float32],
    query_len: Int,
    limit: Int
) raises -> List[QdrantResult]:
    let search_fn = lib.get_function[
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

    let free_fn = lib.get_function[
        fn(UnsafePointer[CSearchResult], Int) -> None
    ]("qdrant_free_results")

    var results_ptr = UnsafePointer[UnsafePointer[CSearchResult]].alloc(1)
    var count_ptr = UnsafePointer[Int].alloc(1)

    let rc = search_fn(
        client,
        collection._as_ptr(),
        query,
        query_len,
        UInt32(limit),
        results_ptr,
        count_ptr
    )

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
        if item.vector_len > 0:
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
