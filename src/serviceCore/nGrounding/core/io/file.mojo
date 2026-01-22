"""
Minimal file I/O shim for leanShimmy CLIs.
"""

from collections import List
from memory import UnsafePointer, alloc
from sys.ffi import OwnedDLHandle

comptime LIB_PATH_MAC = "./zig-out/lib/libleanshimmy_io.dylib"
comptime LIB_PATH_LINUX = "./zig-out/lib/libleanshimmy_io.so"
comptime LIB_PATH_FALLBACK_MAC = "./libleanshimmy_io.dylib"
comptime LIB_PATH_FALLBACK_LINUX = "./libleanshimmy_io.so"


fn _string_to_c_ptr(value: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var bytes = value.as_bytes()
    var ptr = alloc[UInt8](len(bytes) + 1)
    for i in range(len(bytes)):
        ptr.store(i, bytes[i])
    ptr.store(len(bytes), 0)
    return ptr


fn _load_library() raises -> OwnedDLHandle:
    try:
        return OwnedDLHandle(LIB_PATH_MAC)
    except:
        pass

    try:
        return OwnedDLHandle(LIB_PATH_LINUX)
    except:
        pass

    try:
        return OwnedDLHandle(LIB_PATH_FALLBACK_MAC)
    except:
        pass

    try:
        return OwnedDLHandle(LIB_PATH_FALLBACK_LINUX)
    except:
        pass

    raise Error("Could not load leanShimmy IO library")


fn _ptr_to_string(ptr: UnsafePointer[UInt8, MutExternalOrigin], length: Int) raises -> String:
    var bytes = List[UInt8]()
    for i in range(length):
        bytes.append(ptr.load(i))
    return String(from_utf8=bytes)


fn _read_file(path: String) raises -> String:
    var lib = _load_library()
    var c_path = _string_to_c_ptr(path)
    var len_ptr = alloc[UInt64](1)
    len_ptr.store(0, 0)

    var data_ptr = lib.call["leanshimmy_read_file", UnsafePointer[UInt8, MutExternalOrigin]](
        c_path,
        len_ptr
    )
    c_path.free()

    var length = Int(len_ptr.load(0))
    len_ptr.free()

    if not data_ptr:
        raise Error("Failed to read file: " + path)

    var content = _ptr_to_string(data_ptr, length)
    lib.call["leanshimmy_free", NoneType](data_ptr, UInt64(length))
    return content


struct File:
    var path: String
    var mode: String

    fn __init__(out self, path: String, mode: String):
        self.path = path
        self.mode = mode

    @staticmethod
    fn open_read(path: String) raises -> File:
        return File(path, "r")

    fn read_all(self) raises -> String:
        return _read_file(self.path)

    fn close(self):
        pass


fn read_file(path: String) raises -> String:
    return _read_file(path)
