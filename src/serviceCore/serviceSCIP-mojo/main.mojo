from sys.ffi import OwnedDLHandle
from memory import UnsafePointer, alloc

fn c_str(s: String) -> UnsafePointer[UInt8]:
    var bytes = s.as_bytes()
    var ptr = alloc[UInt8](len(bytes) + 1)
    for i in range(len(bytes)):
        ptr.store(i, bytes[i])
    ptr.store(len(bytes), 0)
    return ptr

struct ScipWriter:
    var handle: OwnedDLHandle

    fn __init__(inout self, lib_path: String):
        self.handle = OwnedDLHandle(lib_path)

    fn init_index(self, filename: String) -> Bool:
        var func = self.handle.get_function[fn(UnsafePointer[UInt8]) -> Int32]("scip_init")
        var ptr = c_str(filename)
        var res = func(ptr)
        return res == 0

    fn write_metadata(self, tool_name: String, tool_version: String, root: String) -> Bool:
        var func = self.handle.get_function[fn(UnsafePointer[UInt8], UnsafePointer[UInt8], UnsafePointer[UInt8]) -> Int32]("scip_write_metadata")
        var p1 = c_str(tool_name)
        var p2 = c_str(tool_version)
        var p3 = c_str(root)
        var res = func(p1, p2, p3)
        return res == 0

    fn close(self):
        var func = self.handle.get_function[fn() -> Int32]("scip_close")
        _ = func()

fn main():
    print("🚀 Starting Mojo SCIP Indexer...")
    
    var writer = ScipWriter("./libzig_scip_writer.dylib")
    
    if not writer.init_index("index.scip"):
        print("❌ Failed to initialize index file.")
        return

    print("📝 Writing metadata...")
    if not writer.write_metadata("mojo-scip", "0.1.0", "file:///Users/user/Project"):
        print("❌ Failed to write metadata.")
        return

    writer.close()
    print("✅ Index generation complete: index.scip")