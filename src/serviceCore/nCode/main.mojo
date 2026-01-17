from sys.ffi import OwnedDLHandle
from memory import UnsafePointer, alloc, memcpy

# Define the exact type we expect for C strings
alias CString = UnsafePointer[UInt8]

fn main() raises:
    print("🚀 Starting Mojo SCIP Indexer...")
    
    var handle = OwnedDLHandle("./libzig_scip_writer.dylib")
    
    # We define the function types explicitly
    var scip_init = handle.get_function[fn(CString) -> Int32]("scip_init")
    var scip_write_metadata = handle.get_function[fn(CString, CString, CString) -> Int32]("scip_write_metadata")
    var scip_close = handle.get_function[fn() -> Int32]("scip_close")

    # 1. Filename
    var filename = "index.scip"
    var fn_bytes = filename.as_bytes()
    # Alloc returns UnsafePointer[UInt8], which matches CString
    var fn_ptr = alloc[UInt8](len(fn_bytes) + 1)
    memcpy(dest=fn_ptr, src=fn_bytes.unsafe_ptr(), count=len(fn_bytes))
    fn_ptr.store(len(fn_bytes), 0)

    # Call init
    if scip_init(fn_ptr) != 0:
        print("❌ Failed to initialize index file.")
        return

    print("📝 Writing metadata...")
    
    # 2. Tool Name
    var tool = "mojo-scip"
    var tool_bytes = tool.as_bytes()
    var tool_ptr = alloc[UInt8](len(tool_bytes) + 1)
    memcpy(dest=tool_ptr, src=tool_bytes.unsafe_ptr(), count=len(tool_bytes))
    tool_ptr.store(len(tool_bytes), 0)

    # 3. Version
    var ver = "0.1.0"
    var ver_bytes = ver.as_bytes()
    var ver_ptr = alloc[UInt8](len(ver_bytes) + 1)
    memcpy(dest=ver_ptr, src=ver_bytes.unsafe_ptr(), count=len(ver_bytes))
    ver_ptr.store(len(ver_bytes), 0)

    # 4. Root
    var root = "file:///Users/user/Project"
    var root_bytes = root.as_bytes()
    var root_ptr = alloc[UInt8](len(root_bytes) + 1)
    memcpy(dest=root_ptr, src=root_bytes.unsafe_ptr(), count=len(root_bytes))
    root_ptr.store(len(root_bytes), 0)

    # Call write_metadata
    if scip_write_metadata(tool_ptr, ver_ptr, root_ptr) != 0:
        print("❌ Failed to write metadata.")
        return

    # Close
    _ = scip_close()
    
    # Free memory
    fn_ptr.free()
    tool_ptr.free()
    ver_ptr.free()
    root_ptr.free()

    print("✅ Index generation complete: index.scip")
