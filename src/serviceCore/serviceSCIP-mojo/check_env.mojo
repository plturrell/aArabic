from memory import UnsafePointer, alloc

fn main():
    # Try to define a pointer with 1 arg
    # var p1: UnsafePointer[UInt8] 
    # This failed before.
    
    # Try to alloc and see type
    var ptr = alloc[UInt8](10)
    print("Allocated")
