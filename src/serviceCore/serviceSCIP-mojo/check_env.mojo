from memory import UnsafePointer, alloc

fn return_ptr() -> UnsafePointer[UInt8, 1]:
    return alloc[UInt8](10)

fn main():
    print("Compiled")