from memory import UnsafePointer, alloc

fn takes_ptr(p: UnsafePointer[UInt8]):
    p.store(0, 1)

fn main():
    var ptr = alloc[UInt8](10)
    takes_ptr(ptr)
    print("Stored")
