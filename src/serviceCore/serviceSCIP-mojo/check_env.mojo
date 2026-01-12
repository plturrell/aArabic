from memory import UnsafePointer, alloc, AddressSpace

fn ret_ptr() -> UnsafePointer[UInt8, AddressSpace.GENERIC]:
    return alloc[UInt8](10)

fn main():
    print("Works")