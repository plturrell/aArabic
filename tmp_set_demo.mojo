from stdlib.collections.set import Set
from builtin import Int

fn main():
    var numbers = Set[Int]()
    numbers.add(1)
    numbers.add(2)
    numbers.add(3)
    numbers.remove(2)
    print("contains 1:", numbers.contains(1))
    print("contains 2:", numbers.contains(2))
    print("size:", numbers.size())
