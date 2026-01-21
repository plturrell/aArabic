"""
List implementation for Lean4 stdlib.
"""

from collections import List


@fieldwise_init
struct LeanList[T: Copyable & Movable](Copyable, Movable):
    """Lean-style list (immutable, cons-based)."""
    var elements: List[T]

    fn __init__(out self):
        self.elements = List[T]()

    fn nil() -> LeanList[T]:
        return LeanList[T]()

    fn cons(head: T, tail: LeanList[T]) -> LeanList[T]:
        var result = LeanList[T]()
        result.elements.append(head)
        for i in range(len(tail.elements)):
            result.elements.append(tail.elements[i])
        return result

    fn is_empty(self) -> Bool:
        return len(self.elements) == 0

    fn head(self) -> Optional[T]:
        if len(self.elements) > 0:
            return self.elements[0]
        return None

    fn tail(self) -> LeanList[T]:
        var result = LeanList[T]()
        for i in range(1, len(self.elements)):
            result.elements.append(self.elements[i])
        return result

    fn length(self) -> Int:
        return len(self.elements)

    fn get(self, index: Int) -> Optional[T]:
        if index >= 0 and index < len(self.elements):
            return self.elements[index]
        return None


fn list_nil[T: Copyable & Movable]() -> LeanList[T]:
    """Create empty list."""
    return LeanList[T].nil()


fn list_cons[T: Copyable & Movable](head: T, tail: LeanList[T]) -> LeanList[T]:
    """Cons an element onto a list."""
    return LeanList[T].cons(head, tail)


fn list_append[T: Copyable & Movable](a: LeanList[T], b: LeanList[T]) -> LeanList[T]:
    """Append two lists."""
    var result = LeanList[T]()
    for i in range(len(a.elements)):
        result.elements.append(a.elements[i])
    for i in range(len(b.elements)):
        result.elements.append(b.elements[i])
    return result


fn list_length[T: Copyable & Movable](lst: LeanList[T]) -> Int:
    """Get list length."""
    return lst.length()


fn list_reverse[T: Copyable & Movable](lst: LeanList[T]) -> LeanList[T]:
    """Reverse a list."""
    var result = LeanList[T]()
    var i = len(lst.elements) - 1
    while i >= 0:
        result.elements.append(lst.elements[i])
        i -= 1
    return result


fn list_map[T: Copyable & Movable, U: Copyable & Movable](
    lst: LeanList[T], 
    f: fn(T) -> U
) -> LeanList[U]:
    """Map a function over a list."""
    var result = LeanList[U]()
    for i in range(len(lst.elements)):
        result.elements.append(f(lst.elements[i]))
    return result


fn list_filter[T: Copyable & Movable](
    lst: LeanList[T],
    pred: fn(T) -> Bool
) -> LeanList[T]:
    """Filter list by predicate."""
    var result = LeanList[T]()
    for i in range(len(lst.elements)):
        if pred(lst.elements[i]):
            result.elements.append(lst.elements[i])
    return result


fn list_take[T: Copyable & Movable](lst: LeanList[T], n: Int) -> LeanList[T]:
    """Take first n elements."""
    var result = LeanList[T]()
    var count = min(n, len(lst.elements))
    for i in range(count):
        result.elements.append(lst.elements[i])
    return result


fn list_drop[T: Copyable & Movable](lst: LeanList[T], n: Int) -> LeanList[T]:
    """Drop first n elements."""
    var result = LeanList[T]()
    for i in range(n, len(lst.elements)):
        result.elements.append(lst.elements[i])
    return result


fn list_zip[T: Copyable & Movable, U: Copyable & Movable](
    a: LeanList[T], 
    b: LeanList[U]
) -> LeanList[Tuple[T, U]]:
    """Zip two lists together."""
    var result = LeanList[Tuple[T, U]]()
    var n = min(len(a.elements), len(b.elements))
    for i in range(n):
        result.elements.append((a.elements[i], b.elements[i]))
    return result
