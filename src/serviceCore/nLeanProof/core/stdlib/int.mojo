"""
Integer implementation for Lean4 stdlib.
"""

from .nat import Nat


@fieldwise_init
struct Int(ImplicitlyCopyable, Copyable, Movable):
    """Integer type (can be negative)."""
    var value: PythonObject  # Using Python's arbitrary precision integers

    fn __init__(out self, value: Int = 0):
        self.value = value

    fn zero() -> Int:
        return Int(0)

    fn one() -> Int:
        return Int(1)

    fn neg_one() -> Int:
        return Int(-1)

    fn from_nat(n: Nat) -> Int:
        return Int(n.to_int())

    fn neg(self) -> Int:
        return Int(-self.value)

    fn abs(self) -> Nat:
        if self.value < 0:
            return Nat(-self.value)
        return Nat(self.value)

    fn sign(self) -> Int:
        if self.value > 0:
            return Int(1)
        elif self.value < 0:
            return Int(-1)
        return Int(0)

    fn is_zero(self) -> Bool:
        return self.value == 0

    fn is_positive(self) -> Bool:
        return self.value > 0

    fn is_negative(self) -> Bool:
        return self.value < 0

    fn to_nat(self) -> Optional[Nat]:
        if self.value >= 0:
            return Nat(self.value)
        return None

    fn __eq__(self, other: Int) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Int) -> Bool:
        return self.value != other.value

    fn __lt__(self, other: Int) -> Bool:
        return self.value < other.value

    fn __le__(self, other: Int) -> Bool:
        return self.value <= other.value

    fn __gt__(self, other: Int) -> Bool:
        return self.value > other.value

    fn __ge__(self, other: Int) -> Bool:
        return self.value >= other.value


fn int_add(a: Int, b: Int) -> Int:
    """Add two integers."""
    return Int(a.value + b.value)


fn int_neg(a: Int) -> Int:
    """Negate an integer."""
    return Int(-a.value)


fn int_sub(a: Int, b: Int) -> Int:
    """Subtract integers."""
    return Int(a.value - b.value)


fn int_mul(a: Int, b: Int) -> Int:
    """Multiply integers."""
    return Int(a.value * b.value)


fn int_div(a: Int, b: Int) -> Int:
    """Divide integers (Euclidean division)."""
    if b.value == 0:
        return Int(0)
    return Int(a.value // b.value)


fn int_mod(a: Int, b: Int) -> Int:
    """Modulo for integers."""
    if b.value == 0:
        return Int(0)
    return Int(a.value % b.value)


fn int_abs(a: Int) -> Nat:
    """Absolute value."""
    return a.abs()


fn int_min(a: Int, b: Int) -> Int:
    """Minimum of two integers."""
    if a.value < b.value:
        return a
    return b


fn int_max(a: Int, b: Int) -> Int:
    """Maximum of two integers."""
    if a.value > b.value:
        return a
    return b
