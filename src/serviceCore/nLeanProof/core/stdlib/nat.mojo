"""
Natural number implementation for Lean4 stdlib.
"""


@fieldwise_init
struct Nat(ImplicitlyCopyable, Copyable, Movable):
    """Natural number type."""
    var value: Int

    fn __init__(out self, value: Int = 0):
        self.value = value if value >= 0 else 0

    fn zero() -> Nat:
        return Nat(0)

    fn one() -> Nat:
        return Nat(1)

    fn succ(self) -> Nat:
        return Nat(self.value + 1)

    fn pred(self) -> Nat:
        if self.value > 0:
            return Nat(self.value - 1)
        return Nat(0)

    fn is_zero(self) -> Bool:
        return self.value == 0

    fn to_int(self) -> Int:
        return self.value

    fn __eq__(self, other: Nat) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Nat) -> Bool:
        return self.value != other.value

    fn __lt__(self, other: Nat) -> Bool:
        return self.value < other.value

    fn __le__(self, other: Nat) -> Bool:
        return self.value <= other.value

    fn __gt__(self, other: Nat) -> Bool:
        return self.value > other.value

    fn __ge__(self, other: Nat) -> Bool:
        return self.value >= other.value


fn nat_add(a: Nat, b: Nat) -> Nat:
    """Add two natural numbers."""
    return Nat(a.value + b.value)


fn nat_mul(a: Nat, b: Nat) -> Nat:
    """Multiply two natural numbers."""
    return Nat(a.value * b.value)


fn nat_sub(a: Nat, b: Nat) -> Nat:
    """Subtract natural numbers (saturating at zero)."""
    if a.value >= b.value:
        return Nat(a.value - b.value)
    return Nat(0)


fn nat_div(a: Nat, b: Nat) -> Nat:
    """Divide natural numbers."""
    if b.value == 0:
        return Nat(0)
    return Nat(a.value // b.value)


fn nat_mod(a: Nat, b: Nat) -> Nat:
    """Modulo for natural numbers."""
    if b.value == 0:
        return Nat(0)
    return Nat(a.value % b.value)


fn nat_pow(base: Nat, exp: Nat) -> Nat:
    """Power for natural numbers."""
    var result = 1
    var b = base.value
    var e = exp.value
    while e > 0:
        if e & 1 == 1:
            result *= b
        b *= b
        e >>= 1
    return Nat(result)


fn nat_gcd(a: Nat, b: Nat) -> Nat:
    """Greatest common divisor."""
    var x = a.value
    var y = b.value
    while y != 0:
        var temp = y
        y = x % y
        x = temp
    return Nat(x)


fn nat_lcm(a: Nat, b: Nat) -> Nat:
    """Least common multiple."""
    if a.value == 0 or b.value == 0:
        return Nat(0)
    var g = nat_gcd(a, b)
    return Nat((a.value // g.value) * b.value)


fn nat_min(a: Nat, b: Nat) -> Nat:
    """Minimum of two naturals."""
    if a.value < b.value:
        return a
    return b


fn nat_max(a: Nat, b: Nat) -> Nat:
    """Maximum of two naturals."""
    if a.value > b.value:
        return a
    return b
