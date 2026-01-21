"""
Boolean operations for Lean4 stdlib.
"""


fn lean_and(a: Bool, b: Bool) -> Bool:
    """Logical AND."""
    return a and b


fn lean_or(a: Bool, b: Bool) -> Bool:
    """Logical OR."""
    return a or b


fn lean_not(a: Bool) -> Bool:
    """Logical NOT."""
    return not a


fn lean_xor(a: Bool, b: Bool) -> Bool:
    """Logical XOR."""
    return (a and not b) or (not a and b)


fn lean_ite[T: AnyType](cond: Bool, then_val: T, else_val: T) -> T:
    """If-then-else expression."""
    if cond:
        return then_val
    return else_val


fn lean_implies(a: Bool, b: Bool) -> Bool:
    """Logical implication."""
    return (not a) or b


fn lean_iff(a: Bool, b: Bool) -> Bool:
    """Logical biconditional (if and only if)."""
    return a == b


fn lean_nand(a: Bool, b: Bool) -> Bool:
    """Logical NAND."""
    return not (a and b)


fn lean_nor(a: Bool, b: Bool) -> Bool:
    """Logical NOR."""
    return not (a or b)


fn bool_to_nat(b: Bool) -> Int:
    """Convert bool to natural number (0 or 1)."""
    return 1 if b else 0


fn nat_to_bool(n: Int) -> Bool:
    """Convert natural number to bool (0 = false, else = true)."""
    return n != 0


fn decide(prop: Bool) -> Bool:
    """Decision procedure for decidable propositions."""
    return prop
