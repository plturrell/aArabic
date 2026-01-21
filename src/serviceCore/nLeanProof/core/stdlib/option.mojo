"""
Option type for Lean4 stdlib.
"""


@fieldwise_init
struct LeanOption[T: Copyable & Movable](Copyable, Movable):
    """Option type (none or some value)."""
    var value: Optional[T]

    fn __init__(out self):
        self.value = None

    fn none() -> LeanOption[T]:
        return LeanOption[T]()

    fn some(value: T) -> LeanOption[T]:
        var opt = LeanOption[T]()
        opt.value = value
        return opt

    fn is_some(self) -> Bool:
        return self.value is not None

    fn is_none(self) -> Bool:
        return self.value is None

    fn get(self) -> Optional[T]:
        return self.value

    fn get_or_else(self, default: T) -> T:
        if self.value:
            return self.value.value()
        return default


fn option_none[T: Copyable & Movable]() -> LeanOption[T]:
    """Create None option."""
    return LeanOption[T].none()


fn option_some[T: Copyable & Movable](value: T) -> LeanOption[T]:
    """Create Some option."""
    return LeanOption[T].some(value)


fn option_map[T: Copyable & Movable, U: Copyable & Movable](
    opt: LeanOption[T],
    f: fn(T) -> U
) -> LeanOption[U]:
    """Map a function over an option."""
    if opt.is_some():
        return LeanOption[U].some(f(opt.value.value()))
    return LeanOption[U].none()


fn option_bind[T: Copyable & Movable, U: Copyable & Movable](
    opt: LeanOption[T],
    f: fn(T) -> LeanOption[U]
) -> LeanOption[U]:
    """Bind (flatMap) for option."""
    if opt.is_some():
        return f(opt.value.value())
    return LeanOption[U].none()


fn option_or_else[T: Copyable & Movable](
    opt: LeanOption[T],
    default: LeanOption[T]
) -> LeanOption[T]:
    """Return opt if some, otherwise default."""
    if opt.is_some():
        return opt
    return default


fn option_filter[T: Copyable & Movable](
    opt: LeanOption[T],
    pred: fn(T) -> Bool
) -> LeanOption[T]:
    """Filter option by predicate."""
    if opt.is_some() and pred(opt.value.value()):
        return opt
    return LeanOption[T].none()


fn option_to_list[T: Copyable & Movable](opt: LeanOption[T]) -> List[T]:
    """Convert option to list."""
    var result = List[T]()
    if opt.is_some():
        result.append(opt.value.value())
    return result


fn option_zip[T: Copyable & Movable, U: Copyable & Movable](
    a: LeanOption[T],
    b: LeanOption[U]
) -> LeanOption[Tuple[T, U]]:
    """Zip two options."""
    if a.is_some() and b.is_some():
        return LeanOption[Tuple[T, U]].some((a.value.value(), b.value.value()))
    return LeanOption[Tuple[T, U]].none()
