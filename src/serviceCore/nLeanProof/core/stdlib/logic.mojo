"""
Logic and propositions for Lean4 stdlib.

In Lean4, Prop is a special universe for propositions.
Proofs are terms of propositions, and proof irrelevance
means all proofs of the same proposition are definitionally equal.
"""


@fieldwise_init
struct Prop(Copyable, Movable):
    """A proposition (type in Prop universe)."""
    var name: String
    var is_proven: Bool

    fn __init__(out self, name: String):
        self.name = name
        self.is_proven = False

    fn proven(self) -> Prop:
        var p = Prop(self.name)
        p.is_proven = True
        return p


@fieldwise_init
struct True_(Copyable, Movable):
    """The true proposition (trivially provable)."""

    fn intro() -> True_:
        """The canonical proof of True."""
        return True_()


@fieldwise_init
struct False_(Copyable, Movable):
    """The false proposition (unprovable)."""

    fn elim[T: AnyType](self) -> T:
        """Ex falso quodlibet - from False, derive anything."""
        # This should never be called in consistent code
        abort("False.elim called - inconsistent proof")


@fieldwise_init
struct And(Copyable, Movable):
    """Conjunction of two propositions."""
    var left: Bool
    var right: Bool

    fn __init__(out self, left: Bool, right: Bool):
        self.left = left
        self.right = right

    fn intro(left: Bool, right: Bool) -> And:
        """Introduce a conjunction from two proofs."""
        return And(left, right)

    fn elim_left(self) -> Bool:
        """Extract the left component."""
        return self.left

    fn elim_right(self) -> Bool:
        """Extract the right component."""
        return self.right


@fieldwise_init
struct Or(Copyable, Movable):
    """Disjunction of two propositions."""
    var is_left: Bool
    var value: Bool

    fn __init__(out self, is_left: Bool, value: Bool):
        self.is_left = is_left
        self.value = value

    fn intro_left(left: Bool) -> Or:
        """Introduce a disjunction from the left."""
        return Or(True, left)

    fn intro_right(right: Bool) -> Or:
        """Introduce a disjunction from the right."""
        return Or(False, right)

    fn elim[T: AnyType](self, left_case: fn(Bool) -> T, right_case: fn(Bool) -> T) -> T:
        """Eliminate a disjunction by case analysis."""
        if self.is_left:
            return left_case(self.value)
        return right_case(self.value)


@fieldwise_init
struct Not(Copyable, Movable):
    """Negation of a proposition (P → False)."""
    var negated: Bool

    fn __init__(out self, p: Bool):
        self.negated = not p

    fn intro(p: Bool) -> Not:
        """Negate a proposition."""
        return Not(p)

    fn elim(self, p: Bool) -> Bool:
        """If we have ¬P and P, derive False."""
        return not p and self.negated


@fieldwise_init
struct Eq[T: AnyType](Copyable, Movable):
    """Propositional equality."""
    var left: T
    var right: T

    fn __init__(out self, left: T, right: T):
        self.left = left
        self.right = right

    fn refl(x: T) -> Eq[T]:
        """Reflexivity: x = x."""
        return Eq[T](x, x)

    fn symm(self) -> Eq[T]:
        """Symmetry: if a = b then b = a."""
        return Eq[T](self.right, self.left)


@fieldwise_init
struct Exists[T: AnyType](Copyable, Movable):
    """Existential quantification."""
    var witness: T
    var satisfied: Bool

    fn intro(witness: T, property_holds: Bool) -> Exists[T]:
        """Introduce an existential from a witness and proof."""
        return Exists[T](witness, property_holds)


@fieldwise_init
struct Decidable(Copyable, Movable):
    """A proposition with a decision procedure."""
    var value: Bool
    var is_true: Bool

    fn decide(b: Bool) -> Decidable:
        return Decidable(b, b)
