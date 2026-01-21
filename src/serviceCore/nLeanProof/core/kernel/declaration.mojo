"""
Declaration types for the Lean4 kernel.
"""

from collections import List
from .name import Name
from .level import Level
from core.elaboration.expr import Expr


@fieldwise_init
struct ConstantInfo(Copyable, Movable):
    """Information about a constant in the environment."""
    var name: Name
    var level_params: List[String]
    var type: Expr
    var value: Optional[Expr]  # None for axioms
    var is_unsafe: Bool
    var is_partial: Bool

    fn __init__(out self, name: Name, type: Expr):
        self.name = name
        self.level_params = List[String]()
        self.type = type
        self.value = None
        self.is_unsafe = False
        self.is_partial = False

    fn axiom_info(name: Name, type: Expr, level_params: List[String]) -> ConstantInfo:
        var info = ConstantInfo(name, type)
        info.level_params = level_params.copy()
        return info

    fn def_info(name: Name, type: Expr, value: Expr, level_params: List[String]) -> ConstantInfo:
        var info = ConstantInfo(name, type)
        info.value = value
        info.level_params = level_params.copy()
        return info

    fn theorem_info(name: Name, type: Expr, proof: Expr, level_params: List[String]) -> ConstantInfo:
        return ConstantInfo.def_info(name, type, proof, level_params)

    fn is_axiom(self) -> Bool:
        return not self.value

    fn is_definition(self) -> Bool:
        return self.value is not None


@fieldwise_init
struct ConstructorInfo(Copyable, Movable):
    """Information about an inductive constructor."""
    var name: Name
    var type: Expr
    var num_params: Int
    var num_fields: Int

    fn __init__(out self, name: Name, type: Expr):
        self.name = name
        self.type = type
        self.num_params = 0
        self.num_fields = 0


@fieldwise_init
struct InductiveInfo(Copyable, Movable):
    """Information about an inductive type."""
    var name: Name
    var level_params: List[String]
    var type: Expr
    var num_params: Int
    var num_indices: Int
    var constructors: List[ConstructorInfo]
    var is_recursive: Bool
    var is_nested: Bool
    var is_unsafe: Bool

    fn __init__(out self, name: Name, type: Expr):
        self.name = name
        self.level_params = List[String]()
        self.type = type
        self.num_params = 0
        self.num_indices = 0
        self.constructors = List[ConstructorInfo]()
        self.is_recursive = False
        self.is_nested = False
        self.is_unsafe = False

    fn add_constructor(mut self, ctor: ConstructorInfo):
        self.constructors.append(ctor)


@fieldwise_init
struct RecursorRule(Copyable, Movable):
    """A recursor computation rule."""
    var ctor: Name
    var num_fields: Int
    var rhs: Expr


@fieldwise_init
struct RecursorInfo(Copyable, Movable):
    """Information about a recursor."""
    var name: Name
    var level_params: List[String]
    var type: Expr
    var num_params: Int
    var num_indices: Int
    var num_motives: Int
    var num_minors: Int
    var rules: List[RecursorRule]
    var is_k: Bool

    fn __init__(out self, name: Name, type: Expr):
        self.name = name
        self.level_params = List[String]()
        self.type = type
        self.num_params = 0
        self.num_indices = 0
        self.num_motives = 1
        self.num_minors = 0
        self.rules = List[RecursorRule]()
        self.is_k = False


@fieldwise_init
struct QuotInfo(Copyable, Movable):
    """Information about quotient types."""
    var kind: Int  # 0=type, 1=ctor, 2=lift, 3=ind

    comptime QUOT = QuotInfo(0)
    comptime QUOT_MK = QuotInfo(1)
    comptime QUOT_LIFT = QuotInfo(2)
    comptime QUOT_IND = QuotInfo(3)
