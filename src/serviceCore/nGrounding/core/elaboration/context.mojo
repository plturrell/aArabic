"""
Local context for elaboration.
"""

from collections import List, Dict
from .expr import Expr


@fieldwise_init
struct LocalDecl(Copyable, Movable):
    var name: String
    var type: Expr
    var value: Optional[Expr]
    var index: Int

    fn __init__(out self, name: String, type: Expr, index: Int, value: Optional[Expr] = None):
        self.name = name
        self.type = type.copy()
        self.value = value.copy() if value else None
        self.index = index

    fn __copyinit__(out self, other: LocalDecl):
        self.name = other.name
        self.type = other.type.copy()
        self.value = other.value.copy() if other.value else None
        self.index = other.index


@fieldwise_init
struct Context(Copyable, Movable):
    var locals: List[LocalDecl]
    var name_map: Dict[String, Int]

    fn __init__(out self):
        self.locals = List[LocalDecl]()
        self.name_map = Dict[String, Int]()

    fn add_local(mut self, name: String, type: Expr, value: Optional[Expr] = None):
        var index = len(self.locals)
        var decl = LocalDecl(name, type, index, value)
        self.locals.append(decl)
        self.name_map[name] = index

    fn get_local(self, index: Int) -> Optional[LocalDecl]:
        if index >= 0 and index < len(self.locals):
            return self.locals[index]
        return None

    fn get_type(self, name: String) -> Optional[Expr]:
        if name in self.name_map:
            var index = self.name_map[name]
            return self.locals[index].type
        return None

    fn size(self) -> Int:
        return len(self.locals)


@fieldwise_init
struct MetavarContext(Copyable, Movable):
    var assignments: Dict[String, Expr]

    fn __init__(out self):
        self.assignments = Dict[String, Expr]()

    fn assign(mut self, name: String, val: Expr):
        self.assignments[name] = val.copy()

    fn get_assignment(self, name: String) -> Optional[Expr]:
        if name in self.assignments:
            return self.assignments[name]
        return None