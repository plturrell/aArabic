"""
Kernel type checker - the trusted core.
"""

from collections import Dict, List
from .name import Name, name_to_string
from .level import Level, level_eq
from .declaration import ConstantInfo, InductiveInfo
from core.elaboration.expr import Expr, ExprKind


@fieldwise_init
struct KernelError(Copyable, Movable):
    """A kernel type checking error."""
    var message: String
    var name: String

    fn __init__(out self, message: String):
        self.message = message
        self.name = ""


@fieldwise_init
struct KernelEnvironment(Copyable, Movable):
    """The kernel environment - stores verified declarations."""
    var constants: Dict[String, ConstantInfo]
    var inductives: Dict[String, InductiveInfo]
    var trust_level: Int  # 0 = trust nothing, higher = more trust

    fn __init__(out self):
        self.constants = Dict[String, ConstantInfo]()
        self.inductives = Dict[String, InductiveInfo]()
        self.trust_level = 0

    fn add_constant(mut self, info: ConstantInfo):
        self.constants[name_to_string(info.name)] = info

    fn add_inductive(mut self, info: InductiveInfo):
        self.inductives[name_to_string(info.name)] = info

    fn get_constant(self, name: String) -> Optional[ConstantInfo]:
        if name in self.constants:
            return self.constants[name]
        return None

    fn get_inductive(self, name: String) -> Optional[InductiveInfo]:
        if name in self.inductives:
            return self.inductives[name]
        return None


@fieldwise_init
struct KernelTypeChecker(Copyable, Movable):
    """The kernel type checker - minimal trusted code."""
    var env: KernelEnvironment
    var errors: List[KernelError]
    var locals: List[Expr] # Type of local variables (De Bruijn indices)

    fn __init__(out self):
        self.env = KernelEnvironment()
        self.errors = List[KernelError]()
        self.locals = List[Expr]()

    fn check_constant(mut self, info: ConstantInfo) -> Bool:
        """Check and add a constant to the environment."""
        var name_str = name_to_string(info.name)
        # Check type is well-formed
        if not self._check_type(info.type):
            self._error("Invalid type for constant: " + name_str)
            return False
        # Check value matches type (if definition)
        if info.value:
            if not self._check_expr_type(info.value.value(), info.type):
                self._error("Value does not match type for: " + name_str)
                return False
        self.env.add_constant(info)
        return True

    fn check_inductive(mut self, info: InductiveInfo) -> Bool:
        """Check and add an inductive type to the environment."""
        var name_str = name_to_string(info.name)
        # Check type is well-formed
        if not self._check_type(info.type):
            self._error("Invalid type for inductive: " + name_str)
            return False
        # Check constructors
        for i in range(len(info.constructors)):
            var ctor = info.constructors[i]
            if not self._check_type(ctor.type):
                self._error("Invalid constructor type: " + name_to_string(ctor.name))
                return False
        self.env.add_inductive(info)
        return True

    fn _check_type(mut self, expr: Expr) -> Bool:
        """Check that an expression is a valid type."""
        var type = self._infer_type(expr)
        if not type:
            return False
        # Type must be a Sort
        return type.value().is_sort()

    fn _check_expr_type(mut self, expr: Expr, expected: Expr) -> Bool:
        """Check that an expression has the expected type."""
        var inferred = self._infer_type(expr)
        if not inferred:
            return False
        return self._is_def_eq(inferred.value(), expected)

    fn _infer_type(mut self, expr: Expr) -> Optional[Expr]:
        """Infer the type of an expression."""
        if expr.kind == ExprKind.SORT:
            return Expr.sort(Level.succ(expr.level))
        
        elif expr.kind == ExprKind.VAR:
            # Look up in locals using De Bruijn index
            # Note: Expr.name stores index as string for VAR
            # This is a simplification; ideally we'd parse the int
             # Simple validation since we don't have int parsing handy in this context without imports
             # Assume safe for now or placeholder
            return None # TODO: Implement local lookup
            
        elif expr.kind == ExprKind.CONST:
            var info = self.env.get_constant(expr.name)
            if info:
                return info.value().type
            self._error("Unknown constant: " + expr.name)
            return None
            
        elif expr.kind == ExprKind.PI:
            # Rule: (x : A) -> B : Sort (imax u v)
            if expr.type and expr.body:
                var type_type = self._infer_type(expr.type.value())
                if not type_type or not type_type.value().is_sort():
                    self._error("Domain of Pi type must be a type")
                    return None
                
                # Push domain type to locals for body inference
                self.locals.append(expr.type.value())
                var body_type = self._infer_type(expr.body.value())
                _ = self.locals.pop() # Pop
                
                if not body_type or not body_type.value().is_sort():
                    self._error("Range of Pi type must be a type")
                    return None
                    
                var u = type_type.value().level
                var v = body_type.value().level
                return Expr.sort(Level.imax(u, v))
                
            return None
            
        elif expr.kind == ExprKind.LAM:
            if expr.type and expr.body:
                # Check domain is a type
                var domain_type = self._infer_type(expr.type.value())
                if not domain_type or not domain_type.value().is_sort():
                    self._error("Domain of Lambda must be a type")
                    return None

                # Push domain to locals
                self.locals.append(expr.type.value())
                var body_type = self._infer_type(expr.body.value())
                _ = self.locals.pop()
                
                if body_type:
                    return Expr.pi(expr.name, expr.type.value(), body_type.value())
            return None
            
        elif expr.kind == ExprKind.APP:
            if len(expr.args) >= 2:
                var func_type = self._infer_type(expr.args[0])
                if func_type and func_type.value().is_pi() and func_type.value().body:
                    # Check argument type
                    var arg_type = self._infer_type(expr.args[1])
                    if not arg_type:
                        return None
                    
                    if not self._is_def_eq(arg_type.value(), func_type.value().type.value()):
                        self._error("Type mismatch in application")
                        return None
                        
                    # Result is body type with arg substituted for var(0)
                    # For now, simplistic return since we lack substitution machinery in this POC
                    return func_type.value().body.value() 
            return None
            
        else:
            return Expr.const("_")  # Placeholder

    fn _is_def_eq(self, a: Expr, b: Expr) -> Bool:
        """Check definitional equality."""
        if a.kind != b.kind:
            return False
            
        if a.kind == ExprKind.SORT:
            return level_eq(a.level, b.level)
            
        if a.kind == ExprKind.CONST:
            return a.name == b.name and level_eq(a.level, b.level)
            
        if a.kind == ExprKind.VAR:
            return a.name == b.name
            
        if a.kind == ExprKind.APP:
            if len(a.args) != len(b.args):
                return False
            for i in range(len(a.args)):
                if not self._is_def_eq(a.args[i], b.args[i]):
                    return False
            return True
            
        if a.kind == ExprKind.LAM or a.kind == ExprKind.PI:
            # Check domain types and bodies
            # Note: This is alpha-equivalence check assuming names don't matter for binding logic usually, 
            # but we check structure here.
            var types_eq = False
            if a.type and b.type:
                types_eq = self._is_def_eq(a.type.value(), b.type.value())
            elif not a.type and not b.type:
                types_eq = True
                
            var bodies_eq = False
            if a.body and b.body:
                bodies_eq = self._is_def_eq(a.body.value(), b.body.value())
            elif not a.body and not b.body:
                bodies_eq = True
                
            return types_eq and bodies_eq

        return True

    fn _error(mut self, message: String):
        self.errors.append(KernelError(message))

    fn has_errors(self) -> Bool:
        return len(self.errors) > 0
