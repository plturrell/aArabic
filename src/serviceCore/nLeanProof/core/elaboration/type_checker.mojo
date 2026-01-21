"""
Type checker for Lean4 expressions.
"""

from collections import List
from .expr import Expr, ExprKind, Level
from .environment import Environment, Declaration
from .context import Context, MetavarContext


@fieldwise_init
struct TypeError(Copyable, Movable):
    """Type checking error."""
    var message: String
    var location: String

    fn __init__(out self, message: String):
        self.message = message
        self.location = ""


@fieldwise_init
struct TypeChecker(Copyable, Movable):
    """Type checker for Lean4 expressions."""
    var env: Environment
    var ctx: Context
    var mctx: MetavarContext
    var errors: List[TypeError]

    fn __init__(out self, env: Environment):
        self.env = env.copy()
        self.ctx = Context()
        self.mctx = MetavarContext()
        self.errors = List[TypeError]()

    fn infer(mut self, expr: Expr) -> Optional[Expr]:
        """Infer the type of an expression."""
        if expr.kind == ExprKind.VAR:
            return self._infer_var(expr)
        elif expr.kind == ExprKind.CONST:
            return self._infer_const(expr)
        elif expr.kind == ExprKind.APP:
            return self._infer_app(expr)
        elif expr.kind == ExprKind.LAM:
            return self._infer_lam(expr)
        elif expr.kind == ExprKind.PI:
            return self._infer_pi(expr)
        elif expr.kind == ExprKind.SORT:
            return self._infer_sort(expr)
        elif expr.kind == ExprKind.LET:
            return self._infer_let(expr)
        else:
            self._error("Unknown expression kind")
            return None

    fn check(mut self, expr: Expr, expected: Expr) -> Bool:
        """Check that an expression has the expected type."""
        var inferred = self.infer(expr)
        if not inferred:
            return False
        return self._is_def_eq(inferred.value(), expected)

    fn _infer_var(mut self, expr: Expr) -> Optional[Expr]:
        """Infer type of a variable."""
        var index = atol(expr.name)
        var decl = self.ctx.get_by_index(index)
        if decl:
            return decl.value().type
        self._error("Unknown variable: " + expr.name)
        return None

    fn _infer_const(mut self, expr: Expr) -> Optional[Expr]:
        """Infer type of a constant."""
        var type = self.env.get_type(expr.name)
        if type:
            return type
        self._error("Unknown constant: " + expr.name)
        return None

    fn _infer_app(mut self, expr: Expr) -> Optional[Expr]:
        """Infer type of function application."""
        if len(expr.args) < 2:
            self._error("Application requires function and argument")
            return None
        var func_type = self.infer(expr.args[0])
        if not func_type:
            return None
        var ft = func_type.value()
        if not ft.is_pi():
            self._error("Expected function type in application")
            return None
        var arg = expr.args[1]
        if ft.type and not self.check(arg, ft.type.value()):
            self._error("Argument type mismatch")
            return None
        if ft.body:
            return self._subst(ft.body.value(), arg)
        return None

    fn _infer_lam(mut self, expr: Expr) -> Optional[Expr]:
        """Infer type of lambda abstraction."""
        if not expr.type:
            self._error("Lambda requires type annotation")
            return None
        var param_type = expr.type.value()
        self.ctx.push(expr.name, param_type)
        var body_type: Optional[Expr] = None
        if expr.body:
            body_type = self.infer(expr.body.value())
        self.ctx.pop()
        if body_type:
            return Expr.pi(expr.name, param_type, body_type.value(), expr.binder_info)
        return None

    fn _infer_pi(mut self, expr: Expr) -> Optional[Expr]:
        """Infer type of pi type (always a Sort)."""
        return Expr.sort(Level.succ(Level.zero()))

    fn _infer_sort(mut self, expr: Expr) -> Optional[Expr]:
        """Infer type of a Sort (the next Sort up)."""
        return Expr.sort(Level.succ(expr.level))

    fn _infer_let(mut self, expr: Expr) -> Optional[Expr]:
        """Infer type of let expression."""
        if expr.type and expr.body:
            var decl_type = expr.type.value()
            self.ctx.push(expr.name, decl_type)
            var result = self.infer(expr.body.value())
            self.ctx.pop()
            return result
        return None

    fn _subst(self, expr: Expr, value: Expr) -> Expr:
        """Substitute a value for the bound variable in an expression."""
        # Simplified substitution - full implementation would handle de Bruijn indices
        return expr

    fn _is_def_eq(self, a: Expr, b: Expr) -> Bool:
        """Check if two expressions are definitionally equal."""
        if a.kind != b.kind:
            return False
        if a.kind == ExprKind.CONST:
            return a.name == b.name
        if a.kind == ExprKind.SORT:
            return a.level.value == b.level.value
        return True  # Simplified for now

    fn _error(mut self, message: String):
        """Record a type error."""
        self.errors.append(TypeError(message))

    fn has_errors(self) -> Bool:
        return len(self.errors) > 0
