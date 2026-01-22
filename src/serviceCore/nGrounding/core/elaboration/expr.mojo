"""
Expression representation for Lean4 elaboration.
"""

from collections import List
from core.kernel.level import Level


@fieldwise_init
struct ExprKind(ImplicitlyCopyable, Copyable, Movable):
    var value: Int

    fn __eq__(self, other: ExprKind) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: ExprKind) -> Bool:
        return self.value != other.value

    # Core expression kinds
    comptime VAR = ExprKind(0)        # Variable reference
    comptime CONST = ExprKind(1)      # Constant
    comptime APP = ExprKind(2)        # Function application
    comptime LAM = ExprKind(3)        # Lambda abstraction
    comptime PI = ExprKind(4)         # Pi type (dependent function type)
    comptime SORT = ExprKind(5)       # Sort (Type, Prop, etc.)
    comptime LET = ExprKind(6)        # Let expression
    comptime MVAR = ExprKind(7)       # Metavariable
    comptime FVAR = ExprKind(8)       # Free variable


@fieldwise_init
struct Expr(Copyable, Movable):
    """Core expression type for Lean4."""
    var kind: ExprKind
    var level: Level
    var name: String
    var type: Optional[Self]
    var body: Optional[Self]
    var args: List[Self]
    var binder_info: Int  # 0=default, 1=implicit, 2=strict_implicit, 3=inst_implicit

    fn __init__(out self, kind: ExprKind):
        self.kind = kind
        self.level = Level.zero()
        self.name = ""
        self.type = None
        self.body = None
        self.args = List[Self]()
        self.binder_info = 0

    @staticmethod
    fn var(index: Int) -> Expr:
        var expr = Expr(ExprKind.VAR)
        expr.name = str(index)
        return expr^

    @staticmethod
    fn const(name: String, level: Level = Level.zero()) -> Expr:
        var expr = Expr(ExprKind.CONST)
        expr.name = name
        expr.level = level.copy()
        return expr^

    @staticmethod
    fn app(func: Expr, arg: Expr) -> Expr:
        var expr = Expr(ExprKind.APP)
        expr.args.append(func.copy())
        expr.args.append(arg.copy())
        return expr^

    @staticmethod
    fn lam(name: String, type: Expr, body: Expr, binder_info: Int = 0) -> Expr:
        var expr = Expr(ExprKind.LAM)
        expr.name = name
        expr.type = type.copy()
        expr.body = body.copy()
        expr.binder_info = binder_info
        return expr^

    @staticmethod
    fn pi(name: String, type: Expr, body: Expr, binder_info: Int = 0) -> Expr:
        var expr = Expr(ExprKind.PI)
        expr.name = name
        expr.type = type.copy()
        expr.body = body.copy()
        expr.binder_info = binder_info
        return expr^

    @staticmethod
    fn sort(level: Level) -> Expr:
        var expr = Expr(ExprKind.SORT)
        expr.level = level.copy()
        return expr^

    fn is_var(self) -> Bool:
        return self.kind == ExprKind.VAR

    fn is_const(self) -> Bool:
        return self.kind == ExprKind.CONST

    fn is_app(self) -> Bool:
        return self.kind == ExprKind.APP

    fn is_lambda(self) -> Bool:
        return self.kind == ExprKind.LAM

    fn is_pi(self) -> Bool:
        return self.kind == ExprKind.PI

    fn is_sort(self) -> Bool:
        return self.kind == ExprKind.SORT


fn expr_to_string(expr: Expr) -> String:
    """Convert expression to string representation."""
    if expr.is_var():
        return "#" + expr.name
    elif expr.is_const():
        return expr.name
    elif expr.is_app():
        if len(expr.args) >= 2:
            return "(" + expr_to_string(expr.args[0]) + " " + expr_to_string(expr.args[1]) + ")"
        else:
            return "(app)"
    elif expr.is_lambda():
        var type_str = ""
        var body_str = ""
        if expr.type:
            type_str = expr_to_string(expr.type.value())
        if expr.body:
            body_str = expr_to_string(expr.body.value())
        return "(λ " + expr.name + " : " + type_str + ", " + body_str + ")"
    elif expr.is_pi():
        var type_str = ""
        var body_str = ""
        if expr.type:
            type_str = expr_to_string(expr.type.value())
        if expr.body:
            body_str = expr_to_string(expr.body.value())
        return "(Π " + expr.name + " : " + type_str + ", " + body_str + ")"
    elif expr.is_sort():
        return "Type"
    else:
        return "?"
