"""
Expression evaluator for the Lean4 runtime.
"""

from collections import Dict, List
from .value import Value, ValueKind, Object, Closure
from core.elaboration.expr import Expr, ExprKind
from core.elaboration.environment import Environment, Declaration


@fieldwise_init
struct EvalError(Copyable, Movable):
    """Evaluation error."""
    var message: String
    var expr_name: String

    fn __init__(out self, message: String):
        self.message = message
        self.expr_name = ""


@fieldwise_init
struct EvalResult(Copyable, Movable):
    """Result of evaluation."""
    var value: Optional[Value]
    var error: Optional[EvalError]

    fn __init__(out self):
        self.value = None
        self.error = None

    fn success(v: Value) -> EvalResult:
        var r = EvalResult()
        r.value = v
        return r

    fn failure(message: String) -> EvalResult:
        var r = EvalResult()
        r.error = EvalError(message)
        return r

    fn is_success(self) -> Bool:
        return self.value is not None


@fieldwise_init
struct Evaluator(Copyable, Movable):
    """Expression evaluator."""
    var env: Environment
    var stack: List[Value]
    var globals: Dict[String, Value]
    var errors: List[EvalError]

    fn __init__(out self, env: Environment):
        self.env = env.copy()
        self.stack = List[Value]()
        self.globals = Dict[String, Value]()
        self.errors = List[EvalError]()
        self._init_builtins()

    fn _init_builtins(mut self):
        """Initialize built-in functions."""
        self.globals["Nat.zero"] = Value.mk_nat(0)
        self.globals["Nat.succ"] = Value.mk_closure("Nat.succ", 1)
        self.globals["Nat.add"] = Value.mk_closure("Nat.add", 2)
        self.globals["Nat.mul"] = Value.mk_closure("Nat.mul", 2)
        self.globals["Bool.true"] = Value.mk_bool(True)
        self.globals["Bool.false"] = Value.mk_bool(False)
        self.globals["Unit.unit"] = Value.mk_unit()

    fn eval(mut self, expr: Expr) -> EvalResult:
        """Evaluate an expression."""
        if expr.kind == ExprKind.CONST:
            return self._eval_const(expr)
        elif expr.kind == ExprKind.VAR:
            return self._eval_var(expr)
        elif expr.kind == ExprKind.APP:
            return self._eval_app(expr)
        elif expr.kind == ExprKind.LAM:
            return self._eval_lam(expr)
        elif expr.kind == ExprKind.LET:
            return self._eval_let(expr)
        else:
            return EvalResult.failure("Unsupported expression kind")

    fn _eval_const(mut self, expr: Expr) -> EvalResult:
        """Evaluate a constant."""
        var name = expr.name
        if name in self.globals:
            return EvalResult.success(self.globals[name])
        var decl = self.env.get_declaration(name)
        if decl:
            var d = decl.value()
            if d.value:
                var result = self.eval(d.value.value())
                if result.is_success():
                    self.globals[name] = result.value.value()
                return result
        return EvalResult.failure("Unknown constant: " + name)

    fn _eval_var(mut self, expr: Expr) -> EvalResult:
        """Evaluate a variable."""
        var index = atol(expr.name)
        if index >= 0 and index < len(self.stack):
            var stack_index = len(self.stack) - 1 - index
            return EvalResult.success(self.stack[stack_index])
        return EvalResult.failure("Variable out of scope")

    fn _eval_app(mut self, expr: Expr) -> EvalResult:
        """Evaluate function application."""
        if len(expr.args) < 2:
            return EvalResult.failure("Application needs function and argument")
        var func_result = self.eval(expr.args[0])
        if not func_result.is_success():
            return func_result
        var func = func_result.value.value()
        var arg_result = self.eval(expr.args[1])
        if not arg_result.is_success():
            return arg_result
        var arg = arg_result.value.value()
        return self._apply(func, arg)

    fn _apply(mut self, func: Value, arg: Value) -> EvalResult:
        """Apply a function to an argument."""
        if func.is_closure():
            var closure = func.closure.value()
            var new_closure = closure.apply(arg)
            if new_closure:
                var v = Value(ValueKind.CLOSURE)
                v.closure = new_closure
                return EvalResult.success(v)
            return self._call_builtin(closure.func_name, closure.partial_args)
        return EvalResult.failure("Cannot apply non-function")

    fn _call_builtin(mut self, name: String, args: List[Value]) -> EvalResult:
        """Call a built-in function."""
        if name == "Nat.succ" and len(args) >= 1:
            return EvalResult.success(Value.mk_nat(args[0].to_nat() + 1))
        elif name == "Nat.add" and len(args) >= 2:
            return EvalResult.success(Value.mk_nat(args[0].to_nat() + args[1].to_nat()))
        elif name == "Nat.mul" and len(args) >= 2:
            return EvalResult.success(Value.mk_nat(args[0].to_nat() * args[1].to_nat()))
        return EvalResult.failure("Unknown builtin: " + name)

    fn _eval_lam(mut self, expr: Expr) -> EvalResult:
        """Evaluate a lambda expression."""
        return EvalResult.success(Value.mk_closure(expr.name, 1))

    fn _eval_let(mut self, expr: Expr) -> EvalResult:
        """Evaluate a let expression."""
        if expr.type and expr.body:
            var val_result = self.eval(expr.type.value())
            if not val_result.is_success():
                return val_result
            self.stack.append(val_result.value.value())
            var result = self.eval(expr.body.value())
            _ = self.stack.pop()
            return result
        return EvalResult.failure("Invalid let expression")

    fn has_errors(self) -> Bool:
        return len(self.errors) > 0
