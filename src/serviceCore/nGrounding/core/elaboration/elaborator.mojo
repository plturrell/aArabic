"""
Elaborator for converting parsed syntax to typed expressions.
"""

from collections import List
from core.parser.syntax import SyntaxNode, NodeKind
from .expr import Expr, ExprKind, Level
from .environment import Environment, Declaration
from .context import Context, MetavarContext
from .type_checker import TypeChecker, TypeError


@fieldwise_init
struct ElaborationError(Copyable, Movable):
    """An error during elaboration."""
    var message: String
    var node_kind: NodeKind

    fn __init__(out self, message: String):
        self.message = message
        self.node_kind = NodeKind.ERROR


@fieldwise_init
struct ElaborationResult(Copyable, Movable):
    """Result of elaboration."""
    var expr: Optional[Expr]
    var type: Optional[Expr]
    var errors: List[ElaborationError]

    fn __init__(out self):
        self.expr = None
        self.type = None
        self.errors = List[ElaborationError]()

    @staticmethod
    fn success(expr: Expr, type: Expr) -> ElaborationResult:
        var result = ElaborationResult()
        result.expr = expr.copy()
        result.type = type.copy()
        return result

    @staticmethod
    fn failure(message: String) -> ElaborationResult:
        var result = ElaborationResult()
        result.errors.append(ElaborationError(message))
        return result


@fieldwise_init
struct Elaborator(Copyable, Movable):
    """Elaborator for Lean4 syntax."""
    var env: Environment
    var ctx: Context
    var mctx: MetavarContext
    var type_checker: TypeChecker
    var errors: List[ElaborationError]

    fn __init__(out self):
        self.env = Environment()
        self.ctx = Context()
        self.mctx = MetavarContext()
        self.type_checker = TypeChecker(self.env)
        self.errors = List[ElaborationError]()

    fn elaborate_program(mut self, node: SyntaxNode) -> List[Declaration]:
        """Elaborate a program (list of commands)."""
        var decls = List[Declaration]()
        if node.kind != NodeKind.PROGRAM:
            self._error("Expected program node")
            return decls.copy()
        for i in range(len(node.children)):
            var child = node.children[i].copy()
            var decl = self.elaborate_command(child)
            if decl:
                decls.append(decl.value().copy())
        return decls.copy()

    fn elaborate_command(mut self, node: SyntaxNode) -> Optional[Declaration]:
        """Elaborate a top-level command."""
        if node.kind == NodeKind.DEF:
            return self.elaborate_def(node)
        elif node.kind == NodeKind.THEOREM:
            return self.elaborate_theorem(node)
        elif node.kind == NodeKind.IMPORT:
            self.elaborate_import(node)
            return None
        elif node.kind == NodeKind.NAMESPACE:
            self.elaborate_namespace(node)
            return None
        else:
            self._error("Unknown command kind")
            return None

    fn elaborate_def(mut self, node: SyntaxNode) -> Optional[Declaration]:
        """Elaborate a definition."""
        if len(node.children) < 2:
            self._error("Definition requires name and body")
            return None
        var name = node.children[0].value
        var body_result = self.elaborate_expr(node.children[len(node.children) - 1])
        if not body_result.expr:
            return None
        var expr = body_result.expr.value().copy()
        var type = body_result.type
        if type:
            var decl = Declaration.definition(name, type.value(), expr)
            self.env.add_declaration(decl)
            return decl.copy()
        return None

    fn elaborate_theorem(mut self, node: SyntaxNode) -> Optional[Declaration]:
        """Elaborate a theorem."""
        if len(node.children) < 2:
            self._error("Theorem requires name and statement")
            return None
        var name = node.children[0].value
        var stmt_result = self.elaborate_expr(node.children[1])
        if not stmt_result.expr:
            return None
        var proof = Expr.const("sorry")  # Placeholder
        if len(node.children) > 2:
            var proof_result = self.elaborate_expr(node.children[2])
            if proof_result.expr:
                proof = proof_result.expr.value().copy()
        var decl = Declaration.theorem(name, stmt_result.expr.value(), proof)
        self.env.add_declaration(decl)
        return decl.copy()

    fn elaborate_expr(mut self, node: SyntaxNode) -> ElaborationResult:
        """Elaborate an expression."""
        if node.kind == NodeKind.IDENT:
            return self._elaborate_ident(node)
        elif node.kind == NodeKind.NUMBER:
            return self._elaborate_number(node)
        elif node.kind == NodeKind.STRING:
            return self._elaborate_string(node)
        elif node.kind == NodeKind.APP:
            return self._elaborate_app(node)
        elif node.kind == NodeKind.INFIX:
            return self._elaborate_infix(node)
        else:
            return ElaborationResult.failure("Unknown expression kind")

    fn _elaborate_ident(mut self, node: SyntaxNode) -> ElaborationResult:
        var name = node.value
        var local_type = self.ctx.get_type(name)
        if local_type:
            return ElaborationResult.success(Expr.var(self.ctx.size() - 1), local_type.value())
        var global_type = self.env.get_type(name)
        if global_type:
            return ElaborationResult.success(Expr.const(name), global_type.value())
        return ElaborationResult.failure("Unknown identifier: " + name)

    fn _elaborate_number(mut self, node: SyntaxNode) -> ElaborationResult:
        var nat_type = Expr.const("Nat")
        return ElaborationResult.success(Expr.const(node.value), nat_type)

    fn _elaborate_string(mut self, node: SyntaxNode) -> ElaborationResult:
        var string_type = Expr.const("String")
        return ElaborationResult.success(Expr.const(node.value), string_type)

    fn _elaborate_app(mut self, node: SyntaxNode) -> ElaborationResult:
        if len(node.children) < 2:
            return ElaborationResult.failure("Application requires function and argument")
        var func = self.elaborate_expr(node.children[0])
        var arg = self.elaborate_expr(node.children[1])
        if func.expr and arg.expr:
            return ElaborationResult.success(
                Expr.app(func.expr.value(), arg.expr.value()),
                Expr.const("_")  # Type inference needed
            )
        return ElaborationResult.failure("Failed to elaborate application")

    fn _elaborate_infix(mut self, node: SyntaxNode) -> ElaborationResult:
        return self._elaborate_app(node)  # Treat as application

    fn elaborate_import(mut self, node: SyntaxNode):
        if len(node.children) > 0:
            self.env.add_import(node.children[0].value)

    fn elaborate_namespace(mut self, node: SyntaxNode):
        pass  # Namespace handling

    fn _error(mut self, message: String):
        self.errors.append(ElaborationError(message))

    fn has_errors(self) -> Bool:
        return len(self.errors) > 0
