"""
Minimal Lean4 parser scaffold (def command subset).
"""

from collections import List
from core.lexer.token import Token, TokenKind
from core.parser.syntax import SyntaxNode, NodeKind


struct Parser:
    var tokens: List[Token]
    var index: Int
    var tmp_nodes: List[SyntaxNode]
    var tmp_ops: List[String]
    var tmp_precs: List[Int]
    var tmp_right_assoc: List[Bool]

    fn __init__(out self, tokens: List[Token]):
        self.tokens = tokens.copy()
        self.index = 0
        self.tmp_nodes = List[SyntaxNode]()
        self.tmp_ops = List[String]()
        self.tmp_precs = List[Int]()
        self.tmp_right_assoc = List[Bool]()

    fn is_at_end(self) -> Bool:
        return self.index >= len(self.tokens) or self.tokens[self.index].kind == TokenKind.EOF

    fn current(self) -> Token:
        return self.tokens[self.index].copy()

    fn advance(mut self) -> Token:
        var token = self.current()
        if not self.is_at_end():
            self.index += 1
        return token.copy()

    fn skip_trivia(mut self):
        while not self.is_at_end() and self.current().kind == TokenKind.COMMENT:
            _ = self.advance()

    fn make_leaf(self, kind: NodeKind, value: String) -> SyntaxNode:
        var children = List[SyntaxNode]()
        return SyntaxNode(kind, value, children.copy())

    fn apply_infix(mut self):
        var rhs = self.tmp_nodes.pop()
        var lhs = self.tmp_nodes.pop()
        var op = self.tmp_ops.pop()
        var _ = self.tmp_precs.pop()
        var _ = self.tmp_right_assoc.pop()

        var children = List[SyntaxNode]()
        children.append(lhs.copy())
        children.append(rhs.copy())
        self.tmp_nodes.append(SyntaxNode(NodeKind.INFIX, op, children.copy()))

    fn is_expr_stop(self, token: Token, stop_on_colon: Bool, stop_on_colon_eq: Bool) -> Bool:
        if token.kind != TokenKind.SYMBOL:
            return False
        if stop_on_colon_eq and token.lexeme == ":=":
            return True
        if stop_on_colon and token.lexeme == ":":
            return True
        if token.lexeme == ")" or token.lexeme == "]" or token.lexeme == "}" or token.lexeme == ",":
            return True
        return False

    fn is_primary_start(self, token: Token) -> Bool:
        if token.kind == TokenKind.IDENT:
            return True
        if token.kind == TokenKind.NUMBER:
            return True
        if token.kind == TokenKind.STRING:
            return True
        if token.kind == TokenKind.CHAR:
            return True
        if token.kind == TokenKind.SYMBOL and token.lexeme == "(":
            return True
        return False

    fn op_precedence(self, op: String) -> Int:
        if op == "->":
            return 1
        if op == "+" or op == "-":
            return 2
        if op == "*" or op == "/":
            return 3
        return -1

    fn is_infix_op(self, token: Token) -> Bool:
        if token.kind != TokenKind.SYMBOL:
            return False
        return self.op_precedence(token.lexeme) >= 0

    fn parse_primary(mut self, stop_on_colon: Bool, stop_on_colon_eq: Bool) -> SyntaxNode:
        self.skip_trivia()
        if self.is_at_end():
            return self.make_leaf(NodeKind.ERROR, "eof")

        if self.is_expr_stop(self.current(), stop_on_colon, stop_on_colon_eq):
            return self.make_leaf(NodeKind.ERROR, "stop")

        var token = self.current()
        if token.kind == TokenKind.IDENT:
            _ = self.advance()
            return self.make_leaf(NodeKind.IDENT, token.lexeme)
        if token.kind == TokenKind.NUMBER:
            _ = self.advance()
            return self.make_leaf(NodeKind.NUMBER, token.lexeme)
        if token.kind == TokenKind.STRING:
            _ = self.advance()
            return self.make_leaf(NodeKind.STRING, token.lexeme)
        if token.kind == TokenKind.CHAR:
            _ = self.advance()
            return self.make_leaf(NodeKind.CHAR, token.lexeme)
        if token.kind == TokenKind.SYMBOL and token.lexeme == "(":
            _ = self.advance()
            var inner = self.parse_expr(stop_on_colon, stop_on_colon_eq)
            self.skip_trivia()
            if not self.is_at_end() and self.current().kind == TokenKind.SYMBOL and self.current().lexeme == ")":
                _ = self.advance()
            return inner.copy()

        _ = self.advance()
        return self.make_leaf(NodeKind.ERROR, token.lexeme)

    fn parse_application(mut self, stop_on_colon: Bool, stop_on_colon_eq: Bool) -> SyntaxNode:
        var node = self.parse_primary(stop_on_colon, stop_on_colon_eq)
        while not self.is_at_end():
            self.skip_trivia()
            if self.is_at_end():
                break
            if self.is_expr_stop(self.current(), stop_on_colon, stop_on_colon_eq):
                break
            if not self.is_primary_start(self.current()):
                break
            var arg = self.parse_primary(stop_on_colon, stop_on_colon_eq)
            var children = List[SyntaxNode]()
            children.append(node.copy())
            children.append(arg.copy())
            node = SyntaxNode(NodeKind.APP, "", children.copy())
        return node.copy()

    fn parse_expr(mut self, stop_on_colon: Bool = True, stop_on_colon_eq: Bool = True) -> SyntaxNode:
        self.skip_trivia()
        if self.is_at_end():
            return self.make_leaf(NodeKind.ERROR, "eof")

        self.tmp_nodes.clear()
        self.tmp_ops.clear()
        self.tmp_precs.clear()
        self.tmp_right_assoc.clear()

        var lhs = self.parse_application(stop_on_colon, stop_on_colon_eq)
        self.tmp_nodes.append(lhs.copy())

        while not self.is_at_end():
            self.skip_trivia()
            if self.is_at_end() or self.is_expr_stop(self.current(), stop_on_colon, stop_on_colon_eq):
                break
            if not self.is_infix_op(self.current()):
                break

            var op = self.current().lexeme
            var prec = self.op_precedence(op)
            var right_assoc = (op == "->")
            _ = self.advance()

            var rhs = self.parse_application(stop_on_colon, stop_on_colon_eq)

            while len(self.tmp_ops) > 0:
                var top_prec = self.tmp_precs[len(self.tmp_precs) - 1]

                if right_assoc:
                    if prec < top_prec:
                        self.apply_infix()
                        continue
                    break
                else:
                    if prec <= top_prec:
                        self.apply_infix()
                        continue
                    break

            self.tmp_ops.append(op)
            self.tmp_precs.append(prec)
            self.tmp_right_assoc.append(right_assoc)
            self.tmp_nodes.append(rhs.copy())

        while len(self.tmp_ops) > 0:
            self.apply_infix()

        if len(self.tmp_nodes) == 0:
            return self.make_leaf(NodeKind.ERROR, "empty")

        return self.tmp_nodes[len(self.tmp_nodes) - 1].copy()

    fn parse_def(mut self) -> SyntaxNode:
        _ = self.advance()  # def
        self.skip_trivia()

        var name: String
        if not self.is_at_end() and self.current().kind == TokenKind.IDENT:
            name = self.advance().lexeme
        else:
            return self.make_leaf(NodeKind.ERROR, "expected_ident")

        self.skip_trivia()
        if not self.is_at_end() and self.current().kind == TokenKind.SYMBOL and self.current().lexeme == ":=":
            _ = self.advance()
        else:
            return self.make_leaf(NodeKind.ERROR, "expected_colon_eq")

        var expr = self.parse_expr()
        var children = List[SyntaxNode]()
        children.append(expr.copy())
        return SyntaxNode(NodeKind.DEF, name, children.copy())

    fn parse_theorem(mut self) -> SyntaxNode:
        _ = self.advance()  # theorem
        self.skip_trivia()

        var name: String
        if not self.is_at_end() and self.current().kind == TokenKind.IDENT:
            name = self.advance().lexeme
        else:
            return self.make_leaf(NodeKind.ERROR, "expected_ident")

        self.skip_trivia()
        if not self.is_at_end() and self.current().kind == TokenKind.SYMBOL and self.current().lexeme == ":":
            _ = self.advance()
        else:
            return self.make_leaf(NodeKind.ERROR, "expected_colon")

        var type_expr = self.parse_expr(False, True)
        self.skip_trivia()
        if not self.is_at_end() and self.current().kind == TokenKind.SYMBOL and self.current().lexeme == ":=":
            _ = self.advance()
        else:
            return self.make_leaf(NodeKind.ERROR, "expected_colon_eq")

        var body_expr: SyntaxNode
        if not self.is_at_end() and self.current().kind == TokenKind.KEYWORD and self.current().lexeme == "by":
            _ = self.advance()
            var by_expr = self.parse_expr()
            var by_children = List[SyntaxNode]()
            by_children.append(by_expr.copy())
            body_expr = SyntaxNode(NodeKind.BY, "", by_children.copy())
        else:
            body_expr = self.parse_expr()

        var children = List[SyntaxNode]()
        children.append(type_expr.copy())
        children.append(body_expr.copy())
        return SyntaxNode(NodeKind.THEOREM, name, children.copy())

    fn parse_module_path(mut self) -> String:
        if self.is_at_end() or self.current().kind != TokenKind.IDENT:
            return ""
        var path = self.advance().lexeme
        while not self.is_at_end():
            if self.current().kind == TokenKind.SYMBOL and self.current().lexeme == ".":
                _ = self.advance()
                if self.is_at_end() or self.current().kind != TokenKind.IDENT:
                    break
                path += "." + self.advance().lexeme
            else:
                break
        return path

    fn parse_import(mut self) -> SyntaxNode:
        _ = self.advance()  # import
        var children = List[SyntaxNode]()
        while not self.is_at_end():
            self.skip_trivia()
            if self.is_at_end():
                break
            if self.current().kind != TokenKind.IDENT:
                break
            var module_path = self.parse_module_path()
            if len(module_path) == 0:
                break
            children.append(self.make_leaf(NodeKind.IDENT, module_path))
            self.skip_trivia()
            if self.is_at_end():
                break
            if self.current().kind != TokenKind.IDENT:
                break
        return SyntaxNode(NodeKind.IMPORT, "", children.copy())

    fn parse_namespace(mut self) -> SyntaxNode:
        _ = self.advance()  # namespace
        self.skip_trivia()
        var name: String
        if not self.is_at_end() and self.current().kind == TokenKind.IDENT:
            name = self.parse_module_path()
        else:
            return self.make_leaf(NodeKind.ERROR, "expected_ident")
        return self.parse_block(NodeKind.NAMESPACE, name)

    fn parse_section(mut self) -> SyntaxNode:
        _ = self.advance()  # section
        self.skip_trivia()
        var name = ""
        if not self.is_at_end() and self.current().kind == TokenKind.IDENT:
            name = self.parse_module_path()
        return self.parse_block(NodeKind.SECTION, name)

    fn parse_block(mut self, kind: NodeKind, name: String) -> SyntaxNode:
        var children = List[SyntaxNode]()
        while not self.is_at_end():
            self.skip_trivia()
            if self.is_at_end():
                break
            if self.current().kind == TokenKind.KEYWORD and self.current().lexeme == "end":
                _ = self.advance()
                if not self.is_at_end() and self.current().kind == TokenKind.IDENT:
                    _ = self.parse_module_path()
                break
            var node = self.parse_command()
            children.append(node.copy())
        return SyntaxNode(kind, name, children.copy())

    fn parse_command(mut self) -> SyntaxNode:
        self.skip_trivia()
        if self.is_at_end():
            return self.make_leaf(NodeKind.ERROR, "eof")

        var token = self.current()
        if token.kind == TokenKind.KEYWORD and token.lexeme == "end":
            _ = self.advance()
            return self.make_leaf(NodeKind.ERROR, "unexpected_end")
        if token.kind == TokenKind.KEYWORD and token.lexeme == "import":
            return self.parse_import()
        if token.kind == TokenKind.KEYWORD and token.lexeme == "namespace":
            return self.parse_namespace()
        if token.kind == TokenKind.KEYWORD and token.lexeme == "section":
            return self.parse_section()
        if token.kind == TokenKind.KEYWORD and token.lexeme == "def":
            return self.parse_def()
        if token.kind == TokenKind.KEYWORD and token.lexeme == "theorem":
            return self.parse_theorem()

        return self.parse_expr()

    fn parse(mut self) -> SyntaxNode:
        var children = List[SyntaxNode]()
        while not self.is_at_end():
            self.skip_trivia()
            if self.is_at_end():
                break
            var node = self.parse_command()
            children.append(node.copy())
        return SyntaxNode(NodeKind.PROGRAM, "", children.copy())
