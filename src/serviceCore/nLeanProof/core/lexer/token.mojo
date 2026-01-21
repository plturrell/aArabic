"""
Token definitions for the Lean4 lexer.
"""

@fieldwise_init
struct TokenKind(ImplicitlyCopyable, Copyable, Movable):
    var value: Int

    fn __eq__(self, other: TokenKind) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: TokenKind) -> Bool:
        return self.value != other.value

    comptime EOF = TokenKind(0)
    comptime IDENT = TokenKind(1)
    comptime KEYWORD = TokenKind(2)
    comptime NUMBER = TokenKind(3)
    comptime STRING = TokenKind(4)
    comptime CHAR = TokenKind(5)
    comptime SYMBOL = TokenKind(6)
    comptime COMMENT = TokenKind(7)
    comptime ERROR = TokenKind(8)


@fieldwise_init
struct SourceSpan(Copyable, Movable, ImplicitlyCopyable):
    var line: Int
    var column: Int
    var length: Int


struct Token(Copyable, Movable, ImplicitlyCopyable):
    var kind: TokenKind
    var lexeme: String
    var span: SourceSpan

    fn __init__(out self, kind: TokenKind, lexeme: String, span: SourceSpan):
        self.kind = kind
        self.lexeme = lexeme
        self.span = span

    fn __copyinit__(out self, other: Token):
        self.kind = other.kind
        self.lexeme = other.lexeme
        self.span = other.span

    @property
    fn line(self) -> Int:
        return self.span.line

    @property
    fn column(self) -> Int:
        return self.span.column

    @property
    fn value(self) -> String:
        return self.lexeme



fn token_kind_name(kind: TokenKind) -> String:
    if kind == TokenKind.EOF:
        return "EOF"
    elif kind == TokenKind.IDENT:
        return "IDENT"
    elif kind == TokenKind.KEYWORD:
        return "KEYWORD"
    elif kind == TokenKind.NUMBER:
        return "NUMBER"
    elif kind == TokenKind.STRING:
        return "STRING"
    elif kind == TokenKind.CHAR:
        return "CHAR"
    elif kind == TokenKind.SYMBOL:
        return "SYMBOL"
    elif kind == TokenKind.COMMENT:
        return "COMMENT"
    else:
        return "ERROR"
