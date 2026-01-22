"""
Lean4 lexer scaffold (ASCII-centric, Unicode-aware placeholders).
"""

from collections import List
from core.lexer.token import Token, TokenKind, SourceSpan

comptime KEYWORDS = [
    "abbrev",
    "axiom",
    "by",
    "def",
    "do",
    "else",
    "end",
    "example",
    "export",
    "forall",
    "have",
    "if",
    "import",
    "in",
    "inductive",
    "lemma",
    "let",
    "macro",
    "match",
    "namespace",
    "notation",
    "open",
    "section",
    "set",
    "show",
    "structure",
    "syntax",
    "then",
    "theorem",
    "where",
    "with",
]


struct Lexer:
    var source: String
    var index: Int
    var line: Int
    var column: Int

    fn __init__(out self, source: String):
        self.source = source
        self.index = 0
        self.line = 1
        self.column = 1

    fn is_at_end(self) -> Bool:
        return self.index >= len(self.source)

    fn peek(self) -> String:
        if self.is_at_end():
            return ""
        return String(self.source[self.index])

    fn peek_next(self) -> String:
        if self.index + 1 >= len(self.source):
            return ""
        return String(self.source[self.index + 1])

    fn advance(mut self) -> String:
        if self.is_at_end():
            return ""
        var ch = String(self.source[self.index])
        self.index += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    fn skip_whitespace(mut self):
        while not self.is_at_end():
            var ch = self.peek()
            if ch == " " or ch == "\t" or ch == "\r" or ch == "\n":
                _ = self.advance()
            else:
                break

    fn make_token(self, kind: TokenKind, start_index: Int, start_line: Int, start_column: Int) -> Token:
        var lexeme = String(self.source[start_index:self.index])
        var span = SourceSpan(start_line, start_column, self.index - start_index)
        return Token(kind, lexeme, span)

    fn is_non_ascii(self, ch: String) -> Bool:
        if len(ch) == 0:
            return False
        var bytes = ch.as_bytes()
        if len(bytes) == 0:
            return False
        return Int(bytes[0]) >= 128

    fn is_digit(self, ch: String) -> Bool:
        return ch >= "0" and ch <= "9"

    fn is_hex_digit(self, ch: String) -> Bool:
        return self.is_digit(ch) or (ch >= "a" and ch <= "f") or (ch >= "A" and ch <= "F")

    fn is_bin_digit(self, ch: String) -> Bool:
        return ch == "0" or ch == "1"

    fn is_oct_digit(self, ch: String) -> Bool:
        return ch >= "0" and ch <= "7"

    fn is_alpha(self, ch: String) -> Bool:
        return (ch >= "a" and ch <= "z") or (ch >= "A" and ch <= "Z")

    fn is_ident_start(self, ch: String) -> Bool:
        return self.is_alpha(ch) or ch == "_" or self.is_non_ascii(ch)

    fn is_ident_continue(self, ch: String) -> Bool:
        return self.is_ident_start(ch) or self.is_digit(ch) or ch == "'"

    fn is_keyword(self, ident: String) -> Bool:
        var keywords = materialize[KEYWORDS]()
        for kw in keywords:
            if ident == kw:
                return True
        return False

    fn read_identifier(mut self, start_index: Int, start_line: Int, start_column: Int) -> Token:
        while not self.is_at_end() and self.is_ident_continue(self.peek()):
            _ = self.advance()
        var token = self.make_token(TokenKind.IDENT, start_index, start_line, start_column)
        if self.is_keyword(token.lexeme):
            token.kind = TokenKind.KEYWORD
        return token.copy()

    fn read_number(mut self, start_index: Int, start_line: Int, start_column: Int) -> Token:
        var first = String(self.source[start_index])
        if first == "0":
            var next = self.peek()
            if next == "x" or next == "X":
                _ = self.advance()
                while not self.is_at_end():
                    var ch = self.peek()
                    if self.is_hex_digit(ch) or ch == "_":
                        _ = self.advance()
                        continue
                    break
                return self.make_token(TokenKind.NUMBER, start_index, start_line, start_column)
            if next == "b" or next == "B":
                _ = self.advance()
                while not self.is_at_end():
                    var ch = self.peek()
                    if self.is_bin_digit(ch) or ch == "_":
                        _ = self.advance()
                        continue
                    break
                return self.make_token(TokenKind.NUMBER, start_index, start_line, start_column)
            if next == "o" or next == "O":
                _ = self.advance()
                while not self.is_at_end():
                    var ch = self.peek()
                    if self.is_oct_digit(ch) or ch == "_":
                        _ = self.advance()
                        continue
                    break
                return self.make_token(TokenKind.NUMBER, start_index, start_line, start_column)

        while not self.is_at_end():
            var ch = self.peek()
            if self.is_digit(ch) or ch == "_":
                _ = self.advance()
                continue
            break

        if self.peek() == "." and self.is_digit(self.peek_next()):
            _ = self.advance()
            while not self.is_at_end():
                var ch = self.peek()
                if self.is_digit(ch) or ch == "_":
                    _ = self.advance()
                    continue
                break

        if self.peek() == "e" or self.peek() == "E":
            _ = self.advance()
            if self.peek() == "+" or self.peek() == "-":
                _ = self.advance()
            while not self.is_at_end():
                var ch = self.peek()
                if self.is_digit(ch) or ch == "_":
                    _ = self.advance()
                    continue
                break

        return self.make_token(TokenKind.NUMBER, start_index, start_line, start_column)

    fn read_string(mut self, start_index: Int, start_line: Int, start_column: Int) -> Token:
        while not self.is_at_end():
            var ch = self.peek()
            if ch == "\"":
                _ = self.advance()
                return self.make_token(TokenKind.STRING, start_index, start_line, start_column)
            if ch == "\\":
                _ = self.advance()
                if not self.is_at_end():
                    _ = self.advance()
                continue
            if ch == "\n":
                break
            _ = self.advance()
        return self.make_token(TokenKind.ERROR, start_index, start_line, start_column)

    fn read_comment(mut self, start_index: Int, start_line: Int, start_column: Int) -> Token:
        while not self.is_at_end() and self.peek() != "\n":
            _ = self.advance()
        return self.make_token(TokenKind.COMMENT, start_index, start_line, start_column)

    fn read_block_comment(mut self, start_index: Int, start_line: Int, start_column: Int) -> Token:
        var depth = 1
        while not self.is_at_end():
            var ch = self.advance()
            if ch == "/" and self.peek() == "-":
                _ = self.advance()
                depth += 1
                continue
            if ch == "-" and self.peek() == "/":
                _ = self.advance()
                depth -= 1
                if depth == 0:
                    return self.make_token(TokenKind.COMMENT, start_index, start_line, start_column)
        return self.make_token(TokenKind.ERROR, start_index, start_line, start_column)

    fn read_char(mut self, start_index: Int, start_line: Int, start_column: Int) -> Token:
        if self.is_at_end():
            return self.make_token(TokenKind.ERROR, start_index, start_line, start_column)
        var ch = self.peek()
        if ch == "\n":
            return self.make_token(TokenKind.ERROR, start_index, start_line, start_column)
        if ch == "\\":
            _ = self.advance()
            if self.is_at_end() or self.peek() == "\n":
                return self.make_token(TokenKind.ERROR, start_index, start_line, start_column)
            _ = self.advance()
        else:
            _ = self.advance()
        if self.peek() == "'":
            _ = self.advance()
            return self.make_token(TokenKind.CHAR, start_index, start_line, start_column)
        return self.make_token(TokenKind.ERROR, start_index, start_line, start_column)

    fn next_token(mut self) -> Token:
        self.skip_whitespace()
        if self.is_at_end():
            return Token(TokenKind.EOF, "", SourceSpan(self.line, self.column, 0))

        var start_index = self.index
        var start_line = self.line
        var start_column = self.column
        var ch = self.advance()

        if ch == "/" and self.peek() == "-":
            _ = self.advance()
            return self.read_block_comment(start_index, start_line, start_column)

        if ch == "-" and self.peek() == "-":
            _ = self.advance()
            return self.read_comment(start_index, start_line, start_column)

        if ch == "\"":
            return self.read_string(start_index, start_line, start_column)

        if ch == "'":
            return self.read_char(start_index, start_line, start_column)

        if self.is_ident_start(ch):
            return self.read_identifier(start_index, start_line, start_column)

        if self.is_digit(ch):
            return self.read_number(start_index, start_line, start_column)

        if ch == "." and self.peek() == "." and self.peek_next() == ".":
            _ = self.advance()
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)
        if ch == "." and self.peek() == ".":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)

        if ch == ":" and self.peek() == "=":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)
        if ch == ":" and self.peek() == ":":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)
        if ch == "-" and self.peek() == ">":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)
        if ch == "=" and self.peek() == ">":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)
        if ch == "<" and self.peek() == "-":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)
        if ch == "<" and self.peek() == "=":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)
        if ch == ">" and self.peek() == "=":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)
        if ch == "!" and self.peek() == "=":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)
        if ch == "=" and self.peek() == "=":
            _ = self.advance()
            return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)

        return self.make_token(TokenKind.SYMBOL, start_index, start_line, start_column)

    fn tokenize(mut self) -> List[Token]:
        var tokens = List[Token]()
        while True:
            var token = self.next_token()
            var is_eof = token.kind == TokenKind.EOF
            tokens.append(token.copy())
            if is_eof:
                break
        return tokens.copy()
