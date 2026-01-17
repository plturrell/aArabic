"""
Basic lexer tests for leanShimmy.
"""

from core.lexer.lexer import Lexer
from core.lexer.token import TokenKind


fn test_empty():
    var lexer = Lexer("")
    var tokens = lexer.tokenize()
    assert(len(tokens) == 1)
    assert(tokens[0].kind == TokenKind.EOF)


fn test_basic():
    var lexer = Lexer("def foo := 42")
    var tokens = lexer.tokenize()
    assert(len(tokens) >= 5)
    assert(tokens[0].lexeme == "def")
    assert(tokens[0].kind == TokenKind.KEYWORD)
    assert(tokens[1].lexeme == "foo")
    assert(tokens[2].lexeme == ":=")
    assert(tokens[3].lexeme == "42")


fn main():
    print("Running lexer tests...")
    test_empty()
    test_basic()
    print("Lexer tests completed.")
