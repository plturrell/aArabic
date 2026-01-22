"""
CLI for the leanShimmy lexer (text input only).
"""

from sys import argv
from core.io.file import read_file
from core.lexer.lexer import Lexer
from core.lexer.token import token_kind_name


fn print_usage():
    print("Usage: mojo lexer_cli.mojo --text \"<source>\" | --file <path>")


fn main():
    var args = argv()
    if len(args) < 3:
        print_usage()
        return

    var text: String
    if args[1] == "--text":
        text = args[2]
    elif args[1] == "--file":
        try:
            text = read_file(args[2])
        except:
            print("Failed to read file: " + args[2])
            return
    else:
        print_usage()
        return
    var lexer = Lexer(text)
    var tokens = lexer.tokenize()

    for token in tokens:
        var loc = String(token.span.line) + ":" + String(token.span.column)
        print(token_kind_name(token.kind) + "\t" + token.lexeme + "\t" + loc)
