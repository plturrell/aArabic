"""
CLI for the leanShimmy parser (text input only).
"""

from sys import argv
from core.io.file import read_file
from core.lexer.lexer import Lexer
from core.parser.parser import Parser
from core.parser.syntax import node_to_string


fn print_usage():
    print("Usage: mojo parser_cli.mojo --text \"<source>\" | --file <path>")


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
    var parser = Parser(tokens)
    var tree = parser.parse()
    print(node_to_string(tree))
