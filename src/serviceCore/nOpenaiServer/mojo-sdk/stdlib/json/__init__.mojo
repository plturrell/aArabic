"""
Generic JSON Support for Mojo SDK

Zero Python dependencies - pure Mojo + Zig std.json backend.

Exports:
    - JsonParser: Main JSON parser class
    - create_json_parser: Factory function

Usage:
    from mojo_sdk.stdlib.json import JsonParser
    
    var parser = JsonParser()
    var data = parser.parse_file("config.json")
    
    # Or use factory
    from mojo_sdk.stdlib.json import create_json_parser
    var parser = create_json_parser(verbose=True)
"""

from .parser import JsonParser, create_json_parser

# Module exports
__all__ = ["JsonParser", "create_json_parser"]
