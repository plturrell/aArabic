"""
Lean4 FFI Bridge for Zig Server.

Exports C-ABI functions for the Zig HTTP server to call:
- lean4_check: Check/typecheck Lean4 source code
- lean4_run: Execute Lean4 code and return output
- lean4_elaborate: Elaborate Lean4 source to typed environment
"""

from collections import List
from memory import UnsafePointer

from core.lexer import Lexer, Token, TokenKind
from core.parser import Parser, SyntaxNode, NodeKind
from core.elaboration import Elaborator, ElaborationResult, Environment, Declaration
from core.runtime.evaluator import Evaluator, EvalResult
from core.utils.json_builder import JsonBuilder


# Response buffer size (shared with Zig)
comptime BUFFER_SIZE: Int = 1024 * 1024  # 1MB


@fieldwise_init
struct CheckResult(Copyable, Movable):
    """Result of checking Lean4 source."""
    var success: Bool
    var error_count: Int
    var warning_count: Int
    var info_count: Int
    var messages: String


@fieldwise_init
struct RunResult(Copyable, Movable):
    """Result of running Lean4 source."""
    var success: Bool
    var stdout: String
    var stderr: String
    var exit_code: Int


struct ElaborateResult(Copyable, Movable): # No @fieldwise_init here
    """Result of elaborating Lean4 source."""
    var success: Bool
    var declarations: List[String]
    var environment_size: Int
    var errors: String

    fn __init__(out self, success: Bool, declarations: List[String], environment_size: Int, errors: String):
        self.success = success
        self.declarations = declarations.copy()
        self.environment_size = environment_size
        self.errors = errors


fn _check_source(source: String) -> CheckResult:
    """Check Lean4 source code."""
    var errors = List[String]()
    var warnings = List[String]()

    # Lexical analysis
    var lexer = Lexer(source)
    var tokens = lexer.tokenize()

    for i in range(len(tokens)):
        var token = tokens[i]
        if token.kind == TokenKind.ERROR:
            errors.append("Lexer error at " + String(token.span.line) + ":" + String(token.span.column) + ": " + token.lexeme)

    if len(errors) > 0:
        return CheckResult(False, len(errors), 0, 0, _join_messages(errors))

    # Parsing
    var parser = Parser(tokens)
    var ast = parser.parse()

    if parser.has_errors():
        for i in range(len(parser.errors)):
            errors.append("Parse error: " + parser.errors[i].message)
        return CheckResult(False, len(errors), 0, 0, _join_messages(errors))

    # Elaboration (type checking)
    var elaborator = Elaborator()
    _ = elaborator.elaborate_program(ast)

    if elaborator.has_errors():
        for i in range(len(elaborator.errors)):
            errors.append("Type error: " + elaborator.errors[i].message)
        return CheckResult(False, len(errors), 0, 0, _join_messages(errors))

    return CheckResult(True, 0, len(warnings), 0, "Check successful")


fn _run_source(source: String) -> RunResult:
    """Run Lean4 source code and capture output."""
    var stdout_lines = List[String]()
    var stderr_lines = List[String]()

    # Lex
    var lexer = Lexer(source)
    var tokens = lexer.tokenize()

    for i in range(len(tokens)):
        if tokens[i].kind == TokenKind.ERROR:
            stderr_lines.append("Lexer error: " + tokens[i].lexeme)
            return RunResult(False, "", _join_messages(stderr_lines), 1)

    # Parse
    var parser = Parser(tokens)
    var ast = parser.parse()

    if parser.has_errors():
        for i in range(len(parser.errors)):
            stderr_lines.append("Parse error: " + parser.errors[i].message)
        return RunResult(False, "", _join_messages(stderr_lines), 1)

    # Elaborate
    var elaborator = Elaborator()
    var decls = elaborator.elaborate_program(ast)

    if elaborator.has_errors():
        for i in range(len(elaborator.errors)):
            stderr_lines.append("Error: " + elaborator.errors[i].message)
        return RunResult(False, "", _join_messages(stderr_lines), 1)

    # Evaluate declarations
    var evaluator = Evaluator(elaborator.env)

    for i in range(len(decls)):
        var decl = decls[i].copy()
        if decl.value:
            var result = evaluator.eval(decl.value.value().copy())
            if result.is_success():
                stdout_lines.append(decl.name + " = " + result.value.value().to_string())
            elif result.error:
                stderr_lines.append("Eval error in " + decl.name + ": " + result.error.value().message)

    # Process #eval and #check commands
    for i in range(len(ast.children)):
        var child = ast.children[i].copy()
        if child.kind == NodeKind.COMMAND:
            var cmd_name = child.value
            if cmd_name == "#check" and len(child.children) > 0:
                var result = elaborator.elaborate_expr(child.children[0].copy())
                if result.type:
                    stdout_lines.append(child.children[0].value + " : " + result.type.value().to_string())
            elif cmd_name == "#eval" and len(child.children) > 0:
                var elab_result = elaborator.elaborate_expr(child.children[0].copy())
                if elab_result.expr:
                    var eval_result = evaluator.eval(elab_result.expr.value().copy())
                    if eval_result.is_success():
                        stdout_lines.append(eval_result.value.value().to_string())

    var stdout = _join_messages(stdout_lines)
    var stderr = _join_messages(stderr_lines)
    var success = len(stderr_lines) == 0

    return RunResult(success, stdout, stderr, 0 if success else 1)


fn _elaborate_source(source: String) -> ElaborateResult:
    """Elaborate Lean4 source to typed environment."""
    var decl_names = List[String]()
    var errors = List[String]()

    # Lex
    var lexer = Lexer(source)
    var tokens = lexer.tokenize()

    for i in range(len(tokens)):
        if tokens[i].kind == TokenKind.ERROR:
            errors.append("Lexer error: " + tokens[i].lexeme)
            return ElaborateResult(False, decl_names.copy(), 0, _join_messages(errors))

    # Parse
    var parser = Parser(tokens)
    var ast = parser.parse()

    if parser.has_errors():
        for i in range(len(parser.errors)):
            errors.append("Parse error: " + parser.errors[i].message)
        return ElaborateResult(False, decl_names.copy(), 0, _join_messages(errors))

    # Elaborate
    var elaborator = Elaborator()
    var decls = elaborator.elaborate_program(ast)

    for i in range(len(decls)):
        decl_names.append(decls[i].name)

    if elaborator.has_errors():
        for i in range(len(elaborator.errors)):
            errors.append("Type error: " + elaborator.errors[i].message)
        return ElaborateResult(False, decl_names.copy(), len(decls), _join_messages(errors))

    return ElaborateResult(True, decl_names.copy(), len(decls), "")


fn _join_messages(messages: List[String]) -> String:
    """Join messages with newlines."""
    if len(messages) == 0:
        return ""
    var result = messages[0]
    for i in range(1, len(messages)):
        result = result + "\n" + messages[i]
    return result


fn _copy_to_buffer(s: String, buf: UnsafePointer[UInt8], buf_size: Int) -> Int:
    """Copy string to buffer, return bytes written."""
    var bytes = s.as_bytes()
    # Ensure we don't overflow the buffer
    var to_copy = min(len(bytes), buf_size - 1)
    if to_copy < 0:
        to_copy = 0
    
    var mut_buf = buf # Assign to mutable local variable
    for i in range(to_copy):
        mut_buf.store(i, bytes[i])
    mut_buf.store(to_copy, 0)  # Null terminate
    return to_copy


# ==============================================================================
# C-ABI Exports for Zig Server
# ==============================================================================

fn lean4_check(
    source_ptr: UnsafePointer[UInt8],
    source_len: Int,
    result_buf: UnsafePointer[UInt8],
    buf_size: Int,
) -> Int:
    """
    Check Lean4 source code.
    Returns: bytes written to result_buf, or -1 on error.
    """
    var source = _ptr_to_string(source_ptr, source_len)
    var result = _check_source(source)

    var jb = JsonBuilder()
    jb.begin_object()
    jb.key("success")
    jb.value_bool(result.success)
    jb.comma()
    jb.key("error_count")
    jb.value_int(result.error_count)
    jb.comma()
    jb.key("warning_count")
    jb.value_int(result.warning_count)
    jb.comma()
    jb.key("info_count")
    jb.value_int(result.info_count)
    jb.comma()
    jb.key("messages")
    jb.value_string(result.messages)
    jb.end_object()

    return _copy_to_buffer(jb.to_string(), result_buf, buf_size)


fn lean4_run(
    source_ptr: UnsafePointer[UInt8],
    source_len: Int,
    result_buf: UnsafePointer[UInt8],
    buf_size: Int,
) -> Int:
    """
    Run Lean4 source code.
    Returns: bytes written to result_buf, or -1 on error.
    """
    var source = _ptr_to_string(source_ptr, source_len)
    var result = _run_source(source)

    var jb = JsonBuilder()
    jb.begin_object()
    jb.key("success")
    jb.value_bool(result.success)
    jb.comma()
    jb.key("stdout")
    jb.value_string(result.stdout)
    jb.comma()
    jb.key("stderr")
    jb.value_string(result.stderr)
    jb.comma()
    jb.key("exit_code")
    jb.value_int(result.exit_code)
    jb.end_object()

    return _copy_to_buffer(jb.to_string(), result_buf, buf_size)


fn lean4_elaborate(
    source_ptr: UnsafePointer[UInt8],
    source_len: Int,
    result_buf: UnsafePointer[UInt8],
    buf_size: Int,
) -> Int:
    """
    Elaborate Lean4 source code.
    Returns: bytes written to result_buf, or -1 on error.
    """
    var source = _ptr_to_string(source_ptr, source_len)
    var result = _elaborate_source(source)

    var jb = JsonBuilder()
    jb.begin_object()
    jb.key("success")
    jb.value_bool(result.success)
    jb.comma()
    jb.key("declarations")
    jb.begin_array()
    for i in range(len(result.declarations)):
        if i > 0: jb.comma()
        jb.value_string(result.declarations[i])
    jb.end_array()
    jb.comma()
    jb.key("environment_size")
    jb.value_int(result.environment_size)
    jb.comma()
    jb.key("errors")
    jb.value_string(result.errors)
    jb.end_object()

    return _copy_to_buffer(jb.to_string(), result_buf, buf_size)


fn _ptr_to_string(ptr: UnsafePointer[UInt8], length: Int) -> String:
    """Convert pointer to string."""
    var bytes = List[UInt8]()
    var mut_ptr = ptr # Assign to mutable local variable
    for i in range(length):
        bytes.append(mut_ptr.load(i))
    try:
        return String(from_utf8=bytes)
    except:
        return ""