"""
leanShimmy core compiler modules (Mojo).

This module provides the Lean4 compiler implementation:
- lexer: Tokenization of Lean4 source code
- parser: Parsing tokens into syntax trees
- elaboration: Type checking and elaboration
- kernel: Trusted type checking core
- io: File I/O utilities
"""

# Lexer components
from .lexer import Lexer, Token, TokenKind

# Parser components
from .parser import Parser, SyntaxNode, NodeKind

# Elaboration components
from .elaboration import (
    Environment, Declaration, Context, LocalDecl, MetavarContext,
    Expr, ExprKind, Level, TypeChecker, TypeError,
    Elaborator, ElaborationResult, ElaborationError, expr_to_string
)

# Kernel components
from .kernel import (
    Level as KernelLevel, LevelKind, level_to_string,
    Name, NameKind, name_to_string, ConstantInfo, InductiveInfo,
    KernelTypeChecker
)
