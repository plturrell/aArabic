"""
Elaboration module for nLeanProof - handles type checking and semantic analysis.

This module provides:
- Environment: Global declaration storage
- Context: Local variable context
- Expr: Core expression representation
- TypeChecker: Type checking for expressions
- Elaborator: Converts parsed syntax to typed expressions
"""

from .expr import Expr, ExprKind, Level, expr_to_string
from .environment import Environment, Declaration
from .context import Context, LocalDecl, MetavarContext
from .type_checker import TypeChecker, TypeError
from .elaborator import Elaborator, ElaborationResult, ElaborationError
