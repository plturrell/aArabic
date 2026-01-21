"""
Kernel module for nLeanProof - the trusted core of the type checker.

This module provides the minimal trusted computing base for Lean4:
- Type checking kernel operations
- Definitional equality checking
- Universe level handling
- Inductive type validation
"""

from .level import Level, LevelKind, level_to_string
from .name import Name, NameKind, name_to_string
from .declaration import ConstantInfo, InductiveInfo
from .type_checker_kernel import KernelTypeChecker
