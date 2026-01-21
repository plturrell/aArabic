"""
SCIP (Source Code Intelligence Protocol) package for Mojo.

This package provides type definitions for working with SCIP indexes,
which are used for code intelligence features like go-to-definition,
find-references, and hover documentation.
"""

from .types import (
    # Enums
    ProtocolVersion,
    TextEncoding,
    PositionEncoding,
    SymbolRole,
    Severity,
    DiagnosticTag,
    Suffix,
    SyntaxKind,
    Kind,
    Language,
    # Core structs
    ToolInfo,
    Package,
    Descriptor,
    Symbol,
    Metadata,
    Diagnostic,
    Relationship,
    Occurrence,
    SignatureDocumentation,
    SymbolInformation,
    Document,
    Index,
)

