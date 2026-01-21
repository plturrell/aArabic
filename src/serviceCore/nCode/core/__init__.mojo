"""
nCode core package.

This package contains core type definitions and utilities for code intelligence.
"""

from .scip import (
    # SCIP types
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

from .indexer import (
    # SCIP Consumer
    ScipConsumer,
    ScipLocation,
    ScipHoverInfo,
    # Indexer constants
    SCIP_TYPESCRIPT,
    SCIP_PYTHON,
    SCIP_JAVA,
    SCIP_RUST,
    SCIP_GO,
    SCIP_DOTNET,
    # Utility types and functions
    IndexerInfo,
    get_supported_languages,
    get_indexer_info,
    get_indexer_command,
    is_language_supported,
)

