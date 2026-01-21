"""
nCode Indexer Module - SCIP file consumer for code intelligence.

This module provides the ability to consume SCIP (Source Code Intelligence Protocol)
index files produced by language-specific indexers. Instead of implementing
language-specific parsing, nCode can work with any SCIP-producing indexer:

Supported Indexers:
    - scip-typescript: TypeScript/JavaScript
    - scip-python: Python
    - scip-java: Java (Gradle/Maven)
    - rust-analyzer: Rust (via SCIP export)
    - scip-go: Go
    - scip-dotnet: C#, F#, VB.NET

Usage:
    from core.indexer import ScipConsumer, get_indexer_command

    # Load an existing SCIP index
    var consumer = ScipConsumer()
    consumer.load("index.scip")
    
    # Query the index
    var docs = consumer.get_documents()
    var definition = consumer.find_definition("src/main.py", 10, 5)
    var hover = consumer.get_hover("src/main.py", 10, 5)
    
    # Get the command to generate an index for a language
    var cmd = get_indexer_command("python")  # Returns: "scip-python index . --output index.scip"

Architecture:
    The indexer module uses a Zig-based FFI layer (scip_reader.zig) to efficiently
    parse protobuf-encoded SCIP files. The Mojo wrapper provides a high-level API
    for code intelligence operations.
"""

# Consumer for SCIP files
from .consumer import (
    ScipConsumer,
    ScipLocation,
    ScipHoverInfo,
)

# Supported indexers and utilities
from .supported import (
    # Indexer constants
    SCIP_TYPESCRIPT,
    SCIP_PYTHON,
    SCIP_JAVA,
    SCIP_RUST,
    SCIP_GO,
    SCIP_DOTNET,
    # Utility functions
    IndexerInfo,
    get_supported_languages,
    get_indexer_info,
    get_indexer_command,
    is_language_supported,
)

