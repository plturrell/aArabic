"""
nCode SCIP Database Loaders

This module provides loaders to export SCIP index data to:
- Qdrant: Vector database for semantic code search
- Memgraph: Graph database for code relationships
- Marquez: Data lineage tracking

Usage:
    from loaders import QdrantLoader, MemgraphLoader, MarquezLoader
    
    # Load SCIP index into Qdrant
    qdrant = QdrantLoader(host="localhost", port=6333)
    await qdrant.load_scip_index("index.scip")
    
    # Load SCIP index into Memgraph
    memgraph = MemgraphLoader(host="localhost", port=7687)
    await memgraph.load_scip_index("index.scip")
    
    # Track lineage in Marquez
    marquez = MarquezLoader(url="http://localhost:5000")
    await marquez.track_indexing_run("index.scip", "my-project")
"""

from .qdrant_loader import QdrantLoader
from .memgraph_loader import MemgraphLoader
from .marquez_loader import MarquezLoader

__all__ = ["QdrantLoader", "MemgraphLoader", "MarquezLoader"]

