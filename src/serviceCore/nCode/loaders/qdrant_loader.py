"""
Qdrant Loader for SCIP Index

Loads SCIP symbols into Qdrant vector database for semantic code search.

Features:
- Stores symbol embeddings for semantic search
- Supports multiple embedding models
- Batch upsert for performance
- Metadata filtering by language, kind, file

Usage:
    loader = QdrantLoader(host="localhost", port=6333)
    await loader.load_scip_index("index.scip", collection="code_symbols")
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from .scip_parser import load_scip_file, ScipIndex, SymbolInfo, Document

logger = logging.getLogger(__name__)

# Symbol kind names from SCIP spec
SYMBOL_KINDS = {
    0: "UnspecifiedSymbolKind",
    1: "Comment", 2: "Package", 3: "Namespace", 4: "Class",
    5: "Method", 6: "Property", 7: "Field", 8: "Constructor",
    9: "Enum", 10: "Interface", 11: "Function", 12: "Variable",
    13: "Constant", 14: "String", 15: "Number", 16: "Boolean",
    17: "Array", 18: "Object", 19: "Key", 20: "Null",
    21: "EnumMember", 22: "Struct", 23: "Event", 24: "Operator",
    25: "TypeParameter", 26: "TypeAlias", 27: "Type", 28: "Macro"
}


@dataclass
class SymbolPoint:
    """Symbol data point for Qdrant"""
    id: str
    symbol: str
    display_name: str
    kind: int
    kind_name: str
    documentation: str
    file_path: str
    language: str
    enclosing_symbol: str
    embedding: Optional[List[float]] = None


class QdrantLoader:
    """Loads SCIP index data into Qdrant vector database"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        api_key: Optional[str] = None
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
        
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.api_key = api_key
        self.client: Optional[QdrantClient] = None
        self._embedding_fn = None
    
    def connect(self) -> None:
        """Connect to Qdrant"""
        if self.prefer_grpc:
            self.client = QdrantClient(
                host=self.host,
                grpc_port=self.grpc_port,
                api_key=self.api_key,
                prefer_grpc=True
            )
        else:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key
            )
        logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
    
    def set_embedding_function(self, fn):
        """Set custom embedding function: fn(text) -> List[float]"""
        self._embedding_fn = fn
    
    def _generate_id(self, symbol: str, file_path: str) -> str:
        """Generate unique ID for symbol"""
        content = f"{symbol}:{file_path}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Simple hash-based embedding (for testing without ML model)"""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Expand hash to desired dimension
        result = []
        for i in range(dim):
            byte_idx = i % len(h)
            result.append((h[byte_idx] - 128) / 128.0)
        return result
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        if self._embedding_fn:
            return self._embedding_fn(text)
        return self._simple_embedding(text)
    
    def _extract_symbols(self, index: ScipIndex) -> List[SymbolPoint]:
        """Extract all symbols from SCIP index"""
        symbols = []
        
        # Process document-local symbols
        for doc in index.documents:
            for sym in doc.symbols:
                point = SymbolPoint(
                    id=self._generate_id(sym.symbol, doc.relative_path),
                    symbol=sym.symbol,
                    display_name=sym.display_name or sym.symbol.split('/')[-1],
                    kind=sym.kind,
                    kind_name=SYMBOL_KINDS.get(sym.kind, "Unknown"),
                    documentation=sym.documentation[0] if sym.documentation else "",
                    file_path=doc.relative_path,
                    language=doc.language,
                    enclosing_symbol=sym.enclosing_symbol
                )
                symbols.append(point)
        
        # Process external symbols
        for sym in index.external_symbols:
            point = SymbolPoint(
                id=self._generate_id(sym.symbol, "external"),
                symbol=sym.symbol,
                display_name=sym.display_name or sym.symbol.split('/')[-1],
                kind=sym.kind,
                kind_name=SYMBOL_KINDS.get(sym.kind, "Unknown"),
                documentation=sym.documentation[0] if sym.documentation else "",
                file_path="external",
                language="",
                enclosing_symbol=sym.enclosing_symbol
            )
            symbols.append(point)

        return symbols

    async def create_collection(
        self,
        collection_name: str = "code_symbols",
        vector_size: int = 384,
        distance: str = "Cosine"
    ) -> bool:
        """Create collection for code symbols"""
        if not self.client:
            self.connect()

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                logger.info(f"Collection {collection_name} already exists")
                return True

            # Create collection
            dist = Distance.COSINE if distance == "Cosine" else Distance.EUCLID
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=dist)
            )
            logger.info(f"Created collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    async def load_scip_index(
        self,
        scip_path: str,
        collection_name: str = "code_symbols",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Load SCIP index into Qdrant"""
        if not self.client:
            self.connect()

        # Parse SCIP file
        logger.info(f"Loading SCIP index from {scip_path}")
        index = load_scip_file(scip_path)

        # Extract symbols
        symbols = self._extract_symbols(index)
        logger.info(f"Extracted {len(symbols)} symbols")

        # Ensure collection exists
        await self.create_collection(collection_name)

        # Generate embeddings and upsert in batches
        points_upserted = 0
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            points = []

            for sym in batch:
                # Create embedding text from symbol info
                embed_text = f"{sym.display_name} {sym.kind_name} {sym.documentation}"
                embedding = self._get_embedding(embed_text)

                point = PointStruct(
                    id=sym.id,
                    vector=embedding,
                    payload={
                        "symbol": sym.symbol,
                        "display_name": sym.display_name,
                        "kind": sym.kind,
                        "kind_name": sym.kind_name,
                        "documentation": sym.documentation,
                        "file_path": sym.file_path,
                        "language": sym.language,
                        "enclosing_symbol": sym.enclosing_symbol,
                        "project_root": index.metadata.project_root,
                        "indexer": index.metadata.tool_info.name
                    }
                )
                points.append(point)

            # Upsert batch
            self.client.upsert(collection_name=collection_name, points=points)
            points_upserted += len(points)
            logger.info(f"Upserted {points_upserted}/{len(symbols)} symbols")

        return {
            "collection": collection_name,
            "symbols_loaded": len(symbols),
            "documents": len(index.documents),
            "project_root": index.metadata.project_root,
            "indexer": index.metadata.tool_info.name
        }

    async def search_symbols(
        self,
        query: str,
        collection_name: str = "code_symbols",
        limit: int = 10,
        language: Optional[str] = None,
        kind: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for symbols by semantic similarity"""
        if not self.client:
            self.connect()

        # Build filter
        filter_conditions = []
        if language:
            filter_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )
        if kind is not None:
            filter_conditions.append(
                FieldCondition(key="kind", match=MatchValue(value=kind))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Get query embedding
        embedding = self._get_embedding(query)

        # Search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=embedding,
            query_filter=query_filter,
            limit=limit
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                **r.payload
            }
            for r in results
        ]

