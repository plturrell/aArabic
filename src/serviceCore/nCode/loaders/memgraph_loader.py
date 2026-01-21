"""
Memgraph Loader for SCIP Index

Loads SCIP symbols and relationships into Memgraph graph database.

Graph Schema:
- Nodes: Symbol (with properties: symbol, display_name, kind, documentation, file_path)
- Nodes: Document (with properties: path, language)
- Edges: DEFINED_IN (Symbol -> Document)
- Edges: REFERENCES (Symbol -> Symbol)
- Edges: IMPLEMENTS (Symbol -> Symbol)
- Edges: TYPE_DEFINITION (Symbol -> Symbol)
- Edges: ENCLOSES (Symbol -> Symbol)

Usage:
    loader = MemgraphLoader(host="localhost", port=7687)
    await loader.load_scip_index("index.scip")
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from .scip_parser import load_scip_file, ScipIndex, SymbolInfo, Document, Relationship

logger = logging.getLogger(__name__)

# Symbol kind names
SYMBOL_KINDS = {
    0: "UnspecifiedSymbolKind", 1: "Comment", 2: "Package", 3: "Namespace",
    4: "Class", 5: "Method", 6: "Property", 7: "Field", 8: "Constructor",
    9: "Enum", 10: "Interface", 11: "Function", 12: "Variable",
    13: "Constant", 14: "String", 15: "Number", 16: "Boolean",
    17: "Array", 18: "Object", 19: "Key", 20: "Null",
    21: "EnumMember", 22: "Struct", 23: "Event", 24: "Operator",
    25: "TypeParameter", 26: "TypeAlias", 27: "Type", 28: "Macro"
}


class MemgraphLoader:
    """Loads SCIP index data into Memgraph graph database"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        username: str = "",
        password: str = "",
        database: str = "memgraph"
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver not installed. Run: pip install neo4j")
        
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
    
    def connect(self) -> None:
        """Connect to Memgraph"""
        uri = f"bolt://{self.host}:{self.port}"
        auth = (self.username, self.password) if self.username else None
        self.driver = GraphDatabase.driver(uri, auth=auth)
        logger.info(f"Connected to Memgraph at {uri}")
    
    def close(self) -> None:
        """Close connection"""
        if self.driver:
            self.driver.close()
    
    def _run_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict]:
        """Execute Cypher query"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    async def create_schema(self) -> None:
        """Create indexes and constraints"""
        if not self.driver:
            self.connect()
        
        queries = [
            "CREATE INDEX ON :Symbol(symbol);",
            "CREATE INDEX ON :Symbol(display_name);",
            "CREATE INDEX ON :Symbol(kind);",
            "CREATE INDEX ON :Document(path);",
            "CREATE INDEX ON :Document(language);",
        ]
        
        for query in queries:
            try:
                self._run_query(query)
            except Exception as e:
                # Index might already exist
                logger.debug(f"Schema query result: {e}")
        
        logger.info("Schema created/verified")
    
    async def clear_graph(self) -> None:
        """Clear all nodes and relationships"""
        if not self.driver:
            self.connect()
        
        self._run_query("MATCH (n) DETACH DELETE n;")
        logger.info("Graph cleared")
    
    async def load_scip_index(
        self,
        scip_path: str,
        clear_existing: bool = False,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Load SCIP index into Memgraph"""
        if not self.driver:
            self.connect()
        
        # Parse SCIP file
        logger.info(f"Loading SCIP index from {scip_path}")
        index = load_scip_file(scip_path)
        
        if clear_existing:
            await self.clear_graph()
        
        await self.create_schema()
        
        # Track statistics
        stats = {
            "documents": 0,
            "symbols": 0,
            "references": 0,
            "implementations": 0,
            "type_definitions": 0,
            "enclosures": 0
        }
        
        # Create Document nodes
        for doc in index.documents:
            self._run_query(
                """
                MERGE (d:Document {path: $path})
                SET d.language = $language,
                    d.project_root = $project_root
                """,
                {
                    "path": doc.relative_path,
                    "language": doc.language,
                    "project_root": index.metadata.project_root
                }
            )
            stats["documents"] += 1
        
        logger.info(f"Created {stats['documents']} Document nodes")
        
        # Create Symbol nodes and relationships
        for doc in index.documents:
            for sym in doc.symbols:
                await self._create_symbol_node(sym, doc.relative_path, stats)
        
        # Create external symbol nodes
        for sym in index.external_symbols:
            await self._create_symbol_node(sym, "external", stats)

        logger.info(f"Loaded: {stats}")
        return stats

    async def _create_symbol_node(
        self,
        sym: SymbolInfo,
        file_path: str,
        stats: Dict[str, int]
    ) -> None:
        """Create symbol node and its relationships"""
        # Create Symbol node
        self._run_query(
            """
            MERGE (s:Symbol {symbol: $symbol})
            SET s.display_name = $display_name,
                s.kind = $kind,
                s.kind_name = $kind_name,
                s.documentation = $documentation
            """,
            {
                "symbol": sym.symbol,
                "display_name": sym.display_name or sym.symbol.split('/')[-1],
                "kind": sym.kind,
                "kind_name": SYMBOL_KINDS.get(sym.kind, "Unknown"),
                "documentation": sym.documentation[0] if sym.documentation else ""
            }
        )
        stats["symbols"] += 1

        # Create DEFINED_IN relationship to Document
        if file_path != "external":
            self._run_query(
                """
                MATCH (s:Symbol {symbol: $symbol})
                MATCH (d:Document {path: $path})
                MERGE (s)-[:DEFINED_IN]->(d)
                """,
                {"symbol": sym.symbol, "path": file_path}
            )

        # Create enclosure relationship
        if sym.enclosing_symbol:
            self._run_query(
                """
                MERGE (parent:Symbol {symbol: $parent_symbol})
                MATCH (child:Symbol {symbol: $child_symbol})
                MERGE (parent)-[:ENCLOSES]->(child)
                """,
                {
                    "parent_symbol": sym.enclosing_symbol,
                    "child_symbol": sym.symbol
                }
            )
            stats["enclosures"] += 1

        # Create relationships from SCIP relationships
        for rel in sym.relationships:
            if rel.is_reference:
                self._run_query(
                    """
                    MATCH (from:Symbol {symbol: $from_symbol})
                    MERGE (to:Symbol {symbol: $to_symbol})
                    MERGE (from)-[:REFERENCES]->(to)
                    """,
                    {"from_symbol": sym.symbol, "to_symbol": rel.symbol}
                )
                stats["references"] += 1

            if rel.is_implementation:
                self._run_query(
                    """
                    MATCH (impl:Symbol {symbol: $impl_symbol})
                    MERGE (iface:Symbol {symbol: $iface_symbol})
                    MERGE (impl)-[:IMPLEMENTS]->(iface)
                    """,
                    {"impl_symbol": sym.symbol, "iface_symbol": rel.symbol}
                )
                stats["implementations"] += 1

            if rel.is_type_definition:
                self._run_query(
                    """
                    MATCH (sym:Symbol {symbol: $sym_symbol})
                    MERGE (type:Symbol {symbol: $type_symbol})
                    MERGE (sym)-[:TYPE_DEFINITION]->(type)
                    """,
                    {"sym_symbol": sym.symbol, "type_symbol": rel.symbol}
                )
                stats["type_definitions"] += 1

    async def get_symbol_graph(
        self,
        symbol: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Get symbol and its neighborhood"""
        if not self.driver:
            self.connect()

        result = self._run_query(
            f"""
            MATCH (s:Symbol {{symbol: $symbol}})-[r*1..{depth}]-(neighbor)
            RETURN s, r, neighbor
            """,
            {"symbol": symbol}
        )
        return {"symbol": symbol, "graph": result}

    async def find_implementations(self, interface_symbol: str) -> List[Dict]:
        """Find all implementations of an interface"""
        if not self.driver:
            self.connect()

        return self._run_query(
            """
            MATCH (impl:Symbol)-[:IMPLEMENTS]->(iface:Symbol {symbol: $symbol})
            RETURN impl.symbol as symbol, impl.display_name as name, impl.kind_name as kind
            """,
            {"symbol": interface_symbol}
        )

    async def find_references(self, symbol: str) -> List[Dict]:
        """Find all references to a symbol"""
        if not self.driver:
            self.connect()

        return self._run_query(
            """
            MATCH (ref:Symbol)-[:REFERENCES]->(target:Symbol {symbol: $symbol})
            RETURN ref.symbol as symbol, ref.display_name as name
            """,
            {"symbol": symbol}
        )

    async def get_call_graph(self, function_symbol: str, depth: int = 3) -> List[Dict]:
        """Get call graph for a function"""
        if not self.driver:
            self.connect()

        return self._run_query(
            f"""
            MATCH path = (caller:Symbol {{symbol: $symbol}})-[:REFERENCES*1..{depth}]->(callee:Symbol)
            WHERE callee.kind IN [5, 8, 11]  // Method, Constructor, Function
            RETURN path
            """,
            {"symbol": function_symbol}
        )

