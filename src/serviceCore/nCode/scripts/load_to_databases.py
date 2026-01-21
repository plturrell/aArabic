#!/usr/bin/env python3
"""
Load SCIP Index to Databases

Loads SCIP index data into Qdrant, Memgraph, and Marquez.

Usage:
    python load_to_databases.py index.scip --all
    python load_to_databases.py index.scip --qdrant
    python load_to_databases.py index.scip --memgraph
    python load_to_databases.py index.scip --marquez
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders import QdrantLoader, MemgraphLoader, MarquezLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def load_to_qdrant(
    scip_path: str,
    host: str = "localhost",
    port: int = 6333,
    collection: str = "code_symbols"
) -> dict:
    """Load SCIP index to Qdrant"""
    logger.info(f"Loading to Qdrant at {host}:{port}")
    
    loader = QdrantLoader(host=host, port=port)
    result = await loader.load_scip_index(scip_path, collection_name=collection)
    
    logger.info(f"✅ Qdrant: Loaded {result['symbols_loaded']} symbols")
    return result


async def load_to_memgraph(
    scip_path: str,
    host: str = "localhost",
    port: int = 7687,
    clear: bool = False
) -> dict:
    """Load SCIP index to Memgraph"""
    logger.info(f"Loading to Memgraph at {host}:{port}")
    
    loader = MemgraphLoader(host=host, port=port)
    result = await loader.load_scip_index(scip_path, clear_existing=clear)
    loader.close()
    
    logger.info(f"✅ Memgraph: Loaded {result['symbols']} symbols, {result['references']} references")
    return result


async def load_to_marquez(
    scip_path: str,
    url: str = "http://localhost:5000",
    project: str = "ncode-project"
) -> dict:
    """Track SCIP indexing in Marquez"""
    logger.info(f"Tracking lineage in Marquez at {url}")
    
    loader = MarquezLoader(url=url)
    result = await loader.track_indexing_run(scip_path, project)
    await loader.close()
    
    logger.info(f"✅ Marquez: Tracked run {result['run_id']}")
    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Load SCIP index to databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load to all databases
    python load_to_databases.py index.scip --all
    
    # Load only to Qdrant
    python load_to_databases.py index.scip --qdrant --qdrant-host localhost --qdrant-port 6333
    
    # Load only to Memgraph
    python load_to_databases.py index.scip --memgraph --memgraph-host localhost --memgraph-port 7687
    
    # Track lineage in Marquez
    python load_to_databases.py index.scip --marquez --marquez-url http://localhost:5000
        """
    )
    
    parser.add_argument("scip_path", help="Path to SCIP index file")
    parser.add_argument("--all", action="store_true", help="Load to all databases")
    parser.add_argument("--qdrant", action="store_true", help="Load to Qdrant")
    parser.add_argument("--memgraph", action="store_true", help="Load to Memgraph")
    parser.add_argument("--marquez", action="store_true", help="Track in Marquez")
    
    # Qdrant options
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--qdrant-collection", default="code_symbols", help="Qdrant collection")
    
    # Memgraph options
    parser.add_argument("--memgraph-host", default="localhost", help="Memgraph host")
    parser.add_argument("--memgraph-port", type=int, default=7687, help="Memgraph port")
    parser.add_argument("--memgraph-clear", action="store_true", help="Clear existing graph")
    
    # Marquez options
    parser.add_argument("--marquez-url", default="http://localhost:5000", help="Marquez URL")
    parser.add_argument("--project", default="ncode-project", help="Project name for lineage")
    
    args = parser.parse_args()
    
    # Validate SCIP file exists
    if not Path(args.scip_path).exists():
        logger.error(f"SCIP file not found: {args.scip_path}")
        sys.exit(1)
    
    # Determine which databases to load
    load_qdrant = args.all or args.qdrant
    load_memgraph = args.all or args.memgraph
    load_marquez = args.all or args.marquez
    
    if not (load_qdrant or load_memgraph or load_marquez):
        logger.error("No database specified. Use --all, --qdrant, --memgraph, or --marquez")
        sys.exit(1)
    
    results = {}
    
    # Load to databases
    if load_qdrant:
        try:
            results["qdrant"] = await load_to_qdrant(
                args.scip_path, args.qdrant_host, args.qdrant_port, args.qdrant_collection
            )
        except Exception as e:
            logger.error(f"Qdrant load failed: {e}")
            results["qdrant"] = {"error": str(e)}
    
    if load_memgraph:
        try:
            results["memgraph"] = await load_to_memgraph(
                args.scip_path, args.memgraph_host, args.memgraph_port, args.memgraph_clear
            )
        except Exception as e:
            logger.error(f"Memgraph load failed: {e}")
            results["memgraph"] = {"error": str(e)}
    
    if load_marquez:
        try:
            results["marquez"] = await load_to_marquez(
                args.scip_path, args.marquez_url, args.project
            )
        except Exception as e:
            logger.error(f"Marquez load failed: {e}")
            results["marquez"] = {"error": str(e)}
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCIP Index Load Summary")
    print("=" * 60)
    for db, result in results.items():
        if "error" in result:
            print(f"❌ {db}: {result['error']}")
        else:
            print(f"✅ {db}: {result}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

