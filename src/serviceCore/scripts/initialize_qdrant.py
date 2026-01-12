#!/usr/bin/env python3
"""
Qdrant Initialization Script
Creates default collections and sets up the vector database for Arabic invoice processing
"""

import asyncio
import logging
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from backend.adapters.qdrant import QdrantAdapter, VectorType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initialize_qdrant_collections():
    """Initialize all required Qdrant collections"""
    try:
        # Initialize Qdrant adapter
        qdrant_adapter = QdrantAdapter()
        success = await qdrant_adapter.initialize()
        
        if not success:
            logger.error("Failed to initialize Qdrant adapter")
            return False
        
        logger.info("Qdrant adapter initialized successfully")
        
        # Define collections to create
        collections_config = {
            "workflows": {
                "vector_size": 768,
                "distance": "Cosine",
                "description": "Workflow embeddings for similarity search and recommendations"
            },
            "documents": {
                "vector_size": 1536,
                "distance": "Cosine", 
                "description": "Document embeddings for semantic search"
            },
            "invoices": {
                "vector_size": 768,
                "distance": "Cosine",
                "description": "Arabic invoice embeddings for similarity matching"
            },
            "tools": {
                "vector_size": 384,
                "distance": "Cosine",
                "description": "Tool embeddings for intelligent orchestration"
            },
            "a2ui_components": {
                "vector_size": 512,
                "distance": "Cosine",
                "description": "A2UI component embeddings for interface generation"
            }
        }
        
        # Create collections
        created_collections = []
        for collection_name, config in collections_config.items():
            try:
                logger.info(f"Creating collection: {collection_name}")
                
                success = await qdrant_adapter.create_collection(
                    collection_name=collection_name,
                    vector_size=config["vector_size"],
                    distance=config["distance"]
                )
                
                if success:
                    created_collections.append(collection_name)
                    logger.info(f"✅ Collection '{collection_name}' created successfully")
                else:
                    logger.warning(f"⚠️  Collection '{collection_name}' may already exist or creation failed")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create collection '{collection_name}': {e}")
        
        # Verify collections exist
        logger.info("Verifying created collections...")
        existing_collections = await qdrant_adapter.list_collections()
        
        for collection_name in collections_config.keys():
            if collection_name in [col.name for col in existing_collections]:
                logger.info(f"✅ Collection '{collection_name}' verified")
            else:
                logger.warning(f"⚠️  Collection '{collection_name}' not found")
        
        # Add some sample data for testing
        await add_sample_data(qdrant_adapter)
        
        await qdrant_adapter.close()
        logger.info("Qdrant initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Qdrant initialization failed: {e}")
        return False

async def add_sample_data(qdrant_adapter: QdrantAdapter):
    """Add sample data to collections for testing"""
    try:
        logger.info("Adding sample data...")
        
        # Sample workflow embedding
        sample_workflow_embedding = [0.1 * i for i in range(768)]
        await qdrant_adapter.store_workflow_embedding(
            workflow_id="sample_workflow_001",
            workflow_name="Sample Arabic Invoice Processing",
            embedding=sample_workflow_embedding,
            metadata={
                "type": "invoice_processing",
                "language": "arabic",
                "status": "template",
                "complexity": "medium",
                "success_rate": 0.95
            }
        )
        
        # Sample tool embeddings
        sample_tools = [
            {
                "id": "arabic_ocr_001",
                "name": "Advanced Arabic OCR",
                "description": "High-accuracy OCR for Arabic text extraction from invoices",
                "category": "ocr",
                "success_rate": 0.92
            },
            {
                "id": "invoice_parser_001", 
                "name": "Arabic Invoice Parser",
                "description": "Extract structured data from Arabic invoices",
                "category": "parsing",
                "success_rate": 0.88
            },
            {
                "id": "data_validator_001",
                "name": "Invoice Data Validator",
                "description": "Validate extracted invoice data for accuracy",
                "category": "validation", 
                "success_rate": 0.94
            }
        ]
        
        for tool in sample_tools:
            # Generate simple embedding based on tool description
            embedding = [hash(tool["description"] + str(i)) % 1000 / 1000.0 for i in range(384)]
            
            await qdrant_adapter.store_tool_embedding(
                tool_id=tool["id"],
                tool_name=tool["name"],
                tool_description=tool["description"],
                embedding=embedding,
                tool_metadata={
                    "category": tool["category"],
                    "success_rate": tool["success_rate"],
                    "language_support": ["arabic", "english"],
                    "input_types": ["image", "pdf", "text"],
                    "output_types": ["json", "xml"]
                }
            )
        
        # Sample invoice embedding
        sample_invoice_embedding = [0.05 * i for i in range(768)]
        await qdrant_adapter.store_invoice_embedding(
            invoice_id="sample_invoice_001",
            invoice_text="فاتورة رقم ١٢٣٤٥ - شركة المثال التجارية",
            embedding=sample_invoice_embedding,
            extracted_data={
                "invoice_number": "12345",
                "vendor_name": "شركة المثال التجارية",
                "total_amount": 1500.00,
                "currency": "SAR",
                "invoice_date": "2024-01-15"
            },
            processing_status="completed"
        )
        
        logger.info("✅ Sample data added successfully")
        
    except Exception as e:
        logger.warning(f"Failed to add sample data: {e}")

async def check_qdrant_health():
    """Check if Qdrant service is healthy"""
    try:
        qdrant_adapter = QdrantAdapter()
        health = await qdrant_adapter.health_check()
        
        if health:
            logger.info("✅ Qdrant service is healthy")
            return True
        else:
            logger.error("❌ Qdrant service is not healthy")
            return False
            
    except Exception as e:
        logger.error(f"❌ Qdrant health check failed: {e}")
        return False

async def main():
    """Main initialization function"""
    logger.info("Starting Qdrant initialization...")
    
    # Check if Qdrant is running
    if not await check_qdrant_health():
        logger.error("Qdrant service is not available. Please start Qdrant first.")
        logger.info("To start Qdrant, run: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        return False
    
    # Initialize collections
    success = await initialize_qdrant_collections()
    
    if success:
        logger.info("🎉 Qdrant initialization completed successfully!")
        logger.info("Collections created:")
        logger.info("  - workflows (768d vectors)")
        logger.info("  - documents (1536d vectors)")
        logger.info("  - invoices (768d vectors)")
        logger.info("  - tools (384d vectors)")
        logger.info("  - a2ui_components (512d vectors)")
        return True
    else:
        logger.error("❌ Qdrant initialization failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
