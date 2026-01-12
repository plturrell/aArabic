"""
Integration tests for Qdrant + Memgraph dual database architecture
Tests the hybrid orchestration system with both vector and graph databases
"""

import pytest
import asyncio
import uuid
from typing import Dict, Any, List
import logging

from backend.adapters.qdrant import QdrantAdapter, VectorPoint, VectorType
from backend.adapters.memgraph import MemgraphAdapter, WorkflowGraphNode, NodeStatus
from backend.adapters.hybrid_orchestration import HybridOrchestrationAdapter

logger = logging.getLogger(__name__)

@pytest.fixture
async def qdrant_adapter():
    """Create and initialize Qdrant adapter for testing"""
    adapter = QdrantAdapter()
    success = await adapter.initialize()
    if not success:
        pytest.skip("Qdrant not available for testing")
    yield adapter
    await adapter.close()

@pytest.fixture
async def memgraph_adapter():
    """Create and initialize Memgraph adapter for testing"""
    adapter = MemgraphAdapter()
    success = await adapter.initialize()
    if not success:
        pytest.skip("Memgraph not available for testing")
    yield adapter
    await adapter.close()

@pytest.fixture
async def hybrid_adapter():
    """Create and initialize hybrid orchestration adapter for testing"""
    adapter = HybridOrchestrationAdapter()
    success = await adapter.initialize()
    if not success:
        pytest.skip("Hybrid orchestration not available for testing")
    yield adapter
    await adapter.close()

@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for testing"""
    return {
        "id": str(uuid.uuid4()),
        "name": "Arabic Invoice Processing Workflow",
        "description": "Process Arabic invoices with OCR and data extraction",
        "type": "invoice_processing",
        "tools": [
            {
                "id": "ocr_tool",
                "name": "Arabic OCR Tool",
                "description": "Extract text from Arabic invoice images",
                "dependencies": []
            },
            {
                "id": "data_extractor",
                "name": "Invoice Data Extractor", 
                "description": "Extract structured data from invoice text",
                "dependencies": ["ocr_tool"]
            },
            {
                "id": "validator",
                "name": "Data Validator",
                "description": "Validate extracted invoice data",
                "dependencies": ["data_extractor"]
            }
        ],
        "metadata": {
            "language": "arabic",
            "domain": "finance",
            "complexity": "medium"
        }
    }

class TestQdrantMemgraphIntegration:
    """Test integration between Qdrant vector database and Memgraph graph database"""
    
    @pytest.mark.asyncio
    async def test_dual_database_workflow_storage(
        self, 
        qdrant_adapter: QdrantAdapter, 
        memgraph_adapter: MemgraphAdapter,
        sample_workflow_data: Dict[str, Any]
    ):
        """Test storing workflow data in both Qdrant and Memgraph"""
        workflow_id = sample_workflow_data["id"]
        
        # Store workflow graph in Memgraph
        graph_success = await memgraph_adapter.create_workflow_graph(
            workflow_id=workflow_id,
            workflow_data=sample_workflow_data
        )
        assert graph_success, "Failed to create workflow graph in Memgraph"
        
        # Generate and store workflow embedding in Qdrant
        # Simple embedding generation for testing
        embedding = [0.1 * i for i in range(768)]  # 768-dimensional embedding
        
        vector_success = await qdrant_adapter.store_workflow_embedding(
            workflow_id=workflow_id,
            workflow_name=sample_workflow_data["name"],
            embedding=embedding,
            metadata=sample_workflow_data["metadata"]
        )
        assert vector_success, "Failed to store workflow embedding in Qdrant"
        
        # Verify data exists in both databases
        # Check Memgraph
        graph_analytics = await memgraph_adapter.get_workflow_analytics(workflow_id)
        assert graph_analytics is not None, "Workflow not found in Memgraph"
        assert graph_analytics["node_count"] == len(sample_workflow_data["tools"])
        
        # Check Qdrant
        workflow_point = await qdrant_adapter.get_point(
            "workflows", workflow_id, with_payload=True
        )
        assert workflow_point is not None, "Workflow not found in Qdrant"
        assert workflow_point.payload["workflow_name"] == sample_workflow_data["name"]
    
    @pytest.mark.asyncio
    async def test_workflow_similarity_search(
        self,
        qdrant_adapter: QdrantAdapter,
        sample_workflow_data: Dict[str, Any]
    ):
        """Test finding similar workflows using vector similarity"""
        # Store multiple similar workflows
        workflows = []
        for i in range(3):
            workflow_data = sample_workflow_data.copy()
            workflow_data["id"] = str(uuid.uuid4())
            workflow_data["name"] = f"Arabic Invoice Workflow {i+1}"
            
            # Generate similar embeddings
            embedding = [0.1 * (i + 1) * j for j in range(768)]
            
            await qdrant_adapter.store_workflow_embedding(
                workflow_id=workflow_data["id"],
                workflow_name=workflow_data["name"],
                embedding=embedding,
                metadata=workflow_data["metadata"]
            )
            workflows.append((workflow_data, embedding))
        
        # Search for similar workflows using the first workflow's embedding
        query_embedding = workflows[0][1]
        similar_workflows = await qdrant_adapter.find_similar_workflows(
            query_embedding=query_embedding,
            limit=5,
            score_threshold=0.5
        )
        
        assert len(similar_workflows) >= 1, "Should find at least the original workflow"
        
        # Verify similarity scores are reasonable
        for result in similar_workflows:
            assert result.score >= 0.5, f"Similarity score too low: {result.score}"
    
    @pytest.mark.asyncio
    async def test_hybrid_workflow_execution(
        self,
        hybrid_adapter: HybridOrchestrationAdapter,
        sample_workflow_data: Dict[str, Any]
    ):
        """Test hybrid workflow execution with both graph and vector processing"""
        result = await hybrid_adapter.execute_hybrid_workflow(
            workflow_data=sample_workflow_data,
            context={"user_id": "test_user", "session_id": "test_session"}
        )
        
        assert result is not None, "Hybrid workflow execution failed"
        assert result.workflow_id == sample_workflow_data["id"]
        assert isinstance(result.execution_time, float)
        assert result.execution_time > 0
        
        # Verify graph analytics are populated
        if result.graph_analytics:
            assert "node_count" in result.graph_analytics
        
        # Verify vector recommendations if Qdrant is available
        if hybrid_adapter.qdrant_adapter:
            assert isinstance(result.vector_recommendations, dict)
            assert isinstance(result.similar_workflows, list)
    
    @pytest.mark.asyncio
    async def test_data_synchronization(
        self,
        qdrant_adapter: QdrantAdapter,
        memgraph_adapter: MemgraphAdapter,
        sample_workflow_data: Dict[str, Any]
    ):
        """Test synchronization between Qdrant and Memgraph"""
        workflow_id = sample_workflow_data["id"]
        
        # Create workflow in Memgraph first
        await memgraph_adapter.create_workflow_graph(
            workflow_id=workflow_id,
            workflow_data=sample_workflow_data
        )
        
        # Update workflow status in Memgraph
        await memgraph_adapter.update_node_status(
            workflow_id=workflow_id,
            node_id="ocr_tool",
            status=NodeStatus.COMPLETED,
            execution_time=2.5
        )
        
        # Sync with Qdrant
        sync_success = await qdrant_adapter.sync_with_memgraph(
            memgraph_adapter=memgraph_adapter,
            workflow_id=workflow_id,
            sync_embeddings=True
        )
        
        assert sync_success, "Failed to sync data between databases"
        
        # Verify synchronized data in Qdrant
        workflow_point = await qdrant_adapter.get_point(
            "workflows", workflow_id, with_payload=True
        )
        
        if workflow_point:
            # Check if Memgraph data was synced to Qdrant payload
            assert "memgraph_node_id" in workflow_point.payload or "sync_timestamp" in workflow_point.payload
    
    @pytest.mark.asyncio
    async def test_intelligent_tool_recommendations(
        self,
        hybrid_adapter: HybridOrchestrationAdapter
    ):
        """Test intelligent tool recommendations using vector similarity"""
        # Store some tool embeddings first
        tools_data = [
            {
                "id": "arabic_ocr",
                "name": "Arabic OCR Tool",
                "description": "Extract text from Arabic documents using advanced OCR",
                "category": "ocr"
            },
            {
                "id": "invoice_parser", 
                "name": "Invoice Parser",
                "description": "Parse and extract structured data from invoices",
                "category": "parsing"
            },
            {
                "id": "data_validator",
                "name": "Data Validator",
                "description": "Validate extracted data for accuracy and completeness", 
                "category": "validation"
            }
        ]
        
        if hybrid_adapter.qdrant_adapter:
            for tool in tools_data:
                # Generate simple embedding for tool
                embedding = [hash(tool["description"] + str(i)) % 1000 / 1000.0 for i in range(384)]
                
                await hybrid_adapter.qdrant_adapter.store_tool_embedding(
                    tool_id=tool["id"],
                    tool_name=tool["name"],
                    tool_description=tool["description"],
                    embedding=embedding,
                    tool_metadata={"category": tool["category"]}
                )
        
        # Get recommendations for a task
        recommendations = await hybrid_adapter.get_intelligent_tool_recommendations(
            task_description="I need to process Arabic invoice images and extract data",
            context={"domain": "finance"}
        )
        
        # Should get some recommendations if Qdrant is available
        if hybrid_adapter.qdrant_adapter:
            assert isinstance(recommendations, list)
            # May be empty if no similar tools found, but should be a list
        else:
            assert recommendations == []  # Empty if Qdrant not available
    
    @pytest.mark.asyncio
    async def test_workflow_health_scoring(
        self,
        hybrid_adapter: HybridOrchestrationAdapter,
        sample_workflow_data: Dict[str, Any]
    ):
        """Test comprehensive workflow health scoring"""
        workflow_id = sample_workflow_data["id"]
        
        # Execute workflow first to have data
        await hybrid_adapter.execute_hybrid_workflow(
            workflow_data=sample_workflow_data,
            context={"test": True}
        )
        
        # Get health score
        health_score = await hybrid_adapter.get_workflow_health_score(workflow_id)
        
        assert isinstance(health_score, dict)
        assert "overall_score" in health_score
        assert isinstance(health_score["overall_score"], float)
        assert 0.0 <= health_score["overall_score"] <= 1.0
        
        assert "graph_health" in health_score
        assert "vector_health" in health_score
        assert "recommendations" in health_score
        assert isinstance(health_score["recommendations"], list)
