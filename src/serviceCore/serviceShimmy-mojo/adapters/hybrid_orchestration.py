"""
Hybrid Orchestration Adapter
Integrates Memgraph graph database with Qdrant vector database
for comprehensive AI workflow management and semantic processing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import uuid
from datetime import datetime
import numpy as np

from backend.adapters.memgraph import MemgraphAdapter, WorkflowGraphNode, NodeStatus
from backend.adapters.qdrant import QdrantAdapter, VectorPoint, VectorType, get_qdrant_adapter
from backend.adapters.orchestration import OrchestrationAdapter

logger = logging.getLogger(__name__)


@dataclass
class HybridWorkflowResult:
    """Result from hybrid orchestration combining graph and vector data"""
    workflow_id: str
    execution_result: Dict[str, Any]
    graph_analytics: Dict[str, Any]
    vector_recommendations: Dict[str, Any]
    similar_workflows: List[Dict[str, Any]]
    execution_time: float
    success: bool


class HybridOrchestrationAdapter:
    """
    Hybrid orchestration adapter that combines:
    - Memgraph for workflow dependency tracking and execution flow
    - Qdrant for semantic similarity and AI-powered recommendations
    - Enhanced workflow optimization using both graph and vector data
    """
    
    def __init__(self):
        self.memgraph_adapter = MemgraphAdapter()
        self.qdrant_adapter: Optional[QdrantAdapter] = None
        self.base_orchestration = OrchestrationAdapter()
        
        # Embedding generation (simplified - in production use proper embedding models)
        self.embedding_dimension = 768
        
    async def initialize(self) -> bool:
        """Initialize both Memgraph and Qdrant adapters"""
        try:
            # Initialize Memgraph
            memgraph_success = await self.memgraph_adapter.initialize()
            if not memgraph_success:
                logger.warning("Memgraph initialization failed")
            
            # Initialize Qdrant
            try:
                self.qdrant_adapter = await get_qdrant_adapter()
                qdrant_success = await self.qdrant_adapter.health_check()
                if not qdrant_success:
                    logger.warning("Qdrant not available - vector features disabled")
                    self.qdrant_adapter = None
            except Exception as e:
                logger.warning(f"Qdrant initialization failed: {e}")
                self.qdrant_adapter = None
            
            # Initialize base orchestration
            base_success = await self.base_orchestration.initialize()
            
            logger.info(f"Hybrid orchestration initialized - Memgraph: {memgraph_success}, Qdrant: {self.qdrant_adapter is not None}, Base: {base_success}")
            return memgraph_success or base_success
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid orchestration: {e}")
            return False
    
    def generate_workflow_embedding(self, workflow_data: Dict[str, Any]) -> List[float]:
        """Generate embedding for workflow (simplified implementation)"""
        try:
            # In production, use proper embedding models like sentence-transformers
            # This is a simplified version for demonstration
            
            # Extract text features
            text_features = []
            text_features.append(workflow_data.get("name", ""))
            text_features.append(workflow_data.get("description", ""))
            
            # Add tool names
            tools = workflow_data.get("tools", [])
            for tool in tools:
                if isinstance(tool, dict):
                    text_features.append(tool.get("name", ""))
                else:
                    text_features.append(str(tool))
            
            # Simple hash-based embedding (replace with real embedding model)
            combined_text = " ".join(text_features).lower()
            
            # Generate pseudo-embedding based on text hash
            embedding = []
            for i in range(self.embedding_dimension):
                hash_val = hash(combined_text + str(i)) % 1000000
                embedding.append((hash_val / 1000000.0) * 2 - 1)  # Normalize to [-1, 1]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate workflow embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    async def execute_hybrid_workflow(
        self,
        workflow_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> HybridWorkflowResult:
        """Execute workflow with hybrid graph + vector processing"""
        start_time = datetime.now()
        workflow_id = workflow_data.get("id", str(uuid.uuid4()))
        
        try:
            # Step 1: Create workflow graph in Memgraph
            graph_created = await self.memgraph_adapter.create_workflow_graph(
                workflow_id=workflow_id,
                workflow_data=workflow_data
            )
            
            # Step 2: Generate and store workflow embedding in Qdrant
            workflow_embedding = None
            if self.qdrant_adapter:
                try:
                    workflow_embedding = self.generate_workflow_embedding(workflow_data)
                    await self.qdrant_adapter.store_workflow_embedding(
                        workflow_id=workflow_id,
                        workflow_name=workflow_data.get("name", "Unknown"),
                        embedding=workflow_embedding,
                        metadata={
                            "type": workflow_data.get("type", "general"),
                            "status": "executing",
                            "created_by": context.get("user_id", "system") if context else "system",
                            "tags": workflow_data.get("tags", []),
                            "tool_count": len(workflow_data.get("tools", [])),
                            "complexity": self.calculate_workflow_complexity(workflow_data)
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to store workflow embedding: {e}")
            
            # Step 3: Execute workflow using base orchestration
            execution_result = await self.base_orchestration.execute_workflow(
                workflow_data, context
            )
            
            # Step 4: Update graph with execution results
            if graph_created:
                await self.update_graph_with_results(workflow_id, execution_result)
            
            # Step 5: Get graph analytics
            graph_analytics = {}
            if graph_created:
                try:
                    graph_analytics = await self.memgraph_adapter.get_workflow_analytics(workflow_id)
                except Exception as e:
                    logger.warning(f"Failed to get graph analytics: {e}")
            
            # Step 6: Get vector-based recommendations
            vector_recommendations = {}
            similar_workflows = []
            
            if self.qdrant_adapter and workflow_embedding:
                try:
                    # Update workflow status in Qdrant
                    await self.qdrant_adapter.store_workflow_embedding(
                        workflow_id=workflow_id,
                        workflow_name=workflow_data.get("name", "Unknown"),
                        embedding=workflow_embedding,
                        metadata={
                            "type": workflow_data.get("type", "general"),
                            "status": "completed" if execution_result.get("success", False) else "failed",
                            "execution_time": (datetime.now() - start_time).total_seconds(),
                            "success_rate": 1.0 if execution_result.get("success", False) else 0.0,
                            **workflow_data.get("metadata", {})
                        }
                    )
                    
                    # Get recommendations
                    vector_recommendations = await self.qdrant_adapter.get_workflow_recommendations(
                        current_workflow_id=workflow_id,
                        limit=3,
                        include_similar_patterns=True
                    )
                    
                    # Find similar workflows
                    similar_results = await self.qdrant_adapter.find_similar_workflows(
                        query_embedding=workflow_embedding,
                        limit=5,
                        score_threshold=0.7
                    )
                    
                    similar_workflows = [
                        {
                            "workflow_id": result.id,
                            "workflow_name": result.payload.get("workflow_name", "Unknown"),
                            "similarity_score": result.score,
                            "success_rate": result.payload.get("success_rate", 0.0)
                        }
                        for result in similar_results
                        if result.id != workflow_id
                    ]
                    
                except Exception as e:
                    logger.warning(f"Failed to get vector recommendations: {e}")
            
            # Step 7: Sync data between Memgraph and Qdrant
            if self.qdrant_adapter and graph_created:
                try:
                    await self.qdrant_adapter.sync_with_memgraph(
                        memgraph_adapter=self.memgraph_adapter,
                        workflow_id=workflow_id,
                        sync_embeddings=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to sync with Memgraph: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return HybridWorkflowResult(
                workflow_id=workflow_id,
                execution_result=execution_result,
                graph_analytics=graph_analytics,
                vector_recommendations=vector_recommendations,
                similar_workflows=similar_workflows,
                execution_time=execution_time,
                success=execution_result.get("success", False)
            )
            
        except Exception as e:
            logger.error(f"Hybrid workflow execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return HybridWorkflowResult(
                workflow_id=workflow_id,
                execution_result={"success": False, "error": str(e)},
                graph_analytics={},
                vector_recommendations={},
                similar_workflows=[],
                execution_time=execution_time,
                success=False
            )

    def calculate_workflow_complexity(self, workflow_data: Dict[str, Any]) -> str:
        """Calculate workflow complexity based on structure"""
        try:
            tool_count = len(workflow_data.get("tools", []))
            has_conditions = bool(workflow_data.get("conditions", []))
            has_loops = bool(workflow_data.get("loops", []))
            has_parallel = bool(workflow_data.get("parallel_execution", False))

            complexity_score = tool_count
            if has_conditions:
                complexity_score += 2
            if has_loops:
                complexity_score += 3
            if has_parallel:
                complexity_score += 2

            if complexity_score <= 3:
                return "low"
            elif complexity_score <= 7:
                return "medium"
            else:
                return "high"

        except Exception:
            return "medium"

    async def update_graph_with_results(
        self,
        workflow_id: str,
        execution_result: Dict[str, Any]
    ):
        """Update Memgraph with execution results"""
        try:
            # Update workflow status
            success = execution_result.get("success", False)
            status = NodeStatus.COMPLETED if success else NodeStatus.FAILED

            # Update individual tool results if available
            tool_results = execution_result.get("tool_results", {})
            for tool_id, result in tool_results.items():
                tool_status = NodeStatus.COMPLETED if result.get("success", False) else NodeStatus.FAILED
                execution_time = result.get("execution_time", 0.0)
                error_message = result.get("error") if not result.get("success", False) else None

                await self.memgraph_adapter.update_node_status(
                    workflow_id=workflow_id,
                    node_id=tool_id,
                    status=tool_status,
                    execution_time=execution_time,
                    error_message=error_message
                )

        except Exception as e:
            logger.error(f"Failed to update graph with results: {e}")

    async def get_intelligent_tool_recommendations(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get intelligent tool recommendations using vector similarity"""
        try:
            if not self.qdrant_adapter:
                return []

            # Generate embedding for task description
            task_embedding = self.generate_workflow_embedding({
                "name": task_description,
                "description": task_description,
                "tools": []
            })

            # Find relevant tools
            relevant_tools = await self.qdrant_adapter.find_relevant_tools(
                task_embedding=task_embedding,
                limit=5
            )

            recommendations = []
            for tool_result in relevant_tools:
                recommendations.append({
                    "tool_id": tool_result.id,
                    "tool_name": tool_result.payload.get("tool_name", "Unknown"),
                    "relevance_score": tool_result.score,
                    "description": tool_result.payload.get("tool_description", ""),
                    "category": tool_result.payload.get("tool_category", "general"),
                    "success_rate": tool_result.payload.get("success_rate", 1.0),
                    "avg_execution_time": tool_result.payload.get("execution_time_avg", 0.0)
                })

            return recommendations

        except Exception as e:
            logger.error(f"Failed to get intelligent tool recommendations: {e}")
            return []

    async def analyze_workflow_patterns(
        self,
        workflow_type: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Analyze workflow patterns using both graph and vector data"""
        try:
            analysis = {
                "pattern_insights": [],
                "optimization_suggestions": [],
                "common_tool_sequences": [],
                "performance_metrics": {}
            }

            # Get graph-based patterns from Memgraph
            if workflow_type:
                # This would involve complex Cypher queries to find patterns
                # Simplified for demonstration
                pass

            # Get vector-based similar workflows from Qdrant
            if self.qdrant_adapter:
                try:
                    # Get all workflows of the specified type
                    filter_conditions = {"workflow_type": workflow_type} if workflow_type else None

                    # This is a simplified approach - in production you'd use more sophisticated analysis
                    analysis["pattern_insights"] = [
                        "Vector similarity analysis would provide insights here"
                    ]

                except Exception as e:
                    logger.warning(f"Vector pattern analysis failed: {e}")

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze workflow patterns: {e}")
            return {}

    async def optimize_workflow_execution(
        self,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize workflow execution using hybrid graph + vector insights"""
        try:
            optimization_result = {
                "original_workflow": workflow_data,
                "optimized_workflow": workflow_data.copy(),
                "optimizations_applied": [],
                "expected_improvement": 0.0
            }

            # Graph-based optimizations
            if workflow_data.get("tools"):
                # Analyze dependencies and suggest parallel execution
                parallel_groups = self.identify_parallel_execution_groups(workflow_data)
                if parallel_groups:
                    optimization_result["optimized_workflow"]["parallel_groups"] = parallel_groups
                    optimization_result["optimizations_applied"].append("parallel_execution")
                    optimization_result["expected_improvement"] += 0.3

            # Vector-based optimizations
            if self.qdrant_adapter:
                try:
                    # Find similar successful workflows
                    workflow_embedding = self.generate_workflow_embedding(workflow_data)
                    similar_workflows = await self.qdrant_adapter.find_similar_workflows(
                        query_embedding=workflow_embedding,
                        limit=3,
                        score_threshold=0.8,
                        status_filter=["completed"]
                    )

                    # Suggest tool replacements based on successful patterns
                    if similar_workflows:
                        optimization_result["optimizations_applied"].append("tool_optimization")
                        optimization_result["expected_improvement"] += 0.2

                        # Add suggestions based on similar workflows
                        optimization_result["similar_workflow_insights"] = [
                            {
                                "workflow_id": result.id,
                                "similarity_score": result.score,
                                "success_rate": result.payload.get("success_rate", 0.0)
                            }
                            for result in similar_workflows
                        ]

                except Exception as e:
                    logger.warning(f"Vector-based optimization failed: {e}")

            return optimization_result

        except Exception as e:
            logger.error(f"Failed to optimize workflow execution: {e}")
            return {"error": str(e)}

    def identify_parallel_execution_groups(
        self,
        workflow_data: Dict[str, Any]
    ) -> List[List[str]]:
        """Identify tools that can be executed in parallel"""
        try:
            tools = workflow_data.get("tools", [])
            if len(tools) < 2:
                return []

            # Simplified dependency analysis
            # In production, this would analyze actual tool dependencies
            parallel_groups = []

            # Group tools that don't depend on each other
            independent_tools = []
            for tool in tools:
                if isinstance(tool, dict):
                    tool_id = tool.get("id", tool.get("name", ""))
                    dependencies = tool.get("dependencies", [])
                    if not dependencies:
                        independent_tools.append(tool_id)

            if len(independent_tools) > 1:
                parallel_groups.append(independent_tools)

            return parallel_groups

        except Exception as e:
            logger.error(f"Failed to identify parallel execution groups: {e}")
            return []

    async def get_workflow_health_score(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive workflow health score using both graph and vector data"""
        try:
            health_score = {
                "overall_score": 0.0,
                "graph_health": 0.0,
                "vector_health": 0.0,
                "recommendations": []
            }

            # Graph-based health metrics
            try:
                graph_analytics = await self.memgraph_adapter.get_workflow_analytics(workflow_id)
                if graph_analytics:
                    completed_nodes = graph_analytics.get("completed_nodes", 0)
                    total_nodes = graph_analytics.get("node_count", 1)
                    failed_nodes = graph_analytics.get("failed_nodes", 0)

                    graph_health = (completed_nodes / total_nodes) * (1 - (failed_nodes / total_nodes))
                    health_score["graph_health"] = max(0.0, min(1.0, graph_health))
            except Exception as e:
                logger.warning(f"Graph health calculation failed: {e}")

            # Vector-based health metrics
            if self.qdrant_adapter:
                try:
                    workflow_point = await self.qdrant_adapter.get_point(
                        "workflows", workflow_id, with_payload=True
                    )

                    if workflow_point:
                        success_rate = workflow_point.payload.get("success_rate", 0.0)
                        execution_time = workflow_point.payload.get("execution_time", 0.0)

                        # Calculate vector health based on success rate and performance
                        vector_health = success_rate * (1.0 / (1.0 + execution_time / 60.0))  # Penalize long execution times
                        health_score["vector_health"] = max(0.0, min(1.0, vector_health))

                except Exception as e:
                    logger.warning(f"Vector health calculation failed: {e}")

            # Calculate overall score
            health_score["overall_score"] = (
                health_score["graph_health"] * 0.6 +
                health_score["vector_health"] * 0.4
            )

            # Generate recommendations based on health score
            if health_score["overall_score"] < 0.7:
                health_score["recommendations"].append("Consider workflow optimization")

            if health_score["graph_health"] < 0.5:
                health_score["recommendations"].append("Review workflow dependencies and error handling")

            if health_score["vector_health"] < 0.5:
                health_score["recommendations"].append("Analyze similar successful workflows for improvements")

            return health_score

        except Exception as e:
            logger.error(f"Failed to calculate workflow health score: {e}")
            return {"overall_score": 0.0, "error": str(e)}

    async def close(self):
        """Close all adapter connections"""
        try:
            await self.memgraph_adapter.close()
            if self.qdrant_adapter:
                await self.qdrant_adapter.close()
            await self.base_orchestration.close()
            logger.info("Hybrid orchestration adapter closed")
        except Exception as e:
            logger.error(f"Error closing hybrid orchestration adapter: {e}")


# Global hybrid orchestration adapter instance
hybrid_orchestration_adapter = HybridOrchestrationAdapter()

async def get_hybrid_orchestration_adapter() -> HybridOrchestrationAdapter:
    """Get the global hybrid orchestration adapter instance"""
    return hybrid_orchestration_adapter

async def initialize_hybrid_orchestration() -> bool:
    """Initialize the global hybrid orchestration adapter"""
    return await hybrid_orchestration_adapter.initialize()

async def shutdown_hybrid_orchestration():
    """Shutdown the global hybrid orchestration adapter"""
    await hybrid_orchestration_adapter.close()
