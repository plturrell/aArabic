"""
Adapter layer for external services.

This module provides integration adapters for all vendor services
in the AI Nucleus platform.
"""

# AI Services
from .shimmy import ShimmyAdapter, check_shimmy_health
from .opencanvas import OpenCanvasAdapter, check_opencanvas_health
from .hyperbooklm import HyperbookLMAdapter, check_hyperbooklm_health

# Data Layer
from .qdrant import QdrantAdapter, check_qdrant_health
from .memgraph import MemgraphAdapter, check_memgraph_health
from .dragonfly import DragonflyAdapter, check_dragonfly_health

# Workflow & Orchestration
from .orchestration import OrchestrationAdapter
from .hybrid_orchestration import HybridOrchestrationAdapter
from .nucleus_flow import NucleusFlowAdapter, check_nucleus_flow_health
from .toolorchestra import ToolOrchestraAdapter

# UI Components
from .a2ui import A2UIAdapter
from .a2ui_enhanced import A2UIEnhancedAdapter

# Infrastructure
# These adapters don't have main classes, only utility functions
from .apisix import check_apisix_health  # APISIXAdapter doesn't exist
from .keycloak import check_keycloak_health  # KeycloakAdapter doesn't exist
from .gitea import GiteaAdapter, check_gitea_health  # This one is OK
from .marquez import check_marquez_health  # MarquezAdapter doesn't exist
from .nucleusgraph import check_nucleusgraph_health  # NucleusGraphAdapter doesn't exist

__all__ = [
    # AI Services
    "ShimmyAdapter",
    "check_shimmy_health",
    "OpenCanvasAdapter",
    "check_opencanvas_health",
    "HyperbookLMAdapter",
    "check_hyperbooklm_health",
    # Data Layer
    "QdrantAdapter",
    "check_qdrant_health",
    "MemgraphAdapter",
    "check_memgraph_health",
    "DragonflyAdapter",
    "check_dragonfly_health",
    # Workflow & Orchestration
    "OrchestrationAdapter",
    "HybridOrchestrationAdapter",
    "NucleusFlowAdapter",
    "check_nucleus_flow_health",
    "ToolOrchestraAdapter",
    # UI Components
    "A2UIAdapter",
    "A2UIEnhancedAdapter",
    # Infrastructure
    "check_apisix_health",  # No APISIXAdapter class
    "check_keycloak_health",  # No KeycloakAdapter class
    "GiteaAdapter",
    "check_gitea_health",
    "check_marquez_health",  # No MarquezAdapter class
    "check_nucleusgraph_health",  # No NucleusGraphAdapter class
]
