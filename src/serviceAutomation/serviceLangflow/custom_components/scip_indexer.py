"""
SCIP Indexer Component for Langflow
Indexes vendor code using SCIP protocol
"""

from langflow.custom import CustomComponent
from langflow.field_typing import Text
from typing import Optional
import requests
import json


class SCIPIndexerComponent(CustomComponent):
    display_name = "SCIP Indexer"
    description = "Index vendor code using SCIP (Source Code Intelligence Protocol)"
    icon = "code"
    
    def build_config(self):
        return {
            "project_path": {
                "display_name": "Project Path",
                "info": "Path to vendor project (e.g., vendor/layerIntelligence/hyperbooklm)",
            },
            "language": {
                "display_name": "Language",
                "options": ["rust", "typescript", "python"],
                "info": "Programming language of the project",
            },
            "output_name": {
                "display_name": "Output Name",
                "info": "Name for the SCIP index file",
            },
            "scip_service_url": {
                "display_name": "SCIP Service URL",
                "value": "http://localhost:8008",
                "advanced": True,
            },
        }
    
    def build(
        self,
        project_path: str,
        language: str,
        output_name: Optional[str] = None,
        scip_service_url: str = "http://localhost:8008",
    ) -> Text:
        """Index vendor code with SCIP"""
        
        if not output_name:
            output_name = project_path.split("/")[-1]
        
        # Call SCIP service
        response = requests.post(
            f"{scip_service_url}/index/{language}",
            json={
                "project_path": project_path,
                "language": language,
                "output_name": output_name,
            },
            timeout=120,
        )
        
        if response.status_code != 200:
            raise Exception(f"SCIP indexing failed: {response.text}")
        
        result = response.json()
        
        # Return formatted result
        return {
            "index_path": result["index_path"],
            "symbols_count": result["symbols_count"],
            "documents_count": result["documents_count"],
            "language": language,
            "project": project_path,
            "message": f"✅ Indexed {result['symbols_count']} symbols in {result['documents_count']} documents",
        }