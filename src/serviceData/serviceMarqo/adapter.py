"""
Marqo Adapter - Semantic Vector Search Integration
Provides tensor search capabilities for the Ouroboros knowledge infrastructure
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime


class MarqoAdapter:
    """
    Adapter for Marqo tensor search engine
    Enables semantic search across multimodal content
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8882",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        if api_key:
            self.client.headers.update({"x-api-key": api_key})
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Marqo service health"""
        try:
            response = await self.client.get(f"{self.url}/health")
            response.raise_for_status()
            return {"status": "healthy", "details": response.json()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def create_index(
        self,
        index_name: str,
        model: str = "hf/e5-base-v2",
        normalize_embeddings: bool = True,
        text_preprocessing: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create a new Marqo index
        
        Args:
            index_name: Name of the index
            model: Model to use for embeddings
            normalize_embeddings: Whether to normalize vectors
            text_preprocessing: Text preprocessing settings
        """
        payload = {
            "index_name": index_name,
            "model": model,
            "normalize_embeddings": normalize_embeddings,
            "text_preprocessing": text_preprocessing or {
                "split_length": 2,
                "split_overlap": 0,
                "split_method": "sentence"
            }
        }
        
        try:
            response = await self.client.post(
                f"{self.url}/indexes/{index_name}",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                return {"status": "exists", "index_name": index_name}
            raise
    
    async def add_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        tensor_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add documents to index
        
        Args:
            index_name: Target index
            documents: List of documents to add
            tensor_fields: Fields to create tensors for
        """
        payload = {
            "documents": documents,
            "tensorFields": tensor_fields or ["content", "description"]
        }
        
        response = await self.client.post(
            f"{self.url}/indexes/{index_name}/documents",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def search(
        self,
        index_name: str,
        query: str,
        limit: int = 10,
        filter_string: Optional[str] = None,
        search_method: str = "TENSOR",
        attributes_to_retrieve: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Semantic search in index
        
        Args:
            index_name: Index to search
            query: Search query
            limit: Number of results
            filter_string: Marqo filter expression
            search_method: TENSOR or LEXICAL
            attributes_to_retrieve: Fields to return
        """
        payload = {
            "q": query,
            "limit": limit,
            "searchMethod": search_method
        }
        
        if filter_string:
            payload["filter"] = filter_string
        if attributes_to_retrieve:
            payload["attributesToRetrieve"] = attributes_to_retrieve
        
        response = await self.client.post(
            f"{self.url}/indexes/{index_name}/search",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def hybrid_search(
        self,
        index_name: str,
        query: str,
        limit: int = 10,
        retrieval_method: str = "disjunction",
        rank_profile: str = "bm25"
    ) -> Dict[str, Any]:
        """
        Hybrid search combining tensor and lexical search
        """
        payload = {
            "q": query,
            "limit": limit,
            "retrievalMethod": retrieval_method,
            "rankingMethod": rank_profile,
            "searchableAttributes": ["content", "description", "metadata"]
        }
        
        response = await self.client.post(
            f"{self.url}/indexes/{index_name}/search",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_document(
        self,
        index_name: str,
        document_id: str
    ) -> Dict[str, Any]:
        """Get a specific document by ID"""
        response = await self.client.get(
            f"{self.url}/indexes/{index_name}/documents/{document_id}"
        )
        response.raise_for_status()
        return response.json()
    
    async def delete_documents(
        self,
        index_name: str,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """Delete documents from index"""
        payload = {"documentIds": document_ids}
        
        response = await self.client.post(
            f"{self.url}/indexes/{index_name}/documents/delete-batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index"""
        response = await self.client.get(
            f"{self.url}/indexes/{index_name}/stats"
        )
        response.raise_for_status()
        return response.json()
    
    async def list_indexes(self) -> List[Dict[str, Any]]:
        """List all indexes"""
        response = await self.client.get(f"{self.url}/indexes")
        response.raise_for_status()
        return response.json().get("results", [])
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    # ─────────────────────────────────────────────────────────────
    # Ouroboros-Specific Methods
    # ─────────────────────────────────────────────────────────────
    
    async def index_code_knowledge(
        self,
        code_snippets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Index code snippets for semantic code search"""
        documents = [
            {
                "_id": snippet["id"],
                "content": snippet["code"],
                "description": snippet.get("description", ""),
                "language": snippet.get("language", "python"),
                "file_path": snippet.get("file_path", ""),
                "indexed_at": datetime.utcnow().isoformat()
            }
            for snippet in code_snippets
        ]
        
        return await self.add_documents(
            "code_knowledge",
            documents,
            tensor_fields=["content", "description"]
        )
    
    async def index_proof_knowledge(
        self,
        proofs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Index mathematical proofs and verifications"""
        documents = [
            {
                "_id": proof["id"],
                "content": proof["proof_text"],
                "description": proof.get("theorem", ""),
                "domain": proof.get("domain", "general"),
                "verified": proof.get("verified", False),
                "indexed_at": datetime.utcnow().isoformat()
            }
            for proof in proofs
        ]
        
        return await self.add_documents(
            "proof_knowledge",
            documents,
            tensor_fields=["content", "description"]
        )
    
    async def semantic_code_search(
        self,
        query: str,
        language: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for semantically similar code"""
        filter_string = f"language:{language}" if language else None
        
        result = await self.search(
            "code_knowledge",
            query,
            limit=limit,
            filter_string=filter_string,
            search_method="TENSOR"
        )
        
        return result.get("hits", [])
    
    async def find_similar_proofs(
        self,
        theorem: str,
        domain: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar mathematical proofs"""
        filter_string = f"domain:{domain}" if domain else None
        
        result = await self.search(
            "proof_knowledge",
            theorem,
            limit=limit,
            filter_string=filter_string,
            search_method="TENSOR"
        )
        
        return result.get("hits", [])


# Singleton instance
_marqo_adapter: Optional[MarqoAdapter] = None


def get_marqo_adapter(
    url: str = "http://localhost:8882",
    api_key: Optional[str] = None
) -> MarqoAdapter:
    """Get or create Marqo adapter instance"""
    global _marqo_adapter
    if _marqo_adapter is None:
        _marqo_adapter = MarqoAdapter(url=url, api_key=api_key)
    return _marqo_adapter
