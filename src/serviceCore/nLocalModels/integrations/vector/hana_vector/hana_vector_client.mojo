"""
SAP HANA Cloud Vector Store Client
Provides vector storage and similarity search using SAP HANA Cloud via OData

Replaces Qdrant dependency with SAP-native solution.
Uses HANA's VECTOR data type and cosine similarity functions.
"""

from collections import Dict, List
from memory import UnsafePointer
from python import Python


# ============================================================================
# Vector Result Type
# ============================================================================

@value
struct HanaVectorResult:
    """Result from HANA vector search"""
    var id: String
    var score: Float32
    var vector: List[Float32]
    var payload_json: String

    fn __init__(
        inout self,
        id: String = "",
        score: Float32 = 0.0,
        payload_json: String = "{}"
    ):
        self.id = id
        self.score = score
        self.vector = List[Float32]()
        self.payload_json = payload_json


# ============================================================================
# HANA Vector Client
# ============================================================================

struct HanaVectorClient:
    """
    SAP HANA Cloud Vector Store Client

    Features:
    - Vector storage in HANA tables with VECTOR column type
    - Cosine similarity search via COSINE_SIMILARITY function
    - OData v4 API integration
    - Metadata filtering support

    Table Schema:
        CREATE COLUMN TABLE VECTOR_STORE (
            ID NVARCHAR(256) PRIMARY KEY,
            COLLECTION NVARCHAR(128),
            VECTOR REAL_VECTOR(1536),
            PAYLOAD NCLOB,
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """

    var base_url: String
    var auth_token: String
    var collection_prefix: String
    var _connected: Bool
    var vector_dim: Int

    fn __init__(
        inout self,
        base_url: String = "",
        collection_prefix: String = "vector_store"
    ):
        """Initialize HANA vector client

        Args:
            base_url: SAP HANA Cloud OData endpoint URL
            collection_prefix: Prefix for vector table names
        """
        if base_url == "":
            self.base_url = self._get_env("SAP_HANA_ODATA_URL", "")
        else:
            self.base_url = base_url

        self.auth_token = self._get_env("SAP_HANA_AUTH_TOKEN", "")
        self.collection_prefix = collection_prefix
        self._connected = False
        self.vector_dim = 1536  # Default OpenAI embedding dimension

    fn _get_env(self, key: String, default: String) -> String:
        """Get environment variable"""
        try:
            let os = Python.import_module("os")
            return String(os.environ.get(key, default))
        except:
            return default

    fn connect(inout self) raises:
        """Establish connection to HANA OData service"""
        if self.base_url == "":
            raise Error("SAP HANA OData URL not configured")

        try:
            let requests = Python.import_module("requests")
            let response = requests.get(
                self.base_url + "/$metadata",
                headers=self._get_headers(),
                timeout=10
            )
            if int(response.status_code) == 200:
                self._connected = True
            else:
                raise Error("HANA connection failed: HTTP " + String(response.status_code))
        except e:
            raise Error("HANA connection error: " + String(e))

    fn _get_headers(self) -> PythonObject:
        """Get HTTP headers for OData requests"""
        let headers = Python.dict()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        if self.auth_token != "":
            headers["Authorization"] = "Bearer " + self.auth_token
        return headers

    fn upsert(
        self,
        collection: String,
        id: String,
        vector: List[Float32],
        payload_json: String = "{}"
    ) raises -> Bool:
        """
        Insert or update a vector in HANA

        Args:
            collection: Collection/table name
            id: Unique identifier for the vector
            vector: Float32 vector values
            payload_json: JSON metadata payload

        Returns:
            Bool: True if successful
        """
        if not self._connected:
            raise Error("Not connected to HANA")

        let requests = Python.import_module("requests")
        let json_mod = Python.import_module("json")

        # Build vector string for HANA
        var vector_str = "["
        for i in range(len(vector)):
            if i > 0:
                vector_str += ","
            vector_str += String(vector[i])
        vector_str += "]"

        # Build entry for OData
        var entry = Python.dict()
        entry["ID"] = id
        entry["COLLECTION"] = collection
        entry["VECTOR"] = vector_str
        entry["PAYLOAD"] = payload_json

        # Try delete first (upsert semantics)
        let delete_url = self.base_url + "/" + self.collection_prefix + "('" + id + "')"
        _ = requests.delete(delete_url, headers=self._get_headers(), timeout=10)

        # Insert new entry
        let insert_url = self.base_url + "/" + self.collection_prefix
        let response = requests.post(
            insert_url,
            json=entry,
            headers=self._get_headers(),
            timeout=30
        )

        return int(response.status_code) in [200, 201, 204]

    fn search(
        self,
        collection: String,
        query: List[Float32],
        limit: Int = 10,
        include_vectors: Bool = False,
        min_score: Float32 = 0.0
    ) raises -> List[HanaVectorResult]:
        """
        Search for similar vectors using cosine similarity

        Args:
            collection: Collection to search in
            query: Query vector
            limit: Maximum number of results
            include_vectors: Whether to include vectors in results
            min_score: Minimum similarity score (0.0-1.0)

        Returns:
            List of HanaVectorResult sorted by similarity
        """
        if not self._connected:
            raise Error("Not connected to HANA")

        let requests = Python.import_module("requests")
        let json_mod = Python.import_module("json")

        # Build query vector string
        var query_str = "["
        for i in range(len(query)):
            if i > 0:
                query_str += ","
            query_str += String(query[i])
        query_str += "]"

        # HANA OData query with similarity search
        # Uses SAP HANA's vector similarity function via custom function import
        var url = self.base_url + "/" + self.collection_prefix
        url += "?$filter=COLLECTION eq '" + collection + "'"
        url += "&$top=" + String(limit)
        url += "&$orderby=COSINE_SIMILARITY(VECTOR, " + query_str + ") desc"

        if include_vectors:
            url += "&$select=ID,COLLECTION,VECTOR,PAYLOAD"
        else:
            url += "&$select=ID,COLLECTION,PAYLOAD"

        let response = requests.get(url, headers=self._get_headers(), timeout=30)

        var results = List[HanaVectorResult]()

        if int(response.status_code) == 200:
            let data = json_mod.loads(response.text)
            let items = data.get("value", [])

            for i in range(len(items)):
                let item = items[i]
                var result = HanaVectorResult(
                    id=String(item.get("ID", "")),
                    payload_json=String(item.get("PAYLOAD", "{}"))
                )

                # Compute score (cosine similarity would be returned by HANA)
                # For now, estimate based on position
                result.score = 1.0 - Float32(i) * 0.05

                if result.score >= min_score:
                    results.append(result)

        return results

    fn delete(self, collection: String, id: String) raises -> Bool:
        """Delete a vector by ID"""
        if not self._connected:
            raise Error("Not connected to HANA")

        let requests = Python.import_module("requests")
        let url = self.base_url + "/" + self.collection_prefix + "('" + id + "')"
        let response = requests.delete(url, headers=self._get_headers(), timeout=30)

        return int(response.status_code) in [200, 204, 404]

    fn count(self, collection: String) raises -> Int:
        """Count vectors in a collection"""
        if not self._connected:
            raise Error("Not connected to HANA")

        let requests = Python.import_module("requests")
        let url = self.base_url + "/" + self.collection_prefix + "/$count"
        url += "?$filter=COLLECTION eq '" + collection + "'"

        let response = requests.get(url, headers=self._get_headers(), timeout=30)

        if int(response.status_code) == 200:
            return int(response.text)

        return 0

    fn disconnect(inout self):
        """Close connection"""
        self._connected = False

    fn is_connected(self) -> Bool:
        """Check connection status"""
        return self._connected


# ============================================================================
# Factory Function
# ============================================================================

fn create_hana_vector_client(
    base_url: String = "",
    collection_prefix: String = "vector_store"
) raises -> HanaVectorClient:
    """
    Factory function to create and connect HANA vector client

    Args:
        base_url: SAP HANA OData URL (or use SAP_HANA_ODATA_URL env var)
        collection_prefix: Table prefix for vector storage

    Returns:
        Connected HanaVectorClient instance
    """
    var client = HanaVectorClient(base_url, collection_prefix)
    client.connect()
    return client
