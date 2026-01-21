"""
nCode Client Library for Mojo
Provides a complete API client for the nCode SCIP-based code intelligence platform

Usage:
    from ncode_client import NCodeClient
    
    var client = NCodeClient("http://localhost:18003")
    var health = client.health()
    print("Status:", health.status)
    
    client.load_index("index.scip")
    var symbols = client.get_symbols("src/main.mojo")
"""

from python import Python
from memory import memcpy
from sys import sizeof


struct Position:
    var line: Int
    var character: Int
    
    fn __init__(inout self, line: Int, character: Int):
        self.line = line
        self.character = character


struct Range:
    var start: Position
    var end: Position
    
    fn __init__(inout self, start: Position, end: Position):
        self.start = start
        self.end = end


struct HealthResponse:
    var status: String
    var version: String
    var uptime_seconds: Float64
    var index_loaded: Bool
    
    fn __init__(inout self, status: String, version: String, uptime: Float64 = 0.0, loaded: Bool = False):
        self.status = status
        self.version = version
        self.uptime_seconds = uptime
        self.index_loaded = loaded


struct LoadIndexResponse:
    var success: Bool
    var message: String
    var documents: Int
    var symbols: Int
    
    fn __init__(inout self, success: Bool, message: String, docs: Int = 0, syms: Int = 0):
        self.success = success
        self.message = message
        self.documents = docs
        self.symbols = syms


struct SymbolInfo:
    var name: String
    var kind: String
    var line: Int
    var character: Int
    var detail: String
    
    fn __init__(inout self, name: String, kind: String, line: Int, char: Int, detail: String = ""):
        self.name = name
        self.kind = kind
        self.line = line
        self.character = char
        self.detail = detail


struct NCodeClient:
    """Main nCode client for interacting with the API"""
    var base_url: String
    var timeout_ms: Int
    var py: Python
    var requests: PythonObject
    
    fn __init__(inout self, base_url: String = "http://localhost:18003", timeout_ms: Int = 30000) raises:
        """Initialize a new nCode client"""
        self.base_url = base_url
        self.timeout_ms = timeout_ms
        self.py = Python()
        self.requests = self.py.import_module("requests")
    
    fn health(self) raises -> HealthResponse:
        """Check the health of the nCode server"""
        let url = self.base_url + "/health"
        let response = self.requests.get(url, timeout=self.timeout_ms / 1000)
        
        if response.status_code != 200:
            raise Error("Health check failed with status: " + String(response.status_code))
        
        let data = response.json()
        return HealthResponse(
            String(data["status"]),
            String(data.get("version", "unknown")),
            Float64(data.get("uptime_seconds", 0.0)),
            Bool(data.get("index_loaded", False))
        )
    
    fn load_index(self, scip_path: String) raises -> LoadIndexResponse:
        """Load a SCIP index file into the server"""
        let url = self.base_url + "/v1/index/load"
        let payload = self.py.evaluate('{"path": "' + scip_path + '"}')
        
        let response = self.requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_ms / 1000
        )
        
        if response.status_code != 200:
            raise Error("Load index failed with status: " + String(response.status_code))
        
        let data = response.json()
        return LoadIndexResponse(
            Bool(data["success"]),
            String(data["message"]),
            Int(data.get("documents", 0)),
            Int(data.get("symbols", 0))
        )
    
    fn find_definition(self, file: String, line: Int, character: Int) raises -> PythonObject:
        """Find the definition of a symbol at the given position"""
        let url = self.base_url + "/v1/definition"
        let payload = self.py.evaluate(
            '{"file": "' + file + '", "line": ' + String(line) + ', "character": ' + String(character) + '}'
        )
        
        let response = self.requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_ms / 1000
        )
        
        if response.status_code != 200:
            raise Error("Find definition failed with status: " + String(response.status_code))
        
        return response.json()
    
    fn find_references(self, file: String, line: Int, character: Int, include_declaration: Bool = True) raises -> PythonObject:
        """Find all references to a symbol at the given position"""
        let url = self.base_url + "/v1/references"
        let include_decl = "true" if include_declaration else "false"
        let payload = self.py.evaluate(
            '{"file": "' + file + '", "line": ' + String(line) + 
            ', "character": ' + String(character) + ', "include_declaration": ' + include_decl + '}'
        )
        
        let response = self.requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_ms / 1000
        )
        
        if response.status_code != 200:
            raise Error("Find references failed with status: " + String(response.status_code))
        
        return response.json()
    
    fn get_hover(self, file: String, line: Int, character: Int) raises -> PythonObject:
        """Get hover information for a symbol at the given position"""
        let url = self.base_url + "/v1/hover"
        let payload = self.py.evaluate(
            '{"file": "' + file + '", "line": ' + String(line) + ', "character": ' + String(character) + '}'
        )
        
        let response = self.requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_ms / 1000
        )
        
        if response.status_code != 200:
            raise Error("Get hover failed with status: " + String(response.status_code))
        
        return response.json()
    
    fn get_symbols(self, file_path: String) raises -> PythonObject:
        """Get all symbols in a file"""
        let url = self.base_url + "/v1/symbols"
        let payload = self.py.evaluate('{"file": "' + file_path + '"}')
        
        let response = self.requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_ms / 1000
        )
        
        if response.status_code != 200:
            raise Error("Get symbols failed with status: " + String(response.status_code))
        
        return response.json()
    
    fn get_document_symbols(self, file_path: String) raises -> PythonObject:
        """Get document outline (hierarchical symbol tree)"""
        let url = self.base_url + "/v1/document-symbols"
        let payload = self.py.evaluate('{"file": "' + file_path + '"}')
        
        let response = self.requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_ms / 1000
        )
        
        if response.status_code != 200:
            raise Error("Get document symbols failed with status: " + String(response.status_code))
        
        return response.json()


struct QdrantClient:
    """Helper client for Qdrant vector database queries"""
    var base_url: String
    var collection_name: String
    var py: Python
    var requests: PythonObject
    
    fn __init__(inout self, base_url: String = "http://localhost:6333", collection_name: String = "ncode") raises:
        self.base_url = base_url
        self.collection_name = collection_name
        self.py = Python()
        self.requests = self.py.import_module("requests")
    
    fn semantic_search(self, query: String, limit: Int = 10) raises -> PythonObject:
        """Perform semantic search across code using embeddings"""
        let url = self.base_url + "/collections/" + self.collection_name + "/points/search"
        
        # In production, you would embed the query text first
        # For now, this is a placeholder
        let payload = self.py.evaluate(
            '{"vector": [], "limit": ' + String(limit) + ', "with_payload": true}'
        )
        
        let response = self.requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Error("Semantic search failed with status: " + String(response.status_code))
        
        return response.json()
    
    fn filter_by_language(self, language: String, limit: Int = 10) raises -> PythonObject:
        """Search for symbols filtered by programming language"""
        let url = self.base_url + "/collections/" + self.collection_name + "/points/scroll"
        
        let payload = self.py.evaluate(
            '{"filter": {"must": [{"key": "language", "match": {"value": "' + language + '"}}]}, ' +
            '"limit": ' + String(limit) + ', "with_payload": true}'
        )
        
        let response = self.requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Error("Filter by language failed with status: " + String(response.status_code))
        
        return response.json()


struct MemgraphClient:
    """Helper client for Memgraph graph database queries"""
    var connection_string: String
    var py: Python
    var neo4j: PythonObject
    var driver: PythonObject
    
    fn __init__(inout self, connection_string: String = "bolt://localhost:7687") raises:
        self.connection_string = connection_string
        self.py = Python()
        self.neo4j = self.py.import_module("neo4j")
        self.driver = self.neo4j.GraphDatabase.driver(connection_string)
    
    fn find_definitions(self, symbol_name: String) raises -> PythonObject:
        """Find all definitions of a symbol in the graph"""
        let query = """
        MATCH (s:Symbol {name: $symbol_name})
        WHERE s.kind = 'definition'
        RETURN s
        """
        
        with self.driver.session() as session:
            let result = session.run(query, symbol_name=symbol_name)
            return result.data()
    
    fn find_references(self, symbol_name: String) raises -> PythonObject:
        """Find all references to a symbol"""
        let query = """
        MATCH (source:Symbol)-[:REFERENCES]->(target:Symbol {name: $symbol_name})
        RETURN source, target
        """
        
        with self.driver.session() as session:
            let result = session.run(query, symbol_name=symbol_name)
            return result.data()
    
    fn get_call_graph(self, function_name: String, depth: Int = 3) raises -> PythonObject:
        """Get the call graph for a function up to a certain depth"""
        let query = """
        MATCH path = (f:Symbol {name: $function_name, kind: 'function'})-[:CALLS*1..""" + String(depth) + """]->(called:Symbol)
        RETURN path
        """
        
        with self.driver.session() as session:
            let result = session.run(query, function_name=function_name)
            return result.data()
    
    fn get_dependencies(self, file_path: String) raises -> PythonObject:
        """Get all dependencies of a file"""
        let query = """
        MATCH (doc:Document {uri: $file_path})-[:IMPORTS]->(dep:Document)
        RETURN dep.uri as dependency
        """
        
        with self.driver.session() as session:
            let result = session.run(query, file_path=file_path)
            return result.data()
    
    fn close(self) raises:
        """Close the database connection"""
        self.driver.close()


fn example() raises:
    """Example usage of the nCode client library"""
    
    # Create client
    var client = NCodeClient("http://localhost:18003", 30000)
    
    # Check health
    var health = client.health()
    print("Status:", health.status)
    print("Version:", health.version)
    print("Index loaded:", health.index_loaded)
    
    # Load SCIP index
    var load_result = client.load_index("index.scip")
    print("Load success:", load_result.success)
    print("Message:", load_result.message)
    print("Documents:", load_result.documents)
    print("Symbols:", load_result.symbols)
    
    # Find definition
    var definition = client.find_definition("src/main.mojo", 10, 5)
    print("Definition:", definition)
    
    # Find references
    var references = client.find_references("src/main.mojo", 10, 5)
    print("References:", references)
    
    # Get symbols
    var symbols = client.get_symbols("src/main.mojo")
    print("Symbols:", symbols)
    
    # Qdrant search example
    var qdrant = QdrantClient("http://localhost:6333", "ncode")
    var search_results = qdrant.semantic_search("authentication function", 5)
    print("Search results:", search_results)
    
    # Memgraph query example
    var memgraph = MemgraphClient("bolt://localhost:7687")
    var defs = memgraph.find_definitions("MyClass")
    print("Definitions:", defs)
    memgraph.close()


fn main() raises:
    """Main entry point for testing"""
    example()
