"""
Unified Rust CLI Adapter for All Services
Provides async Python interface to all 17 Rust CLI clients

This adapter bridges Python services to Rust clients, enabling:
- Vendor → APISIX → Rust Client API → Services → Vendor UI
"""

import subprocess
import json
import asyncio
import shutil
from typing import Optional, Dict, Any, List
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class RustCLIError(Exception):
    """Exception raised when Rust CLI execution fails"""
    pass


class RustCLIAdapter:
    """Base adapter for executing Rust CLI commands from Python services"""
    
    def __init__(self, cli_name: str, default_url: Optional[str] = None):
        self.cli_name = cli_name
        self.default_url = default_url
        self._verify_cli()
    
    def _verify_cli(self):
        """Verify CLI is available"""
        if not shutil.which(self.cli_name):
            logger.warning(f"{self.cli_name} not found in PATH. Install from serviceAutomation/{self.cli_name.replace('-cli', '-api-client')}")
    
    async def execute(
        self, 
        operation: str, 
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute CLI command asynchronously"""
        
        cmd = [self.cli_name]
        
        # Add URL if provided or use default
        if url or self.default_url:
            cmd.extend(["--url", url or self.default_url])
        
        # Add operation
        cmd.append(operation)
        
        # Add additional arguments
        if args:
            cmd.extend(args)
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            )
            
            if result.returncode == 0:
                # Try to parse JSON output
                try:
                    data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    data = {"raw_output": result.stdout}
                
                return {
                    "success": True,
                    "data": data,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                logger.error(f"{self.cli_name} failed: {result.stderr}")
                raise RustCLIError(f"CLI error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RustCLIError(f"CLI timeout after {timeout}s")
        except Exception as e:
            logger.error(f"{self.cli_name} execution failed: {e}")
            raise RustCLIError(f"Execution failed: {str(e)}")


# =============================================================================
# SPECIALIZED ADAPTERS FOR EACH RUST CLIENT
# =============================================================================

class GiteaCLI(RustCLIAdapter):
    """Gitea Git hosting operations"""
    def __init__(self, url: str = "http://localhost:3000"):
        super().__init__("gitea-cli", url)
    
    async def health(self) -> Dict[str, Any]:
        return await self.execute("health")
    
    async def list_repos(self, owner: Optional[str] = None) -> List[Dict]:
        args = [owner] if owner else []
        result = await self.execute("list-repos", args)
        return result["data"]
    
    async def create_repo(self, owner: str, name: str) -> Dict:
        result = await self.execute("create-repo", [owner, name])
        return result["data"]
    
    async def get_repo(self, owner: str, repo: str) -> Dict:
        result = await self.execute("get-repo", [owner, repo])
        return result["data"]


class GitCLI(RustCLIAdapter):
    """Git version control operations"""
    def __init__(self, repo_path: str = "."):
        super().__init__("git-cli")
        self.repo_path = repo_path
    
    async def status(self) -> Dict[str, Any]:
        result = await self.execute("status", [], url=None)
        return result["data"]
    
    async def add(self, files: List[str]) -> Dict:
        result = await self.execute("add", files, url=None)
        return result["data"]
    
    async def commit(self, message: str) -> Dict:
        result = await self.execute("commit", [message], url=None)
        return result["data"]
    
    async def push(self) -> Dict:
        result = await self.execute("push", [], url=None)
        return result["data"]


class PostgresCLI(RustCLIAdapter):
    """PostgreSQL database operations"""
    def __init__(self, host: str = "localhost", port: int = 5432, database: str = "postgres"):
        super().__init__("postgres-cli")
        self.host = host
        self.port = port
        self.database = database
    
    async def query(self, sql: str) -> List[Dict]:
        result = await self.execute("query", [sql], url=None)
        return result["data"]
    
    async def execute(self, sql: str) -> Dict:
        result = await super().execute("execute", [sql], url=None)
        return result["data"]
    
    async def list_tables(self) -> List[str]:
        result = await super().execute("list-tables", [], url=None)
        return result["data"]


class KafkaCLI(RustCLIAdapter):
    """Kafka messaging operations"""
    def __init__(self, broker: str = "localhost:9092"):
        super().__init__("kafka-cli")
        self.broker = broker
    
    async def produce(self, topic: str, message: str) -> Dict:
        result = await self.execute("produce", [topic, message], url=None)
        return result["data"]
    
    async def list_topics(self) -> List[str]:
        result = await self.execute("list-topics", [], url=None)
        return result["data"]


class ApisixCLI(RustCLIAdapter):
    """APISIX API Gateway operations"""
    def __init__(self, url: str = "http://localhost:9180"):
        super().__init__("apisix-cli", url)
    
    async def health(self) -> Dict[str, Any]:
        return await self.execute("health")
    
    async def list_routes(self) -> List[Dict]:
        result = await self.execute("list-routes")
        return result["data"]
    
    async def list_services(self) -> List[Dict]:
        result = await self.execute("list-services")
        return result["data"]


class KeycloakCLI(RustCLIAdapter):
    """Keycloak authentication operations with OAuth2/OIDC support"""
    def __init__(self, url: str = "http://localhost:8080"):
        super().__init__("keycloak-cli", url)
    
    async def list_realms(self) -> List[str]:
        result = await self.execute("list-realms")
        return result["data"]
    
    async def list_users(self, realm: str = "master") -> List[Dict]:
        result = await self.execute("list-users", [realm])
        return result["data"]
    
    # OAuth2/OIDC Operations (Pure Rust)
    async def get_token(
        self, 
        realm: str, 
        username: str, 
        password: str,
        client_id: str = "admin-cli",
        client_secret: Optional[str] = None,
        scope: str = "openid"
    ) -> Dict[str, Any]:
        """Get OAuth2 access token using password grant"""
        args = [realm, username, password, "--client-id", client_id, "--scope", scope]
        if client_secret:
            args.extend(["--client-secret", client_secret])
        result = await self.execute("get-token", args)
        return result["data"]
    
    async def refresh_token(
        self,
        realm: str,
        refresh_token: str,
        client_id: str = "admin-cli",
        client_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refresh OAuth2 access token"""
        args = [realm, refresh_token, "--client-id", client_id]
        if client_secret:
            args.extend(["--client-secret", client_secret])
        result = await self.execute("refresh-token", args)
        return result["data"]
    
    async def validate_token(
        self,
        realm: str,
        token: str,
        client_id: str = "admin-cli",
        client_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate/introspect OAuth2 token"""
        args = [realm, token, "--client-id", client_id]
        if client_secret:
            args.extend(["--client-secret", client_secret])
        result = await self.execute("validate-token", args)
        return result["data"]
    
    async def get_user_info(self, realm: str, access_token: str) -> Dict[str, Any]:
        """Get OIDC user info from access token"""
        result = await self.execute("get-user-info", [realm, access_token])
        return result["data"]
    
    async def logout_via_token(
        self,
        realm: str,
        refresh_token: str,
        client_id: str = "admin-cli",
        client_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """Logout via OIDC refresh token"""
        args = [realm, refresh_token, "--client-id", client_id]
        if client_secret:
            args.extend(["--client-secret", client_secret])
        result = await self.execute("logout-via-token", args)
        return result["data"]


class FilesystemCLI(RustCLIAdapter):
    """Filesystem operations"""
    def __init__(self, base_path: str = "."):
        super().__init__("fs-cli")
        self.base_path = base_path
    
    async def read(self, path: str) -> str:
        result = await self.execute("read", [path], url=None)
        return result["data"]["content"]
    
    async def write(self, path: str, content: str) -> Dict:
        result = await self.execute("write", [path, content], url=None)
        return result["data"]
    
    async def list(self, path: str = ".") -> List[str]:
        result = await self.execute("list", [path], url=None)
        return result["data"]


class MemoryCLI(RustCLIAdapter):
    """In-memory cache operations"""
    def __init__(self):
        super().__init__("memory-cli")
    
    async def get(self, key: str) -> Optional[str]:
        result = await self.execute("get", [key], url=None)
        return result["data"].get("value")
    
    async def set(self, key: str, value: str) -> Dict:
        result = await self.execute("set", [key, value], url=None)
        return result["data"]
    
    async def keys(self) -> List[str]:
        result = await self.execute("keys", [], url=None)
        return result["data"]


class ShimmyCLI(RustCLIAdapter):
    """Shimmy-AI local inference operations"""
    def __init__(self, url: str = "http://localhost:8000"):
        super().__init__("shimmy-cli", url)
    
    async def chat(self, prompt: str, model: Optional[str] = None) -> Dict:
        args = [prompt]
        if model:
            args.extend(["--model", model])
        result = await self.execute("chat", args)
        return result["data"]
    
    async def list_models(self) -> List[str]:
        result = await self.execute("list-models")
        return result["data"]


class MarquezCLI(RustCLIAdapter):
    """Marquez data lineage operations"""
    def __init__(self, url: str = "http://localhost:5000"):
        super().__init__("marquez-cli", url)
    
    async def list_namespaces(self) -> List[str]:
        result = await self.execute("list-namespaces")
        return result["data"]
    
    async def list_datasets(self, namespace: str) -> List[Dict]:
        result = await self.execute("list-datasets", [namespace])
        return result["data"]


class N8nCLI(RustCLIAdapter):
    """n8n workflow automation operations"""
    def __init__(self, url: str = "http://localhost:5678"):
        super().__init__("n8n-cli", url)
    
    async def health(self) -> Dict[str, Any]:
        return await self.execute("health")
    
    async def list_workflows(self) -> List[Dict]:
        result = await self.execute("list-workflows")
        return result["data"]
    
    async def execute_workflow(self, workflow_id: str) -> Dict:
        result = await self.execute("execute-workflow", [workflow_id])
        return result["data"]


class OpenCanvasCLI(RustCLIAdapter):
    """OpenCanvas operations"""
    def __init__(self, url: str = "http://localhost:3001"):
        super().__init__("opencanvas-cli", url)
    
    async def health(self) -> Dict[str, Any]:
        return await self.execute("health")
    
    async def create_canvas(self, name: str) -> Dict:
        result = await self.execute("create-canvas", [name])
        return result["data"]


class HyperbookCLI(RustCLIAdapter):
    """Hyperbook operations"""
    def __init__(self, url: str = "http://localhost:3002"):
        super().__init__("hyperbook-cli", url)
    
    async def health(self) -> Dict[str, Any]:
        return await self.execute("health")
    
    async def list_books(self) -> List[Dict]:
        result = await self.execute("list-books")
        return result["data"]


class NucleusGraphCLI(RustCLIAdapter):
    """Nucleus Graph visualization operations"""
    def __init__(self, url: str = "http://localhost:3005"):
        super().__init__("nucleusgraph-cli", url)
    
    async def health(self) -> Dict[str, Any]:
        return await self.execute("health")
    
    async def get_graph(self) -> Dict:
        result = await self.execute("get-graph")
        return result["data"]


class QdrantCLI(RustCLIAdapter):
    """Qdrant vector database operations"""
    def __init__(self, url: str = "http://localhost:6333"):
        super().__init__("qdrant-cli", url)
    
    async def health(self) -> Dict[str, Any]:
        return await self.execute("health")
    
    async def list_collections(self) -> List[str]:
        result = await self.execute("list-collections")
        return result["data"]
    
    async def search(self, collection: str, vector: List[float], limit: int = 10) -> List[Dict]:
        result = await self.execute("search", [collection, json.dumps(vector), str(limit)])
        return result["data"]


class MemgraphCLI(RustCLIAdapter):
    """Memgraph graph database operations"""
    def __init__(self, url: str = "http://localhost:7687"):
        super().__init__("memgraph-cli", url)
    
    async def health(self) -> Dict[str, Any]:
        return await self.execute("health")
    
    async def query(self, cypher: str) -> List[Dict]:
        result = await self.execute("query", [cypher])
        return result["data"]


class DragonflyCLI(RustCLIAdapter):
    """Dragonfly cache operations"""
    def __init__(self, url: str = "http://localhost:6379"):
        super().__init__("dragonflydb-cli", url)
    
    async def get(self, key: str) -> Optional[str]:
        result = await self.execute("get", [key])
        return result["data"].get("value")
    
    async def set(self, key: str, value: str) -> Dict:
        result = await self.execute("set", [key, value])
        return result["data"]


class GleanCLI(RustCLIAdapter):
    """Glean search operations"""
    def __init__(self, url: str = "http://localhost:8080"):
        super().__init__("glean-cli", url)
    
    async def search(self, query: str) -> List[Dict]:
        result = await self.execute("search", [query])
        return result["data"]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

@lru_cache(maxsize=32)
def get_rust_client(client_name: str, **kwargs) -> RustCLIAdapter:
    """
    Factory function to get Rust CLI adapter by name
    
    Usage:
        gitea = get_rust_client("gitea", url="http://localhost:3000")
        repos = await gitea.list_repos()
    """
    clients = {
        "gitea": GiteaCLI,
        "git": GitCLI,
        "postgres": PostgresCLI,
        "kafka": KafkaCLI,
        "apisix": ApisixCLI,
        "keycloak": KeycloakCLI,
        "filesystem": FilesystemCLI,
        "memory": MemoryCLI,
        "shimmy": ShimmyCLI,
        "marquez": MarquezCLI,
        "n8n": N8nCLI,
        "opencanvas": OpenCanvasCLI,
        "hyperbook": HyperbookCLI,
        "nucleusgraph": NucleusGraphCLI,
        "qdrant": QdrantCLI,
        "memgraph": MemgraphCLI,
        "dragonfly": DragonflyCLI,
        "glean": GleanCLI,
    }
    
    if client_name not in clients:
        raise ValueError(f"Unknown client: {client_name}. Available: {list(clients.keys())}")
    
    return clients[client_name](**kwargs)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def quick_cli(cli_name: str, operation: str, *args, url: Optional[str] = None) -> Dict:
    """
    Quick CLI execution for one-off operations
    
    Usage:
        result = await quick_cli("gitea-cli", "health", url="http://localhost:3000")
        repos = await quick_cli("gitea-cli", "list-repos", "owner")
    """
    adapter = RustCLIAdapter(cli_name, url)
    return await adapter.execute(operation, list(args), url)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    async def main():
        # Example 1: Using specific client
        gitea = GiteaCLI("http://localhost:3000")
        repos = await gitea.list_repos("myuser")
        print(f"Found {len(repos)} repositories")
        
        # Example 2: Using factory
        kafka = get_rust_client("kafka", broker="localhost:9092")
        topics = await kafka.list_topics()
        print(f"Kafka topics: {topics}")
        
        # Example 3: Quick CLI
        status = await quick_cli("git-cli", "status")
        print(f"Git status: {status}")
    
    asyncio.run(main())
