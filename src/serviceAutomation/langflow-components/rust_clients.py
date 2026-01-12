"""
Langflow Custom Components for All 17 Rust API Clients
Enables visual drag-and-drop workflow design using production Rust CLIs

Complete coverage:
1. Langflow    2. Gitea       3. Git         4. Filesystem  5. Memory
6. APISIX      7. Keycloak    8. Glean       9. MarkItDown  10. Marquez
11. PostgreSQL 12. Hyperbook  13. n8n        14. OpenCanvas 15. Kafka
16. Shimmy-AI  17. Lean4

Installation:
1. Copy this file to: ~/.langflow/components/rust_clients.py
2. Restart Langflow
3. Components appear in Custom category
"""

from langflow.custom import Component
from langflow.io import MessageTextInput, Output, DropdownInput, IntInput, BoolInput
from langflow.schema import Data
import subprocess
import json
from typing import Any

class RustClientComponent(Component):
    """Base component for all Rust CLI clients"""
    
    def execute_cli(self, cli_name: str, args: list[str]) -> dict[str, Any]:
        """Execute Rust CLI and return parsed output"""
        try:
            result = subprocess.run(
                [cli_name] + args,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# ============================================================================
# 1. LANGFLOW COMPONENT
# ============================================================================

class LangflowComponent(RustClientComponent):
    display_name = "Langflow Operations"
    description = "Execute Langflow workflow operations"
    icon = "🔄"
    
    inputs = [
        DropdownInput(
            name="operation",
            display_name="Operation",
            options=["list-flows", "get-flow", "run-flow", "health"],
            value="health"
        ),
        MessageTextInput(name="flow_id", display_name="Flow ID", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:7860")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.flow_id and self.operation != "health":
            args.append(self.flow_id)
        
        result = self.execute_cli("langflow-cli", args)
        return Data(data=result)

# ============================================================================
# 2. GITEA COMPONENT
# ============================================================================

class GiteaComponent(RustClientComponent):
    display_name = "Gitea Git Hosting"
    description = "Manage Git repositories via Gitea"
    icon = "🦊"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["list-repos", "create-repo", "get-repo", "list-issues", "list-prs", "health"],
            value="list-repos"
        ),
        MessageTextInput(name="owner", display_name="Owner", required=False),
        MessageTextInput(name="repo", display_name="Repository", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:3000")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.owner:
            args.append(self.owner)
        if self.repo:
            args.append(self.repo)
        
        result = self.execute_cli("gitea-cli", args)
        return Data(data=result)

# ============================================================================
# 3. GIT COMPONENT
# ============================================================================

class GitComponent(RustClientComponent):
    display_name = "Git Operations"
    description = "Execute Git version control commands"
    icon = "🌿"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["init", "clone", "add", "commit", "push", "pull", "branch", "status"],
            value="status"
        ),
        MessageTextInput(name="path", display_name="Repository Path", value="."),
        MessageTextInput(name="message", display_name="Commit Message", required=False),
        MessageTextInput(name="url", display_name="Clone URL", required=False)
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--path", self.path, self.operation]
        if self.operation == "commit" and self.message:
            args.append(self.message)
        elif self.operation == "clone" and self.url:
            args.append(self.url)
        
        result = self.execute_cli("git-cli", args)
        return Data(data=result)

# ============================================================================
# 4. FILESYSTEM COMPONENT
# ============================================================================

class FilesystemComponent(RustClientComponent):
    display_name = "Filesystem Operations"
    description = "Read, write, and manage files"
    icon = "📁"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["read", "write", "list", "delete", "copy", "exists", "mkdir"],
            value="list"
        ),
        MessageTextInput(name="path", display_name="Path", value="."),
        MessageTextInput(name="content", display_name="Content", required=False),
        MessageTextInput(name="dest", display_name="Destination", required=False)
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--base", ".", self.operation, self.path]
        if self.operation == "write" and self.content:
            args.append(self.content)
        elif self.operation == "copy" and self.dest:
            args.append(self.dest)
        
        result = self.execute_cli("fs-cli", args)
        return Data(data=result)

# ============================================================================
# 5. MEMORY COMPONENT
# ============================================================================

class MemoryComponent(RustClientComponent):
    display_name = "Memory Cache"
    description = "In-memory key-value storage with TTL"
    icon = "🧠"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["get", "set", "set-ttl", "delete", "keys", "clear", "stats"],
            value="keys"
        ),
        MessageTextInput(name="key", display_name="Key", required=False),
        MessageTextInput(name="value", display_name="Value", required=False),
        MessageTextInput(name="ttl", display_name="TTL (seconds)", required=False)
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = [self.operation]
        if self.key:
            args.append(self.key)
        if self.value:
            args.append(self.value)
        if self.ttl and self.operation == "set-ttl":
            args.append(self.ttl)
        
        result = self.execute_cli("memory-cli", args)
        return Data(data=result)

# ============================================================================
# 6. APISIX COMPONENT
# ============================================================================

class ApisixComponent(RustClientComponent):
    display_name = "APISIX API Gateway"
    description = "Manage API routes and services"
    icon = "🚪"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["health", "list-routes", "list-services", "list-upstreams", 
                    "create-route", "delete-route"],
            value="health"
        ),
        MessageTextInput(name="route_id", display_name="Route ID", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:9180")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.route_id and self.operation in ["delete-route", "get-route"]:
            args.append(self.route_id)
        
        result = self.execute_cli("apisix-cli", args)
        return Data(data=result)

# ============================================================================
# 7. KEYCLOAK COMPONENT
# ============================================================================

class KeycloakComponent(RustClientComponent):
    display_name = "Keycloak Authentication"
    description = "Manage users, roles, and realms"
    icon = "🔐"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["list-realms", "list-users", "list-roles", "list-groups", 
                    "create-user", "delete-user"],
            value="list-realms"
        ),
        MessageTextInput(name="realm", display_name="Realm", value="master"),
        MessageTextInput(name="username", display_name="Username", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:8080")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.operation != "list-realms":
            args.append(self.realm)
        if self.username and self.operation in ["create-user", "delete-user"]:
            args.append(self.username)
        
        result = self.execute_cli("keycloak-cli", args)
        return Data(data=result)

# ============================================================================
# 8. GLEAN COMPONENT
# ============================================================================

class GleanComponent(RustClientComponent):
    display_name = "Glean Code Intelligence"
    description = "Search and analyze code with Glean"
    icon = "🔍"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["search", "get-definition", "find-references", "analyze", "health"],
            value="health"
        ),
        MessageTextInput(name="query", display_name="Search Query", required=False),
        MessageTextInput(name="file", display_name="File Path", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:8080")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.query and self.operation == "search":
            args.append(self.query)
        if self.file and self.operation in ["get-definition", "analyze"]:
            args.append(self.file)
        
        result = self.execute_cli("glean-cli", args)
        return Data(data=result)

# ============================================================================
# 9. MARKITDOWN COMPONENT
# ============================================================================

class MarkItDownComponent(RustClientComponent):
    display_name = "MarkItDown Converter"
    description = "Convert documents to Markdown"
    icon = "📄"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["convert", "convert-file", "list-formats", "health"],
            value="health"
        ),
        MessageTextInput(name="input_file", display_name="Input File", required=False),
        MessageTextInput(name="output_file", display_name="Output File", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:8000")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.input_file and self.operation in ["convert", "convert-file"]:
            args.append(self.input_file)
        if self.output_file and self.operation == "convert-file":
            args.append(self.output_file)
        
        result = self.execute_cli("markitdown-cli", args)
        return Data(data=result)

# ============================================================================
# 10. MARQUEZ COMPONENT
# ============================================================================

class MarquezComponent(RustClientComponent):
    display_name = "Marquez Data Lineage"
    description = "Track data pipelines and lineage"
    icon = "📊"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["list-namespaces", "list-datasets", "list-jobs", "get-lineage", 
                    "create-namespace", "health"],
            value="health"
        ),
        MessageTextInput(name="namespace", display_name="Namespace", required=False),
        MessageTextInput(name="dataset", display_name="Dataset", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:5000")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.namespace:
            args.append(self.namespace)
        if self.dataset and self.operation in ["get-lineage", "get-dataset"]:
            args.append(self.dataset)
        
        result = self.execute_cli("marquez-cli", args)
        return Data(data=result)

# ============================================================================
# 11. POSTGRESQL COMPONENT
# ============================================================================

class PostgreSQLComponent(RustClientComponent):
    display_name = "PostgreSQL Database"
    description = "Execute SQL queries and manage databases"
    icon = "🐘"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["query", "execute", "list-databases", "list-tables", "health"],
            value="health"
        ),
        MessageTextInput(name="sql", display_name="SQL Query", required=False),
        MessageTextInput(name="database", display_name="Database", value="postgres"),
        MessageTextInput(name="host", display_name="Host", value="localhost"),
        IntInput(name="port", display_name="Port", value=5432)
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = [
            "--host", self.host,
            "--port", str(self.port),
            "--database", self.database,
            self.operation
        ]
        if self.sql and self.operation in ["query", "execute"]:
            args.append(self.sql)
        
        result = self.execute_cli("postgres-cli", args)
        return Data(data=result)

# ============================================================================
# 12. HYPERBOOK COMPONENT
# ============================================================================

class HyperbookComponent(RustClientComponent):
    display_name = "Hyperbook Documentation"
    description = "Manage interactive documentation"
    icon = "📚"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["list-books", "get-book", "create-page", "update-page", "health"],
            value="health"
        ),
        MessageTextInput(name="book_id", display_name="Book ID", required=False),
        MessageTextInput(name="page_id", display_name="Page ID", required=False),
        MessageTextInput(name="content", display_name="Content", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:3000")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.book_id:
            args.append(self.book_id)
        if self.page_id and self.operation in ["update-page", "get-page"]:
            args.append(self.page_id)
        if self.content and self.operation in ["create-page", "update-page"]:
            args.append(self.content)
        
        result = self.execute_cli("hyperbook-cli", args)
        return Data(data=result)

# ============================================================================
# 13. N8N COMPONENT
# ============================================================================

class N8nComponent(RustClientComponent):
    display_name = "n8n Workflow Automation"
    description = "Manage n8n workflows"
    icon = "⚡"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["list-workflows", "get-workflow", "activate", "deactivate", 
                    "execute", "health"],
            value="health"
        ),
        MessageTextInput(name="workflow_id", display_name="Workflow ID", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:5678")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.workflow_id and self.operation != "health":
            args.append(self.workflow_id)
        
        result = self.execute_cli("n8n-cli", args)
        return Data(data=result)

# ============================================================================
# 14. OPENCANVAS COMPONENT
# ============================================================================

class OpenCanvasComponent(RustClientComponent):
    display_name = "OpenCanvas Collaboration"
    description = "Real-time collaborative editing"
    icon = "🎨"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["list-canvases", "create-canvas", "get-canvas", "update-canvas", 
                    "delete-canvas", "health"],
            value="health"
        ),
        MessageTextInput(name="canvas_id", display_name="Canvas ID", required=False),
        MessageTextInput(name="title", display_name="Title", required=False),
        MessageTextInput(name="content", display_name="Content", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:3000")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.canvas_id and self.operation in ["get-canvas", "update-canvas", "delete-canvas"]:
            args.append(self.canvas_id)
        if self.title and self.operation == "create-canvas":
            args.append(self.title)
        if self.content and self.operation in ["create-canvas", "update-canvas"]:
            args.append(self.content)
        
        result = self.execute_cli("opencanvas-cli", args)
        return Data(data=result)

# ============================================================================
# 15. KAFKA COMPONENT
# ============================================================================

class KafkaComponent(RustClientComponent):
    display_name = "Kafka Messaging"
    description = "Publish and consume Kafka messages"
    icon = "📨"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["list-topics", "create-topic", "produce", "consume", "health"],
            value="health"
        ),
        MessageTextInput(name="topic", display_name="Topic", required=False),
        MessageTextInput(name="message", display_name="Message", required=False),
        MessageTextInput(name="broker", display_name="Broker", value="localhost:9092")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--broker", self.broker, self.operation]
        if self.topic and self.operation != "list-topics":
            args.append(self.topic)
        if self.message and self.operation == "produce":
            args.append(self.message)
        
        result = self.execute_cli("kafka-cli", args)
        return Data(data=result)

# ============================================================================
# 16. SHIMMY-AI COMPONENT
# ============================================================================

class ShimmyAIComponent(RustClientComponent):
    display_name = "Shimmy-AI Local Inference"
    description = "OpenAI-compatible local AI inference"
    icon = "🤖"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["chat", "list-models", "health"],
            value="health"
        ),
        MessageTextInput(name="prompt", display_name="Prompt", required=False),
        MessageTextInput(name="model", display_name="Model", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:8000")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.prompt and self.operation == "chat":
            args.append(self.prompt)
        if self.model and self.operation == "chat":
            args.extend(["--model", self.model])
        
        result = self.execute_cli("shimmy-cli", args)
        return Data(data=result)

# ============================================================================
# 17. LEAN4 COMPONENT
# ============================================================================

class Lean4Component(RustClientComponent):
    display_name = "Lean4 Theorem Prover"
    description = "Formal verification with Lean4"
    icon = "🔬"
    
    inputs = [
        DropdownInput(
            name="operation",
            options=["check", "prove", "info", "health"],
            value="health"
        ),
        MessageTextInput(name="file", display_name="Lean File", required=False),
        MessageTextInput(name="theorem", display_name="Theorem", required=False),
        MessageTextInput(name="url", display_name="URL", value="http://localhost:8080")
    ]
    
    outputs = [Output(display_name="Result", name="result", method="execute")]
    
    def execute(self) -> Data:
        args = ["--url", self.url, self.operation]
        if self.file and self.operation in ["check", "prove"]:
            args.append(self.file)
        if self.theorem and self.operation == "prove":
            args.append(self.theorem)
        
        result = self.execute_cli("lean4-cli", args)
        return Data(data=result)

# ============================================================================
# ALL 17 COMPONENTS EXPORTED
# ============================================================================

__all__ = [
    # Infrastructure (1-7)
    "LangflowComponent",
    "GiteaComponent",
    "GitComponent",
    "FilesystemComponent",
    "MemoryComponent",
    "ApisixComponent",
    "KeycloakComponent",
    # AI & Intelligence (8-9, 16-17)
    "GleanComponent",
    "MarkItDownComponent",
    "ShimmyAIComponent",
    "Lean4Component",
    # Data & Orchestration (10-11, 13, 15)
    "MarquezComponent",
    "PostgreSQLComponent",
    "N8nComponent",
    "KafkaComponent",
    # Collaboration & Content (12, 14)
    "HyperbookComponent",
    "OpenCanvasComponent",
]
