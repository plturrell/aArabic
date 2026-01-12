import os
import socket
import time
import uuid

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MEMGRAPH_HOST = os.getenv("MEMGRAPH_HOST", "memgraph")
MEMGRAPH_PORT = int(os.getenv("MEMGRAPH_PORT", "7687"))


def _memgraph_reachable() -> bool:
    try:
        sock = socket.create_connection((MEMGRAPH_HOST, MEMGRAPH_PORT), timeout=1)
        sock.close()
        return True
    except OSError:
        return False


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "nucleus-graph",
            "memgraph_host": MEMGRAPH_HOST,
            "memgraph_port": MEMGRAPH_PORT,
            "memgraph_connected": _memgraph_reachable(),
            "timestamp": int(time.time()),
        }
    )


@app.post("/api/query")
def query():
    payload = request.get_json(silent=True) or {}
    cypher = (payload.get("query") or "").strip()
    if not cypher:
        return jsonify({"success": False, "error": "query is required"}), 400

    results = []
    if "MATCH" in cypher.upper():
        results = [
            {
                "n": {
                    "id": "1",
                    "labels": ["Document"],
                    "properties": {"title": "Sample Document"},
                }
            },
            {
                "n": {
                    "id": "2",
                    "labels": ["Concept"],
                    "properties": {"name": "AI Research"},
                }
            },
        ]

    return jsonify(
        {
            "success": True,
            "data": {
                "query": cypher,
                "results": results,
                "execution_time": "0.001s",
                "nodes_created": 0,
                "relationships_created": 0,
                "properties_set": 0,
                "message": "Query executed (stubbed response)",
            },
        }
    )


@app.get("/api/graph")
def graph():
    limit = int(request.args.get("limit", 50))
    nodes = [
        {
            "id": "1",
            "label": "Document",
            "properties": {"title": "AI Research Paper", "type": "Document"},
        },
        {
            "id": "2",
            "label": "Concept",
            "properties": {"name": "Machine Learning", "type": "Concept"},
        },
        {
            "id": "3",
            "label": "Concept",
            "properties": {"name": "Neural Networks", "type": "Concept"},
        },
    ]

    edges = [
        {"id": "e1", "source": "1", "target": "2", "label": "MENTIONS"},
        {"id": "e2", "source": "1", "target": "3", "label": "MENTIONS"},
    ]

    return jsonify(
        {
            "nodes": nodes[:limit],
            "edges": edges,
        }
    )


@app.post("/api/nodes")
def create_node():
    payload = request.get_json(silent=True) or {}
    node_id = payload.get("id") or str(uuid.uuid4())
    return jsonify(
        {
            "id": node_id,
            "label": payload.get("label", "Node"),
            "properties": payload.get("properties", {}),
            "type": payload.get("type"),
        }
    )


@app.post("/api/edges")
def create_edge():
    payload = request.get_json(silent=True) or {}
    edge_id = payload.get("id") or str(uuid.uuid4())
    return jsonify(
        {
            "id": edge_id,
            "source": payload.get("source"),
            "target": payload.get("target"),
            "label": payload.get("label", "RELATES_TO"),
            "properties": payload.get("properties", {}),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
