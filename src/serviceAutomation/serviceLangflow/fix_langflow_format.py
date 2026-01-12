#!/usr/bin/env python3
"""
Convert custom flow JSON to Langflow-compatible format
"""
import json
import psycopg2
from uuid import uuid4

# Database connection
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="langflow",
    user="postgres",
    password="postgres"
)

def create_simple_langflow_node(node_id, display_name, position, description=""):
    """Create a simplified Langflow-compatible node"""
    return {
        "id": node_id,
        "type": "genericNode",
        "position": position,
        "data": {
            "id": node_id,
            "type": "CustomComponent",
            "display_name": display_name,
            "description": description,
            "node": {
                "display_name": display_name,
                "description": description,
                "base_classes": ["Component"],
                "outputs": [{
                    "name": "output",
                    "display_name": "Output",
                    "types": ["Data"]
                }]
            }
        },
        "width": 320,
        "height": 234
    }

def create_simple_langflow_edge(source, target, source_handle="output", target_handle="input"):
    """Create a simplified Langflow edge"""
    return {
        "id": f"reactflow__edge-{source}{source_handle}-{target}{target_handle}",
        "source": source,
        "target": target,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
        "animated": False,
        "data": {}
    }

def convert_to_langflow_format(custom_flow):
    """Convert custom format to Langflow format"""
    langflow_nodes = []
    langflow_edges = []
    
    # Convert nodes
    for node in custom_flow.get("nodes", []):
        node_id = node["id"]
        name = node.get("name", node_id)
        position = node.get("position", {"x": 0, "y": 0})
        description = node.get("data", {}).get("description", "")
        
        langflow_node = create_simple_langflow_node(
            node_id,
            name,
            position,
            description
        )
        langflow_nodes.append(langflow_node)
    
    # Convert edges
    for edge in custom_flow.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        source_handle = edge.get("sourceHandle", "output")
        target_handle = edge.get("targetHandle", "input")
        
        langflow_edge = create_simple_langflow_edge(
            source, target, source_handle, target_handle
        )
        langflow_edges.append(langflow_edge)
    
    # Create final Langflow format
    langflow_format = {
        "nodes": langflow_nodes,
        "edges": langflow_edges,
        "viewport": {"x": 0, "y": 0, "zoom": 0.8}
    }
    
    return langflow_format

def update_flow_in_db(flow_id, langflow_data):
    """Update flow data in database"""
    cursor = conn.cursor()
    
    # Convert to JSON string
    data_json = json.dumps(langflow_data)
    
    # Update query
    update_query = """
    UPDATE flow 
    SET data = %s::json 
    WHERE id = %s;
    """
    
    cursor.execute(update_query, (data_json, flow_id))
    conn.commit()
    cursor.close()
    
    print(f"✅ Updated flow {flow_id}")

def main():
    """Main conversion function"""
    # Flow IDs from database
    flows = [
        {
            "id": "e34afe6c-ca21-4b63-8cd1-af74df72ac79",
            "file": "src/serviceAutomation/serviceLangflow/flows/arabic_training_pipeline.json",
            "name": "Arabic Translation Training & Fine-tuning Pipeline"
        },
        {
            "id": "924af23a-9f8c-438e-9773-b8016d9fa538",
            "file": "src/serviceAutomation/serviceLangflow/flows/intelligent_training_orchestrator.json",
            "name": "Intelligent Multi-Model Training Orchestrator"
        }
    ]
    
    for flow in flows:
        print(f"\n🔄 Processing: {flow['name']}")
        
        try:
            # Read custom format
            with open(f"/Users/user/Documents/arabic_folder/{flow['file']}", 'r') as f:
                custom_flow = json.load(f)
            
            # Convert to Langflow format
            langflow_format = convert_to_langflow_format(custom_flow)
            
            # Update database
            update_flow_in_db(flow['id'], langflow_format)
            
            print(f"✅ Successfully converted {flow['name']}")
            
        except Exception as e:
            print(f"❌ Error processing {flow['name']}: {e}")
    
    conn.close()
    print("\n🎉 Conversion complete!")

if __name__ == "__main__":
    main()
