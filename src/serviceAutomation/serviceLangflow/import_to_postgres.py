#!/usr/bin/env python3
"""
Import flows directly into Langflow's Postgres database
"""

import psycopg2
import json
import uuid
from pathlib import Path
from datetime import datetime

def import_flow_to_postgres(flow_json_path, user_id="a0382395-5bf8-4f56-8fb8-5079358fad59"):
    """Import a flow JSON into Langflow's Postgres database"""
    
    # Read the flow JSON
    with open(flow_json_path, 'r') as f:
        flow_data = json.load(f)
    
    # Extract flow metadata
    flow_name = flow_data.get('name', Path(flow_json_path).stem)
    flow_description = flow_data.get('description', '')
    
    # Generate IDs (32 char hex without dashes)
    flow_id = uuid.uuid4().hex
    timestamp = datetime.now().isoformat()
    
    # Connect to Postgres
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="langflow",
            user="langflow",
            password="langflow123"
        )
        cur = conn.cursor()
        
        try:
                # Insert into flow table
                cur.execute("""
                    INSERT INTO flow (
                        id, name, description, data, is_component, user_id, 
                        updated_at, webhook, icon_bg_color, gradient, endpoint_name,
                        mcp_enabled, access_type, tags, locked
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    flow_id,
                    flow_name,
                    flow_description,
                    json.dumps(flow_data),
                    False,  # is_component
                    user_id,
                    timestamp,
                    False,  # webhook
                    None,  # icon_bg_color
                    None,  # gradient
                    None,  # endpoint_name
                    False,  # mcp_enabled
                    'PRIVATE',  # access_type
                    '[]',  # tags as JSON array
                    False  # locked
                ))
                
                conn.commit()
                print(f"✅ Successfully imported: {flow_name}")
                print(f"   ID: {flow_id}")
                return True
                
        finally:
            cur.close()
            conn.close()
                
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    # Paths
    flows_dir = Path("/Users/user/Documents/arabic_folder/src/serviceAutomation/serviceLangflow/flows")
    
    flows = [
        flows_dir / "arabic_training_pipeline.json",
        flows_dir / "intelligent_training_orchestrator.json"
    ]
    
    print("🚀 Importing Flows to Langflow Postgres Database")
    print("=" * 60)
    print()
    
    success_count = 0
    for flow_path in flows:
        if flow_path.exists():
            if import_flow_to_postgres(flow_path):
                success_count += 1
        else:
            print(f"⚠️  File not found: {flow_path}")
    
    print()
    print("=" * 60)
    print(f"📊 Imported {success_count}/{len(flows)} flows successfully")
    
    if success_count == len(flows):
        print("\n🎉 All flows imported to Postgres!")
        print("🌐 Open http://localhost:7860 to see them in the UI")
        return 0
    else:
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
