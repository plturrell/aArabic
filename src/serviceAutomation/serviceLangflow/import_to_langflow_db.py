#!/usr/bin/env python3
"""
Import flows directly into Langflow's SQLite database
"""

import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime

def import_flow_to_db(db_path, flow_json_path, user_id="4598668df0314639a9b4267bc0af04e7"):
    """Import a flow JSON into Langflow's database"""
    
    # Read the flow JSON
    with open(flow_json_path, 'r') as f:
        flow_data = json.load(f)
    
    # Extract flow metadata
    flow_name = flow_data.get('name', Path(flow_json_path).stem)
    flow_description = flow_data.get('description', '')
    
    # Generate IDs (32 char hex without dashes)
    flow_id = uuid.uuid4().hex
    timestamp = datetime.now().isoformat()
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Insert into flow table with correct schema
        cursor.execute("""
            INSERT INTO flow (
                id, name, description, data, is_component, user_id, 
                updated_at, webhook, icon_bg_color, gradient, endpoint_name,
                mcp_enabled, access_type, tags, locked
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            flow_id,
            flow_name,
            flow_description,
            json.dumps(flow_data),
            0,  # is_component = False
            user_id,
            timestamp,
            0,  # webhook = False
            None,  # icon_bg_color
            None,  # gradient
            None,  # endpoint_name
            0,  # mcp_enabled = False
            'PRIVATE',  # access_type
            '[]',  # tags as JSON array
            0  # locked = False
        ))
        
        conn.commit()
        print(f"✅ Successfully imported: {flow_name}")
        print(f"   ID: {flow_id}")
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

def main():
    # Paths
    flows_dir = Path("/Users/user/Documents/arabic_folder/src/serviceAutomation/serviceLangflow/flows")
    db_path = "/tmp/langflow.db"
    
    flows = [
        flows_dir / "arabic_training_pipeline.json",
        flows_dir / "intelligent_training_orchestrator.json"
    ]
    
    print("🚀 Importing Flows to Langflow Database")
    print("=" * 60)
    print(f"Database: {db_path}")
    print()
    
    success_count = 0
    for flow_path in flows:
        if flow_path.exists():
            if import_flow_to_db(db_path, flow_path):
                success_count += 1
        else:
            print(f"⚠️  File not found: {flow_path}")
    
    print()
    print("=" * 60)
    print(f"📊 Imported {success_count}/{len(flows)} flows successfully")
    
    if success_count == len(flows):
        print("\n🎉 All flows imported! Restart Langflow to see them in the UI.")
        return 0
    else:
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
