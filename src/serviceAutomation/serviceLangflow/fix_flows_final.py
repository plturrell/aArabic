#!/usr/bin/env python3
"""
Final fix - Copy working Basic Prompting flow structure to Arabic flows
"""
import json
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="langflow",
    user="postgres",
    password="postgres"
)

def get_working_flow():
    """Get the Basic Prompting flow that we know works"""
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM flow WHERE name = 'Basic Prompting' LIMIT 1;")
    result = cursor.fetchone()
    cursor.close()
    
    if result:
        return result[0]
    return None

def create_arabic_flow(base_template, flow_name, description):
    """Modify the working template for Arabic workflows"""
    # Start with working template
    flow = base_template.copy()
    
    # Keep the structure but modify the prompt/purpose
    if 'nodes' in flow:
        for node in flow['nodes']:
            if node.get('data', {}).get('type') == 'Prompt':
                # Modify the prompt for Arabic training
                if 'node' in node['data'] and 'template' in node['data']['node']:
                    template_val = node['data']['node']['template'].get('template', {})
                    if 'value' in template_val:
                        template_val['value'] = f"{description}\\n\\nThis is an Arabic translation training workflow."
    
    return flow

def update_flows():
    """Update both flows with working structure"""
    # Get working template
    working_template = get_working_flow()
    
    if not working_template:
        print("❌ Could not find working template!")
        return
    
    print(f"✅ Got working template: {len(str(working_template))} bytes")
    
    cursor = conn.cursor()
    
    flows = [
        {
            "id": "e34afe6c-ca21-4b63-8cd1-af74df72ac79",
            "name": "Arabic Translation Training & Fine-tuning Pipeline",
