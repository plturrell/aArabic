#!/usr/bin/env python3
"""
Script to add missing adapter aliases and health check functions
"""

adapter_fixes = {
    "memgraph.py": ("MemgraphAdapter", "MemgraphService", "memgraph:7687"),
    "nucleus_flow.py": ("NucleusFlowAdapter", "NucleusFlowService", "nucleus-flow:8000"),
    "nucleusgraph.py": ("NucleusGraphAdapter", "NucleusGraphService", "nucleus-graph:5000"),
    "keycloak.py": ("KeycloakAdapter", "KeycloakService", "keycloak:8080"),
    "marquez.py": ("MarquezAdapter", "MarquezService", "marquez:5000"),
    "apisix.py": ("APISIXAdapter", "APISIXService", "apisix:9080"),
    "a2ui.py": ("A2UIAdapter", "A2UIService", "a2ui:8000"),
    "a2ui_enhanced.py": ("A2UIEnhancedAdapter", "A2UIEnhancedService", "a2ui-enhanced:8000"),
    "toolorchestra.py": ("ToolOrchestraAdapter", "ToolOrchestraService", "toolorchestra:8000"),
}

template = '''

# Alias for backward compatibility
{adapter_name} = {class_name}


async def check_{service_name}_health({service_name}_url: str = "http://{default_url}") -> Dict[str, Any]:
    """
    Check {service_display} service health
    
    Args:
        {service_name}_url: Base URL for {service_display} service
        
    Returns:
        Health check result
    """
    service = {class_name}(base_url={service_name}_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()
'''

for filename, (adapter_name, class_name, default_url) in adapter_fixes.items():
    filepath = f"src/serviceCore/adapters/{filename}"
    service_name = adapter_name.replace("Adapter", "").lower()
    service_name = service_name.replace("a2ui", "a2ui")
    service_display = adapter_name.replace("Adapter", "")
    
    # Read file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already has alias
    if f"{adapter_name} =" in content:
        print(f"✓ {filename} already has alias")
        continue
    
    # Add the fix
    fix_text = template.format(
        adapter_name=adapter_name,
        class_name=class_name,
        service_name=service_name,
        service_display=service_display,
        default_url=default_url
    )
    
    # Append to file
    with open(filepath, 'a') as f:
        f.write(fix_text)
    
    print(f"✓ Fixed {filename}")

print("\nAll adapters fixed!")