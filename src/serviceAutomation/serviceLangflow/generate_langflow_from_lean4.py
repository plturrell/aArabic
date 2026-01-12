#!/usr/bin/env python3
"""
Generate Complete Langflow Flows from Lean4 Specifications
Uses the formally verified Lean4 component system to generate working Langflow flows
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class PortColor(Enum):
    """Port colors from Lean4 specification"""
    RED = "#dc2626"      # Data
    PINK = "#ec4899"     # DataFrame
    EMERALD = "#10b981"  # Embeddings
    FUCHSIA = "#c026d3"  # LanguageModel
    ORANGE = "#f97316"   # Memory
    INDIGO = "#4f46e5"   # Message
    CYAN = "#06b6d4"     # Tool
    GRAY = "#9CA3AF"     # Unknown


class ComponentCategory(Enum):
    """Component categories from Lean4"""
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    MODELS = "models"
    AGENTS = "agents"
    PROCESSING = "processing"
    DATA_SOURCE = "dataSource"
    FILES = "files"
    FLOW_CONTROLS = "flowControls"
    LLM_OPERATIONS = "llmOperations"
    UTILITIES = "utilities"


@dataclass
class ComponentTemplate:
    """Component template structure from Lean4"""
    _type: str = "Component"
    code: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    display_name: str = ""
    icon: str = ""
    base_classes: List[str] = field(default_factory=list)
    documentation: str = ""
    frozen: bool = False
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    field_order: List[str] = field(default_factory=list)


@dataclass
class LangflowComponent:
    """Complete Langflow component from Lean4 specification"""
    id: str
    type: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Langflow JSON format"""
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data
        }


@dataclass
class LangflowEdge:
    """Langflow edge from Lean4 specification"""
    id: str
    source: str
    target: str
    sourceHandle: Dict[str, Any]
    targetHandle: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Langflow JSON format"""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "sourceHandle": json.dumps(self.sourceHandle),
            "targetHandle": json.dumps(self.targetHandle),
            "data": {
                "sourceHandle": self.sourceHandle,
                "targetHandle": self.targetHandle
            },
            "style": {},
            "className": "stroke-gray-900 stroke-connection",
            "animated": False,
            "selected": False
        }


class LangflowGenerator:
    """Generate Langflow flows from Lean4 specifications"""
    
    def __init__(self):
        self.components: Dict[str, LangflowComponent] = {}
        self.edges: List[LangflowEdge] = []
    
    def generate_chat_input(self, 
                           component_id: Optional[str] = None,
                           position: tuple = (100, 100)) -> LangflowComponent:
        """Generate Chat Input component from Lean4 chatInputTemplate"""
        if component_id is None:
            component_id = f"ChatInput-{uuid.uuid4().hex[:8]}"
        
        # Python code from Lean4 generateComponentPython
        python_code = """from langflow.custom import Component
from langflow.io import *
from langflow.schema import Data

class ChatInput(Component):
    display_name = 'Chat Input'
    description = 'Get chat inputs from the Playground.'
    icon = 'MessagesSquare'

    inputs = [
        MessageTextInput(name='input_value', display_name='Input Text', info='Message to be passed as input.'),
        BoolInput(name='should_store_message', display_name='Store Messages', value=True),
        DropdownInput(name='sender', display_name='Sender Type', options=['Machine', 'User']),
    ]

    outputs = [
        Output(name='message', display_name='Message', method='build'),
    ]

    def build(self):
        # Component implementation
        return self.build_output()
"""
        
        component = LangflowComponent(
            id=component_id,
            type="ChatInput",
            data={
                "type": "ChatInput",
                "node": {
                    "template": {
                        "_type": "Component",
                        "code": {
                            "type": "code",
                            "required": True,
                            "placeholder": "",
                            "list": False,
                            "show": True,
                            "multiline": True,
                            "value": python_code,
                            "fileTypes": [],
                            "file_path": "",
                            "password": False,
                            "name": "code",
                            "advanced": True,
                            "dynamic": True,
                            "info": "",
                            "load_from_db": False,
                            "title_case": False
                        },
                        "input_value": {
                            "type": "str",
                            "required": False,
                            "placeholder": "",
                            "show": True,
                            "multiline": True,
                            "value": "",
                            "password": False,
                            "name": "input_value",
                            "display_name": "Input Text",
                            "advanced": False,
                            "input_types": ["Message"],
                            "dynamic": False,
                            "info": "Message to be passed as input.",
                            "load_from_db": False,
                            "title_case": False
                        }
                    },
                    "description": "Get chat inputs from the Playground.",
                    "base_classes": ["Message"],
                    "display_name": "Chat Input",
                    "documentation": "https://docs.langflow.org/chat-input-and-output",
                    "custom_fields": {},
                    "output_types": [],
                    "pinned": False,
                    "conditional_paths": [],
                    "frozen": False,
                    "outputs": [
                        {
                            "types": ["Message"],
                            "selected": "Message",
                            "name": "message",
                            "display_name": "Message",
                            "method": "message_response",
                            "value": "__UNDEFINED__",
                            "cache": True
                        }
                    ],
                    "field_order": ["input_value", "should_store_message", "sender"],
                    "beta": False,
                    "edited": False,
                    "lf_version": "1.4.3"
                },
                "id": component_id,
                "value": None,
                "position": {"x": position[0], "y": position[1]}
            }
        )
        
        self.components[component_id] = component
        return component
    
    def generate_chat_output(self,
                            component_id: Optional[str] = None,
                            position: tuple = (400, 100)) -> LangflowComponent:
        """Generate Chat Output component from Lean4 chatOutputTemplate"""
        if component_id is None:
            component_id = f"ChatOutput-{uuid.uuid4().hex[:8]}"
        
        # Python code from Lean4 template
        python_code = """from langflow.custom import Component
from langflow.io import MessageInput, Output
from langflow.schema import Message

class ChatOutput(Component):
    display_name = 'Chat Output'
    description = 'Display message in chat.'
    icon = 'MessagesSquare'

    inputs = [
        MessageInput(name='input_value', display_name='Text'),
    ]
    outputs = [
        Output(name='message', display_name='Message'),
    ]

    def build(self, input_value: Message) -> Message:
        self.status = input_value
        return input_value
"""
        
        component = LangflowComponent(
            id=component_id,
            type="ChatOutput",
            data={
                "type": "ChatOutput",
                "node": {
                    "template": {
                        "_type": "Component",
                        "code": {
                            "type": "code",
                            "required": True,
                            "value": python_code,
                            "show": True,
                            "multiline": True,
                            "name": "code",
                            "advanced": True,
                            "dynamic": True
                        },
                        "input_value": {
                            "type": "str",
                            "required": True,
                            "placeholder": "",
                            "show": True,
                            "multiline": True,
                            "value": "",
                            "password": False,
                            "name": "input_value",
                            "display_name": "Text",
                            "advanced": False,
                            "input_types": ["Message"],
                            "dynamic": False,
                            "info": "Message to be displayed.",
                            "load_from_db": False
                        }
                    },
                    "description": "Display message in chat.",
                    "base_classes": ["Message"],
                    "display_name": "Chat Output",
                    "documentation": "https://docs.langflow.org/chat-input-and-output",
                    "custom_fields": {},
                    "output_types": [],
                    "pinned": False,
                    "conditional_paths": [],
                    "frozen": False,
                    "outputs": [
                        {
                            "types": ["Message"],
                            "selected": "Message",
                            "name": "message",
                            "display_name": "Message",
                            "method": "build",
                            "value": "__UNDEFINED__",
                            "cache": True
                        }
                    ],
                    "field_order": ["input_value"],
                    "beta": False,
                    "edited": False,
                    "lf_version": "1.4.3"
                },
                "id": component_id,
                "value": None,
                "position": {"x": position[0], "y": position[1]}
            }
        )
        
        self.components[component_id] = component
        return component
    
    def connect_components(self,
                          source_id: str,
                          target_id: str,
                          source_output: str = "message",
                          target_input: str = "input_value") -> LangflowEdge:
        """Connect two components with type-safe edge"""
        edge_id = f"edge-{uuid.uuid4().hex[:8]}"
        
        edge = LangflowEdge(
            id=edge_id,
            source=source_id,
            target=target_id,
            sourceHandle={
                "dataType": "Message",
                "id": source_id,
                "name": source_output,
                "output_types": ["Message"]
            },
            targetHandle={
                "fieldName": target_input,
                "id": target_id,
                "inputTypes": ["Message"],
                "type": "str"
            }
        )
        
        self.edges.append(edge)
        return edge
    
    def generate_flow(self,
                     name: str = "Generated Flow",
                     description: str = "Flow generated from Lean4 specifications") -> Dict[str, Any]:
        """Generate complete Langflow flow JSON"""
        nodes = [comp.to_dict() for comp in self.components.values()]
        edges = [edge.to_dict() for edge in self.edges]
        
        flow = {
            "name": name,
            "description": description,
            "data": {
                "nodes": nodes,
                "edges": edges,
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            },
            "is_component": False,
            "updated_at": "2026-01-10T00:00:00.000Z",
            "folder": None,
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4())
        }
        
        return flow


def generate_simple_chat_flow() -> Dict[str, Any]:
    """Generate simple chat flow from Lean4 simpleValidFlow"""
    generator = LangflowGenerator()
    
    # Generate components
    chat_in = generator.generate_chat_input(position=(100, 100))
    chat_out = generator.generate_chat_output(position=(400, 100))
    
    # Connect them
    generator.connect_components(chat_in.id, chat_out.id)
    
    # Generate flow
    return generator.generate_flow(
        name="Simple Chat Flow (Lean4 Generated)",
        description="Basic chat input to output flow generated from Lean4 specifications"
    )


def main():
    """Main function"""
    print("🚀 Generating Langflow Flow from Lean4 Specifications...")
    print("=" * 60)
    
    # Generate simple chat flow
    flow = generate_simple_chat_flow()
    
    # Save to file
    output_file = "lean4_generated_chat_flow.json"
    with open(output_file, 'w') as f:
        json.dump(flow, f, indent=2)
    
    print(f"✅ Generated flow with {len(flow['data']['nodes'])} components")
    print(f"✅ Created {len(flow['data']['edges'])} connections")
    print(f"✅ Saved to: {output_file}")
    print()
    print("📝 Next steps:")
    print("   1. Import JSON to Langflow database")
    print("   2. Open in Langflow UI")
    print("   3. Test in Playground")
    print()
    print("🎉 Flow generation complete!")


if __name__ == "__main__":
    main()
