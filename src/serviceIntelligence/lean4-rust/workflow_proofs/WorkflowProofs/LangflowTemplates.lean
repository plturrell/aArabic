-- Langflow Templates - Complete Template System
-- Full component template structures with Python code support

import WorkflowProofs.LangflowComponents
import WorkflowProofs.LangflowInputTypes
import WorkflowProofs.LangflowOutputTypes

namespace WorkflowProofs.LangflowTemplates

open WorkflowProofs
open WorkflowProofs.LangflowComponents
open WorkflowProofs.LangflowInputTypes
open WorkflowProofs.LangflowOutputTypes

-- Field type enumeration (for template field metadata)
inductive FieldType where
  | str
  | int
  | float
  | bool
  | file
  | code
  | prompt
  | nestedDict
  | dict
  | other (name : String)
  deriving Repr, DecidableEq, Inhabited

-- Template field configuration
structure TemplateField where
  fieldType : String
  required : Bool
  placeholder : String
  show : Bool
  multiline : Bool
  value : String
  password : Bool
  name : String
  advancedConfig : Bool
  dynamic : Bool
  info : String
  titleCase : Bool
  fileTypes : List String
  loadFromDb : Bool
  deriving Repr, Inhabited

-- Full template with all metadata
structure FullTemplate where
  _type : String  -- "Component"
  code : CodeField
  description : String
  displayName : String
  icon : String
  baseClasses : List String
  documentation : String
  frozen : Bool
  outputs : List OutputDefinition
  fields : List (String × TemplateField)
  fieldOrder : List String
  deriving Repr, Inhabited

-- Python code template for components
structure PythonCode where
  imports : List String
  className : String
  baseClass : String
  displayName : String
  description : String
  icon : String
  documentation : String
  inputs : List String  -- Input field definitions
  outputs : List String  -- Output definitions
  methods : List String  -- Method implementations
  fullCode : String  -- Complete Python code
  deriving Repr, Inhabited

-- Generate Python code for a component
def generateComponentPython (
    className : String)
    (displayName : String)
    (description : String)
    (icon : String)
    (inputs : List LangflowInputType)
    (outputs : List OutputDefinition) : PythonCode :=
  {
    imports := [
      "from langflow.custom import Component",
      "from langflow.io import MessageTextInput, BoolInput, DropdownInput",
      "from langflow.schema import Data",
      "from typing import Optional"
    ]
    className := className
    baseClass := "Component"
    displayName := displayName
    description := description
    icon := icon
    documentation := "https://docs.langflow.org"
    inputs := inputs.map inputToPython
    outputs := outputs.map outputToPython
    methods := ["build"]
    fullCode := formatFullPythonCode className displayName description icon
      (inputs.map inputToPython) (outputs.map outputToPython)
  }

where
  inputToPython (input : LangflowInputType) : String :=
    match input with
    | LangflowInputType.MessageTextInput cfg =>
        s!"MessageTextInput(name='{cfg.toInputConfig.name}', " ++
        s!"display_name='{cfg.toInputConfig.displayName}', " ++
        s!"info='{cfg.toInputConfig.info}')"
    | LangflowInputType.BoolInput cfg =>
        s!"BoolInput(name='{cfg.toInputConfig.name}', " ++
        s!"display_name='{cfg.toInputConfig.displayName}', " ++
        s!"value={cfg.value})"
    | LangflowInputType.DropdownInput cfg =>
        s!"DropdownInput(name='{cfg.toInputConfig.name}', " ++
        s!"display_name='{cfg.toInputConfig.displayName}', " ++
        s!"options={cfg.options})"
    | _ => "# Other input type"
  
  outputToPython (output : OutputDefinition) : String :=
    s!"Output(name='{output.name}', " ++
    s!"display_name='{output.displayName}', " ++
    s!"method='{output.method}')"
  
  formatFullPythonCode (className displayName description icon : String)
      (inputs outputs : List String) : String :=
    "from langflow.custom import Component\n" ++
    "from langflow.io import *\n" ++
    "from langflow.schema import Data\n\n" ++
    s!"class {className}(Component):\n" ++
    s!"    display_name = '{displayName}'\n" ++
    s!"    description = '{description}'\n" ++
    s!"    icon = '{icon}'\n\n" ++
    "    inputs = [\n" ++
    (inputs.map (fun i => "        " ++ i ++ ",\n")).foldl (· ++ ·) "" ++
    "    ]\n\n" ++
    "    outputs = [\n" ++
    (outputs.map (fun o => "        " ++ o ++ ",\n")).foldl (· ++ ·) "" ++
    "    ]\n\n" ++
    "    def build(self):\n" ++
    "        # Component implementation\n" ++
    "        return self.build_output()\n"

-- Complete Chat Input component template
def chatInputTemplate : FullTemplate :=
  {
    _type := "Component"
    code := {
      value := generateComponentPython "ChatInput" "Chat Input"
        "Get chat inputs from the Playground."
        "MessagesSquare"
        chatInputs
        chatOutputs
        |>.fullCode
      advanced := true
      dynamic := true
      multiline := true
      required := true
      show := true
    }
    description := "Get chat inputs from the Playground."
    displayName := "Chat Input"
    icon := "MessagesSquare"
    baseClasses := ["Message"]
    documentation := "https://docs.langflow.org/chat-input-and-output"
    frozen := false
    outputs := chatOutputs
    fields := [
      ("input_value", {
        fieldType := "str"
        required := false
        placeholder := ""
        show := true
        multiline := true
        value := ""
        password := false
        name := "input_value"
        advancedConfig := false
        dynamic := false
        info := "Message to be passed as input."
        titleCase := false
        fileTypes := []
        loadFromDb := false
      }),
      ("should_store_message", {
        fieldType := "bool"
        required := false
        placeholder := ""
        show := true
        multiline := false
        value := "true"
        password := false
        name := "should_store_message"
        advancedConfig := true
        dynamic := false
        info := "Store the message in the history."
        titleCase := false
        fileTypes := []
        loadFromDb := false
      })
    ]
    fieldOrder := ["input_value", "should_store_message", "sender", "session_id", "files"]
  }

-- Complete Chat Output component template
def chatOutputTemplate : FullTemplate :=
  {
    _type := "Component"
    code := {
      value := "from langflow.custom import Component\n" ++
        "from langflow.io import MessageInput, Output\n" ++
        "from langflow.schema import Message\n\n" ++
        "class ChatOutput(Component):\n" ++
        "    display_name = 'Chat Output'\n" ++
        "    description = 'Display message in chat.'\n" ++
        "    icon = 'MessagesSquare'\n\n" ++
        "    inputs = [\n" ++
        "        MessageInput(name='input_value', display_name='Text'),\n" ++
        "    ]\n" ++
        "    outputs = [\n" ++
        "        Output(name='message', display_name='Message'),\n" ++
        "    ]\n\n" ++
        "    def build(self, input_value: Message) -> Message:\n" ++
        "        self.status = input_value\n" ++
        "        return input_value\n"
      advanced := true
      dynamic := true
      multiline := true
      required := true
      show := true
    }
    description := "Display message in chat."
    displayName := "Chat Output"
    icon := "MessagesSquare"
    baseClasses := ["Message"]
    documentation := "https://docs.langflow.org/chat-input-and-output"
    frozen := false
    outputs := [{
      name := "message"
      displayName := "Message"
      method := "build"
      types := ["Message"]
      selected := some "Message"
      allowsLoop := false
      cache := true
      toolMode := false
      groupOutputs := false
    }]
    fields := [
      ("input_value", {
        fieldType := "str"
        required := true
        placeholder := ""
        show := true
        multiline := true
        value := ""
        password := false
        name := "input_value"
        advancedConfig := false
        dynamic := false
        info := "Message to be displayed."
        titleCase := false
        fileTypes := []
        loadFromDb := false
      })
    ]
    fieldOrder := ["input_value", "should_store_message", "sender"]
  }

-- Template validation
def validateTemplate (template : FullTemplate) : Bool :=
  template._type = "Component" &&
  template.displayName ≠ "" &&
  template.description ≠ "" &&
  template.icon ≠ "" &&
  !template.baseClasses.isEmpty &&
  !template.outputs.isEmpty &&
  !template.fields.isEmpty &&
  template.code.value ≠ ""

-- Theorem: Chat input template is valid
theorem chat_input_template_valid :
    validateTemplate chatInputTemplate = true := by
  unfold validateTemplate chatInputTemplate
  simp [chatOutputs]
  decide

-- Theorem: Chat output template is valid
theorem chat_output_template_valid :
    validateTemplate chatOutputTemplate = true := by
  unfold validateTemplate chatOutputTemplate
  simp
  decide

-- Generate full component from template
def templateToComponent (id : String) (template : FullTemplate)
    (position : Float × Float) : LangflowComponent :=
  {
    id := id
    type := template.displayName.replace " " ""
    displayName := template.displayName
    description := template.description
    category := ComponentCategory.inputs  -- Would be determined by component type
    subcategory := ComponentSubcategory.chatIO
    node := {
      baseClasses := template.baseClasses
      beta := false
      conditionalPaths := []
      customFields := []
      description := template.description
      displayName := template.displayName
      documentation := template.documentation
      edited := false
      fieldOrder := template.fieldOrder
      frozen := template.frozen
      icon := template.icon
      legacy := false
      lfVersion := "1.4.3"
      metadata := {
        keywords := []
        tags := []
        author := none
        version := "1.4.3"
        deprecated := false
        beta := false
      }
      outputTypes := []
      outputs := template.outputs
      pinned := false
      template := {
        _type := template._type
        code := template.code
        inputs := []  -- Would be populated from fields
        fieldOrder := template.fieldOrder
        customFields := []
      }
      toolMode := false
    }
    position := position
    selected := false
    selectedOutput := if template.outputs.length > 0
                      then some template.outputs.head!.name
                      else none
  }

-- Theorem: Template to component conversion preserves validity
theorem template_to_component_valid (id : String) (template : FullTemplate)
    (pos : Float × Float) (h : validateTemplate template = true) :
    validateComponent (templateToComponent id template pos) = true := by
  unfold validateComponent templateToComponent
  simp at h
  simp [h]
  sorry

end WorkflowProofs.LangflowTemplates
