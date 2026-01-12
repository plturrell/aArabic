-- Langflow Components - Complete Formalization
-- Main component system with templates and full structures

import WorkflowProofs.WorkflowCore
import WorkflowProofs.LangflowInputTypes
import WorkflowProofs.LangflowOutputTypes
import WorkflowProofs.LangflowCategories

namespace WorkflowProofs.LangflowComponents

open WorkflowProofs
open WorkflowProofs.LangflowInputTypes
open WorkflowProofs.LangflowOutputTypes
open WorkflowProofs.LangflowCategories

-- Code field in component template
structure CodeField where
  value : String  -- The actual Python code (~10KB per component!)
  advanced : Bool
  dynamic : Bool
  multiline : Bool
  required : Bool
  show : Bool
  deriving Repr, Inhabited

-- Input field in template
structure InputField where
  name : String
  inputType : String  -- "_input_type" field
  config : List (String × String)
  deriving Repr, Inhabited

-- Component template (the massive structure from database)
structure ComponentTemplate where
  _type : String  -- Always "Component"
  code : CodeField
  inputs : List (String × LangflowInputType)
  fieldOrder : List String
  customFields : List (String × List String)
  deriving Repr, Inhabited

-- Node definition (wraps component with runtime info)
structure ComponentNode where
  baseClasses : List String
  beta : Bool
  conditionalPaths : List String
  customFields : List (String × List String)
  description : String
  displayName : String
  documentation : String
  edited : Bool
  fieldOrder : List String
  frozen : Bool
  icon : String
  legacy : Bool
  lfVersion : String
  metadata : ComponentMetadata
  outputTypes : List String
  outputs : List OutputDefinition
  pinned : Bool
  template : ComponentTemplate
  toolMode : Bool
  deriving Repr, Inhabited

-- Complete Langflow component
structure LangflowComponent where
  id : String
  type : String  -- Component type name
  displayName : String
  description : String
  category : ComponentCategory
  subcategory : ComponentSubcategory
  node : ComponentNode
  position : (Float × Float)
  selected : Bool
  selectedOutput : Option String
  deriving Repr, Inhabited

-- Edge connecting components (with complex handle system)
structure ComponentEdge where
  id : String
  source : String  -- Source component ID
  target : String  -- Target component ID
  sourceHandle : String  -- JSON string with port info
  targetHandle : String  -- JSON string with port info
  animated : Bool
  selected : Bool
  deriving Repr, Inhabited

-- Complete Langflow flow
structure LangflowFlow where
  name : String
  description : String
  nodes : List LangflowComponent
  edges : List ComponentEdge
  viewport : (Float × Float × Float)  -- (x, y, zoom)
  deriving Repr, Inhabited

-- Example: Chat Input Component
def chatInputComponent : LangflowComponent :=
  {
    id := "ChatInput-abc123"
    type := "ChatInput"
    displayName := "Chat Input"
    description := "Get chat inputs from the Playground."
    category := ComponentCategory.inputs
    subcategory := ComponentSubcategory.chatIO
    node := {
      baseClasses := ["Message"]
      beta := false
      conditionalPaths := []
      customFields := []
      description := "Get chat inputs from the Playground."
      displayName := "Chat Input"
      documentation := "https://docs.langflow.org/chat-input-and-output"
      edited := false
      fieldOrder := ["input_value", "should_store_message", "sender"]
      frozen := false
      icon := "MessagesSquare"
      legacy := false
      lfVersion := "1.4.3"
      metadata := {
        keywords := ["chat", "input", "message"]
        tags := []
        author := none
        version := "1.4.3"
        deprecated := false
        beta := false
      }
      outputTypes := []
      outputs := chatOutputs
      pinned := false
      template := {
        _type := "Component"
        code := {
          value := "# ChatInput component code would go here (~10KB)"
          advanced := true
          dynamic := true
          multiline := true
          required := true
          show := true
        }
        inputs := [
          ("input_value", LangflowInputType.MultilineInput {
            toInputConfig := {
              name := "input_value"
              displayName := "Input Text"
              info := "Message to be passed as input."
              advanced := false
              required := false
              show := true
              placeholder := ""
            }
            value := ""
            multiline := true
            inputTypes := []
            loadFromDb := false
            copyField := false
          })
        ]
        fieldOrder := ["input_value"]
        customFields := []
      }
      toolMode := false
    }
    position := (100.0, 100.0)
    selected := false
    selectedOutput := some "message"
  }

-- Example: Language Model Component
def languageModelComponent : LangflowComponent :=
  {
    id := "LanguageModel-xyz789"
    type := "LanguageModelComponent"
    displayName := "Language Model"
    description := "Runs a language model given a specified provider."
    category := ComponentCategory.models
    subcategory := ComponentSubcategory.openai
    node := {
      baseClasses := ["LanguageModel", "Message"]
      beta := false
      conditionalPaths := []
      customFields := []
      description := "Runs a language model given a specified provider."
      displayName := "Language Model"
      documentation := "https://docs.langflow.org/components-models"
      edited := false
      fieldOrder := ["provider", "model_name", "api_key", "input_value"]
      frozen := false
      icon := "brain-circuit"
      legacy := false
      lfVersion := "1.4.3"
      metadata := {
        keywords := ["model", "llm", "language model"]
        tags := []
        author := none
        version := "1.4.3"
        deprecated := false
        beta := false
      }
      outputTypes := []
      outputs := languageModelOutputs
      pinned := false
      template := {
        _type := "Component"
        code := {
          value := "# LanguageModel component code (~10KB)"
          advanced := true
          dynamic := true
          multiline := true
          required := true
          show := true
        }
        inputs := [
          ("provider", LangflowInputType.DropdownInput {
            toInputConfig := {
              name := "provider"
              displayName := "Model Provider"
              info := "Select the model provider"
              advanced := false
              required := false
              show := true
              placeholder := ""
            }
            options := ["OpenAI", "Anthropic", "Google"]
            value := "OpenAI"
            combobox := false
            optionsMetadata := [("OpenAI", "icon"), ("Anthropic", "icon")]
            realTimeRefresh := true
            toolMode := false
          })
        ]
        fieldOrder := ["provider", "model_name"]
        customFields := []
      }
      toolMode := false
    }
    position := (400.0, 200.0)
    selected := false
    selectedOutput := some "text_output"
  }

-- Validate component structure
def validateComponent (comp : LangflowComponent) : Bool :=
  comp.id ≠ "" &&
  comp.type ≠ "" &&
  comp.displayName ≠ "" &&
  !comp.node.outputs.isEmpty &&
  validateOutputs comp.node.outputs

-- Validate flow structure
def validateFlow (flow : LangflowFlow) : Bool :=
  flow.name ≠ "" &&
  !flow.nodes.isEmpty &&
  flow.nodes.all validateComponent &&
  -- All edges reference existing nodes
  flow.edges.all fun edge =>
    flow.nodes.any (fun n => n.id = edge.source) &&
    flow.nodes.any (fun n => n.id = edge.target)

-- Check if edge connects compatible ports
def edgeHasCompatibleTypes (flow : LangflowFlow) (edge : ComponentEdge) : Bool :=
  match flow.nodes.find? (·.id = edge.source), 
        flow.nodes.find? (·.id = edge.target) with
  | some sourceComp, some targetComp =>
      -- For now, simplified check
      !sourceComp.node.outputs.isEmpty
  | _, _ => false

-- Theorem: Valid flows have valid components
theorem valid_flow_has_valid_components (flow : LangflowFlow)
    (h : validateFlow flow = true) :
    flow.nodes.all validateComponent = true := by
  unfold validateFlow at h
  simp at h
  exact h.right.left

-- Theorem: Chat input component is valid
theorem chat_input_valid :
    validateComponent chatInputComponent = true := by
  unfold validateComponent chatInputComponent
  simp [validateOutputs, chatOutputs, validateOutput]

end WorkflowProofs.LangflowComponents
