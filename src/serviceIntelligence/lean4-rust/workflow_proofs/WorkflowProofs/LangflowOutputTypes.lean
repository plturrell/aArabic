-- Langflow Output Types - Complete Formalization
-- All output data types and port definitions

import WorkflowProofs.WorkflowCore

namespace WorkflowProofs.LangflowOutputTypes

open WorkflowProofs

-- Port colors corresponding to data types
inductive PortColor where
  | red       -- Data
  | pink      -- DataFrame
  | emerald   -- Embeddings
  | fuchsia   -- LanguageModel
  | orange    -- Memory
  | indigo    -- Message
  | cyan      -- Tool
  | gray      -- Unknown/Multiple types
  deriving Repr, DecidableEq, Inhabited

-- Port direction
inductive PortDirection where
  | input
  | output
  deriving Repr, DecidableEq, Inhabited

-- Data types that can flow between components
inductive PortDataType where
  | Message
  | Data
  | DataFrame
  | LanguageModel
  | Tool
  | Embeddings
  | Memory
  | Document
  | Text
  | BaseLanguageModel
  | Retriever
  | VectorStore
  | Unknown
  deriving Repr, DecidableEq, Inhabited

-- Map data type to port color (from Langflow docs)
def dataTypeToColor (dt : PortDataType) : PortColor :=
  match dt with
  | PortDataType.Data => PortColor.red
  | PortDataType.DataFrame => PortColor.pink
  | PortDataType.Embeddings => PortColor.emerald
  | PortDataType.LanguageModel => PortColor.fuchsia
  | PortDataType.Memory => PortColor.orange
  | PortDataType.Message => PortColor.indigo
  | PortDataType.Tool => PortColor.cyan
  | _ => PortColor.gray

-- Output definition structure
structure OutputDefinition where
  name : String
  displayName : String
  method : String
  types : List String
  selected : Option String  -- Which type is currently selected
  allowsLoop : Bool
  cache : Bool
  toolMode : Bool
  groupOutputs : Bool  -- If true, all outputs shown; if false, user selects one
  deriving Repr, Inhabited

-- Component port (input or output connection point)
structure ComponentPort where
  name : String
  displayName : String
  dataType : PortDataType
  color : PortColor
  direction : PortDirection
  required : Bool
  multiple : Bool  -- Can accept multiple connections
  componentId : String
  deriving Repr, Inhabited

-- Create output port from output definition
def outputToPort (componentId : String) (output : OutputDefinition) : ComponentPort :=
  let dataType := 
    if output.types.isEmpty then PortDataType.Unknown
    else match output.selected with
      | some sel => stringToDataType sel
      | none => stringToDataType (output.types.head!)
  {
    name := output.name
    displayName := output.displayName
    dataType := dataType
    color := dataTypeToColor dataType
    direction := PortDirection.output
    required := false  -- Outputs are never required
    multiple := true   -- Outputs can connect to multiple inputs
    componentId := componentId
  }

where
  stringToDataType (s : String) : PortDataType :=
    if s = "Message" then PortDataType.Message
    else if s = "Data" then PortDataType.Data
    else if s = "DataFrame" then PortDataType.DataFrame
    else if s = "LanguageModel" then PortDataType.LanguageModel
    else if s = "Tool" then PortDataType.Tool
    else if s = "Embeddings" then PortDataType.Embeddings
    else if s = "Memory" then PortDataType.Memory
    else if s = "Document" then PortDataType.Document
    else if s = "Text" then PortDataType.Text
    else PortDataType.Unknown

-- Check if two data types are compatible for connection
def typesCompatible (sourceType targetType : PortDataType) : Bool :=
  sourceType = targetType ||
  -- Message and Text are compatible
  (sourceType = PortDataType.Message && targetType = PortDataType.Text) ||
  (sourceType = PortDataType.Text && targetType = PortDataType.Message) ||
  -- Data and Document are compatible
  (sourceType = PortDataType.Data && targetType = PortDataType.Document) ||
  (sourceType = PortDataType.Document && targetType = PortDataType.Data) ||
  -- Unknown type is compatible with anything
  sourceType = PortDataType.Unknown ||
  targetType = PortDataType.Unknown

-- Example: Chat Output component outputs
def chatOutputs : List OutputDefinition := [
  {
    name := "message"
    displayName := "Output Message"
    method := "message_response"
    types := ["Message"]
    selected := some "Message"
    allowsLoop := false
    cache := true
    toolMode := true
    groupOutputs := false
  }
]

-- Example: Language Model component outputs (2 outputs)
def languageModelOutputs : List OutputDefinition := [
  {
    name := "text_output"
    displayName := "Model Response"
    method := "text_response"
    types := ["Message"]
    selected := some "Message"
    allowsLoop := false
    cache := true
    toolMode := true
    groupOutputs := false
  },
  {
    name := "model_output"
    displayName := "Language Model"
    method := "build_model"
    types := ["LanguageModel"]
    selected := some "LanguageModel"
    allowsLoop := false
    cache := true
    toolMode := true
    groupOutputs := false
  }
]

-- Example: Memory component outputs (grouped)
def memoryOutputs : List OutputDefinition := [
  {
    name := "messages_text"
    displayName := "Message"
    method := "retrieve_messages_as_text"
    types := ["Message"]
    selected := some "Message"
    allowsLoop := false
    cache := true
    toolMode := true
    groupOutputs := true  -- Both outputs shown simultaneously
  },
  {
    name := "dataframe"
    displayName := "Dataframe"
    method := "retrieve_messages_dataframe"
    types := ["DataFrame"]
    selected := none
    allowsLoop := false
    cache := true
    toolMode := true
    groupOutputs := true
  }
]

-- Validate output definition
def validateOutput (output : OutputDefinition) : Bool :=
  output.name ≠ "" &&
  output.displayName ≠ "" &&
  output.method ≠ "" &&
  !output.types.isEmpty &&
  -- If not group_outputs, must have a selected type
  (output.groupOutputs || output.selected.isSome)

-- Validate all outputs in a list
def validateOutputs (outputs : List OutputDefinition) : Bool :=
  outputs.all validateOutput

-- Theorem: Chat output is valid
theorem chat_output_valid : validateOutputs chatOutputs = true := by
  unfold validateOutputs chatOutputs
  simp [validateOutput]

-- Theorem: Language model outputs are valid
theorem language_model_outputs_valid : 
    validateOutputs languageModelOutputs = true := by
  unfold validateOutputs languageModelOutputs
  simp [validateOutput]

-- Theorem: Type compatibility is reflexive
theorem type_compat_refl (dt : PortDataType) :
    typesCompatible dt dt = true := by
  unfold typesCompatible
  simp

-- Theorem: Type compatibility includes Message-Text
theorem message_text_compatible :
    typesCompatible PortDataType.Message PortDataType.Text = true ∧
    typesCompatible PortDataType.Text PortDataType.Message = true := by
  unfold typesCompatible
  simp

end WorkflowProofs.LangflowOutputTypes
