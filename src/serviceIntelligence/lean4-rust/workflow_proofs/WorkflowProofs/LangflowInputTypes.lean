-- Langflow Input Types - Complete Formalization
-- All 16 input field types used in Langflow components

import WorkflowProofs.WorkflowCore

namespace WorkflowProofs.LangflowInputTypes

open WorkflowProofs

-- Range specification for numeric inputs
structure RangeSpec where
  min : Float
  max : Float
  step : Float
  stepType : String  -- "float" or "int"
  deriving Repr, Inhabited

-- Table column schema
structure TableColumn where
  name : String
  displayName : String
  type : String
  description : String
  default : Option String
  disableEdit : Bool
  formatter : String
  hidden : Bool
  sortable : Bool
  filterable : Bool
  deriving Repr, Inhabited

-- Table schema for TableInput
structure TableSchema where
  columns : List TableColumn
  deriving Repr, Inhabited

-- Base input configuration shared by all types
structure InputConfig where
  name : String
  displayName : String
  info : String
  advanced : Bool
  required : Bool
  show : Bool
  placeholder : String
  deriving Repr, Inhabited

-- Message text input configuration
structure MessageTextConfig extends InputConfig where
  inputTypes : List String  -- ["Message", "Text"]
  isList : Bool
  loadFromDb : Bool
  toolMode : Bool
  traceAsInput : Bool
  traceAsMetadata : Bool
  deriving Repr, Inhabited

-- Handle input configuration
structure HandleInputConfig extends InputConfig where
  inputTypes : List String
  isList : Bool
  traceAsMetadata : Bool
  deriving Repr, Inhabited

-- Dropdown input configuration
structure DropdownConfig extends InputConfig where
  options : List String
  value : String
  combobox : Bool
  optionsMetadata : List (String × String)
  realTimeRefresh : Bool
  toolMode : Bool
  deriving Repr, Inhabited

-- Boolean input configuration
structure BoolConfig extends InputConfig where
  value : Bool
  deriving Repr, Inhabited

-- Integer input configuration
structure IntConfig extends InputConfig where
  value : Nat
  min : Option Nat
  max : Option Nat
  toolMode : Bool
  deriving Repr, Inhabited

-- Slider input configuration
structure SliderConfig extends InputConfig where
  value : Float
  rangeSpec : RangeSpec
  minLabel : String
  maxLabel : String
  minLabelIcon : String
  maxLabelIcon : String
  sliderButtons : Bool
  sliderInput : Bool
  deriving Repr, Inhabited

-- Secret string input configuration (for API keys)
structure SecretConfig extends InputConfig where
  value : String
  password : Bool
  loadFromDb : Bool
  realTimeRefresh : Bool
  deriving Repr, Inhabited

-- File input configuration
structure FileConfig extends InputConfig where
  fileTypes : List String
  isList : Bool
  tempFile : Bool
  deriving Repr, Inhabited

-- Table input configuration
structure TableInputConfig extends InputConfig where
  tableSchema : TableSchema
  value : List (List (String × String))
  inputTypes : List String
  isList : Bool
  triggerIcon : String
  triggerText : String
  deriving Repr, Inhabited

-- Multiline input configuration
structure MultilineConfig extends InputConfig where
  value : String
  multiline : Bool
  inputTypes : List String
  loadFromDb : Bool
  copyField : Bool
  deriving Repr, Inhabited

-- Prompt input configuration
structure PromptConfig extends InputConfig where
  value : String
  toolMode : Bool
  traceAsInput : Bool
  deriving Repr, Inhabited

-- Tab input configuration
structure TabConfig extends InputConfig where
  tabs : List String
  value : String
  realTimeRefresh : Bool
  deriving Repr, Inhabited

-- Dictionary input configuration
structure DictConfig extends InputConfig where
  value : List (String × String)
  deriving Repr, Inhabited

-- Float input configuration
structure FloatConfig extends InputConfig where
  value : Float
  min : Option Float
  max : Option Float
  deriving Repr, Inhabited

-- Data input configuration
structure DataInputConfig extends InputConfig where
  inputTypes : List String
  isList : Bool
  deriving Repr, Inhabited

-- Complete input type enumeration
inductive LangflowInputType where
  | MessageTextInput (config : MessageTextConfig)
  | HandleInput (config : HandleInputConfig)
  | DropdownInput (config : DropdownConfig)
  | BoolInput (config : BoolConfig)
  | IntInput (config : IntConfig)
  | SliderInput (config : SliderConfig)
  | SecretStrInput (config : SecretConfig)
  | FileInput (config : FileConfig)
  | TableInput (config : TableInputConfig)
  | MultilineInput (config : MultilineConfig)
  | PromptInput (config : PromptConfig)
  | MessageInput (config : MessageTextConfig)  -- Similar to MessageTextInput
  | TabInput (config : TabConfig)
  | DictInput (config : DictConfig)
  | FloatInput (config : FloatConfig)
  | DataInput (config : DataInputConfig)
  deriving Repr, Inhabited

-- Get input name from any input type
def getInputName (input : LangflowInputType) : String :=
  match input with
  | LangflowInputType.MessageTextInput cfg => cfg.toInputConfig.name
  | LangflowInputType.HandleInput cfg => cfg.toInputConfig.name
  | LangflowInputType.DropdownInput cfg => cfg.toInputConfig.name
  | LangflowInputType.BoolInput cfg => cfg.toInputConfig.name
  | LangflowInputType.IntInput cfg => cfg.toInputConfig.name
  | LangflowInputType.SliderInput cfg => cfg.toInputConfig.name
  | LangflowInputType.SecretStrInput cfg => cfg.toInputConfig.name
  | LangflowInputType.FileInput cfg => cfg.toInputConfig.name
  | LangflowInputType.TableInput cfg => cfg.toInputConfig.name
  | LangflowInputType.MultilineInput cfg => cfg.toInputConfig.name
  | LangflowInputType.PromptInput cfg => cfg.toInputConfig.name
  | LangflowInputType.MessageInput cfg => cfg.toInputConfig.name
  | LangflowInputType.TabInput cfg => cfg.toInputConfig.name
  | LangflowInputType.DictInput cfg => cfg.toInputConfig.name
  | LangflowInputType.FloatInput cfg => cfg.toInputConfig.name
  | LangflowInputType.DataInput cfg => cfg.toInputConfig.name

-- Check if input is required
def isRequired (input : LangflowInputType) : Bool :=
  match input with
  | LangflowInputType.MessageTextInput cfg => cfg.toInputConfig.required
  | LangflowInputType.HandleInput cfg => cfg.toInputConfig.required
  | LangflowInputType.DropdownInput cfg => cfg.toInputConfig.required
  | LangflowInputType.BoolInput cfg => cfg.toInputConfig.required
  | LangflowInputType.IntInput cfg => cfg.toInputConfig.required
  | LangflowInputType.SliderInput cfg => cfg.toInputConfig.required
  | LangflowInputType.SecretStrInput cfg => cfg.toInputConfig.required
  | LangflowInputType.FileInput cfg => cfg.toInputConfig.required
  | LangflowInputType.TableInput cfg => cfg.toInputConfig.required
  | LangflowInputType.MultilineInput cfg => cfg.toInputConfig.required
  | LangflowInputType.PromptInput cfg => cfg.toInputConfig.required
  | LangflowInputType.MessageInput cfg => cfg.toInputConfig.required
  | LangflowInputType.TabInput cfg => cfg.toInputConfig.required
  | LangflowInputType.DictInput cfg => cfg.toInputConfig.required
  | LangflowInputType.FloatInput cfg => cfg.toInputConfig.required
  | LangflowInputType.DataInput cfg => cfg.toInputConfig.required

-- Check if input is advanced
def isAdvanced (input : LangflowInputType) : Bool :=
  match input with
  | LangflowInputType.MessageTextInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.HandleInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.DropdownInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.BoolInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.IntInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.SliderInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.SecretStrInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.FileInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.TableInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.MultilineInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.PromptInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.MessageInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.TabInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.DictInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.FloatInput cfg => cfg.toInputConfig.advanced
  | LangflowInputType.DataInput cfg => cfg.toInputConfig.advanced

-- Validate input configuration
def validateInput (input : LangflowInputType) : Bool :=
  let name := getInputName input
  let required := isRequired input
  -- Name must not be empty
  name ≠ "" &&
  -- If required, must have show=true
  (required → match input with
    | LangflowInputType.MessageTextInput cfg => cfg.toInputConfig.show
    | LangflowInputType.HandleInput cfg => cfg.toInputConfig.show
    | LangflowInputType.DropdownInput cfg => cfg.toInputConfig.show
    | LangflowInputType.BoolInput cfg => cfg.toInputConfig.show
    | LangflowInputType.IntInput cfg => cfg.toInputConfig.show
    | LangflowInputType.SliderInput cfg => cfg.toInputConfig.show
    | LangflowInputType.SecretStrInput cfg => cfg.toInputConfig.show
    | LangflowInputType.FileInput cfg => cfg.toInputConfig.show
    | LangflowInputType.TableInput cfg => cfg.toInputConfig.show
    | LangflowInputType.MultilineInput cfg => cfg.toInputConfig.show
    | LangflowInputType.PromptInput cfg => cfg.toInputConfig.show
    | LangflowInputType.MessageInput cfg => cfg.toInputConfig.show
    | LangflowInputType.TabInput cfg => cfg.toInputConfig.show
    | LangflowInputType.DictInput cfg => cfg.toInputConfig.show
    | LangflowInputType.FloatInput cfg => cfg.toInputConfig.show
    | LangflowInputType.DataInput cfg => cfg.toInputConfig.show)

-- Example: Chat Input component inputs
def chatInputs : List LangflowInputType := [
  LangflowInputType.MultilineInput {
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
  },
  LangflowInputType.BoolInput {
    toInputConfig := {
      name := "should_store_message"
      displayName := "Store Messages"
      info := "Store the message in the history."
      advanced := true
      required := false
      show := true
      placeholder := ""
    }
    value := true
  },
  LangflowInputType.DropdownInput {
    toInputConfig := {
      name := "sender"
      displayName := "Sender Type"
      info := "Type of sender."
      advanced := true
      required := false
      show := true
      placeholder := ""
    }
    options := ["Machine", "User"]
    value := "User"
    combobox := false
    optionsMetadata := []
    realTimeRefresh := false
    toolMode := false
  },
  LangflowInputType.MessageTextInput {
    toInputConfig := {
      name := "session_id"
      displayName := "Session ID"
      info := "The session ID of the chat. If empty, the current session ID parameter will be used."
      advanced := true
      required := false
      show := true
      placeholder := ""
    }
    inputTypes := ["Message"]
    isList := false
    loadFromDb := false
    toolMode := false
    traceAsInput := true
    traceAsMetadata := true
  },
  LangflowInputType.FileInput {
    toInputConfig := {
      name := "files"
      displayName := "Files"
      info := "Files to be sent with the message."
      advanced := true
      required := false
      show := true
      placeholder := ""
    }
    fileTypes := ["csv", "json", "pdf", "txt", "jpg", "png"]
    isList := true
    tempFile := true
  }
]

-- Validate all chat inputs
def chatInputsValid : Bool :=
  chatInputs.all validateInput

-- Theorem: All chat inputs are valid
theorem chat_inputs_valid : chatInputsValid = true := by
  unfold chatInputsValid chatInputs
  simp [List.all, validateInput, getInputName, isRequired]
  decide
  
end WorkflowProofs.LangflowInputTypes
