-- Langflow Component Categories - Complete Formalization
-- All component categories from Langflow documentation

import WorkflowProofs.WorkflowCore

namespace WorkflowProofs.LangflowCategories

open WorkflowProofs

-- Component categories (from Langflow docs)
inductive ComponentCategory where
  | inputs          -- Chat Input, Text Input, File Input, etc.
  | outputs         -- Chat Output, Text Output, etc.
  | models          -- Language Models (OpenAI, Anthropic, Google, Ollama, etc.)
  | agents          -- Agent components, CrewAI, Tool Calling
  | processing      -- Data operations, Text splitters, Transformers
  | dataSource      -- API Request, Database, File loaders
  | files           -- Directory, File operations
  | flowControls    -- If/Else, Conditional Router, Loops
  | llmOperations   -- Batch Run, Model operations
  | utilities       -- Calculator, Parse, Helper functions
  | helpers         -- Memory, Cache, Parse
  | tools           -- Calculator, Search, API tools
  | vectorStores    -- Qdrant, Pinecone, Chroma, Weaviate
  | retrievers      -- Vector search, Similarity search
  | embeddings      -- OpenAI Embeddings, HuggingFace, etc.
  | memories        -- Chat Memory, Conversation Buffer
  | custom          -- User-created custom components
  | legacy          -- Deprecated components
  deriving Repr, DecidableEq, Inhabited

-- Subcategories for more specific grouping
inductive ComponentSubcategory where
  -- Input/Output subcategories
  | chatIO
  | textIO
  | webhookIO
  
  -- Model providers
  | openai
  | anthropic
  | google
  | ollama
  | huggingface
  | cohere
  | groq
  
  -- Processing types
  | textSplitters
  | documentLoaders
  | dataTransformers
  
  -- Data source types
  | apiConnectors
  | databaseConnectors
  | fileLoaders
  
  -- Agent types
  | simpleAgent
  | crewAI
  | toolCalling
  
  -- Utility types
  | parsing
  | calculation
  | stringOperations
  
  | none  -- No specific subcategory
  deriving Repr, DecidableEq, Inhabited

-- Component metadata
structure ComponentMetadata where
  keywords : List String
  tags : List String
  author : Option String
  version : String
  deprecated : Bool
  beta : Bool
  deriving Repr, Inhabited

-- Category information with description
structure CategoryInfo where
  category : ComponentCategory
  displayName : String
  description : String
  icon : String
  componentCount : Nat
  deriving Repr, Inhabited

-- All core component categories with descriptions
def coreCategories : List CategoryInfo := [
  {
    category := ComponentCategory.inputs
    displayName := "Input / Output"
    description := "Components for receiving and sending data"
    icon := "arrow-right-left"
    componentCount := 5
  },
  {
    category := ComponentCategory.processing
    displayName := "Processing"
    description := "Data operations, text processing, transformations"
    icon := "cpu"
    componentCount := 15
  },
  {
    category := ComponentCategory.dataSource
    displayName := "Data Source"
    description := "API requests, file loaders, database connectors"
    icon := "database"
    componentCount := 10
  },
  {
    category := ComponentCategory.files
    displayName := "Files"
    description := "Directory and file operations"
    icon := "folder"
    componentCount := 5
  },
  {
    category := ComponentCategory.flowControls
    displayName := "Flow Controls"
    description := "If/Else, conditional routing, loops"
    icon := "git-branch"
    componentCount := 8
  },
  {
    category := ComponentCategory.llmOperations
    displayName := "LLM Operations"
    description := "Batch run, model operations"
    icon := "brain"
    componentCount := 6
  },
  {
    category := ComponentCategory.models
    displayName := "Models and Agents"
    description := "Language models and agent components"
    icon := "brain-circuit"
    componentCount := 25
  },
  {
    category := ComponentCategory.utilities
    displayName := "Utilities"
    description := "Calculator, helper functions"
    icon := "wrench"
    componentCount := 12
  }
]

-- Get category display name
def getCategoryDisplayName (cat : ComponentCategory) : String :=
  match coreCategories.find? (·.category = cat) with
  | some info => info.displayName
  | none => match cat with
    | ComponentCategory.inputs => "Inputs"
    | ComponentCategory.outputs => "Outputs"
    | ComponentCategory.models => "Models"
    | ComponentCategory.agents => "Agents"
    | ComponentCategory.processing => "Processing"
    | ComponentCategory.dataSource => "Data Source"
    | ComponentCategory.files => "Files"
    | ComponentCategory.flowControls => "Flow Controls"
    | ComponentCategory.llmOperations => "LLM Operations"
    | ComponentCategory.utilities => "Utilities"
    | ComponentCategory.helpers => "Helpers"
    | ComponentCategory.tools => "Tools"
    | ComponentCategory.vectorStores => "Vector Stores"
    | ComponentCategory.retrievers => "Retrievers"
    | ComponentCategory.embeddings => "Embeddings"
    | ComponentCategory.memories => "Memories"
    | ComponentCategory.custom => "Custom"
    | ComponentCategory.legacy => "Legacy"

-- Check if category is a core category
def isCoreCategory (cat : ComponentCategory) : Bool :=
  coreCategories.any (·.category = cat)

-- Check if component should be hidden by default
def shouldHideByDefault (cat : ComponentCategory) (meta : ComponentMetadata) : Bool :=
  cat = ComponentCategory.legacy || meta.deprecated

-- Total count of core components
def totalCoreComponents : Nat :=
  coreCategories.foldl (fun acc info => acc + info.componentCount) 0

-- Validate category info
def validateCategoryInfo (info : CategoryInfo) : Bool :=
  info.displayName ≠ "" &&
  info.description ≠ "" &&
  info.icon ≠ "" &&
  info.componentCount > 0

-- Theorem: All core categories are valid
theorem all_core_categories_valid :
    coreCategories.all validateCategoryInfo = true := by
  unfold coreCategories
  simp [validateCategoryInfo]

-- Theorem: Core categories have expected component count
theorem core_component_count :
    totalCoreComponents = 86 := by
  unfold totalCoreComponents coreCategories
  simp

end WorkflowProofs.LangflowCategories
