-- n8n Workflow Formalization
import WorkflowProofs.WorkflowCore
import WorkflowProofs.KafkaFormat

namespace WorkflowProofs.N8n

open WorkflowProofs
open WorkflowProofs.Kafka

-- n8n node structure
structure N8nNode where
  name : String
  nodeType : String
  position : List Nat
  parameters : List (String × String)
  deriving Repr, DecidableEq, Inhabited

-- n8n connection
structure N8nConnection where
  sourceNode : String
  targetNode : String
  connectionType : String  -- "main", "ai_memory", etc.
  sourceIndex : Nat
  targetIndex : Nat
  deriving Repr, DecidableEq, Inhabited

-- n8n workflow
structure N8nWorkflow where
  name : String
  nodes : List N8nNode
  connections : List N8nConnection
  settings : List (String × String)
  deriving Repr, Inhabited

-- Convert generic workflow to n8n format
def workflowToN8n (wf : Workflow) : N8nWorkflow :=
  let nodes : List N8nNode :=
    wf.nodes.map fun node =>
      {
        name := node.id
        nodeType := match node.nodeType with
          | "http_request" => "n8n-nodes-base.httpRequest"
          | "webhook" => "n8n-nodes-base.webhook"
          | "condition" => "n8n-nodes-base.if"
          | "kafka_producer" => "n8n-nodes-base.kafka"
          | "kafka_consumer" => "n8n-nodes-base.kafkaTrigger"
          | t => s!"n8n-nodes-base.{t}"
        position := [node.position.1, node.position.2]
        parameters := node.config
      }
  
  let connections : List N8nConnection :=
    wf.edges.map fun edge =>
      {
        sourceNode := edge.source
        targetNode := edge.target
        connectionType := "main"
        sourceIndex := 0
        targetIndex := 0
      }
  
  {
    name := wf.name
    nodes := nodes
    connections := connections
    settings := []
  }

-- Convert n8n to JSON
def n8nWorkflowToJson (n8n : N8nWorkflow) : String :=
  let nodesJson := n8n.nodes.map fun node =>
    let paramsJson := node.parameters.map fun (k, v) =>
      s!"\"{k}\":\"{v}\""
    let paramsStr := "{" ++ String.intercalate "," paramsJson ++ "}"
    s!"{{\"name\":\"{node.name}\",\"type\":\"{node.nodeType}\",\"position\":[{node.position[0]!},{node.position[1]!}],\"parameters\":{paramsStr}}}"
  
  let connectionsObj := n8n.connections.foldl (init := "") fun acc conn =>
    s!"\"{conn.targetNode}\":{{\"main\":[[{{\"node\":\"{conn.sourceNode}\",\"type\":\"main\",\"index\":0}}]]}}"
  
  s!"{{\"name\":\"{n8n.name}\",\"nodes\":[{String.intercalate "," nodesJson}],\"connections\":{{{connectionsObj}}}}}"

-- Arabic Training n8n Workflow
def arabicTrainingN8nWorkflow : N8nWorkflow :=
  {
    name := "Arabic Translation Training (n8n)"
    nodes := [
      {
        name := "Kafka Trigger"
        nodeType := "n8n-nodes-base.kafkaTrigger"
        position := [100, 100]
        parameters := [
          ("topic", "workflow.arabic.training.start"),
          ("groupId", "n8n-training-consumer"),
          ("clientId", "n8n-client"),
          ("brokers", "aimo_kafka:9092")
        ]
      },
      {
        name := "Process Training Request"
        nodeType := "n8n-nodes-base.function"
        position := [350, 100]
        parameters := [
          ("functionCode", "return items.map(item => ({...item.json, processed: true}))")
        ]
      },
      {
        name := "Publish Progress"
        nodeType := "n8n-nodes-base.kafka"
        position := [600, 100]
        parameters := [
          ("topic", "workflow.arabic.training.progress"),
          ("sendInputData", "true"),
          ("brokers", "aimo_kafka:9092")
        ]
      }
    ]
    connections := [
      { sourceNode := "Kafka Trigger", targetNode := "Process Training Request", 
        connectionType := "main", sourceIndex := 0, targetIndex := 0 },
      { sourceNode := "Process Training Request", targetNode := "Publish Progress",
        connectionType := "main", sourceIndex := 0, targetIndex := 0 }
    ]
    settings := []
  }

end WorkflowProofs.N8n
