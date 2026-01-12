-- Kafka Workflow Formalization and Conversion System
import WorkflowProofs.WorkflowCore
import Lean.Data.Json

namespace WorkflowProofs.Kafka

open WorkflowProofs

-- Kafka Topic Definition
structure KafkaTopic where
  name : String
  partitions : Nat
  replicationFactor : Nat
  config : List (String × String)
  deriving Repr, DecidableEq, Inhabited

-- Kafka Message Schema
inductive KafkaSchemaType where
  | string
  | integer
  | float
  | boolean
  | object (fields : List (String × KafkaSchemaType))
  | array (itemType : KafkaSchemaType)
  | union (types : List KafkaSchemaType)
  deriving Repr, DecidableEq

structure KafkaSchema where
  name : String
  version : Nat
  schemaType : KafkaSchemaType
  deriving Repr, DecidableEq, Inhabited

-- Kafka Producer/Consumer/Stream Processor Nodes
structure KafkaNode where
  id : String
  nodeType : String  -- "producer", "consumer", "stream_processor", "connector"
  config : List (String × String)
  inputTopics : List String
  outputTopics : List String
  processingLogic : Option String
  deriving Repr, DecidableEq, Inhabited

-- Kafka Stream Processing Topology
structure KafkaTopology where
  name : String
  topics : List KafkaTopic
  schemas : List KafkaSchema
  nodes : List KafkaNode
  deriving Repr, Inhabited

-- Convert generic workflow to Kafka topology
def workflowToKafka (wf : Workflow) : KafkaTopology :=
  let topics : List KafkaTopic :=
    wf.nodes.filter (fun n => n.nodeType = "kafka_topic") |>.map fun node =>
      let partitions := node.config.find? (fun (k, _) => k = "partitions")
        |>.map (fun (_, v) => v.toNat?) |>.join |>.getD 1
      let replication := node.config.find? (fun (k, _) => k = "replication_factor")
        |>.map (fun (_, v) => v.toNat?) |>.join |>.getD 1
      { 
        name := node.id
        partitions := partitions
        replicationFactor := replication
        config := node.config.filter (fun (k, _) => 
          k ≠ "partitions" && k ≠ "replication_factor")
      }
  
  let nodes : List KafkaNode :=
    wf.nodes.filter (fun n => n.nodeType ≠ "kafka_topic") |>.map fun node =>
      let inputTopics := wf.edges.filter (fun e => e.target = node.id)
        |>.map (fun e => e.source)
      let outputTopics := wf.edges.filter (fun e => e.source = node.id)
        |>.map (fun e => e.target)
      let nodeType := 
        match node.nodeType with
        | "kafka_producer" => "producer"
        | "kafka_consumer" => "consumer"
        | "kafka_stream" => "stream_processor"
        | _ => "processor"
      
      let processingLogic := node.config.find? (fun (k, _) => k = "processing_logic")
        |>.map Prod.snd
      
      {
        id := node.id
        nodeType := nodeType
        config := node.config.filter (fun (k, _) => k ≠ "processing_logic")
        inputTopics := inputTopics
        outputTopics := outputTopics
        processingLogic := processingLogic
      }
  
  {
    name := wf.name
    topics := topics
    schemas := []
    nodes := nodes
  }

-- Convert Kafka topology to JSON
def kafkaTopologyToJson (kafka : KafkaTopology) : String :=
  let topicsJson := kafka.topics.map fun topic =>
    s!"{{\"name\":\"{topic.name}\",\"partitions\":{topic.partitions},\"replicationFactor\":{topic.replicationFactor}}}"
  
  let nodesJson := kafka.nodes.map fun node =>
    let inputTopicsStr := "[" ++ String.intercalate "," (node.inputTopics.map (fun t => s!"\"{t}\"")) ++ "]"
    let outputTopicsStr := "[" ++ String.intercalate "," (node.outputTopics.map (fun t => s!"\"{t}\"")) ++ "]"
    let logicStr := match node.processingLogic with
      | some logic => s!",\"processingLogic\":\"{logic}\""
      | none => ""
    s!"{{\"id\":\"{node.id}\",\"nodeType\":\"{node.nodeType}\",\"inputTopics\":{inputTopicsStr},\"outputTopics\":{outputTopicsStr}{logicStr}}}"
  
  s!"{{\"name\":\"{kafka.name}\",\"topics\":[{String.intercalate "," topicsJson}],\"nodes\":[{String.intercalate "," nodesJson}]}}"

-- Arabic Training with Kafka Integration
def arabicTrainingKafkaTopology : KafkaTopology :=
  {
    name := "Arabic Training with Kafka Events"
    topics := [
      { name := "workflow.arabic.training.start", partitions := 3, replicationFactor := 1, config := [] },
      { name := "workflow.arabic.training.progress", partitions := 3, replicationFactor := 1, config := [] },
      { name := "workflow.arabic.training.complete", partitions := 3, replicationFactor := 1, config := [] }
    ]
    schemas := []
    nodes := [
      {
        id := "training_producer"
        nodeType := "producer"
        config := [("broker", "aimo_kafka:9092"), ("acks", "all")]
        inputTopics := []
        outputTopics := ["workflow.arabic.training.start"]
        processingLogic := none
      },
      {
        id := "progress_publisher"
        nodeType := "producer"
        config := [("broker", "aimo_kafka:9092")]
        inputTopics := []
        outputTopics := ["workflow.arabic.training.progress"]
        processingLogic := none
      },
      {
        id := "completion_publisher"
        nodeType := "producer"
        config := [("broker", "aimo_kafka:9092")]
        inputTopics := []
        outputTopics := ["workflow.arabic.training.complete"]
        processingLogic := none
      }
    ]
  }

end WorkflowProofs.Kafka
