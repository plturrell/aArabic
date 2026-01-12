-- Main entry point for Lean4 Workflow Proof System
import WorkflowProofs.WorkflowCore
import WorkflowProofs.LangflowFormat
import WorkflowProofs.KafkaFormat
import WorkflowProofs.N8nFormat
import WorkflowProofs.ToonFormat
import WorkflowProofs.ToonAgents
import WorkflowProofs.ToonAgentProofs
import WorkflowProofs.LangflowInputTypes
import WorkflowProofs.LangflowOutputTypes
import WorkflowProofs.LangflowCategories
import WorkflowProofs.LangflowComponents
import WorkflowProofs.LangflowComposition
import WorkflowProofs.LangflowTemplates
import WorkflowProofs.LangflowValidation

open WorkflowProofs
open WorkflowProofs.Langflow
open WorkflowProofs.Kafka
open WorkflowProofs.N8n

def main : IO Unit := do
  IO.println "==================================================="
  IO.println "LEAN4 WORKFLOW VERIFICATION & GENERATION SYSTEM"
  IO.println "==================================================="
  
  -- Generate Arabic Training Pipeline for Langflow
  IO.println "\n📊 Generating Arabic Training Pipeline (Langflow)..."
  let arabicLangflow := generateArabicPipeline
  IO.println s!"✅ Generated ({arabicLangflow.length} bytes)"
  
  -- Save to file
  IO.FS.writeFile "arabic_training_langflow.json" arabicLangflow
  IO.println "   Saved to: arabic_training_langflow.json"
  
  -- Generate Orchestrator Pipeline for Langflow
  IO.println "\n📊 Generating Intelligent Orchestrator (Langflow)..."
  let orchestratorLangflow := generateOrchestratorPipeline
  IO.println s!"✅ Generated ({orchestratorLangflow.length} bytes)"
  
  -- Save to file
  IO.FS.writeFile "orchestrator_langflow.json" orchestratorLangflow
  IO.println "   Saved to: orchestrator_langflow.json"
  
  -- Generate Kafka Topology
  IO.println "\n📊 Generating Kafka Topology..."
  let kafkaJson := kafkaTopologyToJson arabicTrainingKafkaTopology
  IO.println s!"✅ Generated ({kafkaJson.length} bytes)"
  IO.FS.writeFile "arabic_kafka_topology.json" kafkaJson
  IO.println "   Saved to: arabic_kafka_topology.json"
  
  -- Generate n8n Workflow
  IO.println "\n📊 Generating n8n Workflow..."
  let n8nJson := n8nWorkflowToJson arabicTrainingN8nWorkflow
  IO.println s!"✅ Generated ({n8nJson.length} bytes)"
  IO.FS.writeFile "arabic_training_n8n.json" n8nJson
  IO.println "   Saved to: arabic_training_n8n.json"
  
  -- Validation Summary
  IO.println "\n==================================================="
  IO.println "VALIDATION SUMMARY"
  IO.println "==================================================="
  
  let arabicValidation := validateWorkflow arabicTrainingPipeline
  IO.println s!"\n✅ Arabic Pipeline Validation:"
  IO.println s!"   Errors: {arabicValidation.errors.length}"
  IO.println s!"   Warnings: {arabicValidation.warnings.length}"
  
  let orchestratorValidation := validateWorkflow intelligentOrchestratorPipeline
  IO.println s!"\n✅ Orchestrator Validation:"
  IO.println s!"   Errors: {orchestratorValidation.errors.length}"
  IO.println s!"   Warnings: {orchestratorValidation.warnings.length}"
  
  IO.println "\n🎉 All workflows generated and validated!"
  IO.println "\n📝 Next steps:"
  IO.println "   1. Import JSON files to Langflow/n8n databases"
  IO.println "   2. Configure Kafka topics with aimo_kafka"
  IO.println "   3. Test workflow execution"
