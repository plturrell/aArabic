import Lake
open Lake DSL

package «workflow_proofs» where
  -- add package configuration options here

lean_lib «WorkflowProofs» where
  -- add library configuration options here

@[default_target]
lean_exe «workflow_proofs» where
  root := `Main
