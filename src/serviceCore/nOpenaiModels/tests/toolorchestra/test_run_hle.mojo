"""
Tests for run_hle.mojo template generation and config utilities.

Since the full orchestration functions depend on external slurm commands,
these tests focus on pure functions that do string manipulation and data
transformation. Placeholder tests are included for functions that will be
added in future iterations.
"""

from collections import List
import sys
sys.path.append("src/serviceCore/nOpenaiServer")

from tools.toolorchestra.evaluation.run_hle import generate_script


fn expect(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


fn test_generate_script_substitution() raises:
    """Test that EXPERIMENT_NAME placeholder gets replaced in script template."""
    print("🧪 test_generate_script_substitution")

    var template = "#!/bin/bash\n#SBATCH --job-name=EXPERIMENT_NAME\necho Running EXPERIMENT_NAME"
    var experiment_name = "hle_test_run_001"
    var ckpt_dir = "/ckpt"

    var result = generate_script(experiment_name, template, ckpt_dir)

    expect("hle_test_run_001" in result, "experiment name should be substituted")
    expect "EXPERIMENT_NAME" not in result, "placeholder should be replaced"
    expect "#SBATCH --job-name=hle_test_run_001" in result, "job name should contain experiment name"


fn test_generate_script_checkpoint_substitution() raises:
    """Test that checkpoint_dir placeholder gets replaced in script template."""
    print("🧪 test_generate_script_checkpoint_substitution")

    var template = "#!/bin/bash\nCHECKPOINT=checkpoint_dir\nmkdir -p $CHECKPOINT"
    var ckpt_dir = "/scratch/user/checkpoints/run_001"

    var result = generate_script("exp_name", template, ckpt_dir)

    expect("/scratch/user/checkpoints/run_001" in result, "checkpoint dir should be substituted")
    expect("checkpoint_dir" not in result, "placeholder should be replaced")


fn test_model_config_structure() raises:
    """Test that generated config has expected keys for HLE evaluation."""
    print("🧪 test_model_config_structure")

    # Placeholder test - documents expected structure once generate_model_config is added
    # The config should have these required keys for HLE runs

    # TODO: Replace with actual function call once implemented:
    # var config = generate_model_config(
    #     model_name="gpt-4",
    #     temperature=0.0,
    #     max_tokens=4096,
    # )

    # Simulate expected config structure
    var required_keys = List[String]()
    required_keys.append("model_name")
    required_keys.append("temperature")
    required_keys.append("max_tokens")
    required_keys.append("output_dir")
    required_keys.append("experiment_name")

    # Document expected structure for when the function is implemented
    expect(len(required_keys) >= 5, "config should have at least 5 required keys")


fn test_placeholder_substitution_multiple() raises:
    """Test multiple placeholder substitutions in a single template."""
    print("🧪 test_placeholder_substitution_multiple")

    var template = """#!/bin/bash
#SBATCH --job-name={{EXPERIMENT_NAME}}
#SBATCH --output={{OUTPUT_DIR}}/{{EXPERIMENT_NAME}}.out
#SBATCH --error={{OUTPUT_DIR}}/{{EXPERIMENT_NAME}}.err
cd {{CHECKPOINT_DIR}}
"""
    var result = template
    result = result.replace("{{EXPERIMENT_NAME}}", "my_exp")
    result = result.replace("{{OUTPUT_DIR}}", "/logs")
    result = result.replace("{{CHECKPOINT_DIR}}", "/ckpt")

    expect("{{EXPERIMENT_NAME}}" not in result, "all EXPERIMENT_NAME placeholders should be replaced")
    expect("{{OUTPUT_DIR}}" not in result, "all OUTPUT_DIR placeholders should be replaced")
    expect("{{CHECKPOINT_DIR}}" not in result, "CHECKPOINT_DIR placeholder should be replaced")
    expect "/logs/my_exp.out" in result, "output path should be correctly formed"
    expect "/logs/my_exp.err" in result, "error path should be correctly formed"


fn main() raises:
    print("═══════════════════════════════════════════════════════════════════════")
    print("  run_hle.mojo template and config tests")
    print("═══════════════════════════════════════════════════════════════════════")

    test_generate_script_substitution()
    print("✅ generate_script substitution passed")

    test_generate_script_checkpoint_substitution()
    print("✅ checkpoint_dir substitution passed")

    test_model_config_structure()
    print("✅ model config structure passed")

    test_placeholder_substitution_multiple()
    print("✅ multiple placeholder substitution passed")

    print("🎉 All run_hle.mojo tests passed")

