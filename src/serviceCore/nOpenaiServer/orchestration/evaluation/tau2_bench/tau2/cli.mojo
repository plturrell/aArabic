# tau2/cli.mojo
# Migrated from tau2/cli.py
# Command-line interface for TAU2-Bench

from sys import argv
from collections import List
from tau2.config import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_ERRORS,
    DEFAULT_SEED,
    DEFAULT_NUM_TRIALS,
)

struct CLIArgs:
    """Command-line arguments for TAU2-Bench."""
    var domain: String
    var max_steps: Int
    var max_errors: Int
    var seed: Int
    var num_trials: Int
    var save_to: String
    var log_level: String
    
    fn __init__(inout self):
        self.domain = ""
        self.max_steps = DEFAULT_MAX_STEPS
        self.max_errors = DEFAULT_MAX_ERRORS
        self.seed = DEFAULT_SEED
        self.num_trials = DEFAULT_NUM_TRIALS
        self.save_to = ""
        self.log_level = "INFO"

fn parse_args() -> CLIArgs:
    """
    Parse command-line arguments.
    
    Returns:
        CLIArgs structure with parsed arguments
    """
    var args = CLIArgs()
    var argv_list = argv()
    
    # TODO: Implement proper argument parsing
    # For now, return defaults
    # Expected args: --domain, --max-steps, --max-errors, --seed, --num-trials, --save-to, --log-level
    
    return args

fn print_help():
    """Print help message for CLI."""
    print("TAU2-Bench - Evaluation Framework")
    print("")
    print("Usage: tau2-bench [OPTIONS]")
    print("")
    print("Options:")
    print("  --domain DOMAIN           Domain to evaluate (required)")
    print("  --max-steps INT           Maximum steps per trial (default: 200)")
    print("  --max-errors INT          Maximum errors allowed (default: 10)")
    print("  --seed INT                Random seed (default: 300)")
    print("  --num-trials INT          Number of trials (default: 1)")
    print("  --save-to PATH            Path to save results")
    print("  --log-level LEVEL         Logging level (default: INFO)")
    print("  --help                    Show this help message")
