# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Main entry point for tau2-bench evaluation framework.
Manages SLURM job submission, VLLM server orchestration, and evaluation runs.
Migrated from run.py to pure Mojo (no Python imports).
"""

from sys import argv, env
from collections import Dict, List
from io import read_file, write_file, file_exists
from .config import Tau2Config, ServerConfig, ModelConfig, DEFAULT_SERVE_REPEAT, DEFAULT_WAIT_TIME


# ============================================================================
# Constants
# ============================================================================

alias SERVE_REPEAT = 1


# ============================================================================
# SLURM Script Template
# ============================================================================

fn get_serve_script_template() -> String:
    """Return the SLURM batch script template for VLLM serving."""
    return """#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition interactive
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name EXPERIMENT_NAME
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=EXPERIMENT_NAME.out
#SBATCH --error=EXPERIMENT_NAME.err

set -x

hostname -i
source ~/.bashrc
source /lustre/fsw/portfolios/llmservice/users/sdiao/anaconda3/bin/activate vllm1
echo SHIZHE DEBUG HF_HOME: $HF_HOME
echo SHIZHE DEBUG USER_PATH: $USER_PATH
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_20"
CUDA_VISIBLE_DEVICES=0 vllm serve CHECKPOINT_DIR --enable-auto-tool-choice --tool-call-parser hermes --port 1900 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_21"
CUDA_VISIBLE_DEVICES=1 vllm serve CHECKPOINT_DIR --enable-auto-tool-choice --tool-call-parser hermes --port 1901 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_22"
CUDA_VISIBLE_DEVICES=2 vllm serve CHECKPOINT_DIR --enable-auto-tool-choice --tool-call-parser hermes --port 1902 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_23"
CUDA_VISIBLE_DEVICES=3 vllm serve CHECKPOINT_DIR --enable-auto-tool-choice --tool-call-parser hermes --port 1903 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_24"
CUDA_VISIBLE_DEVICES=4,5 vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes --port 1904 --tensor-parallel-size 2 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_25"
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes --port 1905 --tensor-parallel-size 2  &
sleep 15000"""


# ============================================================================
# Job Management Structures
# ============================================================================

@value
struct SlurmJob:
    """Represents a SLURM job."""
    var name: String
    var id: String
    var status: String
    var total_time: Int
    var reason: String
    
    fn __init__(out self, name: String = "", id: String = "", status: String = "", 
                total_time: Int = 0, reason: String = ""):
        self.name = name
        self.id = id
        self.status = status
        self.total_time = total_time
        self.reason = reason
    
    fn is_running(self) -> Bool:
        return self.status.lower() == "r"
    
    fn is_held(self) -> Bool:
        return self.reason.lower().strip() == "held)"
    
    fn is_ready(self, min_time: Int = 600) -> Bool:
        """Check if job has been running long enough."""
        return self.is_running() and self.total_time >= min_time


@value
struct ServeInfo:
    """Information about a running server."""
    var name: String
    var total_time: Int
    var ip_addr: String
    
    fn __init__(out self, name: String = "", total_time: Int = 0, ip: String = ""):
        self.name = name
        self.total_time = total_time
        self.ip_addr = ip


# ============================================================================
# Logging
# ============================================================================

fn log(msg: String):
    """Log a message with timestamp."""
    # In pure Mojo, we use print directly
    # Timestamp formatting would require time module
    print("[tau2-bench] " + msg)


fn log_separator(title: String):
    """Log a separator line with title."""
    print("========== " + title + " ==========")


# ============================================================================
# Time Parsing Utilities
# ============================================================================

fn parse_time_to_seconds(time_str: String) -> Int:
    """Parse SLURM time format to seconds.

    Handles formats like: MM:SS, HH:MM:SS, D-HH:MM:SS
    """
    var total_time: Int = 0

    # Check for day format (D-HH:MM:SS)
    if "-" in time_str:
        return 3600  # Simplified: if days present, assume > 1 hour

    # Split by colons
    var parts = time_str.split(":")
    var num_parts = len(parts)

    if num_parts == 2:
        # MM:SS format
        total_time = atoi(parts[0]) * 60 + atoi(parts[1])
    elif num_parts == 3:
        # HH:MM:SS format
        total_time = atoi(parts[0]) * 3600 + atoi(parts[1]) * 60 + atoi(parts[2])

    return total_time


fn atoi(s: String) -> Int:
    """Convert string to integer (simple implementation)."""
    var result: Int = 0
    var negative = False
    var start = 0

    if len(s) > 0 and s[0] == "-":
        negative = True
        start = 1

    for i in range(start, len(s)):
        var c = ord(s[i])
        if c >= ord("0") and c <= ord("9"):
            result = result * 10 + (c - ord("0"))

    if negative:
        result = -result

    return result


# ============================================================================
# Script Generation
# ============================================================================

fn generate_serve_script(exp_name: String, checkpoint_dir: String) -> String:
    """Generate a SLURM serve script for a given experiment."""
    var script = get_serve_script_template()
    script = script.replace("CHECKPOINT_DIR", checkpoint_dir)
    script = script.replace("EXPERIMENT_NAME", exp_name)
    return script


fn write_serve_scripts(checkpoint_dir: String, serve_repeat: Int = SERVE_REPEAT) -> List[String]:
    """Write serve scripts for all repeat instances.

    Returns list of experiment names.
    """
    var serve_collections = List[String]()

    for repeat in range(serve_repeat):
        var exp_name = "eaa_1" + String(repeat)
        serve_collections.append(exp_name)

        var script = generate_serve_script(exp_name, checkpoint_dir)
        var script_path = exp_name + ".sh"
        write_file(script_path, script)

    log("Generated " + String(serve_repeat) + " serve scripts")
    return serve_collections


# ============================================================================
# JSON Utilities (Simple Implementation)
# ============================================================================

fn build_model_config_json(
    checkpoint_dir: String,
    serve_ips: List[String]
) -> String:
    """Build JSON config for model endpoints."""
    var json = "{\n"

    # Add checkpoint model servers
    json += '  "' + checkpoint_dir + '": [\n'
    for i in range(len(serve_ips)):
        var ip = serve_ips[i]
        for port in range(1900, 1904):
            json += '    {"ip_addr": "' + ip + '", "port": "' + String(port) + '"}'
            if port < 1903 or i < len(serve_ips) - 1:
                json += ","
            json += "\n"
    json += "  ],\n"

    # Add Qwen model servers
    json += '  "Qwen/Qwen3-32B": [\n'
    for i in range(len(serve_ips)):
        var ip = serve_ips[i]
        json += '    {"ip_addr": "' + ip + '", "port": "1904"},\n'
        json += '    {"ip_addr": "' + ip + '", "port": "1905"}'
        if i < len(serve_ips) - 1:
            json += ","
        json += "\n"
    json += "  ],\n"

    json += '  "vllm_model_config_path": "eaa.json"\n'
    json += "}"

    return json


fn read_server_ip_from_output(exp_name: String) -> String:
    """Read server IP from job output file."""
    var output_path = exp_name + ".out"
    if not file_exists(output_path):
        return ""

    var content = read_file(output_path)
    var lines = content.split("\n")
    if len(lines) > 0:
        return lines[0].strip()
    return ""


# ============================================================================
# Command Line Argument Parsing
# ============================================================================

@value
struct Args:
    """Parsed command line arguments."""
    var domain: String
    var help: Bool

    fn __init__(out self):
        self.domain = ""
        self.help = False


fn parse_args() -> Args:
    """Parse command line arguments."""
    var args = Args()
    var cli_args = argv()

    var i = 1
    while i < len(cli_args):
        var arg = cli_args[i]
        if arg == "--domain" and i + 1 < len(cli_args):
            args.domain = cli_args[i + 1]
            i += 2
        elif arg == "--help" or arg == "-h":
            args.help = True
            i += 1
        else:
            i += 1

    return args


fn print_usage():
    """Print usage information."""
    print("Usage: mojo run run.mojo [OPTIONS]")
    print("")
    print("Options:")
    print("  --domain DOMAIN    Domain to evaluate (retail, telecom, airline)")
    print("  --help, -h         Show this help message")


# ============================================================================
# Environment Variable Access
# ============================================================================

fn get_env_var(name: String, default: String = "") -> String:
    """Get environment variable with default value.

    Uses Zig FFI for actual implementation.
    """
    # Placeholder - actual implementation uses Zig FFI mojo_getenv
    return default


fn get_checkpoint_dir() -> String:
    """Get checkpoint directory from CKPT_DIR environment variable."""
    return get_env_var("CKPT_DIR", "")


fn get_repo_path() -> String:
    """Get repository path from REPO_PATH environment variable."""
    return get_env_var("REPO_PATH", "")


fn get_user() -> String:
    """Get current user from USER environment variable."""
    return get_env_var("USER", "")


# ============================================================================
# Task Path Helpers
# ============================================================================

fn get_task_path(repo_path: String, domain: String) -> String:
    """Get task file path for a given domain."""
    if domain == "retail":
        return repo_path + "/data/tau2/domains/retail/tasks.json"
    elif domain == "telecom":
        return repo_path + "/data/tau2/domains/telecom/tasks.json"
    elif domain == "airline":
        return repo_path + "/data/tau2/domains/airline/original_tasks.json"
    else:
        return ""


# ============================================================================
# Evaluation Command Builder
# ============================================================================

fn build_evaluation_command(
    domain: String,
    agent_llm: String,
    user_llm: String,
    task_path: String,
    output_file: String,
    model_config_path: String,
    num_trials: Int = 1,
    max_steps: Int = 200
) -> String:
    """Build the CLI command for tau2 evaluation."""
    var cmd = "mojo tau2/cli.mojo"
    cmd += " --domain " + domain
    cmd += " --agent-llm " + agent_llm
    cmd += " --user-llm " + user_llm
    cmd += " --num-trials " + String(num_trials)
    cmd += " --task_path " + task_path
    cmd += " --max-steps " + String(max_steps)
    cmd += " --output_file " + output_file
    cmd += " --model_config_path " + model_config_path
    cmd += " --use_model_tool"
    return cmd


# ============================================================================
# Placeholder Functions for Process Execution
# ============================================================================

fn run_squeue(user: String) -> List[SlurmJob]:
    """Run squeue command and parse job list.

    Note: Actual implementation requires Zig FFI for subprocess execution.
    This is a placeholder that returns empty list.
    """
    var jobs = List[SlurmJob]()
    # Placeholder - would call: squeue -u <user>
    # Parse output lines and create SlurmJob entries
    return jobs


fn run_scancel(job_id: String):
    """Cancel a SLURM job.

    Note: Actual implementation requires Zig FFI for subprocess execution.
    """
    # Placeholder - would call: scancel <job_id>
    log("Would cancel job: " + job_id)


fn run_sbatch(script_path: String):
    """Submit a SLURM batch job.

    Note: Actual implementation requires Zig FFI for subprocess execution.
    """
    # Placeholder - would call: sbatch <script_path>
    log("Would submit job: " + script_path)


fn run_evaluation(cmd: String):
    """Run evaluation command.

    Note: Actual implementation requires Zig FFI for subprocess execution.
    """
    # Placeholder - would call the command
    log("Would run: " + cmd)


fn remove_file(path: String):
    """Remove a file.

    Note: Actual implementation requires Zig FFI.
    """
    # Placeholder - would call unlink
    pass


fn sleep_seconds(seconds: Int):
    """Sleep for specified number of seconds.

    Note: Actual implementation requires Zig FFI.
    """
    # Placeholder - would call sleep
    pass


# ============================================================================
# Main Loop Logic
# ============================================================================

fn process_held_jobs(jobs: List[SlurmJob]):
    """Cancel any held jobs."""
    for i in range(len(jobs)):
        var job = jobs[i]
        if job.is_held():
            run_scancel(job.id)
            sleep_seconds(120)


fn collect_ready_servers(
    jobs: List[SlurmJob],
    serve_collections: List[String],
    min_time: Int = 600
) -> List[ServeInfo]:
    """Collect information about ready servers."""
    var ready_servers = List[ServeInfo]()

    for i in range(len(jobs)):
        var job = jobs[i]
        # Check if job is in our serve collection
        var in_collection = False
        for j in range(len(serve_collections)):
            if serve_collections[j] == job.name:
                in_collection = True
                break

        if in_collection and job.is_running():
            var out_path = job.name + ".out"
            if not file_exists(out_path):
                run_scancel(job.id)
            elif job.total_time >= min_time:
                log("Server " + job.name + " ready after " + String(job.total_time) + "s")
                var ip = read_server_ip_from_output(job.name)
                ready_servers.append(ServeInfo(job.name, job.total_time, ip))
            else:
                var wait_time = min_time - job.total_time
                log("Server " + job.name + " not ready, waiting " + String(wait_time) + "s...")

    return ready_servers


fn run_all_evaluations(checkpoint_dir: String, repo_path: String, config_path: String):
    """Run evaluations for all domains."""
    var domains = List[String]()
    domains.append("retail")
    domains.append("telecom")
    domains.append("airline")

    for i in range(len(domains)):
        var domain = domains[i]
        log_separator("Starting evaluation: " + domain.upper())

        var task_path = get_task_path(repo_path, domain)
        var output_file = "outputs/" + domain + ".json"

        var cmd = build_evaluation_command(
            domain,
            checkpoint_dir,
            "gpt-5",
            task_path,
            output_file,
            config_path
        )

        run_evaluation(cmd)
        log_separator("Finished " + domain.upper())


# ============================================================================
# Main Entry Point
# ============================================================================

fn main():
    """Main entry point for tau2-bench run orchestrator."""
    log_separator("Starting tau2-bench orchestrator")

    # Parse arguments
    var args = parse_args()

    if args.help:
        print_usage()
        return

    # Get environment variables
    var user = get_user()
    var checkpoint_dir = get_checkpoint_dir()
    var repo_path = get_repo_path()

    if checkpoint_dir == "":
        log("Error: CKPT_DIR environment variable not set")
        return

    log("CKPT_DIR = " + checkpoint_dir)
    log("REPO_PATH = " + repo_path)

    # Track previous serve IPs for change detection
    var prev_serve_ips = List[String]()
    var run_done = True
    var loop_count = 0
    var config_path = "eaa.json"

    # Main orchestration loop
    log_separator("Starting main loop")

    while True:
        loop_count += 1
        log(">>> Loop iteration " + String(loop_count) + " started")

        # Get current jobs
        var jobs = run_squeue(user)
        log("Got " + String(len(jobs)) + " jobs from squeue")

        # Cancel held jobs
        process_held_jobs(jobs)

        # Generate serve scripts
        var serve_collections = write_serve_scripts(checkpoint_dir)

        # Refresh job list
        jobs = run_squeue(user)
        var job_names = List[String]()
        for i in range(len(jobs)):
            job_names.append(jobs[i].name)

        # Cancel jobs not in our collection (but start with "eaa")
        for i in range(len(jobs)):
            var job = jobs[i]
            var in_collection = False
            for j in range(len(serve_collections)):
                if serve_collections[j] == job.name:
                    in_collection = True
                    break
            if not in_collection and job.name.startswith("eaa"):
                run_scancel(job.id)

        # Submit new jobs if needed
        for r in range(SERVE_REPEAT):
            var exp_name = "eaa_1" + String(r)
            var found = False
            for j in range(len(job_names)):
                if job_names[j] == exp_name:
                    found = True
                    break
            if not found:
                log("Submitting new job: " + exp_name)
                var out_path = exp_name + ".out"
                var err_path = exp_name + ".err"
                if file_exists(out_path):
                    remove_file(out_path)
                    remove_file(err_path)
                run_sbatch(exp_name + ".sh")

        # Check for ready servers
        var ready_servers = collect_ready_servers(jobs, serve_collections)

        if len(ready_servers) == 0:
            log("No ready servers yet, waiting 30s...")
            sleep_seconds(30)
            continue

        log("Found " + String(len(ready_servers)) + " ready servers")

        # Collect serve IPs
        var serve_ips = List[String]()
        for i in range(len(ready_servers)):
            serve_ips.append(ready_servers[i].ip_addr)
        log("Collected serve IPs: " + String(len(serve_ips)) + " servers")

        # Check if config changed
        var ips_changed = len(serve_ips) != len(prev_serve_ips)
        if not ips_changed:
            for i in range(len(serve_ips)):
                if serve_ips[i] != prev_serve_ips[i]:
                    ips_changed = True
                    break

        var config_changed = False
        if file_exists(config_path):
            var old_config = read_file(config_path)
            if checkpoint_dir not in old_config:
                config_changed = True

        # Update config if needed
        if ips_changed or config_changed:
            log("Config changed, updating " + config_path + "...")
            prev_serve_ips = serve_ips
            var json = build_model_config_json(checkpoint_dir, serve_ips)
            write_file(config_path, json)
            log(config_path + " updated successfully")

        # Run evaluations
        run_all_evaluations(checkpoint_dir, repo_path, config_path)

        log_separator("Loop iteration complete")

