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
Mojo replacement for run_frames.py.
Provides CLI commands and full orchestration loop for FRAMES evaluation jobs.
"""

from sys import argv, env
from collections import List
from time import sleep
from tools.toolorchestra.job_control import job_control
from io import write_file, read_file, file_exists, mkdir

alias SERVE_REPEAT: Int = 1
alias MIN_RUN_TIME: Int = 600
alias OUTPUT_DIR: String = "outputs/frames"

alias SERVE_SCRIPT1: String = """#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition batch_block1,interactive
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name EXPERIMENT_NAME
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=slurm_out/EXPERIMENT_NAME.out
#SBATCH --error=slurm_out/EXPERIMENT_NAME.err

set -x

hostname -i
export HF_HOME=cache/huggingface
source ~/.bashrc
conda activate retriever
CUDA_VISIBLE_DEVICES=0,1 python retrieval_wiki.py --port 1401 &
conda activate vllm1
CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve Qwen/Qwen2.5-Math-72B-Instruct --port 1402 --tensor-parallel-size 4 &
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen3-32B --port 1403 --tensor-parallel-size 2

sleep 15000"""

alias SERVE_SCRIPT2: String = """#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition batch_block1,interactive
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name EXPERIMENT_NAME
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=slurm_out/EXPERIMENT_NAME.out
#SBATCH --error=slurm_out/EXPERIMENT_NAME.err

set -x

hostname -i
export HF_HOME=cache/huggingface
source ~/.bashrc
conda activate vllm1
CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen2.5-Math-7B-Instruct --port 1404 &
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.3-70B-Instruct --port 1405 --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template tool_chat_template_llama3.1_json.jinja --tensor-parallel-size 4 &
CUDA_VISIBLE_DEVICES=4 vllm serve checkpoint_dir --enable-auto-tool-choice --tool-call-parser hermes --port 1406 &
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 1407 --tensor-parallel-size 2

sleep 15000"""


fn generate_script(exp_name: String, template: String, ckpt_dir: String) -> String:
    var script = template.replace("EXPERIMENT_NAME", exp_name)
    script = script.replace("checkpoint_dir", ckpt_dir)
    return script


fn read_serve_ip(exp_name: String) raises -> String:
    let path = "slurm_out/" + exp_name + ".out"
    let content = read_file(path)
    let lines = content.split("\n")
    if len(lines) > 0:
        return lines[0].strip()
    raise Error("No IP found in " + path)


fn write_model_config(serve_ips1: List[String], serve_ips2: List[String], ckpt_dir: String) raises:
    mkdir("model_configs")
    var config = "{\n"
    config += '  "retrieval": ['
    for i in range(len(serve_ips1)):
        if i > 0: config += ","
        config += '{"ip_addr": "' + serve_ips1[i] + '", "port": "1401"}'
    config += '],\n  "Qwen/Qwen2.5-Math-72B-Instruct": ['
    for i in range(len(serve_ips1)):
        if i > 0: config += ","
        config += '{"ip_addr": "' + serve_ips1[i] + '", "port": "1402"}'
    config += '],\n  "Qwen/Qwen3-32B": ['
    for i in range(len(serve_ips1)):
        if i > 0: config += ","
        config += '{"ip_addr": "' + serve_ips1[i] + '", "port": "1403"}'
    config += '],\n  "Qwen/Qwen2.5-Math-7B-Instruct": ['
    for i in range(len(serve_ips2)):
        if i > 0: config += ","
        config += '{"ip_addr": "' + serve_ips2[i] + '", "port": "1404"}'
    config += '],\n  "meta-llama/Llama-3.3-70B-Instruct": ['
    for i in range(len(serve_ips2)):
        if i > 0: config += ","
        config += '{"ip_addr": "' + serve_ips2[i] + '", "port": "1405"}'
    config += '],\n  "' + ckpt_dir + '": ['
    for i in range(len(serve_ips2)):
        if i > 0: config += ","
        config += '{"ip_addr": "' + serve_ips2[i] + '", "port": "1406"}'
    config += '],\n  "Qwen/Qwen2.5-Coder-32B-Instruct": ['
    for i in range(len(serve_ips2)):
        if i > 0: config += ","
        config += '{"ip_addr": "' + serve_ips2[i] + '", "port": "1407"}'
    config += '],\n  "vllm_model_config_path": "model_configs/serve_frames.json"\n}'
    write_file("model_configs/serve_frames.json", config)


fn cancel_held_jobs(jobs: List[job_control.JobInfo]) raises:
    for job in jobs:
        if job.reason.strip().lower() == "held)":
            job_control.cancel(job.id)
            sleep(120)


fn generate_and_write_scripts(ckpt_dir: String) raises -> List[String]:
    var serve_collections = List[String]()
    for repeat in range(SERVE_REPEAT):
        let exp1 = "op_1" + String(repeat)
        serve_collections.append(exp1)
        let script1 = generate_script(exp1, SERVE_SCRIPT1, ckpt_dir)
        write_file(exp1 + ".sh", script1)
        let exp2 = "run_" + String(repeat)
        serve_collections.append(exp2)
        let script2 = generate_script(exp2, SERVE_SCRIPT2, ckpt_dir)
        write_file(exp2 + ".sh", script2)
    return serve_collections


fn submit_missing_jobs(serve_collections: List[String], job_names: List[String]) raises:
    for repeat in range(SERVE_REPEAT):
        let exp1 = "op_1" + String(repeat)
        if exp1 not in job_names:
            let out_path = "slurm_out/" + exp1 + ".out"
            if file_exists(out_path):
                _ = job_control.run_command(List("rm", out_path))
            _ = job_control.submit(exp1 + ".sh")
        let exp2 = "run_" + String(repeat)
        if exp2 not in job_names:
            let out_path = "slurm_out/" + exp2 + ".out"
            if file_exists(out_path):
                _ = job_control.run_command(List("rm", out_path))
            _ = job_control.submit(exp2 + ".sh")


fn count_ready_jobs(jobs: List[job_control.JobInfo], serve_collections: List[String]) raises -> Int:
    var count = 0
    for job in jobs:
        if job.name in serve_collections and job.status.strip().lower() == "r":
            let out_path = "slurm_out/" + job.name + ".out"
            if not file_exists(out_path):
                job_control.cancel(job.id)
            elif job.total_time_seconds >= MIN_RUN_TIME:
                count += 1
    return count


fn run_eval_frames(ckpt_dir: String) raises:
    let cur_output_dir = OUTPUT_DIR + "/26"
    var args = List[String]()
    args.append("python")
    args.append("eval_frames.py")
    args.append("--model_name")
    args.append(ckpt_dir)
    args.append("--output_dir")
    args.append(cur_output_dir)
    args.append("--model_config")
    args.append("model_configs/serve_frames.json")
    _ = job_control.run_command(args)


fn orchestrate(ckpt_dir: String) raises:
    let user = env("USER", "none")
    var prev_ips1 = List[String]()
    var prev_ips2 = List[String]()
    print("Starting FRAMES orchestration for checkpoint:", ckpt_dir)
    while True:
        var jobs = job_control.list_jobs(user)
        cancel_held_jobs(jobs)
        let serve_collections = generate_and_write_scripts(ckpt_dir)
        jobs = job_control.list_jobs(user)
        var job_names = List[String]()
        for job in jobs:
            job_names.append(job.name)
        for job in jobs:
            if job.name not in serve_collections and job.name.startswith("op"):
                job_control.cancel(job.id)
        submit_missing_jobs(serve_collections, job_names)
        let ready_count = count_ready_jobs(jobs, serve_collections)
        if ready_count != 2:
            sleep(30)
            continue
        var serve_ips1 = List[String]()
        for repeat in range(SERVE_REPEAT):
            serve_ips1.append(read_serve_ip("op_1" + String(repeat)))
        var serve_ips2 = List[String]()
        for repeat in range(SERVE_REPEAT):
            serve_ips2.append(read_serve_ip("run_" + String(repeat)))
        var change_flag = False
        if file_exists("model_configs/serve_frames.json"):
            let old_config = read_file("model_configs/serve_frames.json")
            if ckpt_dir not in old_config:
                change_flag = True
        if serve_ips1 != prev_ips1 or serve_ips2 != prev_ips2 or change_flag:
            write_model_config(serve_ips1, serve_ips2, ckpt_dir)
            prev_ips1 = serve_ips1
            prev_ips2 = serve_ips2
            print("Updated model config with IPs:", serve_ips1, serve_ips2)
        run_eval_frames(ckpt_dir)
        sleep(30)


fn usage():
    print("run_frames.mojo commands:")
    print("  list <user>")
    print("  submit <script_path>")
    print("  cancel <job_id>")
    print("  orchestrate <ckpt_dir>")


fn list_jobs(user: String) raises:
    let jobs = job_control.list_jobs(user)
    for job in jobs:
        print(job.id, job.name, job.status, job.total_time_seconds, job.reason)


fn submit_script(path: String) raises:
    let result = job_control.submit(path)
    print(result)


fn cancel_job(job_id: String) raises:
    job_control.cancel(job_id)
    print("Canceled", job_id)


fn main() raises:
    let args = argv()
    if len(args) < 2:
        usage()
        return
    let command = args[1]
    if command == "list" and len(args) >= 3:
        list_jobs(args[2])
    elif command == "submit" and len(args) >= 3:
        submit_script(args[2])
    elif command == "cancel" and len(args) >= 3:
        cancel_job(args[2])
    elif command == "orchestrate" and len(args) >= 3:
        orchestrate(args[2])
    else:
        usage()

