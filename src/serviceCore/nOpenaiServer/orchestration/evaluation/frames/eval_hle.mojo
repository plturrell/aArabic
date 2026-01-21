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
HLE Evaluation Script - Pure Mojo Implementation

This module provides evaluation for the HLE (Humanity's Last Exam) benchmark
using tool orchestration with enhance_reasoning, answer, and search tools.

Usage:
    mojo run eval_hle.mojo --model_name <model> --output_dir <dir>
"""

from sys import argv, env
from collections import Dict, List
from time import now, sleep
from io import read_file, write_file, file_exists, mkdir, listdir

from tools.toolorchestra.LLM_CALL import (
    get_llm_response, ChatMessage, ToolSchema, ToolCall, LLMResponse
)


# ============================================================================
# Constants
# ============================================================================

alias MAX_TURNS: Int = 15
alias MAX_TOKENS: Int = 32768
alias TEMPERATURE: Float32 = 1.0
alias CONCURRENT_LIMIT: Int = 100
alias RETRY_LIMIT: Int = 3

alias SYSTEM_PROMPT: String = """You are a helpful assistant that can use tools to solve problems.

Available tools:
1. enhance_reasoning - Use this to think through complex problems step by step
2. answer - Use this to provide your final answer
3. search - Use this to search for relevant information

Always use enhance_reasoning first to break down the problem, then search if needed, and finally use answer to provide your response."""


# ============================================================================
# Data Models
# ============================================================================

@value
struct Example:
    """A single evaluation example."""
    var id: String
    var question: String
    var answer: String
    var subject: String
    var image_path: String

    fn __init__(inout self, id: String = "", question: String = "", answer: String = "", 
                subject: String = "", image_path: String = ""):
        self.id = id
        self.question = question
        self.answer = answer
        self.subject = subject
        self.image_path = image_path


@value
struct EvalResult:
    """Result of evaluating a single example."""
    var example_id: String
    var predicted: String
    var ground_truth: String
    var correct: Bool
    var turns: Int
    var tool_calls: List[String]

    fn __init__(inout self, example_id: String = "", predicted: String = "", 
                ground_truth: String = "", correct: Bool = False, turns: Int = 0):
        self.example_id = example_id
        self.predicted = predicted
        self.ground_truth = ground_truth
        self.correct = correct
        self.turns = turns
        self.tool_calls = List[String]()


@value
struct ModelConfig:
    """Configuration for model endpoints."""
    var ip_addr: String
    var port: String

    fn __init__(inout self, ip_addr: String = "localhost", port: String = "8000"):
        self.ip_addr = ip_addr
        self.port = port


# ============================================================================
# Configuration
# ============================================================================

@value
struct EvalConfig:
    """Evaluation configuration."""
    var model_name: String
    var output_dir: String
    var model_config_path: String
    var max_turns: Int
    var max_tokens: Int
    var temperature: Float32
    var concurrent_limit: Int
    var data_path: String

    fn __init__(inout self):
        self.model_name = "Qwen/Qwen2.5-Math-72B-Instruct"
        self.output_dir = "outputs/hle"
        self.model_config_path = "model_configs/serve2.json"
        self.max_turns = MAX_TURNS
        self.max_tokens = MAX_TOKENS
        self.temperature = TEMPERATURE
        self.concurrent_limit = CONCURRENT_LIMIT
        self.data_path = "data/hle_test.jsonl"


# ============================================================================
# Tool Definitions
# ============================================================================

fn get_tool_schemas() -> List[ToolSchema]:
    """Get the tool schemas for HLE evaluation."""
    var tools = List[ToolSchema]()
    
    tools.append(ToolSchema(
        name="enhance_reasoning",
        description="Use this tool to think through complex problems step by step. Provide your reasoning process.",
        input_schema='{"type": "object", "properties": {"reasoning": {"type": "string", "description": "Your step-by-step reasoning"}}, "required": ["reasoning"]}'
    ))
    
    tools.append(ToolSchema(
        name="answer",
        description="Use this tool to provide your final answer to the question.",
        input_schema='{"type": "object", "properties": {"answer": {"type": "string", "description": "Your final answer"}}, "required": ["answer"]}'
    ))
    
    tools.append(ToolSchema(
        name="search",
        description="Use this tool to search for relevant information.",
        input_schema='{"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}'
    ))
    
    return tools


# ============================================================================
# JSON Parsing Helpers
# ============================================================================

fn extract_json_field(json_str: String, field: String) -> String:
    """Extract a string field from JSON."""
    var search = '"' + field + '":'
    var pos = json_str.find(search)
    if pos == -1:
        return ""
    
    var start = pos + len(search)
    while start < len(json_str) and json_str[start] == ' ':
        start += 1
    
    if start >= len(json_str):
        return ""
    
    if json_str[start] == '"':
        start += 1
        var end = start
        while end < len(json_str) and json_str[end] != '"':
            if json_str[end] == '\\' and end + 1 < len(json_str):
                end += 2
            else:
                end += 1
        return json_str[start:end]
    else:
        # Handle non-string values
        var end = start
        while end < len(json_str) and json_str[end] != ',' and json_str[end] != '}':
            end += 1
        return json_str[start:end].strip()
    
    return ""

