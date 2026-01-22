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
HLE Basic Evaluation Script - Pure Mojo Implementation

This module provides a basic evaluation script for HLE benchmark using
tool orchestration with enhance_reasoning, answer, and search tools.

Usage:
    mojo run eval_hle_basic.mojo --model_name <model> --output_dir <dir> \\
        --model_config <config> --max_rounds <n> --example_path <path>
"""

from sys import argv, env
from collections import Dict, List
from time import now, sleep
from io import read_file, write_file, file_exists, mkdir

from tools.toolorchestra.LLM_CALL import (
    get_llm_response, ChatMessage, ToolSchema, ToolCall, LLMResponse
)


# ============================================================================
# Constants
# ============================================================================

alias MAX_TURNS: Int = 50
alias MAX_TOKENS: Int = 40000
alias TEMPERATURE: Float32 = 1.0
alias MAX_CODE_LENGTH: Int = 16000
alias MAX_CONTEXT_LENGTH: Int = 24000

alias SYSTEM_PROMPT: String = "You are good at using tools."


# ============================================================================
# Model Mappings and Pricing
# ============================================================================

fn get_model_mapping() -> Dict[String, String]:
    """Get model name mappings."""
    var mapping = Dict[String, String]()
    mapping["search-1"] = "gpt-5"
    mapping["search-2"] = "gpt-5-mini"
    mapping["search-3"] = "Qwen/Qwen3-32B"
    mapping["reasoner-1"] = "gpt-5"
    mapping["reasoner-2"] = "gpt-5-mini"
    mapping["reasoner-3"] = "Qwen/Qwen2.5-Coder-32B-Instruct"
    mapping["answer-math-1"] = "Qwen/Qwen2.5-Math-72B-Instruct"
    mapping["answer-math-2"] = "Qwen/Qwen2.5-Math-7B-Instruct"
    mapping["answer-1"] = "gpt-5"
    mapping["answer-2"] = "gpt-5-mini"
    mapping["answer-3"] = "meta-llama/Llama-3.3-70B-Instruct"
    mapping["answer-4"] = "Qwen/Qwen3-32B"
    return mapping


# ============================================================================
# Data Models
# ============================================================================

@value
struct Example:
    """A single evaluation example."""
    var id: String
    var eid: Int
    var question: String
    var answer: String

    fn __init__(inout self, id: String = "", eid: Int = 0,
                question: String = "", answer: String = ""):
        self.id = id
        self.eid = eid
        self.question = question
        self.answer = answer


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
struct CodeResult:
    """Result of code execution."""
    var code: String
    var output: String

    fn __init__(inout self, code: String = "", output: String = ""):
        self.code = code
        self.output = output


@value
struct AttemptResult:
    """Result of an answer attempt."""
    var model: String
    var answer: String

    fn __init__(inout self, model: String = "", answer: String = ""):
        self.model = model
        self.answer = answer


@value
struct EvalConfig:
    """Evaluation configuration."""
    var model_name: String
    var output_dir: String
    var model_config_path: String
    var max_rounds: Int
    var example_path: String
    var model_type: String

    fn __init__(inout self):
        self.model_name = "Qwen/Qwen3-8B"
        self.output_dir = "outputs/hle_basic"
        self.model_config_path = "model_configs/serve.json"
        self.max_rounds = MAX_TURNS
        self.example_path = ""
        self.model_type = "Qwen/Qwen3-8B"


# ============================================================================
# Tool Definitions
# ============================================================================

fn get_tool_schemas() -> List[ToolSchema]:
    """Get the tool schemas for HLE evaluation."""
    var tools = List[ToolSchema]()

    # enhance_reasoning tool
    tools.append(ToolSchema(
        name="enhance_reasoning",
        description="Use this tool to generate Python code for reasoning through complex problems. The code will be executed to get intermediate results.",
        input_schema='{"type": "object", "properties": {"model": {"type": "string", "enum": ["reasoner-1", "reasoner-2", "reasoner-3"]}}, "required": ["model"]}'
    ))

    # answer tool
    tools.append(ToolSchema(
        name="answer",
        description="Use this tool to provide the final answer to the question.",
        input_schema='{"type": "object", "properties": {"model": {"type": "string", "enum": ["answer-math-1", "answer-math-2", "answer-1", "answer-2", "answer-3", "answer-4"]}}, "required": ["model"]}'
    ))

    # search tool
    tools.append(ToolSchema(
        name="search",
        description="Use this tool to search for relevant information.",
        input_schema='{"type": "object", "properties": {"model": {"type": "string", "enum": ["search-1", "search-2", "search-3"]}}, "required": ["model"]}'
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
        var end = start
        while end < len(json_str) and json_str[end] != ',' and json_str[end] != '}':
            end += 1
        return json_str[start:end].strip()

    return ""


fn parse_example_from_json(json_line: String, eid: Int) -> Example:
    """Parse an Example from a JSON line."""
    var id_str = extract_json_field(json_line, "id")
    var question = extract_json_field(json_line, "question")
    var answer = extract_json_field(json_line, "answer")
    return Example(id=id_str, eid=eid, question=question, answer=answer)


fn result_to_json(result: EvalResult) -> String:
    """Convert EvalResult to JSON string."""
    var tool_calls_json = "["
    for i in range(len(result.tool_calls)):
        if i > 0:
            tool_calls_json += ", "
        tool_calls_json += '"' + result.tool_calls[i] + '"'
    tool_calls_json += "]"

    var correct_str = "false"
    if result.correct:
        correct_str = "true"

    return (
        '{"example_id": "' + result.example_id + '", '
        '"predicted": "' + result.predicted + '", '
        '"ground_truth": "' + result.ground_truth + '", '
        '"correct": ' + correct_str + ', '
        '"turns": ' + String(result.turns) + ', '
        '"all_tool_calls": ' + tool_calls_json + '}'
    )


# ============================================================================
# Text Processing
# ============================================================================

fn truncate_text(text: String, max_chars: Int) -> String:
    """Truncate text to max_chars (simple character-based truncation)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


fn build_context_str(
    doc_list: List[String],
    code_list: List[CodeResult],
    attempt_list: List[AttemptResult],
    max_length: Int
) -> String:
    """Build context string from documents, code, and attempts."""
    var doc_str = String("")
    for i in range(len(doc_list)):
        doc_str += "Doc " + String(i + 1) + ": " + truncate_text(doc_list[i], 1200) + " ...\n\n"

    var code_str = String("")
    for i in range(len(code_list)):
        code_str += "```python\n" + code_list[i].code + "\n```\n\n"
        code_str += "```output\n" + code_list[i].output + "\n```\n\n"

    var attempt_str = String("")
    for i in range(len(attempt_list)):
        attempt_str += "Attempt" + String(i + 1) + " answer by " + attempt_list[i].model + ": " + attempt_list[i].answer + "\n"

    var result = String("")
    if len(doc_str) > 0:
        result += "Documents:\n" + doc_str
    if len(code_str) > 0:
        result += "\npython code and execution outputs:\n" + code_str
    if len(attempt_str) > 0:
        result += "\n" + attempt_str

    return truncate_text(result, max_length)


# ============================================================================
# Tool Execution
# ============================================================================

fn execute_enhance_reasoning(
    context_str: String,
    problem: String,
    model_name: String,
    config: EvalConfig
) -> CodeResult:
    """Execute the enhance_reasoning tool."""
    var prompt = context_str.strip() + "\n\n"
    prompt += "Question: " + problem + "\n"
    prompt += "Instead of directly answering the question, please write additional python code that will give intermidiate results after execution. Wrap the code within ```python and ```. The code should be self-contained with all the import and initialization."

    var messages = List[ChatMessage]()
    messages.append(ChatMessage(role="user", content=prompt))

    var response = get_llm_response(
        model=model_name,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )

    if len(response.content) == 0:
        return CodeResult()

    # Extract code from response
    var content = response.content
    var code_start = content.find("```python")
    if code_start == -1:
        return CodeResult()

    code_start += len("```python")
    var code_end = content.find("```", code_start)
    if code_end == -1:
        return CodeResult()

    var generated_code = content[code_start:code_end].strip()

    # Note: In pure Mojo, we cannot execute Python code
    # This would require external execution or native Mojo code
    return CodeResult(code=generated_code, output="[Code execution not supported in pure Mojo]")


fn execute_answer(
    context_str: String,
    problem: String,
    model_name: String,
    ground_truth: String,
    config: EvalConfig
) raises -> Tuple[String, Bool]:
    """Execute the answer tool and return (prediction, correctness)."""
    var prompt = context_str.strip() + "\n\nProblem:\n" + problem

    var messages = List[ChatMessage]()

    # Different prompts for different model types
    if "qwen" in model_name.lower():
        messages.append(ChatMessage(role="system", content="Please reason step by step, and put your final answer within \\boxed{}."))
        messages.append(ChatMessage(role="user", content=prompt))
    elif "gpt" in model_name.lower():
        prompt += "\n\nTake a deep breath and think hard with high reasoning, wrap the thoughts within <think> and </think>, and wrap only the exact answer without any explanation within <answer> and </answer>."
        prompt += "\nOutput using the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
        messages.append(ChatMessage(role="user", content=prompt))
    else:
        prompt += "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
        messages.append(ChatMessage(role="user", content=prompt))

    var response = get_llm_response(
        model=model_name,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.2
    )

    if len(response.content) == 0:
        return ("", False)

    var response_str = response.content
    var pred = String("")

    # Extract prediction based on format
    if "\\boxed{" in response_str:
        var box_start = response_str.find("\\boxed{")
        if box_start != -1:
            box_start += len("\\boxed{")
            var brace_count = 1
            var box_end = box_start
            while box_end < len(response_str) and brace_count > 0:
                if response_str[box_end] == '{':
                    brace_count += 1
                elif response_str[box_end] == '}':
                    brace_count -= 1
                box_end += 1
            if brace_count == 0:
                pred = response_str[box_start:box_end - 1]
    elif "<answer>" in response_str:
        var ans_start = response_str.find("<answer>")
        if ans_start != -1:
            ans_start += len("<answer>")
            var ans_end = response_str.find("</answer>", ans_start)
            if ans_end != -1:
                pred = response_str[ans_start:ans_end].strip()

    # Check correctness
    var correct = False
    if len(pred.strip()) > 0 and len(pred.split(" ")) <= 500:
        if pred.strip().lower() == ground_truth.strip().lower():
            correct = True
        else:
            # Use LLM to evaluate
            var eval_prompt = "Question: " + problem + "\n\n"
            eval_prompt += "Student answer: " + pred + "\n\n"
            eval_prompt += "Reference answer: " + ground_truth + "\n\n"
            eval_prompt += "Assume that the reference answer is correct. Output <correct>True</correct> if the student answer matches the reference answer. Output <correct>False</correct> if the student answer does not match the reference answer."

            var eval_messages = List[ChatMessage]()
            eval_messages.append(ChatMessage(role="user", content=eval_prompt))

            var eval_response = get_llm_response(
                model="gpt-5",
                messages=eval_messages,
                max_tokens=100,
                temperature=1.0
            )

            if "<correct>True</correct>" in eval_response.content.lower() or "<correct>true</correct>" in eval_response.content:
                correct = True

    return (pred.strip(), correct)


fn execute_search(
    context_str: String,
    problem: String,
    model_name: String,
    config: EvalConfig
) -> List[String]:
    """Execute the search tool and return search results."""
    var prompt = context_str.strip() + "\n\n"
    prompt += "Question: " + problem + "\n"
    prompt += "Instead of directly answering the question, please write a query to search for a piece of relevant and missing information. The query should be a few key words about the information to search or a short sentence. Wrap the query within <query> and </query>."

    var messages = List[ChatMessage]()
    messages.append(ChatMessage(role="user", content=prompt))

    var response = get_llm_response(
        model=model_name,
        messages=messages,
        max_tokens=8000,
        temperature=0.2
    )

    # Extract query
    var query = String("")
    if len(response.content) > 0:
        var query_start = response.content.find("<query>")
        if query_start != -1:
            query_start += len("<query>")
            var query_end = response.content.find("</query>", query_start)
            if query_end != -1:
                query = response.content[query_start:query_end]

    # Note: Actual search functionality requires HTTP client
    # Return placeholder for now
    var results = List[String]()
    if len(query) >= 5:
        results.append("[Search results for: " + query + "]")
    return results



# ============================================================================
# Evaluation Functions
# ============================================================================

fn evaluate_example(example: Example, config: EvalConfig) raises -> EvalResult:
    """Evaluate a single example through the tool orchestration loop."""
    var result = EvalResult(
        example_id=example.id,
        ground_truth=example.answer
    )

    # Check if result already exists
    var result_path = config.output_dir + "/" + example.id + ".json"
    if file_exists(result_path):
        return result

    # Initialize state
    var doc_list = List[String]()
    var code_list = List[CodeResult]()
    var attempt_list = List[AttemptResult]()
    var used_tools = List[String]()
    var final_correct = False
    var final_pred = String("")

    var tools = get_tool_schemas()
    var model_mapping = get_model_mapping()

    # Main evaluation loop
    for step in range(config.max_rounds):
        result.turns = step + 1

        # Ensure output directory exists
        var step_dir = config.output_dir + "/step_" + String(step)
        if not file_exists(step_dir):
            mkdir(step_dir)

        # Build context
        var context_str = build_context_str(doc_list, code_list, attempt_list, MAX_CONTEXT_LENGTH)

        # Create chat message for tool selection
        var chat = List[ChatMessage]()
        chat.append(ChatMessage(role="system", content=SYSTEM_PROMPT))
        chat.append(ChatMessage(role="user", content="Problem: " + example.question + "\n\n" + context_str + "\n\nChoose an appropriate tool."))

        # Get LLM response with tools
        var response = get_llm_response(
            model=config.model_name,
            messages=chat,
            tools=tools,
            max_tokens=12000,
            temperature=TEMPERATURE
        )

        # Check if we got tool calls
        if len(response.tool_calls) == 0:
            result.tool_calls.append("No tool calls returned")
            continue

        # Process first valid tool call
        var tool_call = response.tool_calls[0]
        var tool_name = tool_call.name
        result.tool_calls.append(tool_name)
        used_tools.append(tool_name)

        # Get expert model from arguments (default to config model)
        var expert_model = config.model_name
        var model_arg = extract_json_field(tool_call.arguments, "model")
        if len(model_arg) > 0 and model_arg in model_mapping:
            expert_model = model_mapping[model_arg]

        # Execute the tool
        if tool_name == "enhance_reasoning":
            var code_result = execute_enhance_reasoning(
                context_str, example.question, expert_model, config
            )
            if len(code_result.output) > 0:
                code_list.append(code_result)

        elif tool_name == "answer":
            var answer_result = execute_answer(
                context_str, example.question, expert_model, example.answer, config
            )
            final_pred = answer_result[0]
            final_correct = answer_result[1]
            result.predicted = final_pred
            result.correct = final_correct
            break

        elif tool_name == "search":
            var search_results = execute_search(
                context_str, example.question, expert_model, config
            )
            for i in range(len(search_results)):
                if search_results[i] not in doc_list:
                    doc_list.append(search_results[i])

    # Save result
    var result_json = result_to_json(result)
    write_file(result_path, result_json)

    return result


# ============================================================================
# Argument Parsing
# ============================================================================

fn parse_args() -> EvalConfig:
    """Parse command line arguments."""
    var config = EvalConfig()
    var args = argv()

    var i = 1
    while i < len(args):
        var arg = args[i]
        if arg == "--model_name" and i + 1 < len(args):
            config.model_name = args[i + 1]
            i += 2
        elif arg == "--output_dir" and i + 1 < len(args):
            config.output_dir = args[i + 1]
            i += 2
        elif arg == "--model_config" and i + 1 < len(args):
            config.model_config_path = args[i + 1]
            i += 2
        elif arg == "--max_rounds" and i + 1 < len(args):
            config.max_rounds = atol(args[i + 1])
            i += 2
        elif arg == "--model_type" and i + 1 < len(args):
            config.model_type = args[i + 1]
            i += 2
        elif arg == "--example_path" and i + 1 < len(args):
            config.example_path = args[i + 1]
            i += 2
        else:
            i += 1

    return config


# ============================================================================
# Main Entry Point
# ============================================================================

fn main() raises:
    """Main entry point for HLE basic evaluation."""
    print("HLE Basic Evaluation - Pure Mojo Implementation")
    print("=" * 50)

    # Parse configuration
    var config = parse_args()

    print("Model: " + config.model_name)
    print("Output Dir: " + config.output_dir)
    print("Max Rounds: " + String(config.max_rounds))
    print("Example Path: " + config.example_path)
    print("")

    # Create output directories
    if not file_exists(config.output_dir):
        mkdir(config.output_dir)

    var cache_dir = config.output_dir + "/answer_cache"
    if not file_exists(cache_dir):
        mkdir(cache_dir)

    # Load examples
    if len(config.example_path) == 0:
        print("Error: --example_path is required")
        return

    var content = read_file(config.example_path)
    var lines = content.split("\n")
    var examples = List[Example]()

    for eid in range(len(lines)):
        var line = lines[eid].strip()
        if len(line) > 0:
            var example = parse_example_from_json(line, eid)
            examples.append(example)

    print("Loaded " + String(len(examples)) + " examples")
    print("")

    # Evaluate examples
    var correct_count = 0
    var total_count = 0

    for i in range(len(examples)):
        var example = examples[i]
        print("Evaluating example " + String(i + 1) + "/" + String(len(examples)) + ": " + example.id)

        var result = evaluate_example(example, config)
        total_count += 1
        if result.correct:
            correct_count += 1

        print("  Predicted: " + truncate_text(result.predicted, 50))
        print("  Correct: " + String(result.correct))
        print("")

    # Print summary
    print("=" * 50)
    print("Evaluation Complete")
    print("Total: " + String(total_count))
    print("Correct: " + String(correct_count))
    if total_count > 0:
        var accuracy = Float32(correct_count) / Float32(total_count) * 100.0
        print("Accuracy: " + String(accuracy) + "%")
    print("Results saved to: " + config.output_dir)