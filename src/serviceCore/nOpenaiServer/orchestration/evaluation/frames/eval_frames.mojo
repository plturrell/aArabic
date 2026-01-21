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
FRAMES Benchmark Evaluation Script - Pure Mojo Implementation

This module provides evaluation for the FRAMES benchmark using tool
orchestration with enhance_reasoning, answer, and search tools.

Usage:
    mojo run eval_frames.mojo --model_name <model> --output_dir <dir> \
        --model_config <config> --example_file_path <path> --max_rounds <n>
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
alias RETRY_LIMIT: Int = 3
alias SYSTEM_PROMPT: String = "You are good at using tools."

# Model mappings for different tool types
alias MODEL_MAPPING_SEARCH_1: String = "gpt-5"
alias MODEL_MAPPING_SEARCH_2: String = "gpt-5-mini"
alias MODEL_MAPPING_SEARCH_3: String = "Qwen/Qwen3-32B"
alias MODEL_MAPPING_REASONER_1: String = "gpt-5"
alias MODEL_MAPPING_REASONER_2: String = "gpt-5-mini"
alias MODEL_MAPPING_REASONER_3: String = "Qwen/Qwen2.5-Coder-32B-Instruct"
alias MODEL_MAPPING_ANSWER_MATH_1: String = "Qwen/Qwen2.5-Math-72B-Instruct"
alias MODEL_MAPPING_ANSWER_MATH_2: String = "Qwen/Qwen2.5-Math-7B-Instruct"
alias MODEL_MAPPING_ANSWER_1: String = "gpt-5"
alias MODEL_MAPPING_ANSWER_2: String = "gpt-5-mini"
alias MODEL_MAPPING_ANSWER_3: String = "meta-llama/Llama-3.3-70B-Instruct"
alias MODEL_MAPPING_ANSWER_4: String = "Qwen/Qwen3-32B"


# ============================================================================
# Data Models
# ============================================================================

@value
struct ToolPricing:
    """Pricing information for a model."""
    var input_per_million: Float64
    var output_per_million: Float64

    fn __init__(inout self, input_cost: Float64 = 0.0, output_cost: Float64 = 0.0):
        self.input_per_million = input_cost / 1000000.0
        self.output_per_million = output_cost / 1000000.0


@value
struct Example:
    """A single FRAMES evaluation example."""
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
    var all_tool_calls: List[String]

    fn __init__(inout self, example_id: String = "", predicted: String = "",
                ground_truth: String = "", correct: Bool = False):
        self.example_id = example_id
        self.predicted = predicted
        self.ground_truth = ground_truth
        self.correct = correct
        self.all_tool_calls = List[String]()


@value
struct CodeResult:
    """Result of code generation and execution."""
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
    var example_file_path: String
    var max_rounds: Int
    var model_type: String
    var basic_tools: Bool

    fn __init__(inout self):
        self.model_name = "Qwen/Qwen3-8B"
        self.output_dir = "outputs/frames"
        self.model_config_path = "model_configs/serve_frames.json"
        self.example_file_path = "frames.jsonl"
        self.max_rounds = MAX_TURNS
        self.model_type = "Qwen/Qwen3-8B"
        self.basic_tools = False


# ============================================================================
# Tool Pricing Configuration
# ============================================================================

fn get_tool_pricing(model: String) -> ToolPricing:
    """Get pricing for a given model."""
    if model == "gpt-5":
        return ToolPricing(1.25, 10.0)
    elif model == "gpt-5-mini":
        return ToolPricing(0.25, 2.0)
    elif model == "Qwen/Qwen3-32B":
        return ToolPricing(0.8, 0.8)
    elif model == "Qwen/Qwen2.5-Coder-32B-Instruct":
        return ToolPricing(0.8, 0.8)
    elif model == "Qwen/Qwen2.5-Math-72B-Instruct":
        return ToolPricing(0.9, 0.9)
    elif model == "Qwen/Qwen2.5-Math-7B-Instruct":
        return ToolPricing(0.2, 0.2)
    elif model == "meta-llama/Llama-3.3-70B-Instruct":
        return ToolPricing(0.9, 0.9)
    elif model == "Qwen/Qwen3-8B":
        return ToolPricing(0.2, 0.2)
    elif model == "claude-4.1-opus" or model == "claude-opus-4-20250514":
        return ToolPricing(15.0, 75.0)
    elif model == "claude-4.1-sonnet":
        return ToolPricing(3.0, 15.0)
    return ToolPricing(0.5, 0.5)


fn get_model_for_tool(tool_model: String) -> String:
    """Map tool model alias to actual model name."""
    if tool_model == "search-1":
        return MODEL_MAPPING_SEARCH_1
    elif tool_model == "search-2":
        return MODEL_MAPPING_SEARCH_2
    elif tool_model == "search-3":
        return MODEL_MAPPING_SEARCH_3
    elif tool_model == "reasoner-1":
        return MODEL_MAPPING_REASONER_1
    elif tool_model == "reasoner-2":
        return MODEL_MAPPING_REASONER_2
    elif tool_model == "reasoner-3":
        return MODEL_MAPPING_REASONER_3
    elif tool_model == "answer-math-1":
        return MODEL_MAPPING_ANSWER_MATH_1
    elif tool_model == "answer-math-2":
        return MODEL_MAPPING_ANSWER_MATH_2
    elif tool_model == "answer-1":
        return MODEL_MAPPING_ANSWER_1
    elif tool_model == "answer-2":
        return MODEL_MAPPING_ANSWER_2
    elif tool_model == "answer-3":
        return MODEL_MAPPING_ANSWER_3
    elif tool_model == "answer-4":
        return MODEL_MAPPING_ANSWER_4
    return tool_model


# ============================================================================
# Tool Definitions
# ============================================================================

fn get_tool_schemas() -> List[ToolSchema]:
    """Get the tool schemas for FRAMES evaluation."""
    var tools = List[ToolSchema]()

    var enhance_desc = "tool to enhance answer model reasoning. analyze the problem, write code, execute it and return intermidiate results that will help solve the problem"
    var enhance_schema = '{"type": "object", "properties": {"model": {"type": "string", "description": "Choices: reasoner-1, reasoner-2, reasoner-3"}}, "required": ["model"]}'
    tools.append(ToolSchema(
        name="enhance_reasoning",
        description=enhance_desc,
        input_schema=enhance_schema
    ))

    var answer_desc = "give the final answer. Not allowed to call if documents is empty."
    var answer_schema = '{"type": "object", "properties": {"model": {"type": "string", "description": "Choices: answer-1, answer-2, answer-3, answer-4, answer-math-1, answer-math-2"}}, "required": ["model"]}'
    tools.append(ToolSchema(
        name="answer",
        description=answer_desc,
        input_schema=answer_schema
    ))

    var search_desc = "Search for missing information"
    var search_schema = '{"type": "object", "properties": {"model": {"type": "string", "description": "Choices: search-1, search-2, search-3"}}, "required": ["model"]}'
    tools.append(ToolSchema(
        name="search",
        description=search_desc,
        input_schema=search_schema
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


fn parse_example_line(line: String, eid: Int) -> Example:
    """Parse a single JSONL line into an Example."""
    var example = Example()
    example.eid = eid
    example.id = extract_json_field(line, "id")
    example.question = extract_json_field(line, "question")
    example.answer = extract_json_field(line, "answer")
    return example


fn load_examples(file_path: String) raises -> List[Example]:
    """Load examples from a JSONL file."""
    var examples = List[Example]()
    var content = read_file(file_path)
    var lines = content.split("\n")
    var eid = 0
    for line in lines:
        var stripped = line.strip()
        if len(stripped) > 0:
            examples.append(parse_example_line(stripped, eid))
            eid += 1
    return examples



# ============================================================================
# Result Serialization
# ============================================================================

fn result_to_json(result: EvalResult) -> String:
    """Convert an EvalResult to JSON string."""
    var json = "{\n"
    json += '  "example_id": "' + result.example_id + '",\n'
    json += '  "predicted": "' + result.predicted + '",\n'
    json += '  "ground_truth": "' + result.ground_truth + '",\n'
    json += '  "correct": ' + ("true" if result.correct else "false") + ',\n'
    json += '  "all_tool_calls": ['
    for i in range(len(result.all_tool_calls)):
        if i > 0:
            json += ', '
        json += '"' + result.all_tool_calls[i] + '"'
    json += ']\n}'
    return json


fn cut_sequence(seq: String, max_len: Int) -> String:
    """Cut a sequence to approximately max_len tokens (estimate 4 chars per token)."""
    var max_chars = max_len * 4
    if len(seq) <= max_chars:
        return seq
    return seq[len(seq) - max_chars:]


fn build_context_string(doc_list: List[String], code_list: List[CodeResult],
                        attempt_list: List[AttemptResult], max_tokens: Int) -> String:
    """Build context string from documents, code, and attempts."""
    var doc_str = ""
    for i in range(len(doc_list)):
        doc_str += "Doc " + String(i + 1) + ": " + doc_list[i][:4000] + "\n\n"

    var code_str = ""
    for i in range(len(code_list)):
        code_str += "```python\n" + code_list[i].code + "\n```\n\n"
        code_str += "```output\n" + code_list[i].output + "\n```\n\n"

    var attempt_str = ""
    for i in range(len(attempt_list)):
        attempt_str += "Attempt" + String(i + 1) + " answer by "
        attempt_str += attempt_list[i].model + ": " + attempt_list[i].answer + "\n"

    # Cut sequences to fit within token limits
    attempt_str = cut_sequence(attempt_str, 8000)
    var code_attempt_str = cut_sequence(code_str + attempt_str, 12000)

    var context_str = ""
    if len(doc_str) > 0:
        context_str = "Documents:\n" + cut_sequence(doc_str + "\npython code and execution outputs:\n" + code_attempt_str, max_tokens)
    else:
        context_str = code_attempt_str

    return context_str


# ============================================================================
# Tool Execution
# ============================================================================

fn execute_enhance_reasoning(problem: String, context_str: String, model: String,
                             output_dir: String, example_id: String) -> CodeResult:
    """Execute enhance_reasoning tool - generate and run code."""
    var prompt = context_str.strip() + "\n\n"
    prompt += "Question: " + problem + "\n"
    prompt += "Instead of directly answering the question, please write additional python code "
    prompt += "that will give intermidiate results after execution. Wrap the code within "
    prompt += "```python and ```. The code should be self-contained with all the import and initialization."

    var response = get_llm_response(
        model=model,
        messages=prompt,
        temperature=0.2,
        return_raw_response=True,
        max_length=8000
    )

    if len(response.choices) == 0:
        return CodeResult()

    var content = response.choices[0].message.content

    # Extract code from response
    var code_start = content.find("```python")
    if code_start == -1:
        return CodeResult()
    code_start += 9
    var code_end = content.find("```", code_start)
    if code_end == -1:
        return CodeResult()

    var generated_code = content[code_start:code_end].strip()

    # In pure Mojo, we cannot execute Python code directly
    # Return the code with a placeholder execution result
    return CodeResult(code=generated_code, output="[Code execution not available in pure Mojo]")


fn execute_answer(problem: String, context_str: String, model: String,
                  ground_truth: String) -> (String, Bool):
    """Execute answer tool - get final answer and check correctness."""
    var prompt = context_str.strip() + "\n\n" + problem
    prompt += "\n\nTake a deep breath and think hard with high reasoning, "
    prompt += "wrap the thoughts within <think> and </think>, and wrap only the "
    prompt += "exact answer without any explanation within <answer> and </answer>."
    prompt += "\nOutput using the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"

    var response = get_llm_response(
        model=model,
        messages=prompt,
        temperature=1.0,
        return_raw_response=True,
        max_length=40000
    )

    if len(response.choices) == 0:
        return ("", False)

    var content = response.choices[0].message.content

    # Extract answer from response
    var answer_start = content.find("<answer>")
    if answer_start == -1:
        return ("", False)
    answer_start += 8
    var answer_end = content.find("</answer>", answer_start)
    if answer_end == -1:
        return ("", False)

    var pred = content[answer_start:answer_end].strip()

    # Check correctness
    if len(pred) == 0 or len(pred.split(" ")) > 500:
        return (pred, False)

    if pred.lower() == ground_truth.lower():
        return (pred, True)

    # Use LLM to evaluate correctness
    var eval_prompt = "Question: " + problem + "\n\n"
    eval_prompt += "Student answer: " + pred + "\n\n"
    eval_prompt += "Reference answer: " + ground_truth + "\n\n"
    eval_prompt += "Assume that the reference answer is correct. Output <correct>True</correct> "
    eval_prompt += "if the student answer matches the reference answer. Output <correct>False</correct> "
    eval_prompt += "if the student answer does not match the reference answer."

    var eval_response = get_llm_response(model="gpt-5", messages=eval_prompt)
    var eval_content = ""
    if len(eval_response.choices) > 0:
        eval_content = eval_response.choices[0].message.content

    var correct_start = eval_content.find("<correct>")
    if correct_start != -1:
        correct_start += 9
        var correct_end = eval_content.find("</correct>", correct_start)
        if correct_end != -1:
            var result = eval_content[correct_start:correct_end].lower()
            return (pred, result == "true")

    return (pred, False)


fn execute_search(problem: String, context_str: String, model: String) -> List[String]:
    """Execute search tool - generate query and search for information."""
    var prompt = context_str.strip() + "\n\n"
    prompt += "Question: " + problem + "\n"
    prompt += "Instead of directly answering the question, please think hard and write a "
    prompt += "concise query to search Wikipedia. Wrap the query within <query> and </query>."

    var response = get_llm_response(
        model=model,
        messages=prompt,
        temperature=1.0,
        return_raw_response=True,
        max_length=8000
    )

    var contents = List[String]()

    if len(response.choices) == 0:
        return contents

    var content = response.choices[0].message.content

    # Extract query from response
    var query_start = content.find("<query>")
    if query_start == -1:
        return contents
    query_start += 7
    var query_end = content.find("</query>", query_start)
    if query_end == -1:
        return contents

    var query = content[query_start:query_end].strip()

    # In pure Mojo, we cannot make HTTP requests to retrieval service
    # Return placeholder indicating search was attempted
    contents.append("[Search query: " + query + "] - Retrieval not available in pure Mojo")

    return contents



# ============================================================================
# Evaluation Functions
# ============================================================================

fn evaluate_example(example: Example, config: EvalConfig) raises -> EvalResult:
    """Evaluate a single FRAMES example."""
    var result = EvalResult(
        example_id=example.id,
        ground_truth=example.answer
    )

    var doc_list = List[String]()
    var code_list = List[CodeResult]()
    var attempt_list = List[AttemptResult]()
    var used_tools = List[String]()

    var tools = get_tool_schemas()

    for step in range(config.max_rounds):
        var cur_output_dir = config.output_dir + "/step_" + String(step)

        # Build context string
        var context_str = build_context_string(doc_list, code_list, attempt_list, 24000)

        # Prepare available tools (remove search if docs already exist)
        var available_tools = List[ToolSchema]()
        for t in tools:
            if len(doc_list) > 0 and t.name == "search":
                continue
            available_tools.append(t)

        # Remove last used tool if used twice in a row
        if len(used_tools) > 1:
            if used_tools[len(used_tools) - 1] == used_tools[len(used_tools) - 2]:
                var removed_tool = used_tools[len(used_tools) - 1]
                var filtered_tools = List[ToolSchema]()
                for t in available_tools:
                    if t.name != removed_tool:
                        filtered_tools.append(t)
                available_tools = filtered_tools

        # Build chat messages
        var messages = List[ChatMessage]()
        messages.append(ChatMessage(role="system", content=SYSTEM_PROMPT))
        var user_content = "Problem: " + example.question + "\n\n" + context_str + "\n\nChoose an appropriate tool."
        messages.append(ChatMessage(role="user", content=user_content))

        # Get tool selection from model
        var response = get_llm_response(
            model=config.model_name,
            messages=messages,
            temperature=TEMPERATURE,
            return_raw_response=True,
            tools=available_tools,
            max_length=12000
        )

        if len(response.choices) == 0:
            result.all_tool_calls.append("empty_response_step_" + String(step))
            continue

        var choice = response.choices[0]
        if len(choice.tool_calls) == 0:
            result.all_tool_calls.append("no_tool_calls_step_" + String(step))
            continue

        # Process first valid tool call
        var tool_call = choice.tool_calls[0]
        var tool_name = tool_call.name
        var tool_args = tool_call.arguments

        result.all_tool_calls.append(tool_name + ":" + tool_args)
        used_tools.append(tool_name)

        # Extract model from arguments
        var tool_model = extract_json_field(tool_args, "model")
        var expert_model = get_model_for_tool(tool_model)

        # Execute the tool
        if tool_name == "enhance_reasoning":
            var code_result = execute_enhance_reasoning(
                example.question, context_str, expert_model, cur_output_dir, example.id
            )
            if len(code_result.output) > 0:
                code_list.append(code_result)

        elif tool_name == "answer":
            var (pred, correct) = execute_answer(
                example.question, context_str, expert_model, example.answer
            )
            result.predicted = pred
            result.correct = correct
            break  # Finish evaluation after answer

        elif tool_name == "search":
            var search_results = execute_search(example.question, context_str, expert_model)
            for doc in search_results:
                if doc not in doc_list:
                    doc_list.append(doc)

    return result


fn save_result(result: EvalResult, output_dir: String) raises:
    """Save evaluation result to JSON file."""
    var result_path = output_dir + "/" + result.example_id + ".json"
    var json_content = result_to_json(result)
    write_file(result_path, json_content)


fn run_evaluation(config: EvalConfig) raises:
    """Run evaluation on all examples."""
    print("Starting FRAMES evaluation")
    print("  Model: " + config.model_name)
    print("  Output: " + config.output_dir)
    print("  Max rounds: " + String(config.max_rounds))

    # Create output directory
    if not file_exists(config.output_dir):
        mkdir(config.output_dir)

    var answer_cache_dir = config.output_dir + "/answer_cache"
    if not file_exists(answer_cache_dir):
        mkdir(answer_cache_dir)

    # Load examples
    var examples = load_examples(config.example_file_path)
    print("Loaded " + String(len(examples)) + " examples")

    # Evaluate each example
    var correct_count = 0
    for example in examples:
        print("Evaluating example " + example.id + " (" + String(example.eid + 1) + "/" + String(len(examples)) + ")")

        var result = evaluate_example(example, config)
        save_result(result, config.output_dir)

        if result.correct:
            correct_count += 1

        print("  Result: " + ("CORRECT" if result.correct else "INCORRECT"))
        print("  Predicted: " + result.predicted)

    # Print summary
    var accuracy = Float64(correct_count) / Float64(len(examples)) * 100.0
    print("\n=== Evaluation Summary ===")
    print("Total examples: " + String(len(examples)))
    print("Correct: " + String(correct_count))
    print("Accuracy: " + String(accuracy) + "%")


# ============================================================================
# CLI Argument Parsing
# ============================================================================

fn parse_args() raises -> EvalConfig:
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
        elif arg == "--example_file_path" and i + 1 < len(args):
            config.example_file_path = args[i + 1]
            i += 2
        elif arg == "--max_rounds" and i + 1 < len(args):
            # Simple integer parsing
            var val = args[i + 1]
            var num = 0
            for c in val:
                if c >= '0' and c <= '9':
                    num = num * 10 + (ord(c) - ord('0'))
            config.max_rounds = num
            i += 2
        elif arg == "--model_type" and i + 1 < len(args):
            config.model_type = args[i + 1]
            i += 2
        elif arg == "--basic_tools":
            config.basic_tools = True
            i += 1
        elif arg == "--help" or arg == "-h":
            print_usage()
            raise Error("Help requested")
        else:
            i += 1

    return config


fn print_usage():
    """Print usage information."""
    print("FRAMES Benchmark Evaluation - Pure Mojo Implementation")
    print("")
    print("Usage:")
    print("  mojo run eval_frames.mojo [OPTIONS]")
    print("")
    print("Options:")
    print("  --model_name <model>       Model name for tool selection (default: Qwen/Qwen3-8B)")
    print("  --output_dir <dir>         Output directory for results (default: outputs/frames)")
    print("  --model_config <path>      Path to model config JSON (default: model_configs/serve_frames.json)")
    print("  --example_file_path <path> Path to FRAMES JSONL file (default: frames.jsonl)")
    print("  --max_rounds <n>           Maximum evaluation rounds per example (default: 50)")
    print("  --model_type <type>        Model type (default: Qwen/Qwen3-8B)")
    print("  --basic_tools              Use basic tools mode (all tools use same model)")
    print("  --help, -h                 Show this help message")


# ============================================================================
# Main Entry Point
# ============================================================================

fn main() raises:
    """Main entry point for FRAMES evaluation."""
    var config = parse_args()

    print("=" * 60)
    print("FRAMES Benchmark Evaluation - Pure Mojo Implementation")
    print("=" * 60)
    print("")

    # Load model config if specified
    if file_exists(config.model_config_path):
        print("Loading model config from: " + config.model_config_path)
        var config_content = read_file(config.model_config_path)
        print("  Config loaded (" + String(len(config_content)) + " bytes)")
    else:
        print("Warning: Model config not found: " + config.model_config_path)

    # Run evaluation
    run_evaluation(config)

    print("")
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)