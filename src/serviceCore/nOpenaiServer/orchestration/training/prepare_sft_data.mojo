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

from collections import Dict, List
from pathlib import Path
from sys import run
from io import write_file, read_file, file_exists, mkdir


fn get_search_model_mapping() -> Dict[String, String]:
    """Return the search tool model mapping."""
    var mapping = Dict[String, String]()
    mapping["gpt-5"] = "search-1"
    mapping["gpt-5-mini"] = "search-2"
    mapping["Qwen/Qwen3-32B"] = "search-3"
    return mapping


fn get_enhance_reasoning_model_mapping() -> Dict[String, String]:
    """Return the enhance_reasoning tool model mapping."""
    var mapping = Dict[String, String]()
    mapping["gpt-5"] = "reasoner-1"
    mapping["gpt-5-mini"] = "reasoner-2"
    mapping["Qwen/Qwen2.5-Coder-32B-Instruct"] = "reasoner-3"
    return mapping


fn get_answer_model_mapping() -> Dict[String, String]:
    """Return the answer tool model mapping."""
    var mapping = Dict[String, String]()
    mapping["Qwen/Qwen2.5-Math-72B-Instruct"] = "answer-math-1"
    mapping["Qwen/Qwen2.5-Math-7B-Instruct"] = "answer-math-2"
    mapping["gpt-5"] = "answer-1"
    mapping["gpt-5-mini"] = "answer-2"
    mapping["meta-llama/Llama-3.3-70B-Instruct"] = "answer-3"
    mapping["Qwen/Qwen3-32B"] = "answer-4"
    return mapping


fn escape_json_string(s: String) -> String:
    """Escape special characters for JSON string."""
    var result = s.replace("\\", "\\\\")
    result = result.replace('"', '\\"')
    result = result.replace("\n", "\\n")
    result = result.replace("\r", "\\r")
    result = result.replace("\t", "\\t")
    return result


fn create_message_json(role: String, content: String) -> String:
    """Create a JSON object for a message."""
    return '{"role": "' + role + '", "content": "' + escape_json_string(content) + '"}'


fn messages_to_json(messages: List[String]) -> String:
    """Convert list of message JSON strings to a JSON array with indentation."""
    var result = String("[\n")
    for i in range(len(messages)):
        result += "  " + messages[i]
        if i < len(messages) - 1:
            result += ","
        result += "\n"
    result += "]"
    return result


fn extract_json_value(json_str: String, key: String) -> String:
    """Extract a string value from a JSON object by key."""
    var search_key = '"' + key + '":'
    var key_pos = json_str.find(search_key)
    if key_pos == -1:
        return ""

    var value_start = key_pos + len(search_key)
    # Skip whitespace
    while value_start < len(json_str) and json_str[value_start] == ' ':
        value_start += 1

    if value_start >= len(json_str):
        return ""

    # Check if it's a string value
    if json_str[value_start] == '"':
        var str_start = value_start + 1
        var str_end = str_start
        while str_end < len(json_str):
            if json_str[str_end] == '"' and json_str[str_end - 1] != '\\':
                break
            str_end += 1
        return json_str[str_start:str_end]

    # Otherwise extract until comma or closing brace
    var end = value_start
    while end < len(json_str) and json_str[end] != ',' and json_str[end] != '}':
        end += 1
    return json_str[value_start:end].strip()


fn json_contains_key(json_str: String, key: String) -> Bool:
    """Check if a JSON object contains a key."""
    var search_key = '"' + key + '":'
    return json_str.find(search_key) != -1


fn main() raises:
    var task_id = "66f5e796acadd55c11fb11f5"
    var output_path = "example.json"
    var output_dir = "sft_data"
    var hle_path = "evaluation/hle.jsonl"

    # Read HLE examples and build id to example mapping
    var id2example = Dict[String, String]()  # id -> question
    var hle_content = read_file(hle_path)
    var hle_lines = hle_content.split("\n")
    for line in hle_lines:
        if len(line[]) > 0:
            var example_id = extract_json_value(line[], "id")
            var question = extract_json_value(line[], "question")
            if example_id != "":
                id2example[example_id] = question

    # Read results data
    var results_content = read_file(output_path)

    var problem = id2example.get(task_id, "")

    # Build system message
    var system_content = String("You are good at using tools.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"code_interpreter\", \"description\": \"python executor to execute code and return outputs\", \"parameters\": {\"properties\": {\"code\": {\"description\": \"The code to execute\", \"type\": \"string\"}}, \"required\": [\"code\"], \"title\": \"parameters\", \"type\": \"object\"}}}\n{\"type\": \"function\", \"function\": {\"name\": \"search\", \"description\": \"Search for missing information\", \"parameters\": {\"properties\": {\"query\": {\"description\": \"The query used to search missing information\", \"type\": \"string\"}}, \"required\": [\"query\"], \"title\": \"parameters\", \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")

    # Initialize messages list
    var messages = List[String]()
    messages.append(create_message_json("system", system_content))
    messages.append(create_message_json("user", "Problem: " + problem))

    var seen_documents = List[String]()

    # Process tool responses - find turn_N_response in the results
    for i in range(100):
        var turn_key = "turn_" + String(i) + "_response"
        if not json_contains_key(results_content, turn_key):
            continue

        # Extract the section for this turn
        var turn_start = results_content.find('"' + turn_key + '":')
        if turn_start == -1:
            continue

        # Find the model response from all_tool_calls
        var model_response = extract_json_value(results_content, "model_response")

        # Check what type of response this is
        if json_contains_key(results_content[turn_start:], "search_results_data"):
            var query = extract_json_value(results_content[turn_start:], "query")
            var tool_call_content = String('<tool_call>{"name": "search", "arguments": {"query": "') + escape_json_string(query) + String('"}}</tool_call>')
            messages.append(create_message_json("assistant", model_response + tool_call_content))
            messages.append(create_message_json("user", "Search results:\n[search results]"))

        elif json_contains_key(results_content[turn_start:], "generated_code"):
            var code = extract_json_value(results_content[turn_start:], "generated_code")
            var exec_result = extract_json_value(results_content[turn_start:], "exec_result")
            var tool_call_content = String('<tool_call>{"name": "code_interpreter", "arguments": {"code": "') + escape_json_string(code) + String('"}}</tool_call>')
            messages.append(create_message_json("assistant", model_response + tool_call_content))
            messages.append(create_message_json("user", "Execution results:\n" + exec_result))

        elif json_contains_key(results_content[turn_start:], "answer_response"):
            var answer = extract_json_value(results_content[turn_start:], "answer_response")
            messages.append(create_message_json("assistant", model_response + answer))

    # Create output directory if it doesn't exist
    if not file_exists(output_dir):
        mkdir(output_dir)

    # Write SFT data files
    var data_idx = 0
    var i = 3
    while i <= len(messages):
        data_idx += 1
        var subset = List[String]()
        for j in range(i):
            subset.append(messages[j])

        var output_file = output_dir + "/" + String(data_idx) + ".json"
        write_file(output_file, messages_to_json(subset))
        i += 2

    print("Generated " + String(data_idx) + " SFT data files in " + output_dir)

