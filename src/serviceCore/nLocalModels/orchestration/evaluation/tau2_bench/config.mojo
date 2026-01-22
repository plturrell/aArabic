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
Configuration module for tau2-bench.
Provides configuration structures and utilities for the evaluation framework.
Migrated from config.py to pure Mojo.
"""

from collections import Dict, List


# ============================================================================
# Configuration Structures
# ============================================================================

@value
struct Tau2Config:
    """Main configuration for tau2-bench evaluation."""
    var domain: String
    var agent_llm: String
    var user_llm: String
    var num_trials: Int
    var max_steps: Int
    var output_file: String
    var model_config_path: String
    var use_model_tool: Bool
    var task_path: String
    
    fn __init__(out self):
        self.domain = "retail"
        self.agent_llm = ""
        self.user_llm = "gpt-5"
        self.num_trials = 1
        self.max_steps = 200
        self.output_file = "outputs/results.json"
        self.model_config_path = "eaa.json"
        self.use_model_tool = True
        self.task_path = ""
    
    fn with_domain(self, domain: String) -> Tau2Config:
        var config = self
        config.domain = domain
        return config
    
    fn with_agent_llm(self, llm: String) -> Tau2Config:
        var config = self
        config.agent_llm = llm
        return config
    
    fn with_output_file(self, path: String) -> Tau2Config:
        var config = self
        config.output_file = path
        return config
    
    fn with_task_path(self, path: String) -> Tau2Config:
        var config = self
        config.task_path = path
        return config


@value
struct ServerConfig:
    """Configuration for a VLLM server instance."""
    var ip_addr: String
    var port: String
    
    fn __init__(out self, ip: String = "", port: String = ""):
        self.ip_addr = ip
        self.port = port
    
    fn to_url(self) -> String:
        return "http://" + self.ip_addr + ":" + self.port


@value
struct ModelConfig:
    """Configuration for model endpoints."""
    var servers: List[ServerConfig]
    var model_name: String
    
    fn __init__(out self, name: String = ""):
        self.servers = List[ServerConfig]()
        self.model_name = name
    
    fn add_server(mut self, ip: String, port: String):
        self.servers.append(ServerConfig(ip, port))
    
    fn server_count(self) -> Int:
        return len(self.servers)


# ============================================================================
# Default Constants
# ============================================================================

alias DEFAULT_SERVE_REPEAT = 1
alias DEFAULT_WAIT_TIME = 600  # seconds
alias DEFAULT_LOOP_SLEEP = 30  # seconds
alias DEFAULT_VLLM_PORTS = List[String]("1900", "1901", "1902", "1903", "1904", "1905")


# ============================================================================
# Helper Functions
# ============================================================================

fn get_default_config() -> Tau2Config:
    """Get default tau2 configuration."""
    return Tau2Config()


fn get_domains() -> List[String]:
    """Get list of available domains."""
    var domains = List[String]()
    domains.append("retail")
    domains.append("telecom")
    domains.append("airline")
    return domains

