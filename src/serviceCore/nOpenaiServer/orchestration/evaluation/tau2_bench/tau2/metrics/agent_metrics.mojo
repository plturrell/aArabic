# agent_metrics.mojo
# Migrated from agent_metrics.py
# Agent performance metrics tracking

from collections import Dict, List
from utils.utils import get_now

alias MetricType = String

struct AgentMetrics:
    """Tracks performance metrics for an agent"""
    var agent_id: String
    var total_turns: Int
    var successful_turns: Int
    var failed_turns: Int
    var tool_calls_made: Int
    var tool_calls_successful: Int
    var tool_calls_failed: Int
    var total_tokens_used: Int
    var total_time_ms: Int
    var average_response_time_ms: Float32
    var metrics: Dict[String, Float32]
    var timestamps: List[String]
    
    fn __init__(out self):
        self.agent_id = ""
        self.total_turns = 0
        self.successful_turns = 0
        self.failed_turns = 0
        self.tool_calls_made = 0
        self.tool_calls_successful = 0
        self.tool_calls_failed = 0
        self.total_tokens_used = 0
        self.total_time_ms = 0
        self.average_response_time_ms = 0.0
        self.metrics = Dict[String, Float32]()
        self.timestamps = List[String]()
    
    fn __init__(out self, agent_id: String):
        self.agent_id = agent_id
        self.total_turns = 0
        self.successful_turns = 0
        self.failed_turns = 0
        self.tool_calls_made = 0
        self.tool_calls_successful = 0
        self.tool_calls_failed = 0
        self.total_tokens_used = 0
        self.total_time_ms = 0
        self.average_response_time_ms = 0.0
        self.metrics = Dict[String, Float32]()
        self.timestamps = List[String]()
    
    fn record_turn(mut self, success: Bool, duration_ms: Int):
        """Record a turn execution"""
        self.total_turns += 1
        if success:
            self.successful_turns += 1
        else:
            self.failed_turns += 1
        
        self.total_time_ms += duration_ms
        self.average_response_time_ms = Float32(self.total_time_ms) / Float32(self.total_turns)
        self.timestamps.append(get_now())
    
    fn record_tool_call(mut self, success: Bool):
        """Record a tool call"""
        self.tool_calls_made += 1
        if success:
            self.tool_calls_successful += 1
        else:
            self.tool_calls_failed += 1
    
    fn record_tokens(mut self, tokens: Int):
        """Record token usage"""
        self.total_tokens_used += tokens
    
    fn add_metric(mut self, name: String, value: Float32):
        """Add a custom metric"""
        self.metrics[name] = value
    
    fn get_metric(self, name: String) -> Float32:
        """Get a custom metric"""
        if name in self.metrics:
            return self.metrics[name]
        return 0.0
    
    fn get_success_rate(self) -> Float32:
        """Calculate success rate"""
        if self.total_turns == 0:
            return 0.0
        return Float32(self.successful_turns) / Float32(self.total_turns)
    
    fn get_tool_success_rate(self) -> Float32:
        """Calculate tool call success rate"""
        if self.tool_calls_made == 0:
            return 0.0
        return Float32(self.tool_calls_successful) / Float32(self.tool_calls_made)
    
    fn get_average_tokens_per_turn(self) -> Float32:
        """Calculate average tokens per turn"""
        if self.total_turns == 0:
            return 0.0
        return Float32(self.total_tokens_used) / Float32(self.total_turns)
    
    fn reset(mut self):
        """Reset all metrics"""
        self.total_turns = 0
        self.successful_turns = 0
        self.failed_turns = 0
        self.tool_calls_made = 0
        self.tool_calls_successful = 0
        self.tool_calls_failed = 0
        self.total_tokens_used = 0
        self.total_time_ms = 0
        self.average_response_time_ms = 0.0
        self.metrics = Dict[String, Float32]()
        self.timestamps = List[String]()
    
    fn to_summary(self) -> String:
        """Generate a summary string of metrics"""
        var summary = "Agent Metrics Summary for " + self.agent_id + "\n"
        summary = summary + "=" * 50 + "\n"
        summary = summary + "Total Turns: " + str(self.total_turns) + "\n"
        summary = summary + "Successful: " + str(self.successful_turns) + "\n"
        summary = summary + "Failed: " + str(self.failed_turns) + "\n"
        summary = summary + "Success Rate: " + str(self.get_success_rate() * 100.0) + "%\n"
        summary = summary + "\nTool Calls:\n"
        summary = summary + "Total: " + str(self.tool_calls_made) + "\n"
        summary = summary + "Successful: " + str(self.tool_calls_successful) + "\n"
        summary = summary + "Failed: " + str(self.tool_calls_failed) + "\n"
        summary = summary + "Success Rate: " + str(self.get_tool_success_rate() * 100.0) + "%\n"
        summary = summary + "\nPerformance:\n"
        summary = summary + "Total Time: " + str(self.total_time_ms) + "ms\n"
        summary = summary + "Avg Response Time: " + str(self.average_response_time_ms) + "ms\n"
        summary = summary + "Total Tokens: " + str(self.total_tokens_used) + "\n"
        summary = summary + "Avg Tokens/Turn: " + str(self.get_average_tokens_per_turn()) + "\n"
        summary = summary + "=" * 50 + "\n"
        return summary

struct MetricsAggregator:
    """Aggregates metrics from multiple agents"""
    var agent_metrics: Dict[String, AgentMetrics]
    var global_start_time: String
    var global_end_time: String
    
    fn __init__(out self):
        self.agent_metrics = Dict[String, AgentMetrics]()
        self.global_start_time = get_now()
        self.global_end_time = ""
    
    fn add_agent_metrics(mut self, agent_id: String, metrics: AgentMetrics):
        """Add metrics for an agent"""
        self.agent_metrics[agent_id] = metrics
    
    fn get_agent_metrics(self, agent_id: String) -> AgentMetrics:
        """Get metrics for a specific agent"""
        if agent_id in self.agent_metrics:
            return self.agent_metrics[agent_id]
        return AgentMetrics()
    
    fn finalize(mut self):
        """Finalize metrics collection"""
        self.global_end_time = get_now()
    
    fn get_total_turns(self) -> Int:
        """Get total turns across all agents"""
        var total = 0
        for agent_id in self.agent_metrics:
            let metrics = self.agent_metrics[agent_id]
            total += metrics.total_turns
        return total
    
    fn get_average_success_rate(self) -> Float32:
        """Calculate average success rate across agents"""
        if len(self.agent_metrics) == 0:
            return 0.0
        
        var total_rate = 0.0
        for agent_id in self.agent_metrics:
            let metrics = self.agent_metrics[agent_id]
            total_rate += metrics.get_success_rate()
        
        return total_rate / Float32(len(self.agent_metrics))
    
    fn generate_report(self) -> String:
        """Generate a comprehensive report"""
        var report = "Metrics Aggregation Report\n"
        report = report + "=" * 60 + "\n"
        report = report + "Start Time: " + self.global_start_time + "\n"
        report = report + "End Time: " + self.global_end_time + "\n"
        report = report + "Total Agents: " + str(len(self.agent_metrics)) + "\n"
        report = report + "Total Turns: " + str(self.get_total_turns()) + "\n"
        report = report + "Avg Success Rate: " + str(self.get_average_success_rate() * 100.0) + "%\n"
        report = report + "=" * 60 + "\n\n"
        
        # Add individual agent summaries
        for agent_id in self.agent_metrics:
            let metrics = self.agent_metrics[agent_id]
            report = report + metrics.to_summary() + "\n"
        
        return report

fn create_agent_metrics(agent_id: String) -> AgentMetrics:
    """Factory function to create agent metrics"""
    return AgentMetrics(agent_id)
