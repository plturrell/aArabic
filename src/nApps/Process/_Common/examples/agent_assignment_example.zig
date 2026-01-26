const std = @import("std");

/// Complete Example: Agent Assignment with Hungarian Algorithm + AI
/// Demonstrates integration of:
/// 1. n-c-sdk ProcessEngine and AgentLogic
/// 2. Hungarian algorithm for optimal assignment
/// 3. nLocalModels AI for intelligent recommendations
/// 4. nAgentFlow for workflow orchestration

const ProcessEngine = @import("n-c-sdk").process.ProcessEngine;
const AgentLogic = @import("n-c-sdk").process.AgentLogic;
const Hungarian = @import("n-c-sdk").process.Hungarian;
const AgentFlowBridge = @import("n-c-sdk").process.AgentFlowBridge;
const IntelligentAgentSelector = @import("../backend/ai/intelligent_agent_selector.zig").IntelligentAgentSelector;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("=== Agent Assignment Example ===\n\n", .{});
    
    // =========================================================================
    // Step 1: Initialize Agent Logic with agents
    // =========================================================================
    
    var agent_logic = AgentLogic.init(allocator);
    defer agent_logic.deinit();
    
    // Register agents
    const agents = [_]AgentLogic.Agent{
        .{
            .id = "checker-001",
            .name = "Alice Chen",
            .role = "checker",
            .capabilities = &[_][]const u8{ "accounting", "audit", "compliance" },
            .capacity = 5,
            .current_load = 2,
            .availability = .available,
            .performance_score = 0.92,
        },
        .{
            .id = "checker-002",
            .name = "Bob Smith",
            .role = "checker",
            .capabilities = &[_][]const u8{ "accounting", "tax" },
            .capacity = 5,
            .current_load = 4,
            .availability = .available,
            .performance_score = 0.85,
        },
        .{
            .id = "manager-001",
            .name = "Carol Wang",
            .role = "manager",
            .capabilities = &[_][]const u8{ "accounting", "audit", "compliance", "management" },
            .capacity = 10,
            .current_load = 6,
            .availability = .available,
            .performance_score = 0.95,
        },
    };
    
    for (agents) |agent| {
        try agent_logic.registerAgent(agent);
    }
    
    std.debug.print("✓ Registered {d} agents\n", .{agents.len});
    
    // =========================================================================
    // Step 2: Submit tasks for assignment
    // =========================================================================
    
    const tasks = [_]AgentLogic.Task{
        .{
            .id = "task-001",
            .name = "Review Trial Balance Entry",
            .required_capability = "accounting",
            .priority = 8,
            .estimated_effort = 30,
            .deadline = null,
            .created_at = std.time.timestamp(),
        },
        .{
            .id = "task-002",
            .name = "Audit Compliance Check",
            .required_capability = "compliance",
            .priority = 9,
            .estimated_effort = 45,
            .deadline = std.time.timestamp() + 3600, // 1 hour deadline
            .created_at = std.time.timestamp(),
        },
        .{
            .id = "task-003",
            .name = "Final Approval",
            .required_capability = "management",
            .priority = 7,
            .estimated_effort = 15,
            .deadline = null,
            .created_at = std.time.timestamp(),
        },
    };
    
    for (tasks) |task| {
        try agent_logic.submitTask(task);
    }
    
    std.debug.print("✓ Submitted {d} tasks\n\n", .{tasks.len});
    
    // =========================================================================
    // Step 3: Hungarian Algorithm - Optimal Assignment
    // =========================================================================
    
    std.debug.print("--- Method 1: Hungarian Algorithm ---\n", .{});
    
    const assignments = try agent_logic.assignTasks();
    defer allocator.free(assignments);
    
    for (assignments) |assignment| {
        std.debug.print("Task {s} → Agent {s} (score: {d:.2})\n", .{
            assignment.task_id,
            assignment.agent_id,
            assignment.score,
        });
        std.debug.print("  Reason: {s}\n", .{assignment.reason});
    }
    
    // =========================================================================
    // Step 4: AI-Enhanced Selection (using nLocalModels)
    // =========================================================================
    
    std.debug.print("\n--- Method 2: AI-Enhanced Selection ---\n", .{});
    
    var ai_selector = try IntelligentAgentSelector.init(
        allocator,
        "http://localhost:8000",
        "qwen-2.5-7b",
    );
    defer ai_selector.deinit();
    
    // Create task context for AI
    const ai_task = IntelligentAgentSelector.TaskContext{
        .id = "task-004",
        .name = "Complex Reconciliation",
        .task_type = "reconciliation",
        .priority = 9,
        .complexity = "high",
        .estimated_effort = 60,
        .required_skills = &[_][]const u8{ "accounting", "audit", "analytics" },
        .historical_data = null,
    };
    
    // Create agent contexts
    const agent_contexts = [_]IntelligentAgentSelector.AgentContext{
        .{
            .id = "checker-001",
            .name = "Alice Chen",
            .role = "checker",
            .capabilities = &[_][]const u8{ "accounting", "audit", "compliance" },
            .capacity = 5,
            .current_load = 2,
            .performance_score = 0.92,
            .avg_completion_time = 28,
            .historical_data = null,
        },
        .{
            .id = "manager-001",
            .name = "Carol Wang",
            .role = "manager",
            .capabilities = &[_][]const u8{ "accounting", "audit", "compliance", "management" },
            .capacity = 10,
            .current_load = 6,
            .performance_score = 0.95,
            .avg_completion_time = 25,
            .historical_data = null,
        },
    };
    
    const recommendation = try ai_selector.getRecommendation(ai_task, &agent_contexts);
    
    std.debug.print("AI Recommendation:\n", .{});
    std.debug.print("  Agent: {s}\n", .{recommendation.agent_id});
    std.debug.print("  Confidence: {d:.1}%\n", .{recommendation.confidence * 100});
    std.debug.print("  Reasoning: {s}\n", .{recommendation.reasoning});
    
    // =========================================================================
    // Step 5: Register with nAgentFlow
    // =========================================================================
    
    std.debug.print("\n--- nAgentFlow Integration ---\n", .{});
    
    var bridge = try AgentFlowBridge.init(
        allocator,
        "http://localhost:8090",
        "trial-balance",
        "Trial Balance",
    );
    defer bridge.deinit();
    
    std.debug.print("✓ Connected to nAgentFlow\n", .{});
    
    // Publish assignment events
    try bridge.publishEvent(
        .task_assigned,
        "task-001",
        "{\"agent_id\": \"checker-001\", \"method\": \"hungarian\"}",
    );
    
    std.debug.print("✓ Published task assignment event\n", .{});
    
    // =========================================================================
    // Step 6: Display Statistics
    // =========================================================================
    
    std.debug.print("\n--- Agent Statistics ---\n", .{});
    
    const stats = agent_logic.getAgentStatistics();
    std.debug.print("Total Agents: {d}\n", .{stats.total_agents});
    std.debug.print("Available: {d}\n", .{stats.available_agents});
    std.debug.print("Busy: {d}\n", .{stats.busy_agents});
    std.debug.print("Average Utilization: {d:.1}%\n", .{stats.average_utilization * 100});
    std.debug.print("Pending Tasks: {d}\n", .{stats.total_pending_tasks});
    
    std.debug.print("\n=== Example Complete ===\n", .{});
}

// =========================================================================
// Workflow Definition Example
// =========================================================================

/// Example workflow: Trial Balance Entry with intelligent agent assignment
pub const TrialBalanceWorkflow = struct {
    pub fn defineWorkflow() !void {
        // This would be registered in nAgentFlow as a workflow definition
        
        const workflow_json =
            \\{
            \\  "id": "trial-balance-approval",
            \\  "name": "Trial Balance Entry Approval",
            \\  "nodes": [
            \\    {
            \\      "id": "n1",
            \\      "type": "process/agent_assignment",
            \\      "name": "Assign to Checker",
            \\      "config": {
            \\        "assignment_strategy": "hungarian",
            \\        "capability_required": "accounting",
            \\        "consider_workload": true,
            \\        "consider_performance": true
            \\      }
            \\    },
            \\    {
            \\      "id": "n2",
            \\      "type": "process/approval_step",
            \\      "name": "Checker Review",
            \\      "config": {
            \\        "role": "checker",
            \\        "timeout_hours": 24
            \\      }
            \\    },
            \\    {
            \\      "id": "n3",
            \\      "type": "process/agent_assignment",
            \\      "name": "Assign to Manager",
            \\      "config": {
            \\        "assignment_strategy": "highest_performance",
            \\        "capability_required": "management"
            \\      }
            \\    },
            \\    {
            \\      "id": "n4",
            \\      "type": "process/approval_step",
            \\      "name": "Manager Approval",
            \\      "config": {
            \\        "role": "manager",
            \\        "timeout_hours": 48
            \\      }
            \\    },
            \\    {
            \\      "id": "n5",
            \\      "type": "process/finalize",
            \\      "name": "Finalize Entry"
            \\    }
            \\  ],
            \\  "connections": [
            \\    {"source": "n1", "target": "n2"},
            \\    {"source": "n2", "target": "n3"},
            \\    {"source": "n3", "target": "n4"},
            \\    {"source": "n4", "target": "n5"}
            \\  ]
            \\}
        ;
        
        _ = workflow_json;
        // In production: POST this to nAgentFlow /api/v1/workflows
    }
};

// =========================================================================
// Integration Test
// =========================================================================

test "agent assignment integration" {
    const allocator = std.testing.allocator;
    
    // Test that all components work together
    var agent_logic = AgentLogic.init(allocator);
    defer agent_logic.deinit();
    
    const agent = AgentLogic.Agent{
        .id = "test-agent",
        .name = "Test Agent",
        .role = "checker",
        .capabilities = &[_][]const u8{"accounting"},
        .capacity = 5,
        .current_load = 0,
        .availability = .available,
        .performance_score = 0.8,
    };
    
    try agent_logic.registerAgent(agent);
    
    const stats = agent_logic.getAgentStatistics();
    try std.testing.expectEqual(@as(usize, 1), stats.total_agents);
}