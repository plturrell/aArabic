//! Process Orchestrator - Revolutionary Integration
//! Petri nets control process execution workflows
//! 
//! Use Petri nets to model process dependencies and execution!
//! When a transition fires → spawn a process
//! When process completes → update Petri marking

const std = @import("std");
const Allocator = std.mem.Allocator;
const petri_core = @import("../petri/core.zig");
const petri_types = @import("../petri/types.zig");
const process = @import("../process/lib.zig");
const hungarian = @import("hungarian.zig");

/// Process handle for tracking
pub const ProcessHandle = struct {
    pid: std.posix.pid_t,
    transition_id: [256]u8,
    cmd: []const u8,
    started_at: i64,
    output_place_id: ?[256]u8 = null,
    
    pub fn deinit(self: *ProcessHandle, allocator: Allocator) void {
        allocator.free(self.cmd);
    }
};

/// Process orchestrator - binds Petri nets to OS processes
pub const ProcessOrchestrator = struct {
    allocator: Allocator,
    petri_net: *petri_types.pn_net_t,
    process_map: std.StringHashMap(ProcessHandle),
    transition_bindings: std.StringHashMap([]const u8), // transition_id -> command
    running: bool,
    mutex: std.Thread.Mutex,
    
    pub fn init(allocator: Allocator, petri_net: *petri_types.pn_net_t) !*ProcessOrchestrator {
        const orch = try allocator.create(ProcessOrchestrator);
        orch.* = .{
            .allocator = allocator,
            .petri_net = petri_net,
            .process_map = std.StringHashMap(ProcessHandle).init(allocator),
            .transition_bindings = std.StringHashMap([]const u8).init(allocator),
            .running = false,
            .mutex = .{},
        };
        return orch;
    }
    
    pub fn deinit(self: *ProcessOrchestrator) void {
        // Clean up bindings
        var it = self.transition_bindings.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.transition_bindings.deinit();
        
        // Clean up processes
        var proc_it = self.process_map.iterator();
        while (proc_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.process_map.deinit();
        
        self.allocator.destroy(self);
    }
    
    /// Bind a Petri transition to a shell command
    /// When transition fires, this command executes
    pub fn bindTransitionToCommand(
        self: *ProcessOrchestrator,
        transition_id: []const u8,
        command: []const u8,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const id_copy = try self.allocator.dupe(u8, transition_id);
        const cmd_copy = try self.allocator.dupe(u8, command);
        
        try self.transition_bindings.put(id_copy, cmd_copy);
    }
    
    /// Set the callback on the Petri net to spawn processes
    pub fn registerCallbacks(self: *ProcessOrchestrator) !void {
        // Register transition fired callback
        const rc = petri_core.pn_callback_set(
            self.petri_net,
            .transition_fired,
            onTransitionFired,
        );
        if (rc != 0) return error.CallbackRegistrationFailed;
    }
    
    /// Callback when transition fires
    fn onTransitionFired(
        net: ?*petri_types.pn_net_t,
        event: petri_types.pn_event_type_t,
        ctx: ?*anyopaque,
    ) callconv(.C) void {
        _ = net;
        _ = event;

        // Get orchestrator from context
        const self: *ProcessOrchestrator = @ptrCast(@alignCast(ctx orelse return));

        // Get the transition ID from the event data
        // In a full implementation, the event would contain transition info
        // For now, we trigger any bound transition

        self.mutex.lock();
        defer self.mutex.unlock();

        // Spawn processes for all ready transition bindings
        var iter = self.transition_bindings.iterator();
        while (iter.next()) |entry| {
            const transition_id = entry.key_ptr.*;
            const cmd = entry.value_ptr.*;

            // Spawn the process for this transition
            self.spawnProcessForTransition(transition_id, cmd) catch |err| {
                std.debug.print("Failed to spawn process for transition: {}\n", .{err});
            };
        }

        std.debug.print("Transition fired! Spawning process...\n", .{});
    }

    /// Spawn a process for a specific transition
    fn spawnProcessForTransition(self: *ProcessOrchestrator, transition_id: []const u8, cmd: []const u8) !void {
        // Fork and exec the command
        const argv = [_:null]?[*:0]const u8{
            @ptrCast(cmd.ptr),
            null,
        };

        const pid = std.posix.fork() catch return error.ForkFailed;

        if (pid == 0) {
            // Child process - exec the command
            const result = std.posix.execveZ("/bin/sh", &argv, std.c.environ);
            _ = result;
            std.posix.exit(127);
        }

        // Parent - register the process
        var handle = ProcessHandle{
            .pid = pid,
            .transition_id = undefined,
            .cmd = try self.allocator.dupe(u8, cmd),
            .started_at = std.time.timestamp(),
        };
        @memcpy(handle.transition_id[0..transition_id.len], transition_id);
        handle.transition_id[transition_id.len] = 0;

        const key = try self.allocator.dupe(u8, transition_id);
        try self.process_map.put(key, handle);
    }
    
    /// Execute Petri net with process spawning
    pub fn execute(self: *ProcessOrchestrator) !void {
        self.running = true;
        
        // Enable tracing on Petri net
        _ = petri_core.pn_trace_enable(self.petri_net);
        
        // Register callbacks
        try self.registerCallbacks();
        
        // Execute Petri net in single-step mode
        while (self.running) {
            const rc = petri_core.pn_step(self.petri_net);
            if (rc != 0) break; // Deadlock or completion
            
            // Check for completed processes
            try self.checkCompletedProcesses();
            
            std.time.sleep(100 * std.time.ns_per_ms); // 100ms poll interval
        }
    }
    
    /// Spawn a process for a transition
    fn spawnProcess(
        self: *ProcessOrchestrator,
        transition_id: []const u8,
    ) !std.posix.pid_t {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const command = self.transition_bindings.get(transition_id) orelse {
            return error.NoCommandForTransition;
        };
        
        // Fork and exec
        const pid = try std.posix.fork();
        if (pid == 0) {
            // Child process - execute command
            const argv = [_:null]?[*:0]const u8{ "/bin/sh", "-c", @ptrCast(command.ptr), null };
            const envp = [_:null]?[*:0]const u8{null};
            _ = process.execve("/bin/sh", &argv, &envp);
            std.posix.exit(127); // Should not reach here
        }
        
        // Parent - track the process
        var handle = ProcessHandle{
            .pid = @intCast(pid),
            .transition_id = undefined,
            .cmd = try self.allocator.dupe(u8, command),
            .started_at = std.time.timestamp(),
        };
        
        @memset(&handle.transition_id, 0);
        @memcpy(handle.transition_id[0..@min(transition_id.len, 255)], transition_id);
        
        const key = try std.fmt.allocPrint(self.allocator, "{d}", .{pid});
        try self.process_map.put(key, handle);
        
        std.debug.print("Spawned process {d} for transition '{s}'\n", .{ pid, transition_id });
        
        return @intCast(pid);
    }
    
    /// Check for completed processes and update Petri net
    fn checkCompletedProcesses(self: *ProcessOrchestrator) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var to_remove = std.ArrayList([]const u8).init(self.allocator);
        defer {
            for (to_remove.items) |key| {
                self.allocator.free(key);
            }
            to_remove.deinit();
        }
        
        var it = self.process_map.iterator();
        while (it.next()) |entry| {
            const handle = entry.value_ptr;
            
            // Non-blocking wait
            const result = std.posix.waitpid(handle.pid, std.posix.W.NOHANG) catch continue;
            
            if (result.pid == handle.pid) {
                // Process completed!
                const exit_code = std.posix.W.EXITSTATUS(result.status);
                
                std.debug.print("Process {d} completed with code {d}\n", .{ handle.pid, exit_code });
                
                // Update Petri net marking
                if (handle.output_place_id) |place_id| {
                    // Add success/failure token to output place
                    const place_id_str = std.mem.sliceTo(&place_id, 0);

                    // Find the place in the Petri net and add a token
                    if (petri_core.pn_place_find(self.petri_net, place_id_str.ptr)) |place| {
                        // Add token representing completion status
                        // Token value: 1 for success (exit_code == 0), 0 for failure
                        const token_value: u32 = if (exit_code == 0) 1 else 0;
                        _ = petri_core.pn_token_put(place, token_value);
                    }
                }
                
                try to_remove.append(try self.allocator.dupe(u8, entry.key_ptr.*));
            }
        }
        
        // Remove completed processes
        for (to_remove.items) |key| {
            if (self.process_map.fetchRemove(key)) |kv| {
                var handle = kv.value;
                handle.deinit(self.allocator);
                self.allocator.free(kv.key);
            }
        }
    }
    
    /// Stop orchestration
    pub fn stop(self: *ProcessOrchestrator) void {
        self.running = false;
    }
};

/// Optimal process assignment using Hungarian algorithm
pub fn assignProcessesToWorkers(
    allocator: Allocator,
    processes: [][]const u8,
    workers: [][]const u8,
    cost_fn: *const fn (process: []const u8, worker: []const u8) f64,
) !hungarian.AssignmentResult {
    const n = @max(processes.len, workers.len);
    const solver = try hungarian.HungarianSolver.init(allocator, n);
    defer solver.deinit();
    
    // Build cost matrix
    for (processes, 0..) |proc, i| {
        for (workers, 0..) |worker, j| {
            const cost = cost_fn(proc, worker);
            solver.setCost(i, j, cost);
        }
    }
    
    // Solve optimal assignment
    return solver.solve();
}

// Tests
test "ProcessOrchestrator: initialization" {
    const allocator = std.testing.allocator;
    
    const net = petri_core.pn_create("test_net", 0);
    defer _ = petri_core.pn_destroy(net);
    
    const orch = try ProcessOrchestrator.init(allocator, net.?);
    defer orch.deinit();
    
    try std.testing.expect(orch.running == false);
}

test "ProcessOrchestrator: bind transition" {
    const allocator = std.testing.allocator;
    
    const net = petri_core.pn_create("test_net", 0);
    defer _ = petri_core.pn_destroy(net);
    
    const orch = try ProcessOrchestrator.init(allocator, net.?);
    defer orch.deinit();
    
    try orch.bindTransitionToCommand("t1", "echo 'Hello World'");
    
    try std.testing.expect(orch.transition_bindings.count() == 1);
}
