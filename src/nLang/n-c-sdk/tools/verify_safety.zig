//! Safety Contract Verification Tool
//! Verifies that all unsafe blocks have documented safety contracts
//!
//! Usage:
//!   zig run tools/verify_safety.zig -- src/
//!   zig run tools/verify_safety.zig -- src/ --strict

const std = @import("std");
const mem = std.mem;
const fs = std.fs;

const stdout = std.fs.File.stdout().deprecatedWriter();
const stderr = std.fs.File.stderr().deprecatedWriter();

const UnsafeBlock = struct {
    file_path: []const u8,
    line_start: usize,
    line_end: ?usize,
    has_contract: bool,
    has_profiling: bool,
    has_testing: bool,
    contract_line: ?usize,
};

const VerificationResult = struct {
    allocator: std.mem.Allocator,
    total_files: usize = 0,
    total_unsafe_blocks: usize = 0,
    blocks_with_contracts: usize = 0,
    blocks_with_profiling: usize = 0,
    blocks_with_testing: usize = 0,
    unclosed_blocks: usize = 0,
    violations: std.ArrayList(UnsafeBlock),
    
    pub fn init(allocator: std.mem.Allocator) VerificationResult {
        return .{
            .allocator = allocator,
            .violations = std.ArrayList(UnsafeBlock){},
        };
    }
    
    pub fn deinit(self: *VerificationResult) void {
        self.violations.deinit(self.allocator);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2) {
        try stderr.writeAll("Usage: verify_safety <src_dir> [--strict]\n");
        try stderr.writeAll("\nOptions:\n");
        try stderr.writeAll("  --strict    Require profiling and testing documentation\n");
        std.process.exit(1);
    }
    
    const src_dir = args[1];
    const strict = args.len > 2 and mem.eql(u8, args[2], "--strict");
    
    var result = VerificationResult.init(allocator);
    defer result.deinit();
    
    try verifyDirectory(allocator, src_dir, &result, strict);
    
    try printReport(&result, strict);
    
    // Exit with error code if violations found
    if (result.violations.items.len > 0 or result.unclosed_blocks > 0) {
        std.process.exit(1);
    }
}

fn verifyDirectory(
    allocator: std.mem.Allocator,
    path: []const u8,
    result: *VerificationResult,
    strict: bool,
) !void {
    var dir = fs.cwd().openDir(path, .{ .iterate = true }) catch |err| {
        try stderr.print("⚠️  Cannot open directory '{s}': {s}\n", .{ path, @errorName(err) });
        return;
    };
    defer dir.close();
    
    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind == .file and mem.endsWith(u8, entry.name, ".zig")) {
            try verifyFile(allocator, path, entry.name, result, strict);
        } else if (entry.kind == .directory) {
            // Skip some directories
            if (mem.eql(u8, entry.name, ".git") or
                mem.eql(u8, entry.name, "zig-cache") or
                mem.eql(u8, entry.name, "zig-out"))
            {
                continue;
            }
            
            const sub_path = try fs.path.join(allocator, &.{ path, entry.name });
            defer allocator.free(sub_path);
            try verifyDirectory(allocator, sub_path, result, strict);
        }
    }
}

fn verifyFile(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    file_name: []const u8,
    result: *VerificationResult,
    _: bool,
) !void {
    const file_path = try fs.path.join(allocator, &.{ dir_path, file_name });
    defer allocator.free(file_path);
    
    const source = fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024) catch |err| {
        try stderr.print("⚠️  Cannot read '{s}': {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);
    
    result.total_files += 1;
    
    var line_num: usize = 1;
    var lines = mem.tokenizeScalar(u8, source, '\n');
    
    var in_unsafe = false;
    var unsafe_start: usize = 0;
    var has_contract = false;
    var has_profiling = false;
    var has_testing = false;
    var contract_line: ?usize = null;
    var recent_comment_lines: usize = 0;
    
    while (lines.next()) |line| : (line_num += 1) {
        const trimmed = mem.trim(u8, line, " \t\r");
        
        // Track comments above unsafe blocks
        if (mem.startsWith(u8, trimmed, "///") or mem.startsWith(u8, trimmed, "//")) {
            recent_comment_lines = line_num;
            
            // Check for safety contract markers
            if (mem.indexOf(u8, trimmed, "SAFETY CONTRACT:") != null or
                mem.indexOf(u8, trimmed, "SAFETY:") != null)
            {
                has_contract = true;
                contract_line = line_num;
            }
            
            // Check for profiling documentation
            if (mem.indexOf(u8, trimmed, "PROFILING:") != null or
                mem.indexOf(u8, trimmed, "PROFILE:") != null or
                mem.indexOf(u8, trimmed, "CPU") != null)
            {
                has_profiling = true;
            }
            
            // Check for testing documentation
            if (mem.indexOf(u8, trimmed, "TESTING:") != null or
                mem.indexOf(u8, trimmed, "TESTED:") != null or
                mem.indexOf(u8, trimmed, "fuzz") != null)
            {
                has_testing = true;
            }
        } else if (trimmed.len > 0 and !mem.startsWith(u8, trimmed, "@")) {
            // Non-comment, non-attribute line - reset if too far from unsafe block
            if (line_num > recent_comment_lines + 5) {
                has_contract = false;
                has_profiling = false;
                has_testing = false;
                contract_line = null;
            }
        }
        
        // Check for @setRuntimeSafety(false)
        if (mem.indexOf(u8, line, "@setRuntimeSafety(false)")) |_| {
            if (!in_unsafe) {
                in_unsafe = true;
                unsafe_start = line_num;
                result.total_unsafe_blocks += 1;
                
                // Check if we have recent contract documentation
                if (line_num > recent_comment_lines + 10) {
                    has_contract = false;
                    has_profiling = false;
                    has_testing = false;
                    contract_line = null;
                }
            }
        }
        
        // Check for @setRuntimeSafety(true)
        if (mem.indexOf(u8, line, "@setRuntimeSafety(true)")) |_| {
            if (in_unsafe) {
                // Unsafe block ended
                if (has_contract) {
                    result.blocks_with_contracts += 1;
                } else {
                    try result.violations.append(result.allocator, .{
                        .file_path = try allocator.dupe(u8, file_path),
                        .line_start = unsafe_start,
                        .line_end = line_num,
                        .has_contract = false,
                        .has_profiling = has_profiling,
                        .has_testing = has_testing,
                        .contract_line = contract_line,
                    });
                }
                
                if (has_profiling) result.blocks_with_profiling += 1;
                if (has_testing) result.blocks_with_testing += 1;
                
                // Reset for next block
                in_unsafe = false;
                has_contract = false;
                has_profiling = false;
                has_testing = false;
                contract_line = null;
            }
        }
    }
    
    // Check for unclosed unsafe blocks
    if (in_unsafe) {
        result.unclosed_blocks += 1;
        try result.violations.append(result.allocator, .{
            .file_path = try allocator.dupe(u8, file_path),
            .line_start = unsafe_start,
            .line_end = null,
            .has_contract = has_contract,
            .has_profiling = has_profiling,
            .has_testing = has_testing,
            .contract_line = contract_line,
        });
    }
}

fn printReport(result: *VerificationResult, strict: bool) !void {
    try stdout.writeAll("\n🔍 Safety Contract Verification Report\n");
    try stdout.writeAll("═══════════════════════════════════════════════════════════════\n\n");
    
    // Summary statistics
    try stdout.writeAll("📊 STATISTICS\n");
    try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
    try stdout.print("Files scanned:        {d}\n", .{result.total_files});
    try stdout.print("Unsafe blocks found:  {d}\n", .{result.total_unsafe_blocks});
    try stdout.print("With contracts:       {d} ({d:.1}%)\n", .{
        result.blocks_with_contracts,
        if (result.total_unsafe_blocks > 0)
            @as(f64, @floatFromInt(result.blocks_with_contracts)) /
            @as(f64, @floatFromInt(result.total_unsafe_blocks)) * 100.0
        else
            0.0,
    });
    
    if (strict) {
        try stdout.print("With profiling info:  {d} ({d:.1}%)\n", .{
            result.blocks_with_profiling,
            if (result.total_unsafe_blocks > 0)
                @as(f64, @floatFromInt(result.blocks_with_profiling)) /
                @as(f64, @floatFromInt(result.total_unsafe_blocks)) * 100.0
            else
                0.0,
        });
        try stdout.print("With testing info:    {d} ({d:.1}%)\n", .{
            result.blocks_with_testing,
            if (result.total_unsafe_blocks > 0)
                @as(f64, @floatFromInt(result.blocks_with_testing)) /
                @as(f64, @floatFromInt(result.total_unsafe_blocks)) * 100.0
            else
                0.0,
        });
    }
    
    if (result.unclosed_blocks > 0) {
        try stdout.print("Unclosed blocks:      {d} ❌\n", .{result.unclosed_blocks});
    }
    try stdout.writeAll("\n");
    
    // Print violations
    if (result.violations.items.len > 0) {
        try stdout.writeAll("❌ VIOLATIONS FOUND\n");
        try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
        
        for (result.violations.items, 0..) |violation, i| {
            try stdout.print("{d}. {s}:{d}", .{ i + 1, violation.file_path, violation.line_start });
            
            if (violation.line_end) |end| {
                try stdout.print("-{d}", .{end});
            } else {
                try stdout.writeAll(" (UNCLOSED)");
            }
            try stdout.writeAll("\n");
            
            if (!violation.has_contract) {
                try stdout.writeAll("   ❌ Missing safety contract\n");
            }
            
            if (strict) {
                if (!violation.has_profiling) {
                    try stdout.writeAll("   ⚠️  Missing profiling justification\n");
                }
                if (!violation.has_testing) {
                    try stdout.writeAll("   ⚠️  Missing testing documentation\n");
                }
            }
            
            try stdout.writeAll("\n");
        }
        
        try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
        try stdout.writeAll("\n💡 RECOMMENDATIONS\n");
        try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
        try stdout.writeAll("Each unsafe block should have:\n\n");
        try stdout.writeAll("/// SAFETY CONTRACT:\n");
        try stdout.writeAll("/// - Precondition 1 (e.g., bounds validated)\n");
        try stdout.writeAll("/// - Precondition 2 (e.g., inputs validated)\n");
        try stdout.writeAll("/// - Invariant maintained (e.g., no concurrent access)\n");
        
        if (strict) {
            try stdout.writeAll("///\n");
            try stdout.writeAll("/// PROFILING:\n");
            try stdout.writeAll("/// - XX% of function runtime\n");
            try stdout.writeAll("/// - YY% of total CPU time\n");
            try stdout.writeAll("///\n");
            try stdout.writeAll("/// TESTING:\n");
            try stdout.writeAll("/// - Unit tested with N test cases\n");
            try stdout.writeAll("/// - Fuzz tested with M iterations\n");
            try stdout.writeAll("/// - Matches safe version\n");
        }
        
        try stdout.writeAll("@setRuntimeSafety(false);\n");
        try stdout.writeAll("// ... unsafe code ...\n");
        try stdout.writeAll("@setRuntimeSafety(true);\n");
        try stdout.writeAll("\n");
    } else {
        try stdout.writeAll("✅ NO VIOLATIONS FOUND\n");
        try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
        
        if (result.total_unsafe_blocks > 0) {
            const safety_score = @as(f64, @floatFromInt(result.blocks_with_contracts)) /
                @as(f64, @floatFromInt(result.total_unsafe_blocks)) * 100.0;
            
            try stdout.print("All {d} unsafe block{s} properly documented\n", .{
                result.total_unsafe_blocks,
                if (result.total_unsafe_blocks == 1) "" else "s",
            });
            try stdout.print("Safety Contract Score: {d:.1}%\n", .{safety_score});
            
            if (strict and result.blocks_with_profiling < result.total_unsafe_blocks) {
                try stdout.writeAll("\n⚠️  RECOMMENDATIONS (--strict mode)\n");
                try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
                
                const missing_profiling = result.total_unsafe_blocks - result.blocks_with_profiling;
                const missing_testing = result.total_unsafe_blocks - result.blocks_with_testing;
                
                if (missing_profiling > 0) {
                    try stdout.print("• {d} block{s} missing profiling justification\n", .{
                        missing_profiling,
                        if (missing_profiling == 1) "" else "s",
                    });
                }
                
                if (missing_testing > 0) {
                    try stdout.print("• {d} block{s} missing testing documentation\n", .{
                        missing_testing,
                        if (missing_testing == 1) "" else "s",
                    });
                }
            }
        } else {
            try stdout.writeAll("No unsafe blocks found in scanned files\n");
        }
    }
    
    try stdout.writeAll("\n");
}

fn printUsageExamples() !void {
    try stdout.writeAll("📚 EXAMPLE SAFETY CONTRACT\n");
    try stdout.writeAll("───────────────────────────────────────────────────────────────\n");
    try stdout.writeAll(
        \\/// Process array with optimized inner loop
        \\///
        \\/// SAFETY CONTRACT:
        \\/// - data.len >= min_size (validated at line 42)
        \\/// - data is properly aligned (ensured by allocator)
        \\/// - no concurrent access (protected by mutex)
        \\///
        \\/// PROFILING:
        \\/// - 45.2% of processBuffer() runtime
        \\/// - 23.1% of total application CPU time
        \\/// - Called 1.2M times during profile run
        \\///
        \\/// TESTING:
        \\/// - Unit tested: 100+ cases
        \\/// - Fuzz tested: 10M random inputs
        \\/// - Property tested: matches safe version
        \\/// - Memory sanitizer: clean
        \\pub fn processArrayFast(data: []u8) void {
        \\    @setRuntimeSafety(false);
        \\    defer @setRuntimeSafety(true);
        \\    
        \\    // Optimized processing...
        \\}
        \\
    );
    try stdout.writeAll("\n");
}