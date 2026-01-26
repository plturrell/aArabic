#!/bin/bash
# Fix remaining 5 API issues

cd "$(dirname "$0")"

echo "Fixing remaining API issues..."

# The remaining errors are:
# 1. docgen.zig:40 - ArrayList(DocItem).init → initCapacity
# 2. formatter.zig:152 - ArrayList(u8).init → initCapacity  
# 3. runner.zig:124 - toOwnedSlice() → toOwnedSlice(allocator)
# 4. tester.zig:71 - ArrayList(TestCase).init → initCapacity
# 5. tester.zig:135 - append(.{...}) → append(allocator, .{...})
# 6. repl.zig:92 - getStdOut() issue

echo "Files to fix:"
echo "- tools/cli/docgen.zig"
echo "- tools/cli/formatter.zig"
echo "- tools/cli/runner.zig"
echo "- tools/cli/tester.zig"
echo "- tools/cli/repl.zig"

echo ""
echo "Please manually apply these fixes:"
echo "1. docgen.zig line 40: ArrayList(DocItem).init(allocator) → .initCapacity(allocator, 10)"
echo "2. formatter.zig line 152: ArrayList(u8).init(allocator) → .initCapacity(allocator, 256)"
echo "3. runner.zig line 124: result.toOwnedSlice() → result.toOwnedSlice(allocator)"
echo "4. tester.zig line 71: ArrayList(TestCase).init(allocator) → .initCapacity(allocator, 10)"
echo "5. tester.zig line 135: append(.{...}) → append(allocator, .{...})"
echo "6. repl.zig line 92: Remove std.io. prefix, just use getStdOut()"
