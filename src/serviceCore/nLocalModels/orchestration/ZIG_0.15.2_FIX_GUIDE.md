# Zig 0.15.2 API Fix Guide for nLocalModels Orchestration

## Quick Reference: Common Fixes

### 1. std.io API Changes

**OLD (doesn't work):**
```zig
const stdout = std.io.getStdOut().writer();
const stderr = std.io.getStdErr();
```

**NEW (Zig 0.15.2):**
```zig
const stdout = std.fs.File.stdout().deprecatedWriter();
const stderr = std.fs.File.stderr();
```

### 2. ArrayList.init - No Method, Use Function

**OLD (doesn't work):**
```zig
.validation_errors = std.ArrayList([]const u8).init(allocator),
```

**NEW (correct):**
```zig
.validation_errors = std.ArrayList([]const u8).init(allocator),
```

Note: This is actually correct! The error might be elsewhere in the struct initialization.

### 3. ArrayList.deinit() on struct with allocator field

When ArrayList is in a struct, track allocator separately:

**Pattern:**
```zig
pub const MyStruct = struct {
    allocator: Allocator,  // Add this!
    my_list: std.ArrayList(T),
    
    pub fn deinit(self: *MyStruct) void {
        for (self.my_list.items) |item| {
            self.allocator.free(item);  // Use self.allocator
        }
        self.my_list.deinit();
    }
};
```

### 4. Format String Changes

**OLD:**
```zig
try stdout.print("{'=':**60}\n", .{});
```

**NEW:**
```zig
try stdout.print("{'=':**<60}\n", .{});  // Add < for left-align
```

### 5. Unused Variables/Constants

**Error:** `unused local constant`

**Fix:** Remove unused variables or use `_` to discard:
```zig
_ = unused_var;
```

**Or change const to var if it will be mutated:**
```zig
const sorted = ...;  // Error: never mutated
var sorted = ...;    // Still error if never mutated!
// Better: just use const and actually use it
```

### 6. HTTP Client API (Completely Changed)

**OLD:**
```zig
var headers = std.http.Headers{ .allocator = allocator };
var req = try client.open(.GET, uri, headers, .{});
```

**NEW (stub for now):**
```zig
// TODO: Reimplement with Zig 0.15.2 HTTP API
return error.HTTPNotImplemented;
```

## Specific File Fixes Needed

### analytics.zig (3 errors)
- Line 471-472: Remove or use `epoch_day` and `day_seconds`
- Line 489: Change `std.io.getStdErr()` to `std.fs.File.stderr()`

### benchmark_cli.zig (5 errors)
- Line 41: Change `std.io.getStdOut()` to `std.fs.File.stdout().deprecatedWriter()`
- Line 276: Change `var sorted` to `const sorted` if never mutated

### benchmark_validator.zig (1 error)
- Line 56: ArrayList.init is correct - check struct initialization context

### gpu_monitor.zig (1 error)
- Line 67: ArrayList.init - same pattern as above

### hf_model_card_extractor.zig (3 errors)
- Lines 57, 251: ArrayList.init patterns
- Line 89: `.deinit()` needs no arguments in 0.15.2

### model_selector.zig (2 errors)
- Lines 453-454: Remove `_ = self;` and `_ = task_category;` - actually use them

### multi_category.zig (1 error)
- Line 63: Change `var new_score` to `const new_score`

## Build Command

```bash
cd /Users/user/Documents/arabic_folder/src/serviceCore/nLocalModels/orchestration
zig build
```

## Success Criteria

Target: Build Summary: 14/14 steps succeeded; 0 failed

Current: Build Summary: 3/14 steps succeeded; 5 failed

## Already Fixed Files

✅ dataset_loader.zig - Working with Arena allocator optimization!
✅ build.zig - LTO + ReleaseSafe applied
✅ Parts of hf_model_card_extractor.zig
✅ Parts of benchmark_validator.zig  
✅ Parts of gpu_monitor_cli.zig
