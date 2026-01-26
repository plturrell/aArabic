# Security Guidelines - n-c-sdk

**Version:** 1.0  
**Last Updated:** January 24, 2026  
**Classification:** Internal Development Guidelines

---

## Overview

This document outlines security best practices and guidelines for developing with the n-c-sdk. All contributors should follow these guidelines to maintain the high security standards established in this project.

---

## Security Principles

### 1. Memory Safety First 🛡️

**Always leverage Zig's memory safety features:**

```zig
// ✅ GOOD: Use proper allocator pattern
var list = std.ArrayList(u64){};
defer list.deinit(allocator);

// ❌ BAD: Manual memory management without cleanup
var ptr = malloc(1024);
// Missing free()
```

**Key Practices:**
- Use `defer` for all cleanup operations
- Never use raw pointer arithmetic without bounds checking
- Prefer stack allocation when possible
- Use GeneralPurposeAllocator in tests to detect leaks

### 2. Type Safety ✅

**Leverage Zig's strong type system:**

```zig
// ✅ GOOD: Explicit type conversions
const result: u64 = @intCast(value);

// ❌ BAD: Implicit conversions (Zig won't allow this)
const result: u64 = value;  // Compilation error
```

**Key Practices:**
- Always use explicit type casts with `@intCast`, `@floatCast`, etc.
- Use optional types (`?T`) instead of null pointers
- Validate all type conversions

### 3. Integer Overflow Protection 🔢

**Handle arithmetic operations safely:**

```zig
// ✅ GOOD: Explicit overflow behavior
sum +%= value;  // Wrapping addition (intentional overflow)
sum = try std.math.add(u64, sum, value);  // Checked addition

// ✅ GOOD: Bounded operations
if (index >= array.len) return error.IndexOutOfBounds;

// ❌ BAD: Unchecked arithmetic
sum = sum + value;  // May panic on overflow in ReleaseSafe
```

**Key Practices:**
- Use `+%`, `-%`, `*%` for intentional wrapping
- Use `std.math.add()`, `std.math.mul()` for checked operations
- Always validate array indices

### 4. Resource Management 📦

**Prevent resource exhaustion:**

```zig
// ✅ GOOD: Bounded allocations
const max_size = 1_000_000;
if (requested_size > max_size) return error.TooLarge;
const data = try allocator.alloc(u8, requested_size);

// ❌ BAD: Unbounded allocations
const data = try allocator.alloc(u8, user_input);  // No validation
```

**Key Practices:**
- Set maximum limits on allocations
- Use bounded loops
- Implement timeouts for long-running operations
- Monitor memory usage with tracking allocators

### 5. Error Handling 🚨

**Handle all errors properly:**

```zig
// ✅ GOOD: Explicit error handling
const file = try std.fs.cwd().openFile("config.json", .{});
defer file.close();

// ✅ GOOD: Error propagation
pub fn processData() !void {
    try validateInput();
    try computeResult();
}

// ❌ BAD: Ignoring errors
const file = std.fs.cwd().openFile("config.json", .{}) catch unreachable;
```

**Key Practices:**
- Use `try` to propagate errors
- Use `catch` only with proper handling
- Avoid `catch unreachable` except in test code
- Document possible error conditions

---

## Security Checklist for New Code

### Before Committing Code

- [ ] All allocations have corresponding `defer` cleanup
- [ ] No raw pointer arithmetic without bounds checking
- [ ] All integer conversions are explicit with `@intCast`
- [ ] Overflow behavior is explicitly defined (wrapping or checked)
- [ ] All array accesses are bounds-checked
- [ ] Optional types used instead of null pointers
- [ ] Error handling is complete (no ignored errors)
- [ ] No sensitive information in debug output
- [ ] No unbounded loops or allocations
- [ ] Build tested in ReleaseSafe mode
- [ ] Memory leaks checked with GPA
- [ ] Documentation updated

---

## Build Mode Security

### ReleaseSafe (Recommended for Production) ✅

**Security Features:**
- ✅ Bounds checking enabled
- ✅ Integer overflow detection
- ✅ Null pointer checks
- ✅ Undefined behavior detection
- ✅ Good performance (2-3x faster than Debug)

```bash
zig build -Doptimize=ReleaseSafe
```

### Debug (Development Only) 🔧

**Security Features:**
- ✅ All safety checks enabled
- ✅ Maximum debug information
- ⚠️ Slower performance
- ⚠️ Large binaries

```bash
zig build -Doptimize=Debug
```

### ReleaseFast (Use with Caution) ⚡

**Security Features:**
- ❌ Safety checks disabled
- ⚠️ No bounds checking
- ⚠️ No overflow detection
- ✅ Maximum performance

**Only use when:**
- Performance is absolutely critical
- Code is thoroughly tested
- Security implications are understood

```bash
zig build -Doptimize=ReleaseFast
```

---

## Dependency Management

### Current Status: Zero Dependencies ✅

**Benefits:**
- ✅ Minimal attack surface
- ✅ No supply chain risks
- ✅ Easy to audit
- ✅ Fast builds

### If Adding Dependencies

**Required Checks:**
1. Audit dependency source code
2. Check for known vulnerabilities
3. Verify cryptographic signatures
4. Review dependency's dependencies
5. Monitor for security advisories
6. Pin to specific versions

**Example:**
```zig
// build.zig.zon
.dependencies = .{
    .package_name = .{
        .url = "https://github.com/org/repo/archive/v1.0.0.tar.gz",
        .hash = "1220abcd...",  // Pin exact version
    },
},
```

---

## Input Validation

### Validate All External Input

```zig
// ✅ GOOD: Validate before use
pub fn processConfig(data: []const u8) !Config {
    if (data.len > MAX_CONFIG_SIZE) return error.ConfigTooLarge;
    if (data.len == 0) return error.EmptyConfig;
    
    return try std.json.parseFromSlice(Config, allocator, data, .{});
}

// ❌ BAD: No validation
pub fn processConfig(data: []const u8) !Config {
    return try std.json.parseFromSlice(Config, allocator, data, .{});
}
```

**Key Practices:**
- Validate size limits
- Check for empty inputs
- Sanitize string data
- Validate numeric ranges

---

## Information Disclosure

### Sensitive Data Handling

**DO NOT:**
- ❌ Log passwords, tokens, or secrets
- ❌ Include sensitive data in error messages
- ❌ Expose internal paths in production
- ❌ Print stack traces in production

**DO:**
- ✅ Use generic error messages in production
- ✅ Log only non-sensitive metadata
- ✅ Sanitize debug output
- ✅ Use build-time conditionals for debug info

```zig
// ✅ GOOD: Build-time debug info
if (@import("builtin").mode == .Debug) {
    std.debug.print("Debug: {any}\n", .{internal_state});
}

// ❌ BAD: Always printing sensitive info
std.debug.print("Token: {s}\n", .{api_token});
```

---

## Testing Requirements

### Security Testing

**Required Tests:**
1. **Fuzz Testing** - Run edge case tests
   ```bash
   zig build fuzz
   ```

2. **Memory Leak Detection** - Use GPA
   ```zig
   var gpa = std.heap.GeneralPurposeAllocator(.{}){};
   defer {
       const leaked = gpa.deinit();
       if (leaked == .leak) return error.MemoryLeak;
   }
   ```

3. **ReleaseSafe Verification** - Test with safety checks
   ```bash
   zig build -Doptimize=ReleaseSafe
   zig build test
   ```

---

## Incident Response

### If You Discover a Vulnerability

1. **Do Not** commit the vulnerable code
2. **Do** report to the security contact immediately
3. **Do** document the issue clearly
4. **Do** propose a fix if possible

### Reporting Template

```markdown
## Security Issue Report

**Severity:** [Critical/High/Medium/Low]
**Component:** [Affected component]
**Description:** [Clear description]
**Impact:** [Security impact]
**Reproduction:** [Steps to reproduce]
**Proposed Fix:** [Suggested solution]
```

---

## Code Review Security Checklist

### For Reviewers

When reviewing code, check for:

- [ ] Memory safety (allocations have cleanup)
- [ ] Type safety (no unsafe casts)
- [ ] Integer overflow protection
- [ ] Input validation
- [ ] Error handling completeness
- [ ] Information disclosure risks
- [ ] Resource exhaustion potential
- [ ] Dependency security
- [ ] Documentation completeness

---

## Security Tools

### Available Tools

1. **Zig Compiler Safety Checks** (Built-in)
   - Bounds checking
   - Overflow detection
   - Null pointer protection

2. **GeneralPurposeAllocator** (Memory Leak Detection)
   ```zig
   var gpa = std.heap.GeneralPurposeAllocator(.{}){};
   defer _ = gpa.deinit();  // Reports leaks
   ```

3. **Fuzz Testing Suite** (Custom)
   ```bash
   zig build fuzz
   ```

4. **Performance Profiler** (Custom)
   ```bash
   ./zig-out/bin/performance_profiler
   ```

---

## References

### External Resources

- [Zig Security](https://ziglang.org/documentation/master/#Memory-Safety)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Secure Software Development](https://csrc.nist.gov/publications/detail/sp/800-218/final)

### Internal Documentation

- `SECURITY_AUDIT_REPORT.md` - Latest security audit
- `KNOWN_LIMITATIONS_FIXED.md` - Fixed security issues
- `README.md` - General security considerations

---

## Updates and Maintenance

### This Document

- **Review Frequency:** Quarterly
- **Update Triggers:** New vulnerabilities, code changes, tool updates
- **Owner:** Security team / Lead developer

### Zig Version Updates

When updating Zig version:
1. Review changelog for security-related changes
2. Test all code with new version
3. Run full security audit
4. Update this document if needed

---

## Contact

**Security Questions:** Project maintainer  
**Vulnerability Reports:** Use secure channel  
**General Inquiries:** GitHub issues (for non-sensitive topics)

---

**Remember:** Security is everyone's responsibility. When in doubt, ask!