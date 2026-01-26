# Contributing to Zig SDK - Production Optimized ⚡

Thank you for your interest in contributing to this performance-optimized Zig SDK! This document provides guidelines for contributing to the project.

---

## 🎯 Project Goals

This SDK aims to provide:

1. **Production-Ready Defaults** - ReleaseSafe mode with safety intact
2. **Automatic Optimization** - LTO enabled without configuration
3. **Compatibility** - Full Zig 0.15.2 compatibility maintained
4. **Performance** - 20-30% faster execution vs Debug builds
5. **Developer Experience** - Zero-config production optimization

---

## 🤝 Ways to Contribute

### 1. Report Issues 🐛

**Found a bug or performance regression?**

Create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, architecture)
- Minimal code example if applicable

**Template:**
```markdown
## Bug Description
Brief description here

## Steps to Reproduce
1. Create file with content...
2. Run command...
3. Observe result...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: macOS 14.0 (aarch64)
- Zig SDK Version: 0.15.2-optimized
- Build Command: zig build -Doptimize=ReleaseSafe
```

### 2. Submit Benchmarks 📊

Help us validate and improve performance claims!

**We're looking for:**
- Real-world application benchmarks
- Comparison with official Zig builds
- Platform-specific performance data
- Build time measurements

**Benchmark Submission:**
```markdown
## Benchmark: [Name]

**Platform:** aarch64-macos (M1)
**Workload:** [Description]
**Iterations:** 1000

| Metric | Official Zig | This SDK | Improvement |
|--------|--------------|----------|-------------|
| Time | 500ms | 380ms | 24% faster |
| Size | 2.1MB | 1.7MB | 19% smaller |
```

### 3. Improve Documentation 📖

**Documentation improvements welcome:**
- Fix typos or unclear explanations
- Add usage examples
- Create tutorials
- Improve code comments
- Add architecture diagrams

**Good documentation:**
- Clear and concise
- Includes examples
- Explains the "why" not just "what"
- Has visual aids where helpful

### 4. Code Contributions 💻

**Before submitting code:**

1. **Discuss First** - Open an issue to discuss your idea
2. **Keep It Focused** - One feature/fix per PR
3. **Test Thoroughly** - Ensure changes don't break compatibility
4. **Document Changes** - Update relevant docs

---

## 🔧 Development Setup

### Prerequisites

```bash
# Official Zig 0.15.2 (to build this SDK)
curl -LO https://ziglang.org/download/0.15.2/zig-macos-aarch64-0.15.2.tar.xz
tar xf zig-macos-aarch64-0.15.2.tar.xz
export PATH=$PWD/zig-macos-aarch64-0.15.2:$PATH

# Git
git --version
```

### Build from Source

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/zig_0.1.0_sdk_aarch64.git
cd zig_0.1.0_sdk_aarch64

# Create feature branch
git checkout -b feature/my-improvement

# Build
zig build -Doptimize=ReleaseSafe

# Test
zig build test
```

### Testing Your Changes

```bash
# Run all tests
zig build test

# Test specific component
zig build test --summary all

# Build with your changes
./zig-out/bin/zig version
./zig-out/bin/zig build --help
```

---

## 📝 Contribution Guidelines

### Code Style

- Follow official Zig style guide
- Run `zig fmt` before committing
- Keep lines under 100 characters when reasonable
- Add comments for complex logic

### Commit Messages

**Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `perf:` Performance improvement
- `test:` Test additions or fixes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

**Example:**
```
feat: add profile-guided optimization support

Adds PGO instrumentation and usage flags to build system.
Includes example showing 15% performance improvement on
real-world HTTP server workload.

Closes #42
```

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Commit** your changes with clear messages
4. **Test** thoroughly
5. **Document** your changes
6. **Submit** pull request with description

**PR Template:**
```markdown
## Description
What does this PR do?

## Motivation
Why is this change needed?

## Testing
How was this tested?

## Performance Impact
Any performance implications? (include benchmarks if applicable)

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] MODIFICATIONS.md updated (if applicable)
- [ ] No breaking changes (or clearly documented)
```

---

## 🎯 Priority Areas

### High Priority

1. **Performance Benchmarks**
   - More real-world benchmarks
   - Platform coverage (Linux, Intel Mac)
   - Comparison methodology

2. **Platform Support**
   - Linux x86_64 testing
   - Intel macOS verification
   - Windows support exploration

3. **Documentation**
   - Usage examples
   - Migration guide from standard Zig
   - Performance tuning tips

### Medium Priority

1. **Build System**
   - Pre-built binaries for releases
   - Automated release process
   - Version management

2. **Testing**
   - Regression tests for optimizations
   - Performance test suite
   - Cross-platform validation

3. **Examples**
   - CLI tool example
   - Server application example
   - Library development example

---

## ⚠️ What NOT to Submit

### Changes We Won't Accept

❌ **Breaking Zig 0.15.2 Compatibility**
- Must remain compatible with standard Zig code
- No API changes from upstream

❌ **Removing Safety Checks**
- ReleaseSafe maintains all safety
- No shortcuts that compromise reliability

❌ **Unnecessary Features**
- Keep it focused on optimization
- Avoid feature creep

❌ **Undocumented Changes**
- All changes must be documented
- Update MODIFICATIONS.md for any optimization changes

---

## 🧪 Testing Requirements

### Before Submitting

✅ **All tests pass:**
```bash
zig build test
```

✅ **No regressions:**
```bash
# Compare performance with previous version
# Document any changes (positive or negative)
```

✅ **Cross-platform (if applicable):**
```bash
# Test on relevant platforms
# Document platform-specific behavior
```

✅ **Documentation updated:**
- README.md if user-facing
- MODIFICATIONS.md if technical change
- Code comments if complex logic

---

## 📊 Performance Testing

### Running Benchmarks

```bash
# Create benchmark
# benchmarks/my_benchmark.zig
const std = @import("std");

pub fn main() !void {
    const start = std.time.nanoTimestamp();
    
    // Your benchmark code here
    var i: usize = 0;
    while (i < 1_000_000) : (i += 1) {
        // workload
    }
    
    const end = std.time.nanoTimestamp();
    const duration = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    std.debug.print("Duration: {d:.2}ms\n", .{duration});
}
```

```bash
# Build with this SDK
zig build-exe benchmarks/my_benchmark.zig
./my_benchmark

# Build with official Zig (Debug)
official-zig build-exe benchmarks/my_benchmark.zig -O Debug
./my_benchmark

# Compare results
```

### Benchmark Standards

- **Run 1000+ iterations** - Get median, not just single run
- **Warm up** - Run several times before measuring
- **Document environment** - CPU, RAM, OS version
- **Include variance** - Min, max, stddev if significant
- **Fair comparison** - Same hardware, same workload

---

## 🔍 Code Review Process

### What Reviewers Look For

1. **Correctness**
   - Does the code work as intended?
   - Are edge cases handled?
   - Any potential bugs?

2. **Performance**
   - Does it improve performance?
   - Any regressions?
   - Benchmarks included?

3. **Compatibility**
   - Works with standard Zig code?
   - No breaking changes?
   - Cross-platform considerations?

4. **Code Quality**
   - Readable and maintainable?
   - Well-commented?
   - Follows Zig idioms?

5. **Documentation**
   - Changes documented?
   - Examples provided?
   - Clear explanation?

### Review Timeline

- **Initial Response:** Within 3 days
- **Review:** Within 1 week
- **Merge Decision:** Based on discussion and testing

---

## 🌟 Recognition

Contributors will be:
- Listed in release notes
- Credited in repository
- Thanked in documentation

**Top contributors** may receive:
- Commit access (after sustained contributions)
- Collaborator status
- Input on project direction

---

## 📞 Getting Help

### Questions?

1. **Check Documentation** - README.md, MODIFICATIONS.md
2. **Search Issues** - Someone may have asked already
3. **Ask in Discussions** - Community Q&A
4. **Create Issue** - For specific problems

### Discussion Topics

- **General Help:** "How do I...?"
- **Feature Requests:** "Would be great if..."
- **Performance:** "I noticed that..."
- **Best Practices:** "What's the recommended way to...?"

---

## 🚀 Quick Contribution Checklist

Before submitting your contribution:

- [ ] Tests pass (`zig build test`)
- [ ] Code formatted (`zig fmt`)
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] PR description is complete
- [ ] No breaking changes (or clearly documented)
- [ ] Benchmarks included (if performance-related)

---

## 📜 Code of Conduct

### Our Standards

- **Be Respectful** - Treat everyone with respect
- **Be Constructive** - Provide helpful feedback
- **Be Patient** - Everyone is learning
- **Be Professional** - Keep discussions technical

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Off-topic or spam content

**Violations:** Report to project maintainers. We reserve the right to remove, edit, or reject contributions that don't align with these standards.

---

## 🎉 Thank You!

Every contribution, no matter how small, helps make this SDK better. Whether it's:

- Fixing a typo
- Adding a benchmark
- Submitting a feature
- Helping others in discussions

**Your contribution matters!** 🙏

---

**Questions?** Open an issue or start a discussion!  
**Ready to contribute?** Fork the repo and submit a PR!  
**Want to help but not sure how?** Check the "good first issue" label!

Happy optimizing! ⚡
