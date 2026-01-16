# Quick Start Guide

Get up and running with `mojo-pkg` in 5 minutes!

## Installation

### 1. Build from Source

```bash
cd src/serviceCore/serviceShimmy-mojo/mojo-sdk/tools/pkg
zig build
```

The executable is at: `zig-out/bin/mojo-pkg`

### 2. Add to PATH (Optional)

```bash
# Add to your shell profile (~/.zshrc, ~/.bashrc, etc.)
export PATH="/path/to/tools/pkg/zig-out/bin:$PATH"

# Or copy to system location
sudo cp zig-out/bin/mojo-pkg /usr/local/bin/
```

### 3. Verify Installation

```bash
mojo-pkg help
```

You should see the help text!

---

## Your First Package (2 minutes)

### Create a Package

```bash
# Create and enter directory
mkdir my-first-mojo
cd my-first-mojo

# Initialize package
mojo-pkg init my-first-mojo
```

**Result:** Creates `mojo.toml`:
```toml
[package]
name = "my-first-mojo"
version = "0.1.0"
authors = []

[dependencies]
```

### Add a Dependency (Optional)

```bash
mojo-pkg add http-client ^1.0.0
```

**Result:** Updates `mojo.toml`:
```toml
[dependencies]
http-client = "^1.0.0"
```

### Install Dependencies

```bash
mojo-pkg install
```

**Result:** 
- Resolves all dependencies
- Creates `build.zig.zon` with dependency info

### Build Your Package

```bash
# Debug build
mojo-pkg build

# Release build  
mojo-pkg build --release
```

**Result:** Executable in `zig-out/bin/`

### Run Tests

```bash
mojo-pkg test
```

---

## Your First Workspace (3 minutes)

### Create Workspace Structure

```bash
# Create workspace root
mkdir my-workspace
cd my-workspace

# Initialize as workspace
mojo-pkg workspace new my-workspace
```

**Result:** Creates workspace `mojo.toml`:
```toml
[workspace]
members = []
```

### Add Member Packages

```bash
# Create core library
mkdir core
cd core
mojo-pkg init core
cd ..

# Create app that uses core
mkdir app
cd app
mojo-pkg init app

# Add core as dependency
mojo-pkg add core path:../core

cd ..
```

### Update Workspace Members

Edit `mojo.toml` in workspace root:
```toml
[workspace]
members = [
    "core",
    "app"
]
```

### Install & Build

```bash
# From workspace root
mojo-pkg install

# Build specific member
cd app
mojo-pkg build

# Or build from root
cd ..
mojo-pkg build
```

---

## Common Commands

### Package Commands

```bash
# Initialize new package
mojo-pkg init <name>

# Add dependency
mojo-pkg add <name> <version>
mojo-pkg add <name> path:<path>

# Install all dependencies
mojo-pkg install

# Build package
mojo-pkg build              # Debug
mojo-pkg build --release    # Optimized

# Run tests
mojo-pkg test

# Show help
mojo-pkg help
```

### Workspace Commands

```bash
# Create workspace
mojo-pkg workspace new <name>

# List workspace members
mojo-pkg workspace list
```

---

## Version Constraints Quick Reference

```toml
[dependencies]
# Caret: compatible updates (>=1.2.0 <2.0.0)
http-client = "^1.2.0"

# Exact version only
logger = "1.5.3"

# Minimum version
database = ">=3.0.0"

# Path dependency (workspace member)
my-lib = { path = "../my-lib" }
```

---

## Real-World Examples

### Example 1: Simple CLI Tool

```bash
mkdir cli-tool
cd cli-tool
mojo-pkg init cli-tool
mojo-pkg add args-parser ^2.0.0
mojo-pkg install
mojo-pkg build --release
```

### Example 2: Web Service

```bash
mkdir web-api
cd web-api
mojo-pkg init web-api
mojo-pkg add http-server ^3.0.0
mojo-pkg add json-parser ^1.0.0
mojo-pkg add logger ^0.5.0
mojo-pkg install
mojo-pkg build --release
```

### Example 3: Library

```bash
mkdir utils-lib
cd utils-lib
mojo-pkg init utils-lib
# No dependencies needed
mojo-pkg build
mojo-pkg test
```

### Example 4: Monorepo

```bash
mkdir monorepo
cd monorepo
mojo-pkg workspace new monorepo

# Create shared library
mkdir lib && cd lib
mojo-pkg init shared-lib
cd ..

# Create services
mkdir services && cd services

mkdir api && cd api
mojo-pkg init api-service
mojo-pkg add shared-lib path:../../lib
cd ..

mkdir worker && cd worker
mojo-pkg init worker-service
mojo-pkg add shared-lib path:../../lib
cd ../..

# Update workspace members
cat > mojo.toml << EOF
[workspace]
members = [
    "lib",
    "services/api",
    "services/worker"
]
EOF

# Build all
mojo-pkg install
```

---

## Troubleshooting

### Command Not Found

**Issue:** `mojo-pkg: command not found`

**Solution:**
```bash
# Use full path
/path/to/tools/pkg/zig-out/bin/mojo-pkg help

# Or add to PATH
export PATH="/path/to/tools/pkg/zig-out/bin:$PATH"
```

### Build Fails

**Issue:** `zig build` fails

**Solution:**
1. Ensure Zig 0.15.2+ is installed
2. Run `mojo-pkg install` first
3. Check for errors in `mojo.toml`

### Dependency Not Found

**Issue:** Can't resolve dependency

**Solution:**
1. Check dependency name spelling
2. Verify version exists
3. For path deps, check path is correct

---

## Next Steps

### Learn More

- **Full Documentation:** See [README.md](README.md)
- **Practical Examples:** See [EXAMPLES.md](EXAMPLES.md)
- **API Reference:** See [API.md](API.md)
- **Changelog:** See [CHANGELOG.md](CHANGELOG.md)

### Use Cases

- **Standalone Package:** Perfect for mojo-sdk
- **Workspace:** Perfect for serviceShimmy-mojo
- **Monorepo:** Manage multiple related packages

### Advanced Topics

- Workspace-aware dependency resolution
- Path dependencies between packages
- Build modes (Debug, Release, etc.)
- Custom build configurations

---

## Tips & Best Practices

### 1. Use Workspaces for Related Packages

If you have multiple packages that work together, use a workspace!

### 2. Use Path Dependencies for Local Development

Perfect for developing multiple packages simultaneously.

### 3. Pin Critical Dependencies

Use exact versions for critical dependencies:
```toml
important-lib = "1.2.3"  # Exact
```

### 4. Use Caret for Libraries

Allow compatible updates:
```toml
utils = "^2.0.0"  # >=2.0.0 <3.0.0
```

### 5. Run Tests Regularly

```bash
mojo-pkg test
```

### 6. Build Release for Production

```bash
mojo-pkg build --release
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│          MOJO PACKAGE MANAGER                   │
│                mojo-pkg v0.1.0                  │
├─────────────────────────────────────────────────┤
│ PACKAGE COMMANDS                                │
│   mojo-pkg init <name>       New package        │
│   mojo-pkg add <name> <ver>  Add dependency     │
│   mojo-pkg install           Install deps       │
│   mojo-pkg build             Build (debug)      │
│   mojo-pkg build --release   Build (optimized)  │
│   mojo-pkg test              Run tests          │
├─────────────────────────────────────────────────┤
│ WORKSPACE COMMANDS                              │
│   mojo-pkg workspace new     Create workspace   │
│   mojo-pkg workspace list    List members       │
├─────────────────────────────────────────────────┤
│ VERSION CONSTRAINTS                             │
│   ^1.2.0    Compatible (>=1.2.0 <2.0.0)        │
│   1.2.0     Exact version                       │
│   >=1.0.0   Minimum version                     │
│   path:..   Local path                          │
├─────────────────────────────────────────────────┤
│ FILES                                           │
│   mojo.toml      Package/workspace manifest     │
│   build.zig      Generated build script         │
│   build.zig.zon  Generated dependencies         │
└─────────────────────────────────────────────────┘
```

---

**That's it! You're ready to use mojo-pkg! 🚀**

For more examples and details, see the full documentation.
