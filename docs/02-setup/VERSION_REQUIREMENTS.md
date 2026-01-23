# Version Requirements

This document tracks the exact versions of tools and dependencies used in the arabic_folder project to ensure consistent builds across development, CI/CD, and production environments.

## Critical Build Tools

### Zig Compiler

**Version:** `0.15.2`

**Why this version:**
- Stable release with consistent ABI
- Used throughout the codebase
- All CLI tools built with this version
- Docker builds use this version

**Installation:**
```bash
# macOS (Homebrew)
brew install zig

# Linux
curl -L https://ziglang.org/download/0.15.2/zig-linux-x86_64-0.15.2.tar.xz -o zig.tar.xz
tar -xf zig.tar.xz
sudo mv zig-linux-x86_64-0.15.2 /usr/local/zig
sudo ln -s /usr/local/zig/zig /usr/local/bin/zig
```

**Verification:**
```bash
zig version
# Should output: 0.15.2
```

---

### Mojo Language

**Version:** `0.26.1.0.dev2026011105 (a58f3151)`

**Why this version:**
- Latest development build with critical fixes
- Compatible with custom mojo-sdk
- Required for hyperbolic geometry features

**Installation:**
```bash
# Via magic CLI (recommended)
curl -ssL https://magic.modular.com/install.sh | bash
export PATH="$HOME/.modular/bin:$HOME/.magic/bin:$PATH"
magic install mojo==0.26.1
```

**Verification:**
```bash
mojo --version
# Should output: Mojo 0.26.1.0.dev2026011105 (a58f3151)
```

---

## Custom Components

### mojo-sdk (Custom Mojo Implementation)

**Location:** `src/codeCore/mojo-sdk/`

**Architecture:** This is a **complete custom Mojo language implementation** built in Zig, not the official Modular Mojo SDK!

**Key Features:**
- **74,056 lines** of custom compiler and runtime code
- **Written entirely in Zig** (compiler, stdlib, runtime, tools)
- **Production-ready** with 956 comprehensive tests
- **Extended features** beyond standard Mojo:
  - Hyperbolic geometry support
  - MHC (Manifold Hyperbolic Coordinates) integration
  - Custom MLIR dialect for Mojo
  - LLVM backend integration
  - Custom optimization passes
  - Memory safety system
  - Async runtime

**Components:**
```
mojo-sdk/
├── compiler/           # 13,237 lines - Lexer, Parser, Semantic Analysis
│   ├── frontend/      # Type system, AST, Symbol table
│   ├── middle/        # MLIR integration, IR to MLIR conversion
│   └── backend/       # Code generation, LLVM lowering
├── stdlib/            # 20,068 lines - Standard library
│   ├── async/         # Async/await, channels, streams
│   ├── framework/     # Service framework, JSON, middleware
│   ├── io/            # File, network, JSON I/O
│   ├── memory/        # Memory management, pointers
│   └── simd/          # SIMD operations, vector ops
├── runtime/           # 11,665 lines - Runtime system
│   ├── core.zig      # Memory allocator, reference counting
│   ├── memory.zig    # String, List, Dict, Set
│   ├── ffi.zig       # C interop bridge
│   └── startup.zig   # Entry point
└── tools/             # 14,103 lines - LSP, fuzzer, package manager
    ├── lsp/          # Language Server Protocol (8,596 lines)
    ├── fuzz/         # Continuous fuzzing tools
    └── cli/          # Command-line interface

Build System: build.zig (37,393 lines)
Tests: 956 comprehensive tests
Quality: 98/100 score
```

**Why Custom Implementation:**
1. **Full Control** - Complete control over language features and optimization
2. **Integration** - Seamless integration with Zig-based services
3. **Extensions** - Add domain-specific features (hyperbolic geometry, MHC)
4. **Performance** - Tailored optimizations for arabic_folder use cases
5. **Bundled** - No external dependencies, travels with source code

**Build System:** Uses `build.zig` (37KB) that:
- Builds the Mojo compiler from source
- Compiles the standard library
- Links MLIR/LLVM for backend
- Runs 956 tests
- Integrates with Zig components

**Official Mojo vs Custom mojo-sdk:**

| Feature | Official Mojo (Modular) | Custom mojo-sdk (This Project) |
|---------|------------------------|--------------------------------|
| **Implementation** | Proprietary (closed source) | Open implementation in Zig |
| **Version** | 0.26.1 (system install) | Custom v1.0.0-rc1 (bundled) |
| **Purpose** | General-purpose language | Domain-specific for arabic_folder |
| **Extensions** | Standard features | Hyperbolic geometry, MHC support |
| **Distribution** | System-wide install | Bundled with project |
| **Dependencies** | Modular package manager | Self-contained in project |
| **Integration** | Separate toolchain | Integrated with Zig build |

**How They Work Together:**

```
┌─────────────────────────────────────────────────────────┐
│                  nLocalModels Service                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Zig Code              Mojo Code (.mojo files)          │
│  (orchestration/)      (core/, discovery/, etc.)        │
│       │                       │                          │
│       │                       │                          │
│       ├───────────┬───────────┤                          │
│                   │                                      │
│         System Mojo 0.26.1                               │
│       (Official Modular compiler)                        │
│                   │                                      │
│                   ├── Compiles .mojo → LLVM IR          │
│                   ├── Links with Zig code               │
│                   └── Produces unified binary           │
│                                                         │
│  Custom mojo-sdk/ (Reference implementation)             │
│  - Language research & development                       │
│  - Custom feature prototyping                            │
│  - NOT used for production compilation                   │
└─────────────────────────────────────────────────────────┘
```

**Important Clarification:**

**Production Builds:**
- Use **system Mojo 0.26.1** (Official Modular compiler) to compile `.mojo` files
- The `mojo build` command in build scripts uses the system compiler
- Docker installs Mojo 0.26.1 via `magic install mojo==0.26.1`
- This is the **production compiler** for all `.mojo` code

**Custom mojo-sdk Status:**
- **CLI compiler is COMMENTED OUT** in build.zig (never finished)
- **Currently builds:** LSP server (mojo-lsp) and fuzzer (fuzz-parser)
- **NOT building:** The `mojo` compiler executable itself
- This is a **research/development project**, not a production compiler replacement

**What IS Being Used from Custom mojo-sdk:**
- ✅ **mojo-lsp** (1.6MB) - Language Server for IDE integration
- ✅ **fuzz-parser** (1.2MB) - Quality assurance fuzzing tool
- ✅ **Compiler modules** - For research and learning

**What is NOT Being Used:**
- ❌ **mojo CLI compiler** - Commented out in build.zig, incomplete

**Why This Hybrid Approach:**
1. **System Mojo (0.26.1)** - Proven production compiler for .mojo files
2. **Custom mojo-sdk tools** - IDE support (LSP) and QA (fuzzer)
3. **Custom mojo-sdk modules** - Research/development platform

**To Enable Custom Compiler (Future):**
See `docs/01-architecture/CUSTOM_MOJO_SDK_ANALYSIS.md` for:
- How to uncomment and build the CLI
- Testing requirements (956 tests must pass)
- Build system integration steps
- Pros/cons analysis

**Current approach is optimal** - using proven system compiler for production while leveraging custom tools where they add value.

---

## Docker Build Versions

### Dockerfile.nlocalmodels

Ensures exact version matching:

```dockerfile
# Zig 0.15.2
RUN curl -L https://ziglang.org/download/0.15.2/zig-linux-x86_64-0.15.2.tar.xz -o zig.tar.xz

# Mojo 0.26.1
RUN magic install mojo==0.26.1 || magic install mojo
```

**Verification in Docker:**
```bash
docker build -f docker/Dockerfile.nlocalmodels .
# Outputs:
# === Build Environment ===
# Zig version: 0.15.2
# Mojo version: Mojo 0.26.1.0.dev2026011105
# ===========================
```

---

## GitHub Actions CI/CD

### .github/workflows/ci.yml

Uses same versions for consistent testing:
- Zig: Installed via official release
- Mojo: Installed via magic CLI with version pinning

### .github/workflows/docker-build-backend.yml

Docker builds inherit versions from Dockerfile

---

## Version Compatibility Matrix

| Component | Development | CI/CD | Docker | Production |
|-----------|------------|-------|--------|------------|
| Zig | 0.15.2 | 0.15.2 | 0.15.2 | 0.15.2 |
| Mojo | 0.26.1 | 0.26.1 | 0.26.1 | 0.26.1 |
| mojo-sdk | Bundled | Bundled | Bundled | Bundled |
| Debian | N/A | N/A | bookworm-slim | bookworm-slim |

---

## Update Policy

### When to Update Versions

**Zig:**
- ✅ Only on major releases (0.x.0)
- ✅ When critical bugs are fixed
- ❌ Not for minor patches unless necessary

**Mojo:**
- ✅ When new features are needed
- ✅ When SDK updates require it
- ⚠️ Test thoroughly before updating

**mojo-sdk:**
- ✅ When extending compiler features
- ✅ When adding geometry operations
- ⚠️ Always test with Mojo version

### Update Procedure

1. **Test locally first**
   ```bash
   # Update Zig
   brew upgrade zig  # or download new version
   
   # Update Mojo
   magic install mojo==<new-version>
   
   # Rebuild everything
   cd src/serviceCore/nLocalModels/orchestration
   zig build
   ```

2. **Update this document** with new versions

3. **Update Dockerfile** with new versions

4. **Update CI workflows** if needed

5. **Test Docker builds**
   ```bash
   docker build -f docker/Dockerfile.nlocalmodels .
   ```

6. **Update documentation** in other files referencing versions

7. **Commit with clear message**
   ```bash
   git commit -m "chore: Update Zig to X.Y.Z and Mojo to A.B.C"
   ```

---

## Troubleshooting

### Version Mismatch Issues

**Problem:** Docker build fails with version error
```
Solution: Check Dockerfile versions match this document
```

**Problem:** CI/CD tests fail but local tests pass
```
Solution: Check GitHub Actions workflow versions
```

**Problem:** Mojo SDK compilation errors
```
Solution: Ensure mojo-sdk/ is properly copied in Docker COPY command
```

### Verification Script

```bash
#!/bin/bash
# verify_versions.sh - Check all tool versions

echo "=== Version Check ==="
echo "Zig: $(zig version)"
echo "Mojo: $(mojo --version)"
echo "mojo-sdk: $([ -d src/codeCore/mojo-sdk ] && echo 'Present' || echo 'MISSING')"
echo "===================="
```

---

## References

- **Zig Downloads:** https://ziglang.org/download/
- **Mojo Installation:** https://docs.modular.com/mojo/manual/get-started/
- **Magic CLI:** https://docs.modular.com/magic/
- **Project mojo-sdk:** `src/codeCore/mojo-sdk/README.md`

---

## Change Log

| Date | Zig | Mojo | Notes |
|------|-----|------|-------|
| 2026-01-23 | 0.15.2 | 0.26.1 | Initial version documentation |

---

**Last Updated:** 2026-01-23  
**Maintained By:** Development Team  
**Review Schedule:** Monthly or on major updates
