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

### mojo-sdk

**Location:** `src/serviceCore/nLocalModels/mojo-sdk/`

**Purpose:** Custom Mojo SDK with:
- Extended compiler features
- Hyperbolic geometry support
- MHC (Manifold Hyperbolic Coordinates) integration
- Custom optimization passes

**Important:** This is a **bundled component** that travels with the source code. Docker builds use this exact SDK to ensure consistency.

**Build System:** Uses `build.zig` for integration with Zig components

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
echo "mojo-sdk: $([ -d src/serviceCore/nLocalModels/mojo-sdk ] && echo 'Present' || echo 'MISSING')"
echo "===================="
```

---

## References

- **Zig Downloads:** https://ziglang.org/download/
- **Mojo Installation:** https://docs.modular.com/mojo/manual/get-started/
- **Magic CLI:** https://docs.modular.com/magic/
- **Project mojo-sdk:** `src/serviceCore/nLocalModels/mojo-sdk/README.md`

---

## Change Log

| Date | Zig | Mojo | Notes |
|------|-----|------|-------|
| 2026-01-23 | 0.15.2 | 0.26.1 | Initial version documentation |

---

**Last Updated:** 2026-01-23  
**Maintained By:** Development Team  
**Review Schedule:** Monthly or on major updates
