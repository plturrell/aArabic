# nLang - SCB Custom Language SDKs
## Standard Chartered Bank - Nucleus Project

This directory contains Standard Chartered Bank's custom, audited, and compliance-approved language SDKs for the Nucleus project.

**Service Name:** nLang (following Nucleus naming convention: nLocalModels, nWebServe, nLang)

---

## Directory Structure

```
src/nLang/
├── scb-mojo-sdk/          # SCB Custom Mojo SDK
│   ├── compiler/          # Mojo compiler implementation
│   ├── stdlib/            # Standard library
│   ├── runtime/           # Runtime system
│   ├── tools/             # LSP, fuzzer, CLI tools
│   └── docs/              # Documentation
│
├── scb-zig-sdk/           # SCB Custom Zig SDK
│   ├── lib/               # Zig standard library
│   ├── src/               # Zig compiler source
│   ├── test/              # Test suite
│   ├── doc/               # Documentation
│   └── security/          # Security audit reports
│
└── README.md              # This file
```

---

## SDK Overview

### scb-mojo-sdk

**Purpose:** Custom Mojo SDK for ML/AI workloads
**Version:** Based on Mojo 0.26.1
**Tag Convention:** `scb-mojo-{version}-nucleus-{release}`
**Status:** Research & Development

**Key Components:**
- ✅ **Compiler:** Full compiler implementation (13,237 lines)
- ✅ **Standard Library:** Complete stdlib (20,068 lines)
- ✅ **Runtime:** Runtime system (11,665 lines)
- ✅ **LSP Server:** `mojo-lsp` for IDE integration
- ✅ **Fuzzer:** `fuzz-parser` for quality assurance
- ⏸️ **CLI:** Commented out (use system Mojo for production)

**Use Cases:**
- Language research and prototyping
- Custom ML/AI optimizations
- IDE integration (LSP)
- Quality assurance (fuzzing)
- Understanding Mojo internals

**Documentation:** See `scb-mojo-sdk/README.md`

---

### scb-zig-sdk

**Purpose:** Custom Zig SDK for system tools and infrastructure
**Version:** Zig 0.15.2
**Tag Convention:** `scb-zig-{version}-nucleus-{release}`
**Current Tag:** `scb-zig-0.15.2-nucleus-1`
**Status:** Phase 1 Complete (Repository Setup)

**Key Components:**
- ✅ **Source Code:** Full Zig 0.15.2 source (20,511 files)
- ✅ **Compiler:** Zig compiler source
- ✅ **Standard Library:** Complete std library
- ✅ **Test Suite:** Comprehensive tests
- ✅ **Security:** Audit structure initialized

**Use Cases:**
- System-level tools (dataset_loader, benchmark_validator)
- Performance-critical infrastructure
- Compliance-approved compilation toolchain
- Reproducible builds for regulated environment

**Documentation:** See `scb-zig-sdk/PHASE1_COMPLETE.md`

---

## Naming Convention

### Tag Format

Both SDKs follow the same naming convention:

```
scb-{language}-{version}-nucleus-{release}
```

**Examples:**
- `scb-mojo-0.26.1-nucleus-1`
- `scb-zig-0.15.2-nucleus-1`
- `scb-zig-0.15.2-nucleus-2` (after updates)

**Components:**
- `scb` = Standard Chartered Bank
- `{language}` = mojo | zig
- `{version}` = Upstream version number
- `nucleus` = Project name
- `{release}` = SCB internal release number

---

## Compliance & Security

### Banking Requirements

Both SDKs are subject to:
- ✅ **Supply Chain Security** - Full source code audit
- ✅ **Compliance Review** - SOX, PCI-DSS, Basel III/IV
- ✅ **Internal CI/CD** - Controlled build and deployment
- ✅ **Binary Signing** - Code signing with SCB certificates
- ✅ **Audit Trail** - Complete documentation and approvals

### Security Audit Status

| SDK | Phase | Status | Next Steps |
|-----|-------|--------|------------|
| **scb-mojo-sdk** | Research | ✅ Active | Continue development |
| **scb-zig-sdk** | Phase 1 | ✅ Complete | Phase 2: Security Audit |

---

## Usage

### scb-mojo-sdk

```bash
# Built tools (already available)
cd src/nLang/scb-mojo-sdk/zig-out/bin
./mojo-lsp        # LSP server for IDEs
./fuzz-parser     # Fuzzing tool

# For production Mojo compilation
# Use system Mojo 0.26.1 (not custom SDK)
mojo build your-file.mojo
```

### scb-zig-sdk

```bash
# Recommended: Use system Zig binary
export PATH="/opt/homebrew/bin:$PATH"
zig version  # Should show 0.15.2

# Build Nucleus tools
cd src/serviceCore/nLocalModels/orchestration
zig build

# Artifacts
./zig-out/bin/dataset_loader
./zig-out/bin/benchmark_validator
./zig-out/bin/hf_extractor
```

---

## Development Workflow

### For Mojo Development

1. Use custom SDK for research/prototyping
2. Use LSP for IDE integration
3. Use system Mojo for production builds
4. Maintain audit trail for changes

### For Zig Development

1. Use system Zig 0.15.2 for compilation
2. Reference scb-zig-sdk for source code audit
3. Build tools with system Zig
4. Maintain compliance documentation

---

## Git Configuration

### scb-mojo-sdk

```bash
# Check remotes
cd src/nLang/scb-mojo-sdk
git remote -v

# Internal repository (to be configured)
# upstream: https://github.com/modular/mojo.git (if available)
# origin: git@internal-gitlab.scb.com:nucleus/mojo-sdk.git
```

### scb-zig-sdk

```bash
# Check remotes
cd src/nLang/scb-zig-sdk
git remote -v

# Already configured
# upstream: https://github.com/ziglang/zig.git
# origin: git@internal-gitlab.scb.com:nucleus/zig-sdk.git
```

---

## Maintenance

### Update Schedule

- **Security Patches:** As needed (within 48 hours for critical)
- **Version Updates:** Quarterly review
- **Audit Cycle:** Annual
- **Compliance Review:** Per release

### Version Management

```bash
# List available tags
cd src/nLang/scb-zig-sdk
git tag | grep scb-zig

# Checkout specific version
git checkout scb-zig-0.15.2-nucleus-1

# Create new release
git tag -a scb-zig-0.15.2-nucleus-2 -m "Release notes"
```

---

## Integration with Nucleus Project

### Project Structure

```
arabic_folder/
├── src/
│   ├── nLang/                 # Custom SDKs (this directory)
│   │   ├── scb-mojo-sdk/     # ML/AI language
│   │   └── scb-zig-sdk/      # Systems language
│   │
│   └── serviceCore/           # Built with these SDKs
│       └── nLocalModels/
│           └── orchestration/
│               ├── dataset_loader.zig
│               ├── benchmark_validator.zig
│               └── hf_model_card_extractor.zig
```

### Build Dependencies

**Mojo SDK:**
- Used for: ML/AI research, custom optimizations
- Production: System Mojo 0.26.1

**Zig SDK:**
- Used for: Source audit, compliance
- Production: System Zig 0.15.2 (builds tools)

---

## Documentation

### Available Documents

**scb-mojo-sdk:**
- `scb-mojo-sdk/README.md` - SDK overview
- `docs/01-architecture/CUSTOM_MOJO_SDK_ANALYSIS.md` - Detailed analysis

**scb-zig-sdk:**
- `scb-zig-sdk/PHASE1_COMPLETE.md` - Phase 1 completion report
- `docs/01-architecture/CUSTOM_ZIG_SDK_ANALYSIS.md` - Banking compliance analysis

**General:**
- `docs/02-setup/VERSION_REQUIREMENTS.md` - Version requirements
- `scripts/setup_scb_zig_sdk.sh` - Automated setup script

---

## Support

### Technical Issues
- **Email:** nucleus-platform@scb.com
- **Slack:** #nucleus-platform

### Security Questions
- **Email:** nucleus-security@scb.com
- **Escalation:** Security Team Lead

### Compliance
- **Email:** nucleus-compliance@scb.com
- **Escalation:** Compliance Officer

---

## Roadmap

### Q1 2026
- [x] scb-zig-sdk Phase 1: Repository setup
- [ ] scb-zig-sdk Phase 2: Security audit
- [ ] scb-zig-sdk Phase 3: Build pipeline
- [ ] scb-zig-sdk Phase 4: Compliance approval

### Q2 2026
- [ ] scb-mojo-sdk: Rename and restructure
- [ ] scb-mojo-sdk: Security audit
- [ ] Both: Production approval
- [ ] Integration testing

### Q3-Q4 2026
- [ ] Automated update pipeline
- [ ] Quarterly version updates
- [ ] Annual audit preparation
- [ ] Team training program

---

## Approval Status

### scb-mojo-sdk
- **Platform Engineering:** ✅ Approved (Research use)
- **Security Team:** ⏳ Pending (Full audit)
- **Compliance:** ⏳ Pending (Review)
- **Production:** ❌ Not approved (use system Mojo)

### scb-zig-sdk
- **Platform Engineering:** ✅ Approved (Phase 1)
- **Security Team:** ⏳ Pending (Phase 2)
- **Compliance:** ⏳ Pending (Phase 4)
- **Production:** ⏳ Pending (Full approval)

---

**Last Updated:** 2026-01-23  
**Owner:** Platform Engineering Team  
**Review Cycle:** Quarterly  
**Classification:** Internal - SCB Confidential
