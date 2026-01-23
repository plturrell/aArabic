# nLang - Custom Language SDKs
## Nucleus Project

This directory contains Custom, audited, and compliance-approved language SDKs for the Nucleus project.

**Service Name:** nLang (following Nucleus naming convention: nLocalModels, nWebServe, nLang)

---

## Directory Structure

```
src/nLang/
├── n-python-sdk/          # Custom Mojo SDK
│   ├── compiler/          # Mojo compiler implementation
│   ├── stdlib/            # Standard library
│   ├── runtime/           # Runtime system
│   ├── tools/             # LSP, fuzzer, CLI tools
│   └── docs/              # Documentation
│
├── n-c-sdk/           # Custom Zig SDK
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

### n-python-sdk

**Purpose:** Custom Mojo SDK for ML/AI workloads
**Version:** Based on Mojo 0.26.1
**Tag Convention:** `n-python-{version}-nucleus-{release}`
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

**Documentation:** See `n-python-sdk/README.md`

---

### n-c-sdk

**Purpose:** Custom Zig SDK for system tools and infrastructure
**Version:** Zig 0.15.2
**Tag Convention:** `n-c-{version}-nucleus-{release}`
**Current Tag:** `n-c-0.15.2-nucleus-1`
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

**Documentation:** See `n-c-sdk/PHASE1_COMPLETE.md`

---

## Naming Convention

### Tag Format

Both SDKs follow the same naming convention:

```
scb-{language}-{version}-nucleus-{release}
```

**Examples:**
- `n-python-0.26.1-nucleus-1`
- `n-c-0.15.2-nucleus-1`
- `n-c-0.15.2-nucleus-2` (after updates)

**Components:**
- `scb` = Bank
- `{language}` = mojo | zig
- `{version}` = Upstream version number
- `nucleus` = Project name
- `{release}` = Bank internal release number

---

## Compliance & Security

### Banking Requirements

Both SDKs are subject to:
- ✅ **Supply Chain Security** - Full source code audit
- ✅ **Compliance Review** - SOX, PCI-DSS, Basel III/IV
- ✅ **Internal CI/CD** - Controlled build and deployment
- ✅ **Binary Signing** - Code signing with Bank certificates
- ✅ **Audit Trail** - Complete documentation and approvals

### Security Audit Status

| SDK | Phase | Status | Next Steps |
|-----|-------|--------|------------|
| **n-python-sdk** | Research | ✅ Active | Continue development |
| **n-c-sdk** | Phase 1 | ✅ Complete | Phase 2: Security Audit |

---

## Usage

### n-python-sdk

```bash
# Built tools (already available)
cd src/nLang/n-python-sdk/zig-out/bin
./mojo-lsp        # LSP server for IDEs
./fuzz-parser     # Fuzzing tool

# For production Mojo compilation
# Use system Mojo 0.26.1 (not custom SDK)
mojo build your-file.mojo
```

### n-c-sdk

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
2. Reference n-c-sdk for source code audit
3. Build tools with system Zig
4. Maintain compliance documentation

---

## Git Configuration

### n-python-sdk

```bash
# Check remotes
cd src/nLang/n-python-sdk
git remote -v

# Internal repository (to be configured)
# upstream: https://github.com/modular/mojo.git (if available)
# origin: git@internal-gitlab.scb.com:nucleus/mojo-sdk.git
```

### n-c-sdk

```bash
# Check remotes
cd src/nLang/n-c-sdk
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
cd src/nLang/n-c-sdk
git tag | grep n-c

# Checkout specific version
git checkout n-c-0.15.2-nucleus-1

# Create new release
git tag -a n-c-0.15.2-nucleus-2 -m "Release notes"
```

---

## Integration with Nucleus Project

### Project Structure

```
arabic_folder/
├── src/
│   ├── nLang/                 # Custom SDKs (this directory)
│   │   ├── n-python-sdk/     # ML/AI language
│   │   └── n-c-sdk/      # Systems language
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

**n-python-sdk:**
- `n-python-sdk/README.md` - SDK overview
- `docs/01-architecture/CUSTOM_MOJO_SDK_ANALYSIS.md` - Detailed analysis

**n-c-sdk:**
- `n-c-sdk/PHASE1_COMPLETE.md` - Phase 1 completion report
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
- [x] n-c-sdk Phase 1: Repository setup
- [ ] n-c-sdk Phase 2: Security audit
- [ ] n-c-sdk Phase 3: Build pipeline
- [ ] n-c-sdk Phase 4: Compliance approval

### Q2 2026
- [ ] n-python-sdk: Rename and restructure
- [ ] n-python-sdk: Security audit
- [ ] Both: Production approval
- [ ] Integration testing

### Q3-Q4 2026
- [ ] Automated update pipeline
- [ ] Quarterly version updates
- [ ] Annual audit preparation
- [ ] Team training program

---

## Approval Status

### n-python-sdk
- **Platform Engineering:** ✅ Approved (Research use)
- **Security Team:** ⏳ Pending (Full audit)
- **Compliance:** ⏳ Pending (Review)
- **Production:** ❌ Not approved (use system Mojo)

### n-c-sdk
- **Platform Engineering:** ✅ Approved (Phase 1)
- **Security Team:** ⏳ Pending (Phase 2)
- **Compliance:** ⏳ Pending (Phase 4)
- **Production:** ⏳ Pending (Full approval)

---

**Last Updated:** 2026-01-23  
**Owner:** Platform Engineering Team  
**Review Cycle:** Quarterly  
**Classification:** Internal - Bank Confidential
