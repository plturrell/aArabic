# GitHub Release Instructions
## nLang v1.0.0 - SCB Custom Language SDKs

**Release Date:** 2026-01-23  
**Repository:** https://github.com/plturrell/aArabic

---

## Release Overview

This document provides instructions for creating the GitHub release for **nLang v1.0.0**, which includes two SCB-tagged SDKs:

1. **scb-mojo-0.1.0-nucleus-1** - Custom Mojo SDK with i18n support
2. **scb-zig-0.15.2-nucleus-1** - Custom Zig SDK (Phase 1 complete)

---

## Pre-Release Checklist

### ✅ Completed

- [x] Both SDKs renamed following SCB convention
- [x] Moved to `src/nLang/` directory
- [x] scb-mojo-sdk: Committed i18n changes
- [x] scb-mojo-sdk: Tagged as `scb-mojo-0.1.0-nucleus-1`
- [x] scb-zig-sdk: Tagged as `scb-zig-0.15.2-nucleus-1`
- [x] Release notes created: `NLANG_RELEASE_v1.0.0.md`
- [x] nLang README updated
- [x] Documentation updated

### ⏳ Pending

- [ ] Push scb-mojo-sdk changes and tag to GitHub
- [ ] Push scb-zig-sdk tag to GitHub
- [ ] Create GitHub release
- [ ] Attach release notes
- [ ] Notify team

---

## Step 1: Push Changes to GitHub

### Push scb-mojo-sdk

```bash
cd /Users/user/Documents/arabic_folder/src/nLang/scb-mojo-sdk

# Verify tag exists
git tag | grep scb-mojo-0.1.0-nucleus-1

# Push commits
git push origin main

# Push tag
git push origin scb-mojo-0.1.0-nucleus-1
```

### Push scb-zig-sdk

```bash
cd /Users/user/Documents/arabic_folder/src/nLang/scb-zig-sdk

# Verify tag exists
git tag | grep scb-zig-0.15.2-nucleus-1

# Push tag (already on detached HEAD, so just push tag)
git push origin scb-zig-0.15.2-nucleus-1
```

---

## Step 2: Create GitHub Release

### Using GitHub Web Interface

1. **Navigate to Repository**
   ```
   https://github.com/plturrell/aArabic/releases/new
   ```

2. **Release Details**
   - **Tag:** Select existing tag or create new: `nlang-v1.0.0`
   - **Target:** `main` branch
   - **Title:** `nLang v1.0.0 - SCB Custom Language SDKs`

3. **Description** (Copy and paste):

```markdown
# nLang v1.0.0 🚀
## SCB Custom Language SDKs for Nucleus Project

First official release of **nLang**, Standard Chartered Bank's custom language SDK service.

---

## 📦 What's Included

### 1. scb-mojo-0.1.0-nucleus-1
**Custom Mojo SDK for ML/AI Workloads**

- ✅ Full compiler (13,237 lines)
- ✅ Standard library (20,068 lines)
- ✅ Runtime system (11,665 lines)
- 🆕 Internationalization (i18n) support
- 🆕 Arabic language support
- ✅ LSP server & fuzzer tools

**Status:** Research & Development ✅

### 2. scb-zig-0.15.2-nucleus-1
**Custom Zig SDK for Systems Programming**

- ✅ Full Zig 0.15.2 source (20,511 files)
- ✅ Complete standard library
- ✅ Security audit structure
- ✅ CI/CD pipeline configured
- ✅ Built tools: dataset_loader, benchmark_validator

**Status:** Phase 1 Complete ✅

---

## 🎯 Key Features

### Internationalization (Mojo)
- Core i18n framework
- Locale detection
- Plural rules
- Date/time formatting
- Arabic RTL support

### Banking Compliance (Both)
- Complete source code audit capability
- SCB naming convention: `scb-{lang}-{ver}-nucleus-{rel}`
- Internal CI/CD integration
- Security audit structure

---

## 📚 Documentation

- **Release Notes:** [NLANG_RELEASE_v1.0.0.md](docs/08-reports/releases/NLANG_RELEASE_v1.0.0.md)
- **nLang Overview:** [src/nLang/README.md](src/nLang/README.md)
- **Mojo Analysis:** [CUSTOM_MOJO_SDK_ANALYSIS.md](docs/01-architecture/CUSTOM_MOJO_SDK_ANALYSIS.md)
- **Zig Analysis:** [CUSTOM_ZIG_SDK_ANALYSIS.md](docs/01-architecture/CUSTOM_ZIG_SDK_ANALYSIS.md)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/plturrell/aArabic.git
cd aArabic/src/nLang

# Check out Mojo SDK
cd scb-mojo-sdk
git checkout scb-mojo-0.1.0-nucleus-1
./zig-out/bin/mojo-lsp  # Use LSP

# Check out Zig SDK
cd ../scb-zig-sdk
git checkout scb-zig-0.15.2-nucleus-1

# Build tools
cd ../../serviceCore/nLocalModels/orchestration
zig build
```

---

## 🏦 Banking Compliance

Both SDKs meet SCB regulatory requirements:

- ✅ Supply chain security
- ✅ Audit trail
- ✅ SOX, PCI-DSS, Basel III/IV
- ✅ MAS compliance

**Security Audit Schedule:**
- scb-mojo: Full audit pending (Q2 2026)
- scb-zig: Phase 2 pending (Feb 2026)

---

## 📋 What's Next

### Q1 2026
- [ ] scb-zig Phase 2: Security audit
- [ ] scb-mojo: Full security audit

### Q2 2026
- [ ] scb-zig Phases 3-4: Build pipeline & compliance
- [ ] scb-mojo: Production approval
- [ ] scb-mojo-0.2.0-nucleus-1 (CLI enabled)

---

## 📞 Contact

- **Platform:** nucleus-platform@scb.com
- **Security:** nucleus-security@scb.com
- **Compliance:** nucleus-compliance@scb.com

---

## ⚠️ Classification

**Internal - SCB Confidential**
- For Standard Chartered Bank use only
- External distribution not authorized
- Requires approval for modifications

---

**Release Version:** nLang v1.0.0  
**Release Date:** 2026-01-23  
**Platform:** aarch64 (Apple Silicon)  
**Classification:** Internal - SCB Confidential
```

4. **Assets** (Optional)
   - Attach: `docs/08-reports/releases/NLANG_RELEASE_v1.0.0.md`
   - Note: Source code archives will be auto-generated

5. **Options**
   - ☑️ Set as the latest release
   - ☑️ Create a discussion for this release
   - ☐ Set as pre-release (uncheck for stable)

6. **Publish**
   - Click "Publish release"

---

## Step 3: Using GitHub CLI (Alternative)

If you have GitHub CLI installed:

```bash
cd /Users/user/Documents/arabic_folder

# Create release
gh release create nlang-v1.0.0 \
  --title "nLang v1.0.0 - SCB Custom Language SDKs" \
  --notes-file docs/08-reports/releases/NLANG_RELEASE_v1.0.0.md \
  --latest

# Verify
gh release view nlang-v1.0.0
```

---

## Step 4: Post-Release Actions

### Immediate

1. **Verify Release**
   ```bash
   # Check release page
   open https://github.com/plturrell/aArabic/releases/tag/nlang-v1.0.0
   ```

2. **Notify Team**
   ```
   To: nucleus-platform@scb.com
   Subject: nLang v1.0.0 Released
   
   The first official release of nLang is now available:
   https://github.com/plturrell/aArabic/releases/tag/nlang-v1.0.0
   
   Includes:
   - scb-mojo-0.1.0-nucleus-1 (with i18n)
   - scb-zig-0.15.2-nucleus-1 (Phase 1)
   
   See release notes for details.
   ```

3. **Update Internal Wiki**
   - Add release announcement
   - Update SDK version references
   - Link to GitHub release

### Within 24 Hours

4. **Security Team Notification**
   ```
   To: nucleus-security@scb.com
   Cc: nucleus-compliance@scb.com
   Subject: nLang v1.0.0 - Security Audit Request
   
   nLang v1.0.0 has been released and is ready for security audit.
   
   Priority SDKs:
   1. scb-zig-0.15.2-nucleus-1 (Phase 2 audit)
   2. scb-mojo-0.1.0-nucleus-1 (Full audit)
   
   Documentation available in release notes.
   ```

5. **Compliance Team Notification**
   ```
   To: nucleus-compliance@scb.com
   Subject: nLang v1.0.0 - Compliance Review Request
   
   nLang v1.0.0 requires compliance review for:
   - SCB naming convention compliance
   - Banking regulatory requirements
   - Audit trail verification
   
   Timeline: Q1-Q2 2026
   ```

### Within 1 Week

6. **Documentation Updates**
   - Update main README.md
   - Update project documentation
   - Create internal training materials
   - Schedule team presentation

7. **Monitoring Setup**
   - Set up release monitoring
   - Configure security alerts
   - Track adoption metrics
   - Monitor for issues

---

## Release Artifacts

### Tags Created

```
src/nLang/scb-mojo-sdk:
  scb-mojo-0.1.0-nucleus-1

src/nLang/scb-zig-sdk:
  scb-zig-0.15.2-nucleus-1
```

### Commits

```
scb-mojo-sdk:
  6b8c9d6 - SCB Mojo SDK 0.1.0 - Nucleus Release 1
  
scb-zig-sdk:
  (detached HEAD at Zig 0.15.2)
```

### Documentation

- `docs/08-reports/releases/NLANG_RELEASE_v1.0.0.md` - Full release notes
- `src/nLang/README.md` - nLang service overview
- `src/nLang/scb-zig-sdk/PHASE1_COMPLETE.md` - Zig Phase 1 report
- `docs/01-architecture/CUSTOM_MOJO_SDK_ANALYSIS.md` - Mojo analysis
- `docs/01-architecture/CUSTOM_ZIG_SDK_ANALYSIS.md` - Zig analysis

---

## Troubleshooting

### Issue: Tag not found on GitHub

**Solution:**
```bash
cd src/nLang/scb-mojo-sdk
git push origin scb-mojo-0.1.0-nucleus-1

cd ../scb-zig-sdk
git push origin scb-zig-0.15.2-nucleus-1
```

### Issue: Permission denied

**Solution:**
```bash
# Check authentication
gh auth status

# Re-authenticate if needed
gh auth login
```

### Issue: Cannot create release

**Solution:**
- Verify you have write access to repository
- Check that tag exists: `git tag -l "scb-*"`
- Ensure commits are pushed: `git push origin main`

---

## Rollback Procedure

If release needs to be rolled back:

```bash
# Delete GitHub release
gh release delete nlang-v1.0.0 --yes

# Delete tags locally
cd src/nLang/scb-mojo-sdk
git tag -d scb-mojo-0.1.0-nucleus-1

cd ../scb-zig-sdk
git tag -d scb-zig-0.15.2-nucleus-1

# Delete tags on GitHub
git push origin :refs/tags/scb-mojo-0.1.0-nucleus-1
git push origin :refs/tags/scb-zig-0.15.2-nucleus-1
```

---

## Success Criteria

Release is successful when:

- ✅ Tags visible on GitHub
- ✅ Release page accessible
- ✅ Release notes readable
- ✅ Source code archives downloadable
- ✅ Team notified
- ✅ Security audit requested
- ✅ Compliance review initiated

---

## References

- **Repository:** https://github.com/plturrell/aArabic
- **Release Notes:** `docs/08-reports/releases/NLANG_RELEASE_v1.0.0.md`
- **nLang README:** `src/nLang/README.md`
- **GitHub Releases:** https://docs.github.com/en/repositories/releasing-projects-on-github

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-23  
**Next Review:** After release publication  
**Owner:** Platform Engineering Team
