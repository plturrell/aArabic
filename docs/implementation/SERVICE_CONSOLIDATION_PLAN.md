# Service Consolidation Plan - Remove Duplicates

**Date:** 2026-01-11  
**Goal:** Consolidate replicated services (RAG, Embedding, Translation) to single implementations  

---

## 🎯 Current State Analysis

### Identified Duplicate Services

#### 1. **Embedding Services** (3 versions)
- `serviceEmbedding/` - Original Python implementation
- `serviceEmbedding-mojo/` - **✅ PRODUCTION READY** (10-25x faster, documented, deployed)
- `serviceEmbedding-rust/` - Rust/Burn implementation (experimental)

#### 2. **Translation Services** (2 versions)
- `serviceTranslation-mojo/` - Mojo implementation
- `serviceTranslation-rust/` - Rust implementation

#### 3. **RAG Services** (3 versions)
- `serviceRAG-mojo/` - Mojo RAG implementation
- `serviceRAG-rust/` - Rust RAG implementation
- `serviceRAG-zig-mojo/` - **🚧 IN PROGRESS** Zig+Mojo hybrid (not built, needs Zig)

---

## 📋 Consolidation Strategy

### Decision Criteria:
1. **Production Readiness**: Is it documented, tested, deployed?
2. **Performance**: Does it show significant improvements?
3. **Maintenance**: Is it actively maintained?
4. **Dependencies**: Are all dependencies available?

---

## 🎯 Recommended Actions

### **Embedding Services**

**KEEP:** `serviceEmbedding-mojo/`
- ✅ Production ready
- ✅ Fully documented (33+ pages)
- ✅ Docker deployment complete
- ✅ 10-25x performance gains verified
- ✅ Real ML models integrated
- ✅ Redis caching ready
- ✅ Port 8007 operational

**REMOVE:**
- ❌ `serviceEmbedding/` - Original Python (superseded by Mojo)
- ❌ `serviceEmbedding-rust/` - Experimental (Burn framework, unfinished)

### **Translation Services**

**NEED TO EVALUATE:**
- Both `serviceTranslation-mojo/` and `serviceTranslation-rust/` need review
- Check which is production-ready
- Check which has better documentation
- Likely keep Mojo version for consistency with embedding service

**ACTION:** Investigate both before deciding

### **RAG Services**

**KEEP:** `serviceRAG-zig-mojo/` (Future potential)
- 🚧 Most advanced architecture (Zig I/O + Mojo SIMD)
- Code complete, just needs Zig compiler
- Best performance potential (10-100x)
- Native binary, no Python runtime

**REMOVE:**
- ❌ `serviceRAG-mojo/` - Superseded by Zig+Mojo hybrid
- ❌ `serviceRAG-rust/` - Experimental, Rust not chosen as primary language

---

## 📦 Detailed Removal Plan

### Phase 1: Verify What to Keep

```bash
# Check Mojo embedding service status
curl http://localhost:8007/health

# Check if any services are currently running
docker-compose ps

# Review translation services
ls -la src/serviceCore/serviceTranslation-*/
```

### Phase 2: Safe Removal Process

1. **Create backup** of entire serviceCore directory
2. **Document dependencies** - check if any scripts reference removed services
3. **Update docker-compose files** to remove old service references
4. **Remove directories** one at a time
5. **Test remaining services** after each removal
6. **Update documentation** to reflect new structure

### Phase 3: Services to Remove

```bash
# Embedding - Remove originals, keep Mojo
rm -rf src/serviceCore/serviceEmbedding/
rm -rf src/serviceCore/serviceEmbedding-rust/

# RAG - Remove Mojo/Rust, keep Zig+Mojo hybrid
rm -rf src/serviceCore/serviceRAG-mojo/
rm -rf src/serviceCore/serviceRAG-rust/

# Translation - TBD after evaluation
# (Likely remove one, keep the other)
```

---

## ⚠️ Pre-Removal Checklist

Before removing any service, verify:

- [ ] No active Docker containers using the service
- [ ] No docker-compose.yml references
- [ ] No scripts in `scripts/` calling the service
- [ ] No other services depending on it
- [ ] No environment variables pointing to it
- [ ] Backup created
- [ ] Alternative service tested and working

---

## 🔍 Dependency Check Commands

```bash
# Search for references to old embedding service
grep -r "serviceEmbedding" --include="*.yml" --include="*.yaml" --include="*.sh" --include="*.py" .

# Search for references to old RAG services
grep -r "serviceRAG-mojo\|serviceRAG-rust" --include="*.yml" --include="*.yaml" --include="*.sh" --include="*.py" .

# Check docker-compose files
grep -r "serviceEmbedding\|serviceRAG" docker/compose/

# Check deployment scripts
grep -r "serviceEmbedding\|serviceRAG" scripts/
```

---

## 📊 Expected Results After Consolidation

### Before:
```
src/serviceCore/
├── serviceEmbedding/           (Python - old)
├── serviceEmbedding-mojo/       (Mojo - production)
├── serviceEmbedding-rust/       (Rust - experimental)
├── serviceRAG-mojo/             (Mojo)
├── serviceRAG-rust/             (Rust)
├── serviceRAG-zig-mojo/         (Zig+Mojo - future)
├── serviceTranslation-mojo/     (Mojo)
└── serviceTranslation-rust/     (Rust)
```

### After:
```
src/serviceCore/
├── serviceEmbedding-mojo/       ✅ (Production - 10-25x faster)
├── serviceRAG-zig-mojo/         ✅ (Future - needs Zig compiler)
└── serviceTranslation-mojo/     ✅ (To be verified)
    OR
    serviceTranslation-rust/     ✅ (TBD)
```

**Reduction:** 8 services → 3 services (62% reduction in duplication)

---

## 🎯 Benefits of Consolidation

1. **Clarity**: Single source of truth for each service type
2. **Maintenance**: Easier to maintain fewer codebases
3. **Performance**: Keep only the fastest implementations
4. **Documentation**: Focus docs on what's actually used
5. **Disk Space**: Remove ~500MB+ of build artifacts
6. **Cognitive Load**: Developers know which service to use

---

## 🚀 Next Steps

### Immediate (Do Now):
1. Review translation services to decide which to keep
2. Search for dependencies on services to be removed
3. Create backup of serviceCore directory
4. Test that Mojo embedding service is working

### After Review:
1. Remove `serviceEmbedding/` (old Python)
2. Remove `serviceEmbedding-rust/` (experimental)
3. Remove `serviceRAG-mojo/` (superseded)
4. Remove `serviceRAG-rust/` (not primary language)
5. Remove one of the translation services (TBD)

### Future:
1. Install Zig compiler
2. Build `serviceRAG-zig-mojo/` 
3. Deploy native Zig+Mojo RAG service
4. Achieve 10-100x performance gains

---

## ⚡ Quick Start - Execute Consolidation

```bash
# 1. Create backup
mkdir -p ~/backups
cp -r src/serviceCore ~/backups/serviceCore-backup-$(date +%Y%m%d)

# 2. Check dependencies (must return nothing or handle updates)
./scripts/check-service-dependencies.sh

# 3. Remove duplicates (after verification)
rm -rf src/serviceCore/serviceEmbedding/
rm -rf src/serviceCore/serviceEmbedding-rust/
rm -rf src/serviceCore/serviceRAG-mojo/
rm -rf src/serviceCore/serviceRAG-rust/
# rm -rf src/serviceCore/serviceTranslation-rust/  # OR -mojo, TBD

# 4. Test remaining services
curl http://localhost:8007/health  # Embedding
# Test translation service
# Test RAG when Zig is installed

# 5. Update documentation
# Update README.md with new service structure
```

---

## 📝 Notes

- **serviceEmbedding-mojo** is the clear winner for embeddings (production tested)
- **serviceRAG-zig-mojo** has the most potential but needs Zig installation
- Translation services need evaluation before removal decision
- All removed code is backed up and in git history if needed

---

**Status:** Ready for execution pending dependency verification  
**Next Action:** Check translation services and verify no dependencies on old services
