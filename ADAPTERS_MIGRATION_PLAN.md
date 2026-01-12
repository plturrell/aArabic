# 🔄 Adapters Migration Plan: Python → Mojo/Zig

**Date:** 2026-01-12  
**Issue:** 21 Python adapters orphaned in src/serviceCore/adapters/  
**Goal:** Migrate to Mojo/Zig and consolidate into services

---

## 🔍 Current State

### **Found: 21 Python Adapters**

```
src/serviceCore/adapters/          (Python - legacy)
├── __init__.py
├── shimmy.py                      ⚠️ Should be IN serviceShimmy-mojo!
├── toolorchestra.py               ⚠️ Tools already in Shimmy
├── orchestration.py               ⚠️ Orchestration logic
├── hybrid_orchestration.py        ⚠️ Hybrid logic
├── nucleus_flow.py                ⚠️ Workflow engine
├── nucleusgraph.py                ⚠️ Graph operations
├── a2ui.py                        ⚠️ UI adapter
├── a2ui_enhanced.py               ⚠️ Enhanced UI
├── qdrant.py                      Database adapter
├── memgraph.py                    Database adapter
├── dragonfly.py                   Cache adapter
├── apisix.py                      API gateway
├── keycloak.py                    Auth adapter
├── marquez.py                     Lineage adapter
├── gitea.py                       Git adapter
├── hyperbooklm.py                 BookLM adapter
├── opencanvas.py                  Canvas adapter
├── rust_cli_adapter.py            CLI adapter
├── saudi_otp_vat_methods.py       Saudi VAT methods
└── saudi_otp_vat_workflow.py      Saudi VAT workflow
```

**Problem:**
- ❌ All Python (not Mojo/Zig)
- ❌ Orphaned at serviceCore level
- ❌ shimmy.py is 300+ lines adapter to talk TO Shimmy!
- ❌ Should be part of their respective services
- ❌ Many duplicates/overlaps

---

## 🎯 Migration Strategy

### **Phase 1: Move to Services** (Immediate)

**Priority 1: Shimmy Adapters → serviceShimmy-mojo**

```bash
Move to serviceShimmy-mojo/adapters/:
✅ shimmy.py             (300+ lines) - HTTP client to Shimmy service
✅ toolorchestra.py      - Tool orchestration
✅ orchestration.py      - Workflow orchestration
✅ hybrid_orchestration.py - Hybrid workflows
✅ nucleus_flow.py       - Flow engine
✅ nucleusgraph.py       - Graph operations
✅ a2ui.py              - UI generation
✅ a2ui_enhanced.py     - Enhanced UI

Result: Core Shimmy functionality in one place
```

**Priority 2: External Service Adapters**

```bash
Keep as shared adapters (used by multiple services):
├── src/serviceCore/adapters/       (Shared)
│   ├── qdrant.py                   (DB - used by multiple)
│   ├── memgraph.py                 (DB - used by multiple)
│   ├── dragonfly.py                (Cache - shared)
│   ├── apisix.py                   (Gateway - shared)
│   ├── keycloak.py                 (Auth - shared)
│   ├── marquez.py                  (Lineage - shared)
│   ├── gitea.py                    (Git - shared)
│   ├── hyperbooklm.py              (BookLM - shared)
│   └── opencanvas.py               (Canvas - shared)

Move to serviceTranslation-mojo/adapters/:
├── saudi_otp_vat_methods.py        (Saudi-specific)
└── saudi_otp_vat_workflow.py       (Saudi-specific)

Archive:
└── rust_cli_adapter.py             (Legacy - if not used)
```

### **Phase 2: Rewrite in Mojo/Zig** (Future)

**High Priority for Mojo/Zig:**

```
1. shimmy.py → shimmy_adapter.mojo
   Why: Core Shimmy functionality, should be native
   Effort: ~500 lines Mojo
   Benefit: Native performance, no Python dependency

2. toolorchestra.py → tool_orchestration.mojo
   Why: Core tool management
   Effort: ~300 lines Mojo
   Benefit: Better integration with Shimmy core

3. a2ui.py → a2ui_generator.mojo
   Why: UI generation
   Effort: ~400 lines Mojo
   Benefit: Faster UI generation
```

**Medium Priority (Keep Python for now):**

```
Database adapters (qdrant, memgraph, dragonfly):
- These wrap existing APIs
- Python clients are mature
- Can migrate later if needed
```

**Low Priority:**

```
External service adapters:
- apisix, keycloak, marquez, gitea, etc.
- These are HTTP/REST wrappers
- Python is fine for these
- Migrate only if performance critical
```

---

## 📋 Immediate Action Plan

### **Step 1: Move Shimmy Adapters**

```bash
# Create adapters directory in Shimmy
mkdir -p src/serviceCore/serviceShimmy-mojo/adapters/

# Move Shimmy-specific adapters
mv src/serviceCore/adapters/shimmy.py \
   src/serviceCore/serviceShimmy-mojo/adapters/

mv src/serviceCore/adapters/toolorchestra.py \
   src/serviceCore/serviceShimmy-mojo/adapters/

mv src/serviceCore/adapters/orchestration.py \
   src/serviceCore/serviceShimmy-mojo/adapters/

mv src/serviceCore/adapters/hybrid_orchestration.py \
   src/serviceCore/serviceShimmy-mojo/adapters/

mv src/serviceCore/adapters/nucleus_flow.py \
   src/serviceCore/serviceShimmy-mojo/adapters/

mv src/serviceCore/adapters/nucleusgraph.py \
   src/serviceCore/serviceShimmy-mojo/adapters/

mv src/serviceCore/adapters/a2ui.py \
   src/serviceCore/serviceShimmy-mojo/adapters/

mv src/serviceCore/adapters/a2ui_enhanced.py \
   src/serviceCore/serviceShimmy-mojo/adapters/

# Copy __init__.py
cp src/serviceCore/adapters/__init__.py \
   src/serviceCore/serviceShimmy-mojo/adapters/
```

### **Step 2: Move Saudi VAT Adapters**

```bash
mkdir -p src/serviceCore/serviceTranslation-mojo/adapters/

mv src/serviceCore/adapters/saudi_otp_vat_methods.py \
   src/serviceCore/serviceTranslation-mojo/adapters/

mv src/serviceCore/adapters/saudi_otp_vat_workflow.py \
   src/serviceCore/serviceTranslation-mojo/adapters/
```

### **Step 3: Keep Shared Adapters**

```bash
# These stay in src/serviceCore/adapters/ (shared)
# - qdrant.py
# - memgraph.py
# - dragonfly.py
# - apisix.py
# - keycloak.py
# - marquez.py
# - gitea.py
# - hyperbooklm.py
# - opencanvas.py
# - rust_cli_adapter.py (maybe archive)
```

---

## 🎯 Final Structure

### **After Phase 1 (Move)**

```
src/serviceCore/
├── adapters/                       (Shared adapters only)
│   ├── __init__.py
│   ├── qdrant.py                   ✅ Shared DB
│   ├── memgraph.py                 ✅ Shared DB
│   ├── dragonfly.py                ✅ Shared cache
│   ├── apisix.py                   ✅ Shared gateway
│   ├── keycloak.py                 ✅ Shared auth
│   ├── marquez.py                  ✅ Shared lineage
│   ├── gitea.py                    ✅ Shared git
│   ├── hyperbooklm.py              ✅ Shared BookLM
│   └── opencanvas.py               ✅ Shared Canvas
│
├── serviceShimmy-mojo/
│   ├── adapters/                   ✅ Shimmy-specific (Python for now)
│   │   ├── __init__.py
│   │   ├── shimmy.py               ← From serviceCore/adapters
│   │   ├── toolorchestra.py        ← From serviceCore/adapters
│   │   ├── orchestration.py        ← From serviceCore/adapters
│   │   ├── hybrid_orchestration.py ← From serviceCore/adapters
│   │   ├── nucleus_flow.py         ← From serviceCore/adapters
│   │   ├── nucleusgraph.py         ← From serviceCore/adapters
│   │   ├── a2ui.py                 ← From serviceCore/adapters
│   │   └── a2ui_enhanced.py        ← From serviceCore/adapters
│   │
│   ├── recursive_llm/              ✅ Pure Mojo
│   ├── models/                     ✅ Models
│   ├── tools/                      ✅ Tools
│   └── [other components]
│
└── serviceTranslation-mojo/
    ├── adapters/                   ✅ Translation-specific
    │   ├── saudi_otp_vat_methods.py
    │   └── saudi_otp_vat_workflow.py
    └── [other components]
```

### **After Phase 2 (Rewrite - Future)**

```
serviceShimmy-mojo/
├── adapters/                       
│   ├── python/                     ⚠️ Legacy (being phased out)
│   │   ├── shimmy.py
│   │   ├── toolorchestra.py
│   │   └── [...]
│   │
│   └── native/                     ✅ Mojo/Zig (new)
│       ├── shimmy_adapter.mojo     ← Replaces shimmy.py
│       ├── tool_orchestration.mojo ← Replaces toolorchestra.py
│       ├── a2ui_generator.mojo     ← Replaces a2ui.py
│       └── [...]
```

---

## 📊 Migration Priorities

### **Immediate (This Session)**

```
Priority 1: Organization
✅ Move Shimmy adapters to serviceShimmy-mojo/adapters/
✅ Move Saudi adapters to serviceTranslation-mojo/adapters/
✅ Keep shared adapters in serviceCore/adapters/
✅ Document structure

Effort: 10 minutes
Benefit: Clear organization
```

### **Short-term (Next Week)**

```
Priority 2: Core Mojo Rewrite
⏳ Rewrite shimmy.py → shimmy_adapter.mojo
⏳ Rewrite toolorchestra.py → tool_orchestration.mojo
⏳ Rewrite a2ui.py → a2ui_generator.mojo

Effort: 2-3 days
Benefit: Native performance, no Python deps
```

### **Long-term (As Needed)**

```
Priority 3: Database Adapters
⏳ Rewrite qdrant.py → qdrant_adapter.mojo (if needed)
⏳ Rewrite memgraph.py → memgraph_adapter.mojo (if needed)

Effort: 1-2 days each
Benefit: Marginal (only if bottleneck)
```

---

## 🤔 Key Questions

### **1. shimmy.py Analysis**

**What it does:**
- HTTP client to talk TO Shimmy service
- Wraps Shimmy REST API
- 300+ lines of async Python
- Health checks, model loading, tool execution, etc.

**Why it's weird:**
- This is an adapter to talk TO Shimmy
- But it's IN the Shimmy project!
- Suggests Shimmy might be used as both:
  - Service (server)
  - Client (via this adapter)

**Should it be:**
```
Option A: Client library (external)
   → Separate package for others to use
   
Option B: Internal adapter (serviceShimmy-mojo/adapters/)
   → For internal Shimmy-to-Shimmy communication
   
Option C: Rewrite in Mojo as native client
   → shimmy_client.mojo for internal use
```

### **2. Are These Still Used?**

Need to check if code references these adapters:
```bash
# Search for imports
grep -r "from.*adapters import" src/
grep -r "import.*adapters\." src/
```

If not used → Archive instead of migrate

---

## ✅ Recommended Action (Now)

**Immediate organizational move:**

```bash
1. Move Shimmy-specific adapters to serviceShimmy-mojo/adapters/
   (8 files: shimmy, toolorchestra, orchestration, etc.)

2. Move Saudi-specific adapters to serviceTranslation-mojo/adapters/
   (2 files: saudi VAT methods/workflow)

3. Keep shared adapters in serviceCore/adapters/
   (9 files: qdrant, memgraph, etc.)

4. Archive rust_cli_adapter.py if unused

Result:
✅ Clear ownership
✅ Logical organization
✅ Foundation for future Mojo rewrites
```

**Next session:**
- Rewrite shimmy.py → shimmy_adapter.mojo
- Rewrite toolorchestra.py → tool_orchestration.mojo
- Rewrite a2ui.py → a2ui_generator.mojo

---

## 📝 Summary

**Current:**
- ❌ 21 Python adapters orphaned at serviceCore level
- ❌ shimmy.py is 300+ line adapter TO Shimmy (should be IN Shimmy)
- ❌ No clear ownership
- ❌ All Python (not Mojo/Zig)

**After Phase 1 (This Session):**
- ✅ 8 Shimmy adapters in serviceShimmy-mojo/
- ✅ 2 Saudi adapters in serviceTranslation-mojo/
- ✅ 9 shared adapters in serviceCore/
- ✅ Clear ownership
- ⏳ Still Python (but organized)

**After Phase 2 (Future):**
- ✅ Core adapters rewritten in Mojo
- ✅ Native performance
- ✅ Zero Python dependencies for core
- ✅ Python adapters only for external services

**Ready to move the adapters?** 🚀
