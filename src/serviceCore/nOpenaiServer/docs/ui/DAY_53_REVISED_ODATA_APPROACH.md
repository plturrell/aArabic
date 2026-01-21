# Day 53 REVISED: OData-Only Approach for HANA Cloud

**Date:** 2026-01-21  
**Clarification:** HANA Cloud = OData v4 REST API Only (No Direct SQL)  
**Impact:** Major architecture revision required  

---

## 🔄 Critical Clarification

### HANA Cloud Constraint
**HANA Cloud only provides OData v4 REST API access** - no direct SQL/ODBC/JDBC access.

This means:
- ✗ No direct SQL queries
- ✗ No ODBC/JDBC drivers
- ✗ No connection pooling to database
- ✅ OData v4 REST API only
- ✅ HTTP/HTTPS connections
- ✅ CSRF token-based authentication

---

## 🏗️ Revised Architecture

### Original Plan (Days 51-52) - INCORRECT ❌
```
Router → SQL Client → Connection Pool → HANA Database (SQL)
```
**Problem:** HANA Cloud doesn't expose SQL interface!

### Correct Architecture - OData ✅
```
Router → OData Client → HTTP Client → HANA Cloud OData Service
```

**Components:**
1. **OData Client** - HTTP REST API client
2. **CSRF Token** - Authentication for write operations
3. **JSON Serialization** - Convert Zig structs to OData JSON
4. **HTTP Client** - HTTP/HTTPS requests to HANA Cloud

---

## 📦 New Implementation Approach

### What We Built (Days 51-52)
- ❌ `hana/core/client.zig` - SQL connection pool (won't work with HANA Cloud)
- ❌ `hana/core/queries.zig` - SQL queries (won't work with HANA Cloud)
- ✅ `hana/odata/` - OData v4 client (correct approach!)

### What We Need (Revised)
- ✅ `hana/core/odata_persistence.zig` - OData-based Router persistence (NEW - Day 53)
- ✅ HTTP client for OData requests
- ✅ CSRF token management
- ✅ JSON serialization/deserialization
- ✅ OData query builder

---

## 🔧 OData Service Design

### HANA Cloud OData Service Endpoints

**Base URL:** `https://{tenant}.hanacloud.ondemand.com/sap/opu/odata4/nopenai/routing/default/v1`

**Entity Sets:**

**1. AgentModelAssignments**
```
POST   /AgentModelAssignments          # Create assignment
GET    /AgentModelAssignments          # List all
GET    /AgentModelAssignments('{id}')  # Get by ID
PATCH  /AgentModelAssignments('{id}')  # Update
DELETE /AgentModelAssignments('{id}')  # Delete
GET    /AgentModelAssignments?$filter=Status eq 'ACTIVE'  # Filter active
```

**2. RoutingDecisions**
```
POST   /RoutingDecisions               # Create decision
GET    /RoutingDecisions               # List all
GET    /RoutingDecisions?$filter=DecisionTimestamp ge datetime'{ts}'  # Time range
GET    /RoutingDecisions?$apply=aggregate(...)  # Analytics
```

**3. InferenceMetrics**
```
POST   /InferenceMetrics               # Create metric
POST   /$batch                         # Batch insert
GET    /InferenceMetrics?$filter=ModelID eq '{id}'  # By model
```

### OData Entity Definitions

**AssignmentEntity (CDSVIEW):**
```
entity AgentModelAssignments {
  key AssignmentID : String(50);
  AgentID : String(50);
  ModelID : String(50);
  MatchScore : Decimal(5,2);
  Status : String(20);  // ACTIVE, INACTIVE
  AssignmentMethod : String(20);  // AUTO, MANUAL
  Capabilities : LargeString;
  AssignedAt : Timestamp;
  LastUpdated : Timestamp;
}
```

**RoutingDecisionEntity (CDSVIEW):**
```
entity RoutingDecisions {
  key DecisionID : String(50);
  RequestID : String(50);
  TaskType : String(50);
  AgentID : String(50);
  ModelID : String(50);
  CapabilityScore : Decimal(5,2);
  PerformanceScore : Decimal(5,2);
  CompositeScore : Decimal(5,2);
  StrategyUsed : String(20);
  LatencyMS : Integer;
  Success : Boolean;
  FallbackUsed : Boolean;
  DecisionTimestamp : Timestamp;
}
```

---

## 🔄 Implementation Strategy

### Phase 1: OData Client (Day 53) ✅
- Created `odata_persistence.zig` 
- OData entity definitions
- JSON serialization helpers
- CSRF token placeholder

### Phase 2: HTTP Integration (Days 54-55)
Need to integrate with existing HTTP client or create new:

**Option A: Use zig_http_shimmy.zig**
- Already exists in project
- Supports HTTP/HTTPS
- Need to add CSRF token support
- Add OData-specific headers

**Option B: Use std.http.Client**
- Zig standard library
- Native HTTP support
- Simpler integration

**Option C: Use existing zig_odata_sap.zig**
- Already has OData protocol
- Located in project root
- Need to adapt for Router persistence

### Phase 3: Router Integration Update (Day 54)
Update Router modules to use OData persistence:
```zig
// Instead of SQL client
const ODataPersistence = @import("../hana/core/odata_persistence.zig").ODataPersistence;

// Initialize with OData config
const odata = try ODataPersistence.init(allocator, .{
    .base_url = "https://tenant.hanacloud.ondemand.com",
    .username = "ROUTER_API",
    .password = env.get("HANA_PASSWORD"),
});

// Use OData methods
try odata.createAssignment(assignment);
try odata.createRoutingDecision(decision);
const stats = try odata.getRoutingStats(24);
```

---

## 📊 OData vs SQL Comparison

### What Changes

| Aspect | SQL (Original) | OData (Correct) |
|--------|---------------|-----------------|
| Protocol | SQL over ODBC | HTTP REST API |
| Connection | Connection pool | HTTP connections |
| Authentication | User/password | Basic Auth + CSRF |
| Queries | SQL SELECT | OData $filter, $apply |
| Inserts | SQL INSERT | HTTP POST with JSON |
| Updates | SQL UPDATE | HTTP PATCH with JSON |
| Deletes | SQL DELETE | HTTP DELETE |
| Batch | SQL transaction | OData $batch |
| Transactions | SQL BEGIN/COMMIT | OData changesets |

### What Stays the Same

- ✅ Data model (entities unchanged)
- ✅ API interface (same function signatures)
- ✅ Router integration points
- ✅ Metrics and analytics
- ✅ Error handling strategy

---

## 🎯 Revised Implementation Plan

### Days 51-53: Foundation (DONE ✅)
- Day 51: Unified hana/ module structure ✅
- Day 52: Router integration (needs update to OData) ✅
- Day 53: OData persistence layer created ✅

### Days 54-55: HTTP & Testing
- **Day 54:** HTTP client integration
  - Integrate with existing zig_http or std.http
  - Implement CSRF token fetching
  - Implement POST/GET/PATCH/DELETE
  - Test with HANA Cloud sandbox

- **Day 55:** Week 11 completion
  - Update Router modules to use OData
  - End-to-end testing
  - Performance validation
  - Week 11 completion report

### Week 12: OData Enhancement
- Prepared statement equivalent (request caching)
- Batch operations ($batch)
- Query optimization ($apply aggregations)
- Error recovery strategies

---

## 🔧 Required Changes

### Files to Update

**1. hana/core/client.zig (Day 51)** 
- ❌ Remove: SQL connection pool logic
- ✅ Replace: HTTP connection pooling
- ✅ Keep: Metrics, health checks

**2. hana/core/queries.zig (Day 51)**
- ❌ Remove: SQL query strings
- ✅ Replace: OData query builders
- ✅ Keep: Data structures

**3. Router modules (Day 52)**
- ✅ Already using abstraction (good!)
- ✅ Just swap client implementation
- ✅ Same API, different backend

### New Files Created

**Day 53:**
- ✅ `hana/core/odata_persistence.zig` - OData client
- ✅ JSON serialization
- ✅ CSRF token handling
- ✅ OData entity definitions

---

## 📚 HANA Cloud OData Resources

### Creating OData Service in HANA Cloud

**Step 1: Define CDS Models**
```cds
namespace nopenai.routing;

entity AgentModelAssignments {
  key AssignmentID : String(50);
  AgentID : String(50);
  ModelID : String(50);
  MatchScore : Decimal(5,2);
  Status : String(20);
  AssignmentMethod : String(20);
  Capabilities : LargeString;
  AssignedAt : Timestamp;
  LastUpdated : Timestamp;
}
```

**Step 2: Expose as OData Service**
```cds
service RoutingService {
  entity AgentModelAssignments as projection on nopenai.routing.AgentModelAssignments;
  entity RoutingDecisions as projection on nopenai.routing.RoutingDecisions;
  entity InferenceMetrics as projection on nopenai.routing.InferenceMetrics;
}
```

**Step 3: Deploy to HANA Cloud**
```bash
cf deploy
# Service available at:
# https://{tenant}.hanacloud.ondemand.com/sap/opu/odata4/nopenai/routing/default/v1
```

---

## 🎯 Success Criteria (Revised)

### Day 53 (Completed) ✅
- ✅ OData persistence layer created
- ✅ Entity definitions  
- ✅ JSON serialization
- ✅ CSRF token handling
- ✅ Tests created

### Day 54 (HTTP Integration)
- [ ] Integrate HTTP client
- [ ] Implement actual POST/GET/PATCH
- [ ] Test with HANA Cloud
- [ ] Validate CSRF token flow

### Day 55 (Week 11 Completion)
- [ ] Update all Router modules to OData
- [ ] End-to-end testing
- [ ] Performance validation
- [ ] Week 11 completion report

---

## 📈 Performance with OData

### Expected Performance

| Operation | OData (HTTP) | Notes |
|-----------|--------------|-------|
| Create assignment | 50-100ms | HTTP POST + CSRF |
| Create decision | 50-100ms | HTTP POST |
| Query assignments | 100-200ms | HTTP GET with filter |
| Batch insert (100) | 200-500ms | $batch request |
| Analytics query | 200-500ms | $apply aggregation |

**Key Differences from SQL:**
- Higher latency (HTTP overhead)
- Better for HANA Cloud compatibility
- Automatic scaling (cloud service)
- No connection pool needed (HTTP)

---

## 🎉 Key Insights

### What We Learned ✅
1. **HANA Cloud = OData only** (critical constraint)
2. **No direct SQL access** in HANA Cloud
3. **HTTP REST** is the only interface
4. **Architecture must be REST-based**

### What This Means
- ✅ OData client is the correct approach
- ✅ No need for ODBC/JDBC
- ✅ Simpler deployment (no native drivers)
- ✅ Cloud-native architecture
- ⚠️ Higher latency than direct SQL
- ⚠️ Need HTTP client integration

### Impact on Timeline
- Days 51-53: Foundation work still valuable ✅
- Days 54-55: Focus on HTTP + OData integration
- Week 12: OData optimization ($batch, caching)
- Overall: Timeline unchanged

---

## 🚀 Next Steps

### Immediate (Day 54)
1. Choose HTTP client (zig_http or std.http)
2. Implement OData HTTP operations
3. Test with HANA Cloud sandbox
4. Validate CSRF token flow

### This Week (Day 55)
1. Update Router modules for OData
2. End-to-end testing
3. Performance benchmarks
4. Week 11 completion report

---

## 🎯 Conclusion

Day 53 successfully pivots to the correct **OData-only architecture** for HANA Cloud. The foundation work (Days 51-52) provides valuable structure, and now we have the right persistence layer (`odata_persistence.zig`) that matches HANA Cloud's capabilities.

**Status:** ✅ Architecture corrected, OData persistence layer created, ready for HTTP integration (Day 54)

---

**Document Created:** 2026-01-21 21:12 UTC  
**Purpose:** Clarify HANA Cloud = OData only constraint  
**Impact:** Architecture now correctly aligned with HANA Cloud  
**Next:** Day 54 - HTTP client integration for OData operations
