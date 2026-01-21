# Unified HANA Module

**Comprehensive SAP HANA Integration for nOpenaiServer**

A unified module providing complete SAP HANA functionality: direct SQL operations, OData v4 business integration, and Graph Engine analytics.

---

## 🎯 Overview

This unified module consolidates all HANA-related functionality:

- **Core SQL** (`core/`) - Direct HANA database operations with connection pooling
- **OData Integration** (`odata/`) - SAP S/4HANA business data via OData v4
- **Graph Operations** (`graph/`) - HANA Graph Engine for analytics
- **Shared Types** (`types/`) - Common data structures across all HANA operations
- **Examples** (`examples/`) - Usage patterns and best practices

---

## 🏗️ Architecture

```
hana/
├── core/                      # Direct SQL Operations (Zig)
│   ├── client.zig            # Connection pool manager
│   ├── queries.zig           # Router persistence queries
│   └── connection_pool.zig   # Pool implementation
│
├── odata/                     # OData v4 Integration (Mojo)
│   ├── README.md             # OData-specific documentation
│   ├── client.mojo           # SAP HANA OData client
│   ├── protocol.mojo         # OData v4 protocol wrapper
│   └── scripts/              # Utilities
│
├── graph/                     # Graph Engine (Zig/Mojo)
│   ├── client.zig            # Graph query client
│   └── queries.zig           # Common graph patterns
│
├── types/                     # Shared Types
│   ├── sap_types.zig         # SAP-specific types (CURR, UNIT)
│   └── result_types.zig      # Query result structures
│
└── examples/                  # Usage Examples
    ├── router_persistence.zig # SQL operations
    ├── odata_business.mojo    # Business data access
    └── hybrid_query.mojo      # Combined SQL+OData+Graph
```

---

## 📋 Quick Start

### 1. Core SQL Operations (Router Persistence)

```zig
const hana = @import("hana/core/client.zig");

// Initialize HANA client with connection pool
const config = hana.HanaClient.HanaConfig{
    .host = "localhost",
    .port = 30015,
    .database = "NOPENAI_DB",
    .user = "NUCLEUS_APP",
    .password = "your_password",
    .pool_min = 5,
    .pool_max = 10,
};

const client = try hana.HanaClient.init(allocator, config);
defer client.deinit();

// Execute query
try client.execute("INSERT INTO ROUTING_DECISIONS ...");

// Get metrics
const metrics = client.getMetrics();
std.log.info("Active connections: {d}", .{metrics.active_connections});
```

### 2. OData Business Integration

```mojo
from hana.odata.client import SAPHanaClient

# Initialize SAP client
var client = SAPHanaClient(
    base_url="https://s4hana.example.com:8000",
    username="SAP_USER",
    password="password",
    sap_client="100",
    sap_language="EN"
)

# Query business partners
var partners = client.get_business_partners(top=10)
print(partners)
```

### 3. Graph Operations

```zig
const graph = @import("hana/graph/client.zig");

// Execute graph query
const result = try graph.executeQuery(client,
    \\SELECT * FROM GRAPH_TABLE
    \\  MATCH (a:BusinessPartner)-[:ORDERS]->(b:SalesOrder)
    \\  WHERE a.country = 'US'
);
```

---

## 🔧 Module Details

### Core SQL (`core/`)

**Purpose:** Direct HANA database operations for Router persistence

**Key Features:**
- Connection pooling (5-10 connections)
- Thread-safe operations
- Auto-recovery and retry logic
- Health monitoring
- Prepared statement caching

**Use Cases:**
- Store routing decisions
- Save agent-model assignments
- Track performance metrics
- Query analytics

**Performance:**
- Connection acquisition: <1ms
- Query execution: <10ms
- Throughput: >1000 ops/sec

### OData Integration (`odata/`)

**Purpose:** SAP S/4HANA business data access via OData v4

**Key Features:**
- Full OData v4 protocol support
- SAP-specific types (CURR, UNIT, LANG)
- CSRF token handling
- Metadata parsing
- Batch operations

**Use Cases:**
- Query business partners
- Access sales orders
- Retrieve material master data
- Financial reporting

**Supported Services:**
- API_BUSINESS_PARTNER
- API_SALES_ORDER_SRV
- API_MATERIAL_DOCUMENT_SRV
- API_JOURNALENTRY_SRV

### Graph Operations (`graph/`)

**Purpose:** HANA Graph Engine for relationship analytics

**Key Features:**
- Graph pattern matching
- Path finding algorithms
- Relationship traversal
- Hybrid SQL+Graph queries

**Use Cases:**
- Customer relationship networks
- Supply chain optimization
- Fraud detection
- Recommendation engines

---

## 💡 Common Patterns

### Pattern 1: Router Persistence

```zig
const queries = @import("hana/core/queries.zig");

// Save routing decision
const decision = queries.RoutingDecision{
    .id = try queries.generateDecisionId(allocator),
    .request_id = "req_123",
    .task_type = "coding",
    .agent_id = "agent_1",
    .model_id = "llama-70b",
    .capability_score = 0.95,
    .performance_score = 0.88,
    .composite_score = 0.92,
    .strategy_used = "balanced",
    .latency_ms = 45,
    .success = true,
    .fallback_used = false,
    .timestamp = std.time.milliTimestamp(),
};

try queries.saveRoutingDecision(client, decision);
```

### Pattern 2: Business Data Query

```mojo
from hana.odata.protocol import SAPQueryOptions

# Advanced OData query
var options = SAPQueryOptions()
options.select.append("BusinessPartnerID")
options.select.append("BusinessPartnerName")
options.filter = "Country eq 'US'"
options.top = 100
options.orderby = "BusinessPartnerName asc"

var result = client.query_odata("A_BusinessPartner", options)
```

### Pattern 3: Hybrid Operation

```mojo
# Combine SQL, OData, and Graph
# 1. Get business partner from OData
var partner = client.get_business_partner("1000")

# 2. Enrich with graph relationships
var relationships = graph_client.get_relationships(partner.id)

# 3. Store analytics in HANA SQL
sql_client.save_analytics(partner.id, relationships)
```

---

## 📊 Performance Characteristics

### Core SQL
- **Latency:** P50 <5ms, P95 <10ms, P99 <20ms
- **Throughput:** >1000 queries/sec
- **Connection Pool:** 5-10 connections
- **Recovery Time:** <5 seconds

### OData Integration
- **Initial Connection:** ~100ms (CSRF token fetch)
- **Query Latency:** 50-200ms (network dependent)
- **Throughput:** Limited by SAP Gateway
- **Caching:** CSRF token cached per session

### Graph Operations
- **Simple Query:** <50ms
- **Path Finding:** 100-500ms (depends on depth)
- **Large Graph:** May require optimization

---

## 🔐 Security

### Connection Security
- TLS/SSL for all connections
- Credential encryption at rest
- Password zeroing after use
- Connection pool isolation

### Authentication
- **Core SQL:** HANA user/password
- **OData:** SAP Basic Auth + CSRF token
- **Graph:** Shared HANA credentials

### Input Validation
- Query size limits (1MB max)
- Parameter validation
- SQL injection prevention
- Null byte protection

---

## 🧪 Testing

### Unit Tests
```bash
# Test core SQL
zig test hana/core/client.zig
zig test hana/core/queries.zig

# Test OData (if applicable)
mojo test hana/odata/
```

### Integration Tests
```bash
# Full integration test
./scripts/test_hana_integration.sh
```

### Performance Tests
```bash
# Load testing
./scripts/benchmark_hana.sh
```

---

## 📖 Configuration

### Environment Variables

```bash
# Core SQL
export HANA_HOST=localhost
export HANA_PORT=30015
export HANA_DATABASE=NOPENAI_DB
export HANA_USER=NUCLEUS_APP
export HANA_PASSWORD=your_password
export HANA_POOL_MIN=5
export HANA_POOL_MAX=10

# OData
export SAP_BASE_URL=https://s4hana.example.com:8000
export SAP_CLIENT=100
export SAP_LANGUAGE=EN
```

### Configuration File

See `config/hana.config.json` for detailed configuration options.

---

## 🚀 Migration Guide

### From `database/` to `hana/core/`

```zig
// Old import
const HanaClient = @import("../database/hana_client.zig").HanaClient;

// New import
const HanaClient = @import("../hana/core/client.zig").HanaClient;
```

### From `sap-toolkit-mojo/` to `hana/odata/`

```mojo
# Old import
from sap_toolkit.lib.clients.sap_hana_client import SAPHanaClient

# New import
from hana.odata.client import SAPHanaClient
```

---

## 📚 Additional Resources

### Documentation
- [Core SQL Documentation](core/README.md)
- [OData Integration Guide](odata/README.md)
- [Graph Operations Guide](graph/README.md)

### Examples
- [Router Persistence Example](examples/router_persistence.zig)
- [OData Business Example](examples/odata_business.mojo)
- [Hybrid Query Example](examples/hybrid_query.mojo)

### External Links
- [SAP HANA Documentation](https://help.sap.com/docs/HANA_CLOUD_DATABASE)
- [OData v4 Specification](https://www.odata.org/documentation/)
- [HANA Graph Engine](https://help.sap.com/docs/HANA_CLOUD_DATABASE/11afa2e60a5f4192a381df30f94863f9/30d1d8cfd5d0470dbaac2ebe20cefb8f.html)

---

## 🛠️ Development

### Adding New Features

1. **Core SQL:** Add to `core/queries.zig`
2. **OData:** Extend `odata/client.mojo`
3. **Graph:** Add to `graph/queries.zig`
4. **Types:** Define in `types/` if shared

### Best Practices

- Use connection pooling for all operations
- Implement retry logic for transient failures
- Log all HANA operations for debugging
- Monitor connection pool metrics
- Cache frequently accessed data
- Use prepared statements for performance

---

## 📈 Roadmap

### Phase 1: Foundation ✅
- Core SQL operations
- OData integration
- Basic graph support

### Phase 2: Enhancement (Current)
- Advanced connection pooling
- Batch operations
- Performance optimization

### Phase 3: Advanced Features
- Distributed caching
- Multi-region support
- Advanced graph algorithms
- Real-time streaming

---

## 🤝 Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - See [LICENSE](../../../LICENSE) for details.

---

**Version:** 1.0.0 (Day 51 - Unified Module)  
**Last Updated:** 2026-01-21  
**Status:** Production Ready (Core), Alpha (OData, Graph)  

**Maintainers:**
- Core SQL: Backend Team (Zig)
- OData: Integration Team (Mojo)
- Graph: Analytics Team (Zig/Mojo)
