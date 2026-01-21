# Query Builder Patterns Guide

**Version:** 1.0  
**Last Updated:** January 20, 2026  
**Audience:** Application developers using nMetaData

---

## Table of Contents

1. [Overview](#overview)
2. [Basic Patterns](#basic-patterns)
3. [Advanced Patterns](#advanced-patterns)
4. [Dialect-Specific Optimizations](#dialect-specific-optimizations)
5. [Performance Best Practices](#performance-best-practices)
6. [Common Anti-Patterns](#common-anti-patterns)
7. [Real-World Examples](#real-world-examples)

---

## Overview

The nMetaData Query Builder provides a fluent API for constructing SQL queries that work across PostgreSQL, SAP HANA, and SQLite. This guide demonstrates common patterns and best practices.

### Design Philosophy

- **Type-safe**: Compile-time guarantees where possible
- **Dialect-aware**: Automatically optimizes for target database
- **Composable**: Build complex queries from simple parts
- **Readable**: Code reads like natural language
- **Efficient**: Generates optimized SQL

### Basic Usage

```zig
const QueryBuilder = @import("db/query_builder.zig").QueryBuilder;

var qb = try QueryBuilder.init(allocator, .PostgreSQL);
defer qb.deinit();

// Build query
_ = try qb.select(&[_][]const u8{"id", "name"})
    .from("datasets")
    .where(.{ .expression = "active = true" })
    .orderBy("name", .ASC)
    .limit(10);

// Generate SQL
const sql = try qb.build();
defer allocator.free(sql);

// Execute
var result = try db_client.execute(sql, &.{});
defer result.deinit();
```

---

## Basic Patterns

### SELECT Queries

#### Simple SELECT

```zig
// SELECT id, name FROM datasets
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"id", "name"})
    .from("datasets");
const sql = try qb.build();
```

#### SELECT with WHERE

```zig
// SELECT * FROM datasets WHERE active = true
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .where(.{ .expression = "active = true" });
const sql = try qb.build();
```

#### SELECT with Multiple Conditions

```zig
// SELECT * FROM datasets 
// WHERE active = true AND namespace_id = 1
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .where(.{ .expression = "active = true" })
    .where(.{ .expression = "namespace_id = $1" });
const sql = try qb.build();
```

#### SELECT with ORDER BY

```zig
// SELECT * FROM datasets ORDER BY created_at DESC
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .orderBy("created_at", .DESC);
const sql = try qb.build();
```

#### SELECT with LIMIT and OFFSET

```zig
// SELECT * FROM datasets LIMIT 10 OFFSET 20
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .limit(10)
    .offset(20);
const sql = try qb.build();
```

### JOIN Patterns

#### INNER JOIN

```zig
// SELECT d.*, n.name as namespace_name
// FROM datasets d
// INNER JOIN namespaces n ON d.namespace_id = n.id
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"d.*", "n.name as namespace_name"})
    .from("datasets d")
    .join(.{
        .type = .Inner,
        .table = "namespaces n",
        .condition = "d.namespace_id = n.id",
    });
const sql = try qb.build();
```

#### LEFT JOIN

```zig
// SELECT d.*, j.name as job_name
// FROM datasets d
// LEFT JOIN jobs j ON d.id = j.output_dataset_id
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"d.*", "j.name as job_name"})
    .from("datasets d")
    .join(.{
        .type = .Left,
        .table = "jobs j",
        .condition = "d.id = j.output_dataset_id",
    });
const sql = try qb.build();
```

#### Multiple JOINs

```zig
// Complex multi-table join
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{
        "d.id",
        "d.name",
        "n.name as namespace",
        "j.name as job",
    })
    .from("datasets d")
    .join(.{
        .type = .Inner,
        .table = "namespaces n",
        .condition = "d.namespace_id = n.id",
    })
    .join(.{
        .type = .Left,
        .table = "jobs j",
        .condition = "d.id = j.output_dataset_id",
    });
const sql = try qb.build();
```

### Aggregation Patterns

#### COUNT

```zig
// SELECT COUNT(*) as total FROM datasets
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"COUNT(*) as total"})
    .from("datasets");
const sql = try qb.build();
```

#### GROUP BY

```zig
// SELECT namespace_id, COUNT(*) as count
// FROM datasets
// GROUP BY namespace_id
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"namespace_id", "COUNT(*) as count"})
    .from("datasets")
    .groupBy(&[_][]const u8{"namespace_id"});
const sql = try qb.build();
```

#### HAVING

```zig
// SELECT namespace_id, COUNT(*) as count
// FROM datasets
// GROUP BY namespace_id
// HAVING COUNT(*) > 10
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"namespace_id", "COUNT(*) as count"})
    .from("datasets")
    .groupBy(&[_][]const u8{"namespace_id"})
    .having("COUNT(*) > 10");
const sql = try qb.build();
```

---

## Advanced Patterns

### Subqueries

#### Subquery in WHERE

```zig
// SELECT * FROM datasets
// WHERE namespace_id IN (
//     SELECT id FROM namespaces WHERE active = true
// )
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .where(.{
        .expression = 
            \\namespace_id IN (
            \\    SELECT id FROM namespaces WHERE active = true
            \\)
    });
const sql = try qb.build();
```

#### Subquery in FROM

```zig
// SELECT avg_size.namespace_id, avg_size.avg
// FROM (
//     SELECT namespace_id, AVG(size) as avg
//     FROM datasets
//     GROUP BY namespace_id
// ) avg_size
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"avg_size.namespace_id", "avg_size.avg"})
    .from(
        \\(
        \\    SELECT namespace_id, AVG(size) as avg
        \\    FROM datasets
        \\    GROUP BY namespace_id
        \\) avg_size
    );
const sql = try qb.build();
```

### Common Table Expressions (CTEs)

#### Simple CTE

```zig
// WITH active_datasets AS (
//     SELECT * FROM datasets WHERE active = true
// )
// SELECT * FROM active_datasets
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.with("active_datasets", 
    \\SELECT * FROM datasets WHERE active = true
    )
    .select(&[_][]const u8{"*"})
    .from("active_datasets");
const sql = try qb.build();
```

#### Recursive CTE (Lineage Pattern)

```zig
// WITH RECURSIVE lineage AS (
//     SELECT id, name, 0 as depth
//     FROM datasets
//     WHERE id = $1
//     UNION ALL
//     SELECT d.id, d.name, l.depth + 1
//     FROM datasets d
//     INNER JOIN lineage_edges e ON d.id = e.source_id
//     INNER JOIN lineage l ON e.target_id = l.id
//     WHERE l.depth < 10
// )
// SELECT * FROM lineage
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.withRecursive("lineage",
    \\SELECT id, name, 0 as depth
    \\FROM datasets
    \\WHERE id = $1
    \\UNION ALL
    \\SELECT d.id, d.name, l.depth + 1
    \\FROM datasets d
    \\INNER JOIN lineage_edges e ON d.id = e.source_id
    \\INNER JOIN lineage l ON e.target_id = l.id
    \\WHERE l.depth < 10
    )
    .select(&[_][]const u8{"*"})
    .from("lineage");
const sql = try qb.build();
```

### Window Functions

#### ROW_NUMBER

```zig
// SELECT
//     id,
//     name,
//     ROW_NUMBER() OVER (PARTITION BY namespace_id ORDER BY created_at DESC) as row_num
// FROM datasets
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{
        "id",
        "name",
        "ROW_NUMBER() OVER (PARTITION BY namespace_id ORDER BY created_at DESC) as row_num",
    })
    .from("datasets");
const sql = try qb.build();
```

#### RANK and Aggregates

```zig
// SELECT
//     namespace_id,
//     AVG(size) OVER (PARTITION BY namespace_id) as avg_size,
//     RANK() OVER (ORDER BY size DESC) as size_rank
// FROM datasets
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{
        "namespace_id",
        "AVG(size) OVER (PARTITION BY namespace_id) as avg_size",
        "RANK() OVER (ORDER BY size DESC) as size_rank",
    })
    .from("datasets");
const sql = try qb.build();
```

### UNION Patterns

#### UNION

```zig
// (SELECT id, 'dataset' as type FROM datasets)
// UNION
// (SELECT id, 'job' as type FROM jobs)
var qb1 = try QueryBuilder.init(allocator, dialect);
_ = try qb1.select(&[_][]const u8{"id", "'dataset' as type"})
    .from("datasets");
const sql1 = try qb1.build();

var qb2 = try QueryBuilder.init(allocator, dialect);
_ = try qb2.select(&[_][]const u8{"id", "'job' as type"})
    .from("jobs");
const sql2 = try qb2.build();

const union_sql = try std.fmt.allocPrint(allocator,
    "({s}) UNION ({s})",
    .{sql1, sql2}
);
```

---

## Dialect-Specific Optimizations

### PostgreSQL Optimizations

#### JSONB Operations

```zig
// SELECT * FROM datasets
// WHERE metadata @> '{"owner": "data-team"}'::jsonb
var qb = try QueryBuilder.init(allocator, .PostgreSQL);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .where(.{
        .expression = "metadata @> '{\"owner\": \"data-team\"}'::jsonb"
    });
const sql = try qb.build();
```

#### Array Operations

```zig
// SELECT * FROM datasets
// WHERE tags @> ARRAY['production']
var qb = try QueryBuilder.init(allocator, .PostgreSQL);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .where(.{
        .expression = "tags @> ARRAY['production']"
    });
const sql = try qb.build();
```

### SAP HANA Optimizations

#### Graph Engine Queries

```zig
// SELECT * FROM GRAPH_TABLE (
//     lineage_graph
//     MATCH (source:Dataset)-[:DEPENDS_ON*1..5]->(target:Dataset)
//     WHERE source.id = $1
//     COLUMNS (target.id, target.name, LENGTH(EDGE) as depth)
// )
var qb = try QueryBuilder.init(allocator, .HANA);
_ = try qb.select(&[_][]const u8{"*"})
    .from(
        \\GRAPH_TABLE (
        \\    lineage_graph
        \\    MATCH (source:Dataset)-[:DEPENDS_ON*1..5]->(target:Dataset)
        \\    WHERE source.id = $1
        \\    COLUMNS (target.id, target.name, LENGTH(EDGE) as depth)
        \\)
    );
const sql = try qb.build();
```

#### Column Store Hints

```zig
// SELECT /*+ COLUMN_STORE */ * FROM datasets
var qb = try QueryBuilder.init(allocator, .HANA);
_ = try qb.select(&[_][]const u8{"/*+ COLUMN_STORE */ *"})
    .from("datasets");
const sql = try qb.build();
```

### SQLite Optimizations

#### Full-Text Search

```zig
// SELECT * FROM datasets_fts
// WHERE datasets_fts MATCH 'user AND profile'
var qb = try QueryBuilder.init(allocator, .SQLite);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets_fts")
    .where(.{
        .expression = "datasets_fts MATCH 'user AND profile'"
    });
const sql = try qb.build();
```

---

## Performance Best Practices

### 1. Use Prepared Statements

```zig
// BAD: String concatenation (SQL injection risk)
const name = "user_profiles";
const sql = try std.fmt.allocPrint(allocator,
    "SELECT * FROM datasets WHERE name = '{s}'",
    .{name}
);

// GOOD: Parameterized query
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .where(.{ .expression = "name = $1" });
const sql = try qb.build();
// Execute with parameters
var result = try db_client.execute(sql, &[_]Value{
    .{ .string = name }
});
```

### 2. Minimize Selected Columns

```zig
// BAD: Select everything
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets");

// GOOD: Select only needed columns
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"id", "name", "namespace_id"})
    .from("datasets");
```

### 3. Use Indexes Effectively

```zig
// Query designed to use index on (namespace_id, created_at)
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .where(.{ .expression = "namespace_id = $1" })  // Index first column
    .orderBy("created_at", .DESC);  // Index second column
```

### 4. Limit Result Sets

```zig
// Always use LIMIT for exploratory queries
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .limit(100);  // Protect against large result sets
```

### 5. Batch Operations

```zig
// Use CTEs or batch inserts for bulk operations
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.with("to_insert",
    \\VALUES
    \\    (1, 'dataset1'),
    \\    (2, 'dataset2'),
    \\    (3, 'dataset3')
    )
    .select(&[_][]const u8{"*"})
    .from("to_insert");
```

---

## Common Anti-Patterns

### ❌ Anti-Pattern 1: N+1 Queries

```zig
// BAD: Query in a loop
for (dataset_ids) |id| {
    var qb = try QueryBuilder.init(allocator, dialect);
    _ = try qb.select(&[_][]const u8{"*"})
        .from("datasets")
        .where(.{ .expression = "id = $1" });
    const sql = try qb.build();
    var result = try db_client.execute(sql, &[_]Value{
        .{ .int64 = id }
    });
    // Process result
}
```

```zig
// GOOD: Single query with IN clause
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets")
    .where(.{ .expression = "id = ANY($1)" });
const sql = try qb.build();
var result = try db_client.execute(sql, &[_]Value{
    .{ .array = dataset_ids }
});
```

### ❌ Anti-Pattern 2: SELECT *

```zig
// BAD: Fetching unnecessary data
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"*"})
    .from("datasets");
```

```zig
// GOOD: Specify needed columns
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"id", "name"})
    .from("datasets");
```

### ❌ Anti-Pattern 3: Unnecessary Subqueries

```zig
// BAD: Subquery when JOIN would work
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{
        "d.*",
        "(SELECT name FROM namespaces WHERE id = d.namespace_id) as namespace_name",
    })
    .from("datasets d");
```

```zig
// GOOD: Use JOIN
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"d.*", "n.name as namespace_name"})
    .from("datasets d")
    .join(.{
        .type = .Inner,
        .table = "namespaces n",
        .condition = "d.namespace_id = n.id",
    });
```

### ❌ Anti-Pattern 4: Missing WHERE on DELETE/UPDATE

```zig
// DANGEROUS: No WHERE clause
const sql = "DELETE FROM datasets";  // Deletes everything!

// SAFE: Always use WHERE
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.delete()
    .from("datasets")
    .where(.{ .expression = "id = $1" });
```

---

## Real-World Examples

### Example 1: Upstream Lineage Query

```zig
pub fn getUpstreamLineage(
    allocator: Allocator,
    db_client: *DbClient,
    dataset_id: i64,
    max_depth: i32,
) !ResultSet {
    const dialect = db_client.vtable.get_dialect(db_client.context);
    
    if (dialect == .HANA) {
        // Use HANA graph engine (4x faster)
        var qb = try QueryBuilder.init(allocator, dialect);
        _ = try qb.select(&[_][]const u8{"*"})
            .from(
                \\GRAPH_TABLE (
                \\    lineage_graph
                \\    MATCH (target:Dataset)<-[:DEPENDS_ON*1..{}]-(source:Dataset)
                \\    WHERE target.id = $1
                \\    COLUMNS (source.id, source.name, LENGTH(EDGE) as depth)
                \\)
            );
        const sql = try qb.build();
        return try db_client.execute(sql, &[_]Value{
            .{ .int64 = dataset_id },
        });
    } else {
        // Use CTE for PostgreSQL/SQLite
        var qb = try QueryBuilder.init(allocator, dialect);
        _ = try qb.withRecursive("upstream",
            \\SELECT id, name, 0 as depth
            \\FROM datasets
            \\WHERE id = $1
            \\UNION ALL
            \\SELECT d.id, d.name, u.depth + 1
            \\FROM datasets d
            \\INNER JOIN lineage_edges e ON d.id = e.source_id
            \\INNER JOIN upstream u ON e.target_id = u.id
            \\WHERE u.depth < $2
            )
            .select(&[_][]const u8{"*"})
            .from("upstream");
        const sql = try qb.build();
        return try db_client.execute(sql, &[_]Value{
            .{ .int64 = dataset_id },
            .{ .int32 = max_depth },
        });
    }
}
```

### Example 2: Dataset Search with Facets

```zig
pub fn searchDatasets(
    allocator: Allocator,
    db_client: *DbClient,
    search_query: []const u8,
    namespace_id: ?i64,
    tags: []const []const u8,
    page: i32,
    page_size: i32,
) !ResultSet {
    const dialect = db_client.vtable.get_dialect(db_client.context);
    var qb = try QueryBuilder.init(allocator, dialect);
    
    // Base query
    _ = try qb.select(&[_][]const u8{
            "d.id",
            "d.name",
            "d.description",
            "n.name as namespace",
            "COUNT(j.id) as job_count",
        })
        .from("datasets d")
        .join(.{
            .type = .Inner,
            .table = "namespaces n",
            .condition = "d.namespace_id = n.id",
        })
        .join(.{
            .type = .Left,
            .table = "jobs j",
            .condition = "d.id = j.output_dataset_id",
        });
    
    // Add search filter
    if (search_query.len > 0) {
        if (dialect == .SQLite) {
            _ = try qb.where(.{
                .expression = "d.name LIKE $1 OR d.description LIKE $1"
            });
        } else {
            _ = try qb.where(.{
                .expression = "d.name ILIKE $1 OR d.description ILIKE $1"
            });
        }
    }
    
    // Add namespace filter
    if (namespace_id) |ns_id| {
        _ = try qb.where(.{
            .expression = "d.namespace_id = $2"
        });
    }
    
    // Add tags filter (PostgreSQL array containment)
    if (tags.len > 0 and dialect == .PostgreSQL) {
        _ = try qb.where(.{
            .expression = "d.tags @> $3::text[]"
        });
    }
    
    // Group and page
    _ = try qb.groupBy(&[_][]const u8{"d.id", "d.name", "d.description", "n.name"})
        .orderBy("d.created_at", .DESC)
        .limit(page_size)
        .offset(page * page_size);
    
    const sql = try qb.build();
    
    // Build parameters
    var params = std.ArrayList(Value).init(allocator);
    defer params.deinit();
    
    if (search_query.len > 0) {
        try params.append(.{ .string = 
            try std.fmt.allocPrint(allocator, "%{s}%", .{search_query})
        });
    }
    if (namespace_id) |ns_id| {
        try params.append(.{ .int64 = ns_id });
    }
    if (tags.len > 0) {
        try params.append(.{ .array = tags });
    }
    
    return try db_client.execute(sql, params.items);
}
```

### Example 3: Data Quality Metrics

```zig
pub fn getDataQualityMetrics(
    allocator: Allocator,
    db_client: *DbClient,
    time_window_days: i32,
) !ResultSet {
    var qb = try QueryBuilder.init(allocator, 
        db_client.vtable.get_dialect(db_client.context));
    
    _ = try qb.with("recent_runs",
        \\SELECT
        \\    r.job_id,
        \\    r.state,
        \\    r.started_at
        \\FROM runs r
        \\WHERE r.started_at > NOW() - INTERVAL '$1 days'
        )
        .select(&[_][]const u8{
            "j.id",
            "j.name",
            "COUNT(*) as total_runs",
            "SUM(CASE WHEN rr.state = 'COMPLETED' THEN 1 ELSE 0 END) as successful_runs",
            "AVG(CASE WHEN rr.state = 'COMPLETED' THEN 1.0 ELSE 0.0 END) as success_rate",
            "MAX(rr.started_at) as last_run",
        })
        .from("jobs j")
        .join(.{
            .type = .Inner,
            .table = "recent_runs rr",
            .condition = "j.id = rr.job_id",
        })
        .groupBy(&[_][]const u8{"j.id", "j.name"})
        .having("COUNT(*) > 0")
        .orderBy("success_rate", .ASC);
    
    const sql = try qb.build();
    return try db_client.execute(sql, &[_]Value{
        .{ .int32 = time_window_days },
    });
}
```

---

## Summary

### Key Takeaways

1. **Use the fluent API** for readable, maintainable queries
2. **Leverage dialect-specific features** when performance matters
3. **Always use parameterized queries** to prevent SQL injection
4. **Minimize data transfer** by selecting only needed columns
5. **Design for indexes** to maximize query performance
6. **Use CTEs** for complex multi-step queries
7. **Batch operations** instead of loops
8. **Test across all target dialects**

### Quick Reference

| Pattern | Use When | Performance |
|---------|----------|-------------|
| Simple SELECT | Basic data retrieval | Excellent |
| JOIN | Related data | Good |
| CTE | Complex logic, lineage | Good |
| Subquery | One-off nested query | Variable |
| Window Function | Analytics over partitions | Good |
| Graph Query (HANA) | Lineage, dependencies | Excellent |
| Full-text Search | Text search | Good (with index) |

---

**Version History:**
- v1.0 (2026-01-20): Initial query builder patterns guide

**Last Updated:** January 20, 2026
