/**
 * HANA Cloud SQL Bridge - Fast Edition (Bun)
 *
 * Features:
 * - Full CRUD REST API
 * - Query caching (LRU)
 * - Batch query support
 * - Prepared statement caching
 * - Connection pooling
 * - Parallel query execution
 */

// ============================================================================
// Types
// ============================================================================

interface CacheEntry<T> {
  value: T;
  expires: number;
}

interface QueryResult {
  success: boolean;
  result?: any;
  rowCount?: number;
  cached?: boolean;
  duration?: number;
  error?: string;
}

interface BatchQuery {
  id: string;
  sql: string;
  params?: any[];
}

interface BatchResult {
  id: string;
  success: boolean;
  result?: any;
  error?: string;
  duration?: number;
}

// ============================================================================
// Configuration
// ============================================================================

const config = {
  port: parseInt(Bun.env.BRIDGE_PORT || "3001"),
  host: Bun.env.BRIDGE_HOST || "0.0.0.0",

  hana: {
    host: Bun.env.HANA_HOST || "d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com",
    port: parseInt(Bun.env.HANA_PORT || "443"),
    user: Bun.env.HANA_USER || "DBADMIN",
    password: Bun.env.HANA_PASSWORD || "Initial@1",
    schema: Bun.env.HANA_SCHEMA || "NUCLEUS",
  },

  pool: {
    min: parseInt(Bun.env.POOL_MIN || "5"),
    max: parseInt(Bun.env.POOL_MAX || "20"),
  },

  cache: {
    enabled: Bun.env.CACHE_ENABLED !== "false",
    maxSize: parseInt(Bun.env.CACHE_MAX_SIZE || "1000"),
    ttlMs: parseInt(Bun.env.CACHE_TTL_MS || "30000"), // 30 seconds default
  },

  preparedStatements: {
    enabled: Bun.env.PREPARED_STATEMENTS !== "false",
    maxSize: parseInt(Bun.env.PREPARED_MAX_SIZE || "100"),
  },
};

// ============================================================================
// LRU Cache for Query Results
// ============================================================================

class LRUCache<T> {
  private cache = new Map<string, CacheEntry<T>>();
  private maxSize: number;

  constructor(maxSize: number) {
    this.maxSize = maxSize;
  }

  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    if (Date.now() > entry.expires) {
      this.cache.delete(key);
      return undefined;
    }

    // Move to end (most recently used)
    this.cache.delete(key);
    this.cache.set(key, entry);
    return entry.value;
  }

  set(key: string, value: T, ttlMs: number): void {
    // Evict oldest if at capacity
    if (this.cache.size >= this.maxSize) {
      const oldest = this.cache.keys().next().value;
      if (oldest) this.cache.delete(oldest);
    }

    this.cache.set(key, {
      value,
      expires: Date.now() + ttlMs,
    });
  }

  invalidate(pattern?: string): number {
    if (!pattern) {
      const size = this.cache.size;
      this.cache.clear();
      return size;
    }

    let count = 0;
    const regex = new RegExp(pattern, "i");
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
        count++;
      }
    }
    return count;
  }

  stats(): { size: number; maxSize: number } {
    return { size: this.cache.size, maxSize: this.maxSize };
  }
}

// ============================================================================
// HANA Connection Pool with Prepared Statements
// ============================================================================

let hanaModule: any = null;
let callbackPumpRunning = false;
let activeCallbacks = 0;

async function startCallbackPump(): Promise<void> {
  if (callbackPumpRunning) return;
  callbackPumpRunning = true;

  while (activeCallbacks > 0) {
    if (hanaModule?.__runCallbacks) {
      hanaModule.__runCallbacks();
    }
    await Bun.sleep(2);
  }
  callbackPumpRunning = false;
}

function wrapCallback<T>(
  executor: (resolve: (v: T) => void, reject: (e: Error) => void) => void
): Promise<T> {
  return new Promise((resolve, reject) => {
    activeCallbacks++;
    startCallbackPump();
    executor(
      (v) => { activeCallbacks--; resolve(v); },
      (e) => { activeCallbacks--; reject(e); }
    );
  });
}

class ConnectionPool {
  private connections: any[] = [];
  private available: any[] = [];
  private preparedCache = new Map<string, any>();

  async init(): Promise<void> {
    hanaModule = require("@sap/hana-client");

    // Create connections in parallel
    const promises = Array(config.pool.min).fill(0).map(() => this.createConnection());
    const conns = await Promise.all(promises);

    this.connections = conns.filter(Boolean);
    this.available = [...this.connections];

    console.log(`Pool initialized: ${this.connections.length} connections`);
  }

  private async createConnection(): Promise<any> {
    const conn = hanaModule.createConnection();

    return wrapCallback((resolve, reject) => {
      conn.connect({
        serverNode: `${config.hana.host}:${config.hana.port}`,
        uid: config.hana.user,
        pwd: config.hana.password,
        encrypt: true,
        sslValidateCertificate: false,
      }, (err: Error | null) => {
        if (err) return reject(err);

        conn.exec(`SET SCHEMA ${config.hana.schema}`, () => resolve(conn));
      });
    });
  }

  async acquire(): Promise<any> {
    if (this.available.length > 0) {
      return this.available.pop();
    }

    if (this.connections.length < config.pool.max) {
      const conn = await this.createConnection();
      this.connections.push(conn);
      return conn;
    }

    // Wait for available connection
    return new Promise((resolve) => {
      const check = () => {
        if (this.available.length > 0) {
          resolve(this.available.pop());
        } else {
          setTimeout(check, 5);
        }
      };
      check();
    });
  }

  release(conn: any): void {
    this.available.push(conn);
  }

  // Get or create prepared statement
  async getPrepared(conn: any, sql: string): Promise<any> {
    if (!config.preparedStatements.enabled) return null;

    const key = `${conn._id || "c"}_${sql}`;
    if (this.preparedCache.has(key)) {
      return this.preparedCache.get(key);
    }

    return wrapCallback((resolve, reject) => {
      conn.prepare(sql, (err: Error | null, stmt: any) => {
        if (err) return reject(err);

        if (this.preparedCache.size >= config.preparedStatements.maxSize) {
          const oldest = this.preparedCache.keys().next().value;
          if (oldest) {
            this.preparedCache.get(oldest)?.drop?.();
            this.preparedCache.delete(oldest);
          }
        }

        this.preparedCache.set(key, stmt);
        resolve(stmt);
      });
    });
  }

  getStats() {
    return {
      total: this.connections.length,
      available: this.available.length,
      active: this.connections.length - this.available.length,
      prepared: this.preparedCache.size,
    };
  }
}

// ============================================================================
// Query Executor
// ============================================================================

const queryCache = new LRUCache<any>(config.cache.maxSize);
const pool = new ConnectionPool();

async function executeQuery(sql: string, useCache = true): Promise<QueryResult> {
  const start = performance.now();
  const cacheKey = sql.trim().toLowerCase();

  // Check cache for SELECT queries
  if (useCache && config.cache.enabled && sql.trim().toUpperCase().startsWith("SELECT")) {
    const cached = queryCache.get(cacheKey);
    if (cached !== undefined) {
      return {
        success: true,
        result: cached,
        rowCount: Array.isArray(cached) ? cached.length : 0,
        cached: true,
        duration: performance.now() - start,
      };
    }
  }

  const conn = await pool.acquire();

  try {
    const result = await wrapCallback<any>((resolve, reject) => {
      conn.exec(sql, (err: Error | null, res: any) => {
        if (err) reject(err);
        else resolve(res);
      });
    });

    // Cache SELECT results
    if (config.cache.enabled && sql.trim().toUpperCase().startsWith("SELECT")) {
      queryCache.set(cacheKey, result, config.cache.ttlMs);
    }

    // Invalidate cache on write operations
    if (/^(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)/i.test(sql.trim())) {
      const tableMatch = sql.match(/(?:INTO|FROM|UPDATE|TABLE)\s+([^\s(]+)/i);
      if (tableMatch) {
        queryCache.invalidate(tableMatch[1]);
      }
    }

    return {
      success: true,
      result,
      rowCount: Array.isArray(result) ? result.length : 0,
      cached: false,
      duration: performance.now() - start,
    };
  } catch (error: any) {
    return {
      success: false,
      error: error.message,
      duration: performance.now() - start,
    };
  } finally {
    pool.release(conn);
  }
}

// Execute multiple queries in parallel
async function executeBatch(queries: BatchQuery[]): Promise<BatchResult[]> {
  return Promise.all(
    queries.map(async (q) => {
      const start = performance.now();
      try {
        const result = await executeQuery(q.sql);
        return {
          id: q.id,
          success: result.success,
          result: result.result,
          error: result.error,
          duration: performance.now() - start,
        };
      } catch (error: any) {
        return {
          id: q.id,
          success: false,
          error: error.message,
          duration: performance.now() - start,
        };
      }
    })
  );
}

// ============================================================================
// CRUD Helpers
// ============================================================================

function buildSelect(table: string, query: URLSearchParams): string {
  const columns = query.get("select") || "*";
  const where = query.get("where");
  const orderBy = query.get("orderBy");
  const limit = query.get("limit") || "100";
  const offset = query.get("offset") || "0";

  let sql = `SELECT ${columns} FROM ${config.hana.schema}.${table}`;
  if (where) sql += ` WHERE ${where}`;
  if (orderBy) sql += ` ORDER BY ${orderBy}`;
  sql += ` LIMIT ${limit} OFFSET ${offset}`;

  return sql;
}

function buildInsert(table: string, data: Record<string, any>): string {
  const columns = Object.keys(data);
  const values = columns.map((k) => {
    const v = data[k];
    return typeof v === "string" ? `'${v.replace(/'/g, "''")}'` : v;
  });

  return `INSERT INTO ${config.hana.schema}.${table} (${columns.join(", ")}) VALUES (${values.join(", ")})`;
}

function buildUpdate(table: string, id: string, data: Record<string, any>): string {
  const sets = Object.entries(data)
    .map(([k, v]) => {
      const val = typeof v === "string" ? `'${v.replace(/'/g, "''")}'` : v;
      return `${k} = ${val}`;
    })
    .join(", ");

  return `UPDATE ${config.hana.schema}.${table} SET ${sets} WHERE ID = ${id}`;
}

function buildDelete(table: string, id: string): string {
  return `DELETE FROM ${config.hana.schema}.${table} WHERE ID = ${id}`;
}

// ============================================================================
// HTTP Server
// ============================================================================

async function main() {
  await pool.init();

  const server = Bun.serve({
    port: config.port,
    hostname: config.host,

    async fetch(req: Request): Promise<Response> {
      const url = new URL(req.url);
      const path = url.pathname;
      const method = req.method;

      const headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      };

      if (method === "OPTIONS") {
        return new Response(null, { status: 204, headers });
      }

      try {
        // ============================================================
        // Health & Metrics
        // ============================================================
        if (path === "/health") {
          return Response.json({
            status: "ok",
            runtime: "bun",
            version: "fast",
            uptime: process.uptime(),
          }, { headers });
        }

        if (path === "/stats") {
          return Response.json({
            pool: pool.getStats(),
            cache: queryCache.stats(),
            config: {
              cacheEnabled: config.cache.enabled,
              cacheTtlMs: config.cache.ttlMs,
              preparedStatements: config.preparedStatements.enabled,
            },
          }, { headers });
        }

        // ============================================================
        // Raw SQL Endpoints
        // ============================================================
        if (path === "/sql" && method === "POST") {
          const body = await req.json() as { sql: string; cache?: boolean };
          const result = await executeQuery(body.sql, body.cache !== false);
          return Response.json(result, { headers });
        }

        if (path === "/batch" && method === "POST") {
          const body = await req.json() as { queries: BatchQuery[] };
          const results = await executeBatch(body.queries);
          return Response.json({ success: true, results }, { headers });
        }

        // ============================================================
        // Cache Management
        // ============================================================
        if (path === "/cache/clear" && method === "POST") {
          const body = await req.json().catch(() => ({})) as { pattern?: string };
          const cleared = queryCache.invalidate(body.pattern);
          return Response.json({ success: true, cleared }, { headers });
        }

        // ============================================================
        // CRUD REST API: /api/v1/{table}
        // ============================================================
        const apiMatch = path.match(/^\/api\/v1\/([a-zA-Z_][a-zA-Z0-9_]*)(?:\/(\d+))?$/);
        if (apiMatch) {
          const table = apiMatch[1].toUpperCase();
          const id = apiMatch[2];

          // GET /api/v1/{table} - List all
          // GET /api/v1/{table}?where=...&orderBy=...&limit=...
          if (method === "GET" && !id) {
            const sql = buildSelect(table, url.searchParams);
            const result = await executeQuery(sql);
            return Response.json(result, { headers });
          }

          // GET /api/v1/{table}/{id} - Get one
          if (method === "GET" && id) {
            const sql = `SELECT * FROM ${config.hana.schema}.${table} WHERE ID = ${id}`;
            const result = await executeQuery(sql);
            if (result.success && Array.isArray(result.result)) {
              result.result = result.result[0] || null;
              result.rowCount = result.result ? 1 : 0;
            }
            return Response.json(result, { headers });
          }

          // POST /api/v1/{table} - Create
          if (method === "POST") {
            const data = await req.json();
            const sql = buildInsert(table, data);
            const result = await executeQuery(sql);
            return Response.json(result, { status: result.success ? 201 : 400, headers });
          }

          // PUT/PATCH /api/v1/{table}/{id} - Update
          if ((method === "PUT" || method === "PATCH") && id) {
            const data = await req.json();
            const sql = buildUpdate(table, id, data);
            const result = await executeQuery(sql);
            return Response.json(result, { headers });
          }

          // DELETE /api/v1/{table}/{id} - Delete
          if (method === "DELETE" && id) {
            const sql = buildDelete(table, id);
            const result = await executeQuery(sql);
            return Response.json(result, { headers });
          }
        }

        return Response.json({ error: "Not found" }, { status: 404, headers });

      } catch (error: any) {
        return Response.json(
          { success: false, error: error.message },
          { status: 500, headers }
        );
      }
    },
  });

  console.log(`
================================================================================
  HANA Bridge - Fast Edition (Bun)
================================================================================
  URL:        http://${config.host}:${config.port}
  Runtime:    Bun ${Bun.version}
  Pool:       ${config.pool.min}-${config.pool.max} connections
  Cache:      ${config.cache.enabled ? `enabled (${config.cache.ttlMs}ms TTL)` : "disabled"}
  Prepared:   ${config.preparedStatements.enabled ? "enabled" : "disabled"}

  Endpoints:
    GET  /health              - Health check
    GET  /stats               - Pool & cache stats
    POST /sql                 - Execute SQL {"sql": "..."}
    POST /batch               - Batch queries {"queries": [...]}
    POST /cache/clear         - Clear cache {"pattern": "..."}

  CRUD API:
    GET    /api/v1/{table}          - List (supports ?where=&orderBy=&limit=)
    GET    /api/v1/{table}/{id}     - Get one
    POST   /api/v1/{table}          - Create
    PUT    /api/v1/{table}/{id}     - Update
    DELETE /api/v1/{table}/{id}     - Delete
================================================================================
`);
}

main().catch(console.error);
