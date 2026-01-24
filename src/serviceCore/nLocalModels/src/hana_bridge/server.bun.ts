/**
 * HANA Cloud SQL Bridge - Bun Version
 *
 * Bun advantages:
 * - 4x faster HTTP server (uWebSockets)
 * - Faster startup (~5ms vs ~50ms)
 * - Native TypeScript
 * - Lower memory footprint
 */

// ============================================================================
// Types
// ============================================================================

interface HanaConfig {
  host: string;
  port: number;
  user: string;
  password: string;
  schema: string;
  encrypt: boolean;
  sslValidateCertificate: boolean;
}

interface PoolConfig {
  min: number;
  max: number;
  acquireTimeout: number;
  idleTimeout: number;
}

interface RetryConfig {
  maxAttempts: number;
  initialDelay: number;
  maxDelay: number;
  factor: number;
}

interface CircuitBreakerConfig {
  threshold: number;
  resetTimeout: number;
}

interface Metrics {
  requestsTotal: number;
  requestsSuccess: number;
  requestsFailed: number;
  requestDurationSum: number;
  requestDurationCount: number;
  connectionPoolSize: number;
  connectionPoolActive: number;
  circuitBreakerState: string;
  circuitBreakerFailures: number;
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
    encrypt: true,
    sslValidateCertificate: Bun.env.HANA_SSL_VALIDATE === "true",
  } as HanaConfig,

  pool: {
    min: parseInt(Bun.env.POOL_MIN || "2"),
    max: parseInt(Bun.env.POOL_MAX || "10"),
    acquireTimeout: parseInt(Bun.env.POOL_ACQUIRE_TIMEOUT || "30000"),
    idleTimeout: parseInt(Bun.env.POOL_IDLE_TIMEOUT || "60000"),
  } as PoolConfig,

  retry: {
    maxAttempts: parseInt(Bun.env.RETRY_MAX_ATTEMPTS || "3"),
    initialDelay: parseInt(Bun.env.RETRY_INITIAL_DELAY || "1000"),
    maxDelay: parseInt(Bun.env.RETRY_MAX_DELAY || "10000"),
    factor: parseFloat(Bun.env.RETRY_FACTOR || "2"),
  } as RetryConfig,

  circuitBreaker: {
    threshold: parseInt(Bun.env.CB_THRESHOLD || "5"),
    resetTimeout: parseInt(Bun.env.CB_RESET_TIMEOUT || "30000"),
  } as CircuitBreakerConfig,

  rateLimit: {
    windowMs: parseInt(Bun.env.RATE_LIMIT_WINDOW || "60000"),
    maxRequests: parseInt(Bun.env.RATE_LIMIT_MAX || "1000"), // Higher for Bun
  },

  corsOrigin: Bun.env.CORS_ORIGIN || "*",
};

// ============================================================================
// Metrics
// ============================================================================

const metrics: Metrics = {
  requestsTotal: 0,
  requestsSuccess: 0,
  requestsFailed: 0,
  requestDurationSum: 0,
  requestDurationCount: 0,
  connectionPoolSize: 0,
  connectionPoolActive: 0,
  circuitBreakerState: "closed",
  circuitBreakerFailures: 0,
};

function getMetricsText(): string {
  return `# HELP hana_bridge_requests_total Total requests
# TYPE hana_bridge_requests_total counter
hana_bridge_requests_total ${metrics.requestsTotal}

# HELP hana_bridge_requests_success_total Successful requests
# TYPE hana_bridge_requests_success_total counter
hana_bridge_requests_success_total ${metrics.requestsSuccess}

# HELP hana_bridge_requests_failed_total Failed requests
# TYPE hana_bridge_requests_failed_total counter
hana_bridge_requests_failed_total ${metrics.requestsFailed}

# HELP hana_bridge_request_duration_seconds Request duration
# TYPE hana_bridge_request_duration_seconds summary
hana_bridge_request_duration_seconds_sum ${metrics.requestDurationSum / 1000}
hana_bridge_request_duration_seconds_count ${metrics.requestDurationCount}

# HELP hana_bridge_pool_size Connection pool size
# TYPE hana_bridge_pool_size gauge
hana_bridge_pool_size ${metrics.connectionPoolSize}

# HELP hana_bridge_pool_active Active connections
# TYPE hana_bridge_pool_active gauge
hana_bridge_pool_active ${metrics.connectionPoolActive}

# HELP hana_bridge_circuit_breaker Circuit breaker (0=closed,1=open,2=half)
# TYPE hana_bridge_circuit_breaker gauge
hana_bridge_circuit_breaker ${metrics.circuitBreakerState === "closed" ? 0 : metrics.circuitBreakerState === "open" ? 1 : 2}
`;
}

// ============================================================================
// Logging
// ============================================================================

type LogLevel = "DEBUG" | "INFO" | "WARN" | "ERROR";

const logLevels: Record<LogLevel, number> = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };
const currentLogLevel = logLevels[(Bun.env.LOG_LEVEL?.toUpperCase() as LogLevel) || "INFO"] ?? 1;

function log(level: LogLevel, message: string, meta: Record<string, unknown> = {}): void {
  if (logLevels[level] < currentLogLevel) return;

  console.log(JSON.stringify({
    timestamp: new Date().toISOString(),
    level,
    message,
    pid: process.pid,
    ...meta,
  }));
}

// ============================================================================
// Circuit Breaker
// ============================================================================

class CircuitBreaker {
  private state: "closed" | "open" | "half-open" = "closed";
  private failures = 0;
  private lastFailure: number | null = null;
  private halfOpenRequests = 0;

  constructor(private config: CircuitBreakerConfig) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === "open") {
      if (Date.now() - (this.lastFailure || 0) > this.config.resetTimeout) {
        this.state = "half-open";
        this.halfOpenRequests = 0;
        log("INFO", "Circuit breaker half-open");
        metrics.circuitBreakerState = "half-open";
      } else {
        throw new Error("Circuit breaker is open");
      }
    }

    if (this.state === "half-open" && this.halfOpenRequests >= 1) {
      throw new Error("Circuit breaker is testing");
    }

    try {
      if (this.state === "half-open") this.halfOpenRequests++;
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failures = 0;
    if (this.state === "half-open") {
      this.state = "closed";
      log("INFO", "Circuit breaker closed");
      metrics.circuitBreakerState = "closed";
    }
    metrics.circuitBreakerFailures = this.failures;
  }

  private onFailure(): void {
    this.failures++;
    this.lastFailure = Date.now();
    metrics.circuitBreakerFailures = this.failures;

    if (this.failures >= this.config.threshold) {
      this.state = "open";
      log("WARN", "Circuit breaker opened", { failures: this.failures });
      metrics.circuitBreakerState = "open";
    }
  }

  getState(): string {
    return this.state;
  }
}

// ============================================================================
// Rate Limiter (using Map for speed)
// ============================================================================

class RateLimiter {
  private requests = new Map<string, number[]>();

  constructor(private windowMs: number, private maxRequests: number) {}

  isAllowed(ip: string): boolean {
    const now = Date.now();
    const windowStart = now - this.windowMs;

    let timestamps = this.requests.get(ip);
    if (timestamps) {
      timestamps = timestamps.filter(t => t > windowStart);
      if (timestamps.length >= this.maxRequests) {
        return false;
      }
      timestamps.push(now);
      this.requests.set(ip, timestamps);
    } else {
      this.requests.set(ip, [now]);
    }

    return true;
  }

  // Periodic cleanup
  cleanup(): void {
    const now = Date.now();
    const windowStart = now - this.windowMs;

    for (const [ip, timestamps] of this.requests) {
      const valid = timestamps.filter(t => t > windowStart);
      if (valid.length === 0) {
        this.requests.delete(ip);
      } else {
        this.requests.set(ip, valid);
      }
    }
  }
}

// ============================================================================
// HANA Callback Pump (Bun Compatibility Fix)
// ============================================================================

let hanaModule: any = null;
let callbackPumpRunning = false;
let activeCallbacks = 0;

// Global callback pump - runs while there are active HANA operations
async function startCallbackPump(): Promise<void> {
  if (callbackPumpRunning) return;
  callbackPumpRunning = true;

  while (activeCallbacks > 0) {
    if (hanaModule && typeof hanaModule.__runCallbacks === "function") {
      hanaModule.__runCallbacks();
    }
    await Bun.sleep(5); // 5ms interval reduces CPU usage
  }

  callbackPumpRunning = false;
}

// Helper to wrap HANA callbacks with Bun event loop pump
function wrapHanaCallback<T>(
  executor: (resolve: (value: T) => void, reject: (error: Error) => void) => void
): Promise<T> {
  return new Promise((resolve, reject) => {
    activeCallbacks++;
    startCallbackPump(); // Ensure pump is running

    executor(
      (value: T) => { activeCallbacks--; resolve(value); },
      (error: Error) => { activeCallbacks--; reject(error); }
    );
  });
}

// ============================================================================
// Connection Pool
// ============================================================================

class ConnectionPool {
  private connections: any[] = [];
  private available: any[] = [];
  private waiting: Array<{ resolve: Function; reject: Function; timeout: Timer }> = [];
  private hana: any = null;

  constructor(private hanaConfig: HanaConfig, private poolConfig: PoolConfig) {}

  async init(): Promise<void> {
    try {
      this.hana = require("@sap/hana-client");
      hanaModule = this.hana; // Store for callback pump
    } catch (e: any) {
      log("ERROR", "HANA client not installed", { error: e.message });
      throw e;
    }

    // Create minimum connections
    const initPromises: Promise<void>[] = [];
    for (let i = 0; i < this.poolConfig.min; i++) {
      initPromises.push(
        this.createConnection()
          .then(conn => {
            this.connections.push(conn);
            this.available.push(conn);
          })
          .catch(e => log("WARN", "Failed to create initial connection", { error: e.message }))
      );
    }

    await Promise.all(initPromises);
    metrics.connectionPoolSize = this.connections.length;
    log("INFO", "Connection pool initialized", { size: this.connections.length, runtime: "bun" });
  }

  private createConnection(): Promise<any> {
    const conn = this.hana.createConnection();

    return wrapHanaCallback<any>((resolve, reject) => {
      conn.connect({
        serverNode: `${this.hanaConfig.host}:${this.hanaConfig.port}`,
        uid: this.hanaConfig.user,
        pwd: this.hanaConfig.password,
        encrypt: this.hanaConfig.encrypt,
        sslValidateCertificate: this.hanaConfig.sslValidateCertificate,
      }, (err: Error | null) => {
        if (err) {
          reject(err);
          return;
        }

        // Set schema with pumped callback
        conn.exec(`SET SCHEMA ${this.hanaConfig.schema}`, (schemaErr: Error | null) => {
          if (schemaErr) {
            log("WARN", "Could not set default schema", { error: schemaErr.message });
          }
          resolve(conn);
        });
      });
    });
  }

  async acquire(): Promise<any> {
    if (this.available.length > 0) {
      const conn = this.available.pop()!;
      metrics.connectionPoolActive = this.connections.length - this.available.length;
      return conn;
    }

    if (this.connections.length < this.poolConfig.max) {
      try {
        const conn = await this.createConnection();
        this.connections.push(conn);
        metrics.connectionPoolSize = this.connections.length;
        metrics.connectionPoolActive = this.connections.length - this.available.length;
        return conn;
      } catch (e: any) {
        log("ERROR", "Failed to create connection", { error: e.message });
        throw e;
      }
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        const idx = this.waiting.findIndex(w => w.timeout === timeout);
        if (idx > -1) this.waiting.splice(idx, 1);
        reject(new Error("Connection acquire timeout"));
      }, this.poolConfig.acquireTimeout);

      this.waiting.push({ resolve, reject, timeout });
    });
  }

  release(conn: any): void {
    if (this.waiting.length > 0) {
      const waiter = this.waiting.shift()!;
      clearTimeout(waiter.timeout);
      waiter.resolve(conn);
      return;
    }

    this.available.push(conn);
    metrics.connectionPoolActive = this.connections.length - this.available.length;
  }

  async shutdown(): Promise<void> {
    for (const conn of this.connections) {
      try {
        conn.disconnect();
      } catch {}
    }
    this.connections = [];
    this.available = [];
    metrics.connectionPoolSize = 0;
    metrics.connectionPoolActive = 0;
  }

  getSize(): number {
    return this.connections.length;
  }
}

// ============================================================================
// SQL Executor with Retry
// ============================================================================

async function executeWithRetry(
  pool: ConnectionPool,
  circuitBreaker: CircuitBreaker,
  sql: string,
  retryConfig: RetryConfig
): Promise<any> {
  let lastError: Error | null = null;
  let delay = retryConfig.initialDelay;

  for (let attempt = 1; attempt <= retryConfig.maxAttempts; attempt++) {
    try {
      return await circuitBreaker.execute(async () => {
        const conn = await pool.acquire();
        try {
          // Use wrapHanaCallback for Bun compatibility
          return await wrapHanaCallback<any>((resolve, reject) => {
            conn.exec(sql, (err: Error | null, result: any) => {
              if (err) reject(err);
              else resolve(result);
            });
          });
        } finally {
          pool.release(conn);
        }
      });
    } catch (error: any) {
      lastError = error;

      if (error.message.includes("Circuit breaker")) {
        throw error;
      }

      if (attempt < retryConfig.maxAttempts) {
        log("WARN", "SQL execution failed, retrying", {
          attempt,
          maxAttempts: retryConfig.maxAttempts,
          delay,
          error: error.message,
        });

        await Bun.sleep(delay);
        delay = Math.min(delay * retryConfig.factor, retryConfig.maxDelay);
      }
    }
  }

  throw lastError;
}

// ============================================================================
// Main Server
// ============================================================================

async function main(): Promise<void> {
  const startTime = Bun.nanoseconds();

  const pool = new ConnectionPool(config.hana, config.pool);
  const circuitBreaker = new CircuitBreaker(config.circuitBreaker);
  const rateLimiter = new RateLimiter(config.rateLimit.windowMs, config.rateLimit.maxRequests);

  await pool.init();

  // Periodic rate limiter cleanup
  setInterval(() => rateLimiter.cleanup(), 60000);

  const server = Bun.serve({
    port: config.port,
    hostname: config.host,

    async fetch(req: Request): Promise<Response> {
      const reqStart = performance.now();
      const url = new URL(req.url);
      const clientIp = req.headers.get("x-forwarded-for")?.split(",")[0] || "unknown";

      metrics.requestsTotal++;

      // CORS headers
      const corsHeaders = {
        "Access-Control-Allow-Origin": config.corsOrigin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
      };

      if (req.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: corsHeaders });
      }

      // Rate limiting
      if (!rateLimiter.isAllowed(clientIp)) {
        metrics.requestsFailed++;
        return Response.json({ error: "Too many requests" }, { status: 429, headers: corsHeaders });
      }

      try {
        // Health check
        if (url.pathname === "/health" || url.pathname === "/healthz") {
          metrics.requestsSuccess++;
          return Response.json({
            status: "ok",
            uptime: process.uptime(),
            timestamp: new Date().toISOString(),
            runtime: "bun",
          }, { headers: corsHeaders });
        }

        // Readiness
        if (url.pathname === "/ready" || url.pathname === "/readyz") {
          const isReady = pool.getSize() > 0 && circuitBreaker.getState() !== "open";
          if (isReady) metrics.requestsSuccess++;
          else metrics.requestsFailed++;

          return Response.json({
            ready: isReady,
            pool: pool.getSize(),
            circuitBreaker: circuitBreaker.getState(),
          }, { status: isReady ? 200 : 503, headers: corsHeaders });
        }

        // Metrics
        if (url.pathname === "/metrics") {
          return new Response(getMetricsText(), {
            headers: { ...corsHeaders, "Content-Type": "text/plain" },
          });
        }

        // Test
        if (url.pathname === "/test" && req.method === "GET") {
          const result = await executeWithRetry(
            pool, circuitBreaker,
            "SELECT 1 AS test FROM DUMMY",
            config.retry
          );
          metrics.requestsSuccess++;
          return Response.json({ success: true, result, runtime: "bun" }, { headers: corsHeaders });
        }

        // SQL execution
        if (url.pathname === "/sql" && req.method === "POST") {
          const body = await req.json() as { sql?: string; schema?: string };

          if (!body.sql || typeof body.sql !== "string") {
            metrics.requestsFailed++;
            return Response.json(
              { error: "Missing or invalid sql parameter" },
              { status: 400, headers: corsHeaders }
            );
          }

          // Schema switch if needed
          if (body.schema && body.schema !== config.hana.schema) {
            await executeWithRetry(
              pool, circuitBreaker,
              `SET SCHEMA ${body.schema}`,
              config.retry
            );
          }

          const result = await executeWithRetry(pool, circuitBreaker, body.sql, config.retry);

          metrics.requestsSuccess++;
          return Response.json({
            success: true,
            result,
            rowCount: Array.isArray(result) ? result.length : 0,
          }, { headers: corsHeaders });
        }

        // Not found
        return Response.json({ error: "Not found" }, { status: 404, headers: corsHeaders });

      } catch (error: any) {
        log("ERROR", "Request failed", {
          url: url.pathname,
          method: req.method,
          error: error.message,
        });

        metrics.requestsFailed++;
        return Response.json(
          { success: false, error: error.message },
          { status: 500, headers: corsHeaders }
        );

      } finally {
        const duration = performance.now() - reqStart;
        metrics.requestDurationSum += duration;
        metrics.requestDurationCount++;
      }
    },
  });

  const startupMs = (Bun.nanoseconds() - startTime) / 1_000_000;

  log("INFO", "HANA Bridge (Bun) started", {
    port: config.port,
    host: config.host,
    poolSize: pool.getSize(),
    startupMs: startupMs.toFixed(2),
  });

  console.log(`
================================================================================
  HANA Cloud SQL Bridge (Bun)
================================================================================
  URL:      http://${config.host}:${config.port}
  Runtime:  Bun ${Bun.version}
  Startup:  ${startupMs.toFixed(2)}ms
  Pool:     ${pool.getSize()} connections
================================================================================
`);

  // Graceful shutdown
  process.on("SIGTERM", async () => {
    log("INFO", "Shutting down...");
    server.stop();
    await pool.shutdown();
    process.exit(0);
  });

  process.on("SIGINT", async () => {
    log("INFO", "Shutting down...");
    server.stop();
    await pool.shutdown();
    process.exit(0);
  });
}

main().catch(err => {
  log("ERROR", "Failed to start", { error: err.message });
  process.exit(1);
});
