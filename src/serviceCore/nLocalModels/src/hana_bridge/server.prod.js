/**
 * HANA Cloud SQL Bridge - Production Server
 *
 * Features:
 * - Connection pooling with automatic reconnection
 * - Health checks (liveness + readiness probes)
 * - Retry logic with exponential backoff
 * - Circuit breaker pattern
 * - Prometheus metrics
 * - Structured JSON logging
 * - Rate limiting
 * - Graceful shutdown
 * - Cluster support via PM2
 */

const http = require('http');
const cluster = require('cluster');
const os = require('os');

// ============================================================================
// Configuration
// ============================================================================

const config = {
    // Server
    port: parseInt(process.env.BRIDGE_PORT) || 3001,
    host: process.env.BRIDGE_HOST || '0.0.0.0',

    // HANA Connection
    hana: {
        host: process.env.HANA_HOST || 'd93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com',
        port: parseInt(process.env.HANA_PORT) || 443,
        user: process.env.HANA_USER || 'DBADMIN',
        password: process.env.HANA_PASSWORD || 'Initial@1',
        schema: process.env.HANA_SCHEMA || 'NUCLEUS',
        encrypt: true,
        sslValidateCertificate: process.env.HANA_SSL_VALIDATE === 'true',
    },

    // Connection Pool
    pool: {
        min: parseInt(process.env.POOL_MIN) || 2,
        max: parseInt(process.env.POOL_MAX) || 10,
        acquireTimeout: parseInt(process.env.POOL_ACQUIRE_TIMEOUT) || 30000,
        idleTimeout: parseInt(process.env.POOL_IDLE_TIMEOUT) || 60000,
    },

    // Retry
    retry: {
        maxAttempts: parseInt(process.env.RETRY_MAX_ATTEMPTS) || 3,
        initialDelay: parseInt(process.env.RETRY_INITIAL_DELAY) || 1000,
        maxDelay: parseInt(process.env.RETRY_MAX_DELAY) || 10000,
        factor: parseFloat(process.env.RETRY_FACTOR) || 2,
    },

    // Circuit Breaker
    circuitBreaker: {
        threshold: parseInt(process.env.CB_THRESHOLD) || 5,
        resetTimeout: parseInt(process.env.CB_RESET_TIMEOUT) || 30000,
    },

    // Rate Limiting
    rateLimit: {
        windowMs: parseInt(process.env.RATE_LIMIT_WINDOW) || 60000,
        maxRequests: parseInt(process.env.RATE_LIMIT_MAX) || 100,
    },

    // Clustering
    cluster: {
        enabled: process.env.CLUSTER_ENABLED === 'true',
        workers: parseInt(process.env.CLUSTER_WORKERS) || os.cpus().length,
    },
};

// ============================================================================
// Logging
// ============================================================================

const LogLevel = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };
const currentLogLevel = LogLevel[process.env.LOG_LEVEL?.toUpperCase()] ?? LogLevel.INFO;

function log(level, message, meta = {}) {
    if (LogLevel[level] < currentLogLevel) return;

    const entry = {
        timestamp: new Date().toISOString(),
        level,
        message,
        worker: cluster.worker?.id || 'master',
        pid: process.pid,
        ...meta,
    };

    console.log(JSON.stringify(entry));
}

// ============================================================================
// Metrics (Prometheus format)
// ============================================================================

const metrics = {
    requestsTotal: 0,
    requestsSuccess: 0,
    requestsFailed: 0,
    requestDurationSum: 0,
    requestDurationCount: 0,
    connectionPoolSize: 0,
    connectionPoolActive: 0,
    circuitBreakerState: 'closed',
    circuitBreakerFailures: 0,
};

function getMetricsText() {
    return `# HELP hana_bridge_requests_total Total number of requests
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

# HELP hana_bridge_circuit_breaker_state Circuit breaker state (0=closed, 1=open, 2=half-open)
# TYPE hana_bridge_circuit_breaker_state gauge
hana_bridge_circuit_breaker_state ${metrics.circuitBreakerState === 'closed' ? 0 : metrics.circuitBreakerState === 'open' ? 1 : 2}
`;
}

// ============================================================================
// Circuit Breaker
// ============================================================================

class CircuitBreaker {
    constructor(options) {
        this.threshold = options.threshold;
        this.resetTimeout = options.resetTimeout;
        this.state = 'closed';
        this.failures = 0;
        this.lastFailure = null;
        this.halfOpenRequests = 0;
    }

    async execute(fn) {
        if (this.state === 'open') {
            if (Date.now() - this.lastFailure > this.resetTimeout) {
                this.state = 'half-open';
                this.halfOpenRequests = 0;
                log('INFO', 'Circuit breaker half-open');
                metrics.circuitBreakerState = 'half-open';
            } else {
                throw new Error('Circuit breaker is open');
            }
        }

        if (this.state === 'half-open' && this.halfOpenRequests >= 1) {
            throw new Error('Circuit breaker is testing');
        }

        try {
            if (this.state === 'half-open') this.halfOpenRequests++;
            const result = await fn();
            this.onSuccess();
            return result;
        } catch (error) {
            this.onFailure();
            throw error;
        }
    }

    onSuccess() {
        this.failures = 0;
        if (this.state === 'half-open') {
            this.state = 'closed';
            log('INFO', 'Circuit breaker closed');
            metrics.circuitBreakerState = 'closed';
        }
        metrics.circuitBreakerFailures = this.failures;
    }

    onFailure() {
        this.failures++;
        this.lastFailure = Date.now();
        metrics.circuitBreakerFailures = this.failures;

        if (this.failures >= this.threshold) {
            this.state = 'open';
            log('WARN', 'Circuit breaker opened', { failures: this.failures });
            metrics.circuitBreakerState = 'open';
        }
    }
}

// ============================================================================
// Rate Limiter
// ============================================================================

class RateLimiter {
    constructor(options) {
        this.windowMs = options.windowMs;
        this.maxRequests = options.maxRequests;
        this.requests = new Map();
    }

    isAllowed(ip) {
        const now = Date.now();
        const windowStart = now - this.windowMs;

        // Clean old entries
        for (const [key, timestamps] of this.requests) {
            const valid = timestamps.filter(t => t > windowStart);
            if (valid.length === 0) {
                this.requests.delete(key);
            } else {
                this.requests.set(key, valid);
            }
        }

        const timestamps = this.requests.get(ip) || [];
        if (timestamps.length >= this.maxRequests) {
            return false;
        }

        timestamps.push(now);
        this.requests.set(ip, timestamps);
        return true;
    }
}

// ============================================================================
// Connection Pool
// ============================================================================

class ConnectionPool {
    constructor(hanaConfig, poolConfig) {
        this.hanaConfig = hanaConfig;
        this.poolConfig = poolConfig;
        this.connections = [];
        this.available = [];
        this.waiting = [];
        this.hana = null;
    }

    async init() {
        try {
            this.hana = require('@sap/hana-client');
        } catch (e) {
            log('ERROR', 'HANA client not installed', { error: e.message });
            throw e;
        }

        // Create minimum connections
        for (let i = 0; i < this.poolConfig.min; i++) {
            try {
                const conn = await this.createConnection();
                this.connections.push(conn);
                this.available.push(conn);
            } catch (e) {
                log('WARN', 'Failed to create initial connection', { error: e.message });
            }
        }

        metrics.connectionPoolSize = this.connections.length;
        log('INFO', 'Connection pool initialized', { size: this.connections.length });
    }

    createConnection() {
        return new Promise((resolve, reject) => {
            const conn = this.hana.createConnection();

            conn.connect({
                serverNode: `${this.hanaConfig.host}:${this.hanaConfig.port}`,
                uid: this.hanaConfig.user,
                pwd: this.hanaConfig.password,
                encrypt: this.hanaConfig.encrypt,
                sslValidateCertificate: this.hanaConfig.sslValidateCertificate,
            }, (err) => {
                if (err) {
                    reject(err);
                    return;
                }

                // Set default schema
                conn.exec(`SET SCHEMA ${this.hanaConfig.schema}`, (schemaErr) => {
                    if (schemaErr) {
                        log('WARN', 'Could not set default schema', { error: schemaErr.message });
                    }
                    resolve(conn);
                });
            });
        });
    }

    async acquire() {
        // Try to get available connection
        if (this.available.length > 0) {
            const conn = this.available.pop();
            metrics.connectionPoolActive = this.connections.length - this.available.length;
            return conn;
        }

        // Create new connection if pool not at max
        if (this.connections.length < this.poolConfig.max) {
            try {
                const conn = await this.createConnection();
                this.connections.push(conn);
                metrics.connectionPoolSize = this.connections.length;
                metrics.connectionPoolActive = this.connections.length - this.available.length;
                return conn;
            } catch (e) {
                log('ERROR', 'Failed to create connection', { error: e.message });
                throw e;
            }
        }

        // Wait for available connection
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                const idx = this.waiting.indexOf(waiter);
                if (idx > -1) this.waiting.splice(idx, 1);
                reject(new Error('Connection acquire timeout'));
            }, this.poolConfig.acquireTimeout);

            const waiter = { resolve, reject, timeout };
            this.waiting.push(waiter);
        });
    }

    release(conn) {
        // Check if anyone waiting
        if (this.waiting.length > 0) {
            const waiter = this.waiting.shift();
            clearTimeout(waiter.timeout);
            waiter.resolve(conn);
            return;
        }

        this.available.push(conn);
        metrics.connectionPoolActive = this.connections.length - this.available.length;
    }

    async destroy(conn) {
        const idx = this.connections.indexOf(conn);
        if (idx > -1) {
            this.connections.splice(idx, 1);
            metrics.connectionPoolSize = this.connections.length;
        }

        try {
            conn.disconnect();
        } catch (e) {
            // Ignore disconnect errors
        }
    }

    async shutdown() {
        for (const conn of this.connections) {
            try {
                conn.disconnect();
            } catch (e) {
                // Ignore
            }
        }
        this.connections = [];
        this.available = [];
        metrics.connectionPoolSize = 0;
        metrics.connectionPoolActive = 0;
    }
}

// ============================================================================
// SQL Executor with Retry
// ============================================================================

async function executeWithRetry(pool, circuitBreaker, sql, retryConfig) {
    let lastError;
    let delay = retryConfig.initialDelay;

    for (let attempt = 1; attempt <= retryConfig.maxAttempts; attempt++) {
        try {
            return await circuitBreaker.execute(async () => {
                const conn = await pool.acquire();
                try {
                    return await new Promise((resolve, reject) => {
                        conn.exec(sql, (err, result) => {
                            if (err) reject(err);
                            else resolve(result);
                        });
                    });
                } finally {
                    pool.release(conn);
                }
            });
        } catch (error) {
            lastError = error;

            // Don't retry circuit breaker errors
            if (error.message.includes('Circuit breaker')) {
                throw error;
            }

            if (attempt < retryConfig.maxAttempts) {
                log('WARN', 'SQL execution failed, retrying', {
                    attempt,
                    maxAttempts: retryConfig.maxAttempts,
                    delay,
                    error: error.message,
                });

                await new Promise(r => setTimeout(r, delay));
                delay = Math.min(delay * retryConfig.factor, retryConfig.maxDelay);
            }
        }
    }

    throw lastError;
}

// ============================================================================
// HTTP Server
// ============================================================================

function createServer(pool, circuitBreaker, rateLimiter) {
    return http.createServer(async (req, res) => {
        const startTime = Date.now();
        const clientIp = req.headers['x-forwarded-for']?.split(',')[0] ||
                         req.socket.remoteAddress || 'unknown';

        metrics.requestsTotal++;

        // CORS
        res.setHeader('Access-Control-Allow-Origin', process.env.CORS_ORIGIN || '*');
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

        if (req.method === 'OPTIONS') {
            res.writeHead(204);
            res.end();
            return;
        }

        // Rate limiting
        if (!rateLimiter.isAllowed(clientIp)) {
            res.writeHead(429, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Too many requests' }));
            metrics.requestsFailed++;
            return;
        }

        try {
            // Health check (liveness)
            if (req.url === '/health' || req.url === '/healthz') {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    status: 'ok',
                    uptime: process.uptime(),
                    timestamp: new Date().toISOString(),
                }));
                metrics.requestsSuccess++;
                return;
            }

            // Readiness check
            if (req.url === '/ready' || req.url === '/readyz') {
                const isReady = pool.connections.length > 0 &&
                               circuitBreaker.state !== 'open';
                res.writeHead(isReady ? 200 : 503, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    ready: isReady,
                    pool: pool.connections.length,
                    circuitBreaker: circuitBreaker.state,
                }));
                if (isReady) metrics.requestsSuccess++;
                else metrics.requestsFailed++;
                return;
            }

            // Metrics endpoint
            if (req.url === '/metrics') {
                res.writeHead(200, { 'Content-Type': 'text/plain' });
                res.end(getMetricsText());
                return;
            }

            // Test connection
            if (req.url === '/test' && req.method === 'GET') {
                const result = await executeWithRetry(
                    pool, circuitBreaker,
                    'SELECT 1 AS test FROM DUMMY',
                    config.retry
                );
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: true, result }));
                metrics.requestsSuccess++;
                return;
            }

            // SQL execution
            if (req.url === '/sql' && req.method === 'POST') {
                let body = '';

                await new Promise((resolve, reject) => {
                    req.on('data', chunk => {
                        body += chunk.toString();
                        if (body.length > 1e7) { // 10MB limit
                            reject(new Error('Request too large'));
                        }
                    });
                    req.on('end', resolve);
                    req.on('error', reject);
                });

                const { sql, schema } = JSON.parse(body);

                if (!sql || typeof sql !== 'string') {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Missing or invalid sql parameter' }));
                    metrics.requestsFailed++;
                    return;
                }

                // Schema switch if needed
                if (schema && schema !== config.hana.schema) {
                    await executeWithRetry(
                        pool, circuitBreaker,
                        `SET SCHEMA ${schema}`,
                        config.retry
                    );
                }

                const result = await executeWithRetry(
                    pool, circuitBreaker, sql, config.retry
                );

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    success: true,
                    result,
                    rowCount: Array.isArray(result) ? result.length : 0,
                }));

                metrics.requestsSuccess++;
                return;
            }

            // Not found
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Not found' }));

        } catch (error) {
            log('ERROR', 'Request failed', {
                url: req.url,
                method: req.method,
                error: error.message,
            });

            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                success: false,
                error: error.message,
            }));

            metrics.requestsFailed++;
        } finally {
            const duration = Date.now() - startTime;
            metrics.requestDurationSum += duration;
            metrics.requestDurationCount++;

            log('DEBUG', 'Request completed', {
                url: req.url,
                method: req.method,
                duration,
                clientIp,
            });
        }
    });
}

// ============================================================================
// Worker Process
// ============================================================================

async function startWorker() {
    const pool = new ConnectionPool(config.hana, config.pool);
    const circuitBreaker = new CircuitBreaker(config.circuitBreaker);
    const rateLimiter = new RateLimiter(config.rateLimit);

    await pool.init();

    const server = createServer(pool, circuitBreaker, rateLimiter);

    server.listen(config.port, config.host, () => {
        log('INFO', 'HANA Bridge worker started', {
            port: config.port,
            host: config.host,
            poolSize: pool.connections.length,
        });
    });

    // Graceful shutdown
    const shutdown = async (signal) => {
        log('INFO', 'Shutting down', { signal });

        server.close(() => {
            log('INFO', 'HTTP server closed');
        });

        await pool.shutdown();
        log('INFO', 'Connection pool closed');

        process.exit(0);
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
}

// ============================================================================
// Master Process (Clustering)
// ============================================================================

function startMaster() {
    log('INFO', 'Starting HANA Bridge cluster', {
        workers: config.cluster.workers,
    });

    for (let i = 0; i < config.cluster.workers; i++) {
        cluster.fork();
    }

    cluster.on('exit', (worker, code, signal) => {
        log('WARN', 'Worker died, restarting', {
            workerId: worker.id,
            code,
            signal,
        });
        cluster.fork();
    });

    // Graceful shutdown
    process.on('SIGTERM', () => {
        log('INFO', 'Master received SIGTERM');
        for (const id in cluster.workers) {
            cluster.workers[id].kill('SIGTERM');
        }
    });
}

// ============================================================================
// Main
// ============================================================================

if (config.cluster.enabled && cluster.isMaster) {
    startMaster();
} else {
    startWorker().catch(err => {
        log('ERROR', 'Failed to start worker', { error: err.message });
        process.exit(1);
    });
}
