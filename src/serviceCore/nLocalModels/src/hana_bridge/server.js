/**
 * HANA Cloud SQL Bridge
 *
 * Accepts HTTP requests and executes SQL on HANA Cloud via native driver.
 * This is needed because HANA Cloud database hosts only accept the native
 * HANA SQL protocol (binary over TLS), not HTTP/REST.
 *
 * Usage:
 *   npm install
 *   npm start
 *
 * Then POST to http://localhost:3001/sql with JSON body:
 *   { "sql": "SELECT * FROM DUMMY" }
 */

const http = require('http');

// Try to load HANA client
let hana;
try {
    hana = require('@sap/hana-client');
} catch (e) {
    console.error('❌ @sap/hana-client not installed!');
    console.error('   Run: npm install @sap/hana-client');
    console.error('   Or:  npm install');
    process.exit(1);
}

// Configuration from environment
const config = {
    host: process.env.HANA_HOST || 'd93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com',
    port: parseInt(process.env.HANA_PORT) || 443,
    user: process.env.HANA_USER || 'DBADMIN',
    password: process.env.HANA_PASSWORD || 'Initial@1',
    schema: process.env.HANA_SCHEMA || 'NUCLEUS',
    encrypt: true,
    sslValidateCertificate: false
};

const BRIDGE_PORT = parseInt(process.env.BRIDGE_PORT) || 3001;

// Connection pool
let connection = null;

async function getConnection() {
    if (connection) {
        return connection;
    }

    return new Promise((resolve, reject) => {
        const conn = hana.createConnection();

        conn.connect({
            serverNode: `${config.host}:${config.port}`,
            uid: config.user,
            pwd: config.password,
            encrypt: config.encrypt,
            sslValidateCertificate: config.sslValidateCertificate
        }, (err) => {
            if (err) {
                console.error('❌ HANA connection failed:', err.message);
                reject(err);
                return;
            }

            console.log('✅ Connected to HANA Cloud');
            connection = conn;

            // Set default schema
            conn.exec(`SET SCHEMA ${config.schema}`, (err) => {
                if (err) {
                    console.warn('⚠️  Could not set schema:', err.message);
                }
                resolve(conn);
            });
        });
    });
}

async function executeSQL(sql) {
    const conn = await getConnection();

    return new Promise((resolve, reject) => {
        conn.exec(sql, (err, result) => {
            if (err) {
                reject(err);
                return;
            }
            resolve(result);
        });
    });
}

// HTTP Server
const server = http.createServer(async (req, res) => {
    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
    }

    // Health check
    if (req.url === '/health' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok', connected: connection !== null }));
        return;
    }

    // SQL execution endpoint
    if (req.url === '/sql' && req.method === 'POST') {
        let body = '';

        req.on('data', chunk => {
            body += chunk.toString();
        });

        req.on('end', async () => {
            try {
                const { sql, schema } = JSON.parse(body);

                if (!sql) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Missing sql parameter' }));
                    return;
                }

                // Optional schema override
                let fullSql = sql;
                if (schema && schema !== config.schema) {
                    fullSql = `SET SCHEMA ${schema}; ${sql}`;
                }

                console.log(`📝 Executing: ${sql.substring(0, 100)}...`);

                const result = await executeSQL(fullSql);

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    success: true,
                    result: result,
                    rowCount: Array.isArray(result) ? result.length : 0
                }));

            } catch (err) {
                console.error('❌ SQL Error:', err.message);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    success: false,
                    error: err.message,
                    code: err.code
                }));
            }
        });

        return;
    }

    // Test connection endpoint
    if (req.url === '/test' && req.method === 'GET') {
        try {
            const result = await executeSQL('SELECT 1 AS test FROM DUMMY');
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                success: true,
                message: 'HANA Cloud connection working',
                result: result
            }));
        } catch (err) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                success: false,
                error: err.message
            }));
        }
        return;
    }

    // Default: not found
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
});

// Start server
server.listen(BRIDGE_PORT, () => {
    console.log('========================================');
    console.log('📡 HANA Cloud SQL Bridge');
    console.log('========================================');
    console.log(`Bridge URL: http://localhost:${BRIDGE_PORT}`);
    console.log(`HANA Host:  ${config.host}`);
    console.log(`User:       ${config.user}`);
    console.log(`Schema:     ${config.schema}`);
    console.log('');
    console.log('Endpoints:');
    console.log('  GET  /health  - Health check');
    console.log('  GET  /test    - Test HANA connection');
    console.log('  POST /sql     - Execute SQL { "sql": "..." }');
    console.log('========================================');
    console.log('');

    // Test connection on startup
    getConnection()
        .then(() => console.log('🚀 Ready to accept requests'))
        .catch(err => console.error('⚠️  Initial connection failed:', err.message));
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down...');
    if (connection) {
        connection.disconnect();
    }
    process.exit(0);
});
