#!/usr/bin/env node
/**
 * ==============================================================================
 * Upload Vectors to HANA Cloud
 * ==============================================================================
 * 
 * Generates embeddings using Ollama (nomic-embed-text) and uploads to HANA Cloud
 * vector store for RAG queries.
 * 
 * Usage:
 *   node upload_vectors_to_hana.js [--limit N] [--batch-size N]
 * 
 * Prerequisites:
 *   - Ollama running with nomic-embed-text model
 *   - HANA Cloud with TRIALBALANCE_VECTORS table created
 *   - @sap/hana-client installed
 * ==============================================================================
 */

const fs = require('fs');
const path = require('path');
const http = require('http');

// Load .env
const envPath = path.resolve(__dirname, '../../../../../.env');
if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf8');
    envContent.split('\n').forEach(line => {
        const match = line.match(/^([^=]+)=(.*)$/);
        if (match && !process.env[match[1]]) {
            process.env[match[1]] = match[2].replace(/^["']|["']$/g, '');
        }
    });
}

// Try to load HANA client
let hana;
try {
    hana = require('@sap/hana-client');
} catch (e) {
    console.error('❌ @sap/hana-client not found');
    console.log('   Run: npm install @sap/hana-client');
    process.exit(1);
}

// Configuration
const CONFIG = {
    ollama: {
        url: process.env.OLLAMA_URL || 'http://localhost:11434',
        model: 'nomic-embed-text',
        dimensions: 768
    },
    hana: {
        host: process.env.HANA_HOST,
        port: parseInt(process.env.HANA_PORT || '443'),
        user: process.env.HANA_USER,
        password: process.env.HANA_PASSWORD
    },
    documentsPath: path.resolve(__dirname, '../.vectorized/documents.json'),
    batchSize: parseInt(process.argv.find(a => a.startsWith('--batch-size='))?.split('=')[1] || '50'),
    limit: parseInt(process.argv.find(a => a.startsWith('--limit='))?.split('=')[1] || '0')
};

// Logging
const log = {
    info: (msg) => console.log(`ℹ️  ${msg}`),
    success: (msg) => console.log(`✅ ${msg}`),
    warn: (msg) => console.log(`⚠️  ${msg}`),
    error: (msg) => console.error(`❌ ${msg}`),
    progress: (current, total) => {
        process.stdout.write(`\r   Processing: ${current}/${total} (${Math.round(current/total*100)}%)`);
    }
};

// Get embedding from Ollama
function getEmbedding(text) {
    return new Promise((resolve, reject) => {
        const url = new URL(`${CONFIG.ollama.url}/api/embeddings`);
        const data = JSON.stringify({
            model: CONFIG.ollama.model,
            prompt: text.substring(0, 8000) // Truncate for model limit
        });

        const req = http.request({
            hostname: url.hostname,
            port: url.port || 11434,
            path: url.pathname,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(data)
            }
        }, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try {
                    const json = JSON.parse(body);
                    if (json.embedding) {
                        resolve(json.embedding);
                    } else {
                        reject(new Error('No embedding in response'));
                    }
                } catch (e) {
                    reject(e);
                }
            });
        });

        req.on('error', reject);
        req.setTimeout(30000, () => {
            req.destroy();
            reject(new Error('Embedding timeout'));
        });
        req.write(data);
        req.end();
    });
}

// Connect to HANA
function connectHana() {
    return new Promise((resolve, reject) => {
        const conn = hana.createConnection();
        const connOpts = {
            serverNode: `${CONFIG.hana.host}:${CONFIG.hana.port}`,
            uid: CONFIG.hana.user,
            pwd: CONFIG.hana.password,
            encrypt: 'true',
            sslValidateCertificate: 'false'
        };

        conn.connect(connOpts, (err) => {
            if (err) reject(err);
            else resolve(conn);
        });
    });
}

// Execute HANA query
function executeQuery(conn, sql, params = []) {
    return new Promise((resolve, reject) => {
        conn.exec(sql, params, (err, result) => {
            if (err) reject(err);
            else resolve(result);
        });
    });
}

// Create table if not exists
async function ensureTable(conn) {
    try {
        // Check if table exists
        const check = await executeQuery(conn, 
            `SELECT COUNT(*) as CNT FROM TABLES WHERE TABLE_NAME = 'TRIALBALANCE_VECTORS'`
        );
        
        if (check[0].CNT === 0) {
            log.info('Creating TRIALBALANCE_VECTORS table...');
            await executeQuery(conn, `
                CREATE COLUMN TABLE TRIALBALANCE_VECTORS (
                    ID NVARCHAR(256) PRIMARY KEY,
                    CONTENT NCLOB,
                    FILE_PATH NVARCHAR(500),
                    FILE_TYPE NVARCHAR(50),
                    MODULE NVARCHAR(100),
                    CHUNK_INDEX INTEGER,
                    TOTAL_CHUNKS INTEGER,
                    EMBEDDING REAL_VECTOR(768),
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            `);
            log.success('Table created');
        } else {
            // Check existing count
            const countResult = await executeQuery(conn, 
                `SELECT COUNT(*) as CNT FROM TRIALBALANCE_VECTORS`
            );
            log.info(`Table exists with ${countResult[0].CNT} vectors`);
        }
    } catch (err) {
        log.warn(`Table check/create error: ${err.message}`);
    }
}

// Insert vector into HANA
async function insertVector(conn, doc, embedding) {
    const sql = `
        UPSERT TRIALBALANCE_VECTORS (ID, CONTENT, FILE_PATH, FILE_TYPE, MODULE, CHUNK_INDEX, TOTAL_CHUNKS, EMBEDDING)
        VALUES (?, ?, ?, ?, ?, ?, ?, TO_REAL_VECTOR(?))
        WITH PRIMARY KEY
    `;
    
    const embeddingStr = `[${embedding.join(',')}]`;
    
    await executeQuery(conn, sql, [
        doc.id,
        doc.content,
        doc.metadata.file_path,
        doc.metadata.file_type,
        doc.metadata.module,
        doc.metadata.chunk_index,
        doc.metadata.total_chunks,
        embeddingStr
    ]);
}

// Main
async function main() {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════════╗');
    console.log('║          Upload Vectors to HANA Cloud                            ║');
    console.log('╚══════════════════════════════════════════════════════════════════╝');
    console.log('');

    // Check Ollama
    log.info('Checking Ollama...');
    try {
        await getEmbedding('test');
        log.success('Ollama nomic-embed-text ready');
    } catch (err) {
        log.error(`Ollama not ready: ${err.message}`);
        log.info('Run: ollama serve && ollama pull nomic-embed-text');
        process.exit(1);
    }

    // Load documents
    log.info(`Loading documents from ${CONFIG.documentsPath}...`);
    if (!fs.existsSync(CONFIG.documentsPath)) {
        log.error('Documents file not found. Run vectorize_with_aicore.js first.');
        process.exit(1);
    }
    
    let documents = JSON.parse(fs.readFileSync(CONFIG.documentsPath, 'utf8'));
    log.success(`Loaded ${documents.length} documents`);

    if (CONFIG.limit > 0) {
        documents = documents.slice(0, CONFIG.limit);
        log.info(`Limited to ${documents.length} documents`);
    }

    // Connect to HANA
    log.info('Connecting to HANA Cloud...');
    let conn;
    try {
        conn = await connectHana();
        log.success('Connected to HANA Cloud');
    } catch (err) {
        log.error(`HANA connection failed: ${err.message}`);
        process.exit(1);
    }

    // Ensure table exists
    await ensureTable(conn);

    // Process documents
    log.info('Generating embeddings and uploading to HANA...');
    console.log('');
    
    let successCount = 0;
    let errorCount = 0;
    const startTime = Date.now();

    for (let i = 0; i < documents.length; i++) {
        const doc = documents[i];
        log.progress(i + 1, documents.length);

        try {
            // Get embedding from Ollama
            const embedding = await getEmbedding(doc.content);
            
            // Insert into HANA
            await insertVector(conn, doc, embedding);
            successCount++;
        } catch (err) {
            errorCount++;
            // Don't log individual errors to keep progress clean
        }
    }

    console.log('');
    console.log('');

    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    
    // Final count
    const finalCount = await executeQuery(conn, 
        `SELECT COUNT(*) as CNT FROM TRIALBALANCE_VECTORS`
    );

    // Summary
    console.log('═══ Summary ═══');
    console.log('');
    console.log(`  Documents processed: ${documents.length}`);
    console.log(`  Successfully uploaded: ${successCount}`);
    console.log(`  Errors: ${errorCount}`);
    console.log(`  Total in HANA: ${finalCount[0].CNT}`);
    console.log(`  Duration: ${duration}s`);
    console.log('');

    // Cleanup
    conn.disconnect();

    console.log('╔══════════════════════════════════════════════════════════════════╗');
    console.log('║                    Upload Complete                               ║');
    console.log('╚══════════════════════════════════════════════════════════════════╝');
    console.log('');
}

main().catch(err => {
    log.error(`Fatal: ${err.message}`);
    process.exit(1);
});