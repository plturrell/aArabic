#!/usr/bin/env node
/**
 * ==============================================================================
 * TrialBalance Codebase Vectorization via SAP AI Core
 * ==============================================================================
 * 
 * Uses SAP AI Core with text-embedding-3-small or similar to vectorize
 * the TrialBalance codebase and ODPS files.
 * 
 * Usage:
 *   node vectorize_with_aicore.js
 * 
 * Environment Variables:
 *   AICORE_CLIENT_ID, AICORE_CLIENT_SECRET, AICORE_AUTH_URL, AICORE_BASE_URL
 * ==============================================================================
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { URL } = require('url');

// Load .env from root
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

// Configuration
const CONFIG = {
    aicore: {
        clientId: process.env.AICORE_CLIENT_ID,
        clientSecret: process.env.AICORE_CLIENT_SECRET,
        authUrl: process.env.AICORE_AUTH_URL,
        baseUrl: process.env.AICORE_BASE_URL,
        resourceGroup: process.env.AICORE_RESOURCE_GROUP || 'default'
    },
    paths: {
        trialBalance: path.resolve(__dirname, '..'),
        output: path.resolve(__dirname, '../.vectorized')
    },
    embedding: {
        model: 'text-embedding-3-small',
        chunkSize: 1500,
        chunkOverlap: 200
    }
};

// Logging
const log = {
    info: (msg) => console.log(`ℹ️  ${msg}`),
    success: (msg) => console.log(`✅ ${msg}`),
    warn: (msg) => console.log(`⚠️  ${msg}`),
    error: (msg) => console.error(`❌ ${msg}`),
    section: (title) => {
        console.log('');
        console.log(`═══ ${title} ═══`);
        console.log('');
    }
};

// HTTP helper
function httpRequest(url, options, data = null) {
    return new Promise((resolve, reject) => {
        const urlObj = new URL(url);
        const reqOptions = {
            hostname: urlObj.hostname,
            port: urlObj.port || 443,
            path: urlObj.pathname + urlObj.search,
            method: options.method || 'GET',
            headers: options.headers || {}
        };

        const req = https.request(reqOptions, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try {
                    resolve({ status: res.statusCode, data: JSON.parse(body) });
                } catch {
                    resolve({ status: res.statusCode, data: body });
                }
            });
        });

        req.on('error', reject);
        req.setTimeout(30000, () => {
            req.destroy();
            reject(new Error('Request timeout'));
        });

        if (data) req.write(data);
        req.end();
    });
}

// Get AI Core OAuth token
async function getAICoreToken() {
    const credentials = Buffer.from(
        `${CONFIG.aicore.clientId}:${CONFIG.aicore.clientSecret}`
    ).toString('base64');

    const response = await httpRequest(
        `${CONFIG.aicore.authUrl}?grant_type=client_credentials`,
        {
            method: 'POST',
            headers: {
                'Authorization': `Basic ${credentials}`,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        },
        'grant_type=client_credentials'
    );

    if (response.status !== 200) {
        throw new Error(`Auth failed: ${JSON.stringify(response.data)}`);
    }

    return response.data.access_token;
}

// Generate embeddings via AI Core
async function generateEmbeddings(token, texts) {
    // AI Core embedding endpoint
    const url = `${CONFIG.aicore.baseUrl}/v2/inference/deployments/default/embeddings`;
    
    const response = await httpRequest(url, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            'AI-Resource-Group': CONFIG.aicore.resourceGroup
        }
    }, JSON.stringify({
        input: texts,
        model: CONFIG.embedding.model
    }));

    if (response.status !== 200) {
        throw new Error(`Embedding failed: ${JSON.stringify(response.data)}`);
    }

    return response.data.data.map(d => d.embedding);
}

// Chunk text
function chunkText(text, maxLength = CONFIG.embedding.chunkSize, overlap = CONFIG.embedding.chunkOverlap) {
    const chunks = [];
    let start = 0;
    
    while (start < text.length) {
        const end = Math.min(start + maxLength, text.length);
        chunks.push(text.slice(start, end));
        start += maxLength - overlap;
    }
    
    return chunks;
}

// Get file type/language
function getFileType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const typeMap = {
        '.zig': 'zig',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.xml': 'xml',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.html': 'html',
        '.css': 'css'
    };
    return typeMap[ext] || 'text';
}

// Collect files to vectorize
function collectFiles(rootDir) {
    const files = [];
    const extensions = ['.zig', '.js', '.ts', '.xml', '.json', '.yaml', '.yml', '.md'];
    const excludeDirs = ['node_modules', '.git', 'zig-out', 'zig-cache', '.zig-cache'];

    function walk(dir) {
        if (!fs.existsSync(dir)) return;
        
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        
        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            
            if (entry.isDirectory()) {
                if (!excludeDirs.includes(entry.name)) {
                    walk(fullPath);
                }
            } else if (entry.isFile()) {
                const ext = path.extname(entry.name).toLowerCase();
                if (extensions.includes(ext)) {
                    files.push(fullPath);
                }
            }
        }
    }

    walk(rootDir);
    return files;
}

// Process a single file
function processFile(filePath, rootDir) {
    const content = fs.readFileSync(filePath, 'utf8');
    const relativePath = path.relative(rootDir, filePath);
    const fileType = getFileType(filePath);
    
    // Determine module based on path
    let module = 'other';
    if (relativePath.startsWith('backend/')) module = 'backend';
    else if (relativePath.startsWith('webapp/controller/')) module = 'controller';
    else if (relativePath.startsWith('webapp/view/')) module = 'view';
    else if (relativePath.startsWith('webapp/service/')) module = 'service';
    else if (relativePath.includes('/odps/')) module = 'odps';
    else if (relativePath.startsWith('BusDocs/')) module = 'busdocs';
    else if (relativePath.startsWith('config/')) module = 'config';

    // Chunk the content
    const chunks = chunkText(content);
    
    return chunks.map((chunk, index) => ({
        id: `${relativePath.replace(/[/\\]/g, '_')}_${index}`,
        content: chunk,
        metadata: {
            file_path: relativePath,
            file_type: fileType,
            module: module,
            chunk_index: index,
            total_chunks: chunks.length
        }
    }));
}

// Main
async function main() {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════════╗');
    console.log('║     TrialBalance Codebase Vectorization (SAP AI Core)            ║');
    console.log('╚══════════════════════════════════════════════════════════════════╝');
    console.log('');

    // Check config
    if (!CONFIG.aicore.clientId || !CONFIG.aicore.authUrl) {
        log.error('Missing AI Core credentials in .env');
        log.info('Required: AICORE_CLIENT_ID, AICORE_CLIENT_SECRET, AICORE_AUTH_URL, AICORE_BASE_URL');
        process.exit(1);
    }

    log.section('Configuration');
    log.info(`Root: ${CONFIG.paths.trialBalance}`);
    log.info(`Output: ${CONFIG.paths.output}`);
    log.info(`Model: ${CONFIG.embedding.model}`);

    // Collect files
    log.section('Collecting Files');
    const files = collectFiles(CONFIG.paths.trialBalance);
    log.success(`Found ${files.length} files to process`);

    // Process files into documents
    log.section('Processing Documents');
    const documents = [];
    for (const file of files) {
        try {
            const docs = processFile(file, CONFIG.paths.trialBalance);
            documents.push(...docs);
        } catch (err) {
            log.warn(`Failed to process ${file}: ${err.message}`);
        }
    }
    log.success(`Created ${documents.length} document chunks`);

    // Ensure output directory exists
    if (!fs.existsSync(CONFIG.paths.output)) {
        fs.mkdirSync(CONFIG.paths.output, { recursive: true });
    }

    // Save documents without embeddings (for later processing)
    const documentsPath = path.join(CONFIG.paths.output, 'documents.json');
    fs.writeFileSync(documentsPath, JSON.stringify(documents, null, 2));
    log.success(`Saved documents to ${documentsPath}`);

    // Try to get embeddings from AI Core
    log.section('Generating Embeddings');
    try {
        log.info('Getting AI Core OAuth token...');
        const token = await getAICoreToken();
        log.success('OAuth token obtained');

        // Process in batches
        const batchSize = 10;
        const embeddedDocs = [];
        
        for (let i = 0; i < documents.length; i += batchSize) {
            const batch = documents.slice(i, i + batchSize);
            const texts = batch.map(d => d.content.substring(0, 8000)); // Truncate for API
            
            try {
                const embeddings = await generateEmbeddings(token, texts);
                
                for (let j = 0; j < batch.length; j++) {
                    embeddedDocs.push({
                        ...batch[j],
                        embedding: embeddings[j]
                    });
                }
                
                process.stdout.write(`\r  Processed ${Math.min(i + batchSize, documents.length)}/${documents.length} documents`);
            } catch (err) {
                log.warn(`Batch ${i}-${i + batchSize} failed: ${err.message}`);
                // Save without embedding
                embeddedDocs.push(...batch);
            }
        }
        console.log('');

        // Save embedded documents
        const embeddedPath = path.join(CONFIG.paths.output, 'embedded_documents.json');
        fs.writeFileSync(embeddedPath, JSON.stringify(embeddedDocs, null, 2));
        log.success(`Saved ${embeddedDocs.length} embedded documents`);

    } catch (err) {
        log.warn(`Could not generate embeddings: ${err.message}`);
        log.info('Documents saved without embeddings - can be embedded later');
    }

    // Summary
    log.section('Summary');
    console.log('');
    console.log(`  Files processed:    ${files.length}`);
    console.log(`  Document chunks:    ${documents.length}`);
    console.log(`  Output directory:   ${CONFIG.paths.output}`);
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════════╗');
    console.log('║                    Vectorization Complete                        ║');
    console.log('╚══════════════════════════════════════════════════════════════════╝');
    console.log('');
}

main().catch(err => {
    log.error(`Fatal error: ${err.message}`);
    process.exit(1);
});