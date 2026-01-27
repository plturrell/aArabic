#!/usr/bin/env node
/**
 * ==============================================================================
 * RAG Query Service - HANA Vectors + AI Core (Claude 4.5 Opus)
 * ==============================================================================
 * 
 * Performs semantic search in HANA Cloud vector store and generates
 * responses using SAP AI Core with Claude 4.5 Opus.
 * 
 * Usage:
 *   node rag_query_service.js "your question here"
 *   node rag_query_service.js --interactive
 *   node rag_query_service.js --server --port 8095
 * 
 * ==============================================================================
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const readline = require('readline');

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
    process.exit(1);
}

// Configuration
const CONFIG = {
    ollama: {
        url: process.env.OLLAMA_URL || 'http://localhost:11434',
        model: 'nomic-embed-text'
    },
    hana: {
        host: process.env.HANA_HOST,
        port: parseInt(process.env.HANA_PORT || '443'),
        user: process.env.HANA_USER,
        password: process.env.HANA_PASSWORD
    },
    aicore: {
        clientId: process.env.AICORE_CLIENT_ID,
        clientSecret: process.env.AICORE_CLIENT_SECRET,
        authUrl: process.env.AICORE_AUTH_URL,
        baseUrl: process.env.AICORE_BASE_URL,
        resourceGroup: process.env.AICORE_RESOURCE_GROUP || 'default',
        model: 'anthropic--claude-4-5-opus',
        deploymentId: process.env.AICORE_DEPLOYMENT_ID || 'd3e75e2d0fc86ceb'
    },
    rag: {
        topK: 5,
        maxContextLength: 8000
    }
};

// Logging
const log = {
    info: (msg) => console.log(`ℹ️  ${msg}`),
    success: (msg) => console.log(`✅ ${msg}`),
    warn: (msg) => console.log(`⚠️  ${msg}`),
    error: (msg) => console.error(`❌ ${msg}`)
};

// HTTP helper
function httpRequest(url, options, data = null) {
    return new Promise((resolve, reject) => {
        const urlObj = new URL(url);
        const isHttps = urlObj.protocol === 'https:';
        const lib = isHttps ? https : http;
        
        const reqOptions = {
            hostname: urlObj.hostname,
            port: urlObj.port || (isHttps ? 443 : 80),
            path: urlObj.pathname + urlObj.search,
            method: options.method || 'GET',
            headers: options.headers || {}
        };

        const req = lib.request(reqOptions, (res) => {
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
        req.setTimeout(60000, () => {
            req.destroy();
            reject(new Error('Request timeout'));
        });

        if (data) req.write(data);
        req.end();
    });
}

// Get embedding from Ollama
async function getEmbedding(text) {
    const response = await httpRequest(`${CONFIG.ollama.url}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    }, JSON.stringify({
        model: CONFIG.ollama.model,
        prompt: text.substring(0, 8000)
    }));

    if (response.data.embedding) {
        return response.data.embedding;
    }
    throw new Error('Failed to get embedding');
}

// Connect to HANA
function connectHana() {
    return new Promise((resolve, reject) => {
        const conn = hana.createConnection();
        conn.connect({
            serverNode: `${CONFIG.hana.host}:${CONFIG.hana.port}`,
            uid: CONFIG.hana.user,
            pwd: CONFIG.hana.password,
            encrypt: 'true',
            sslValidateCertificate: 'false'
        }, (err) => {
            if (err) reject(err);
            else resolve(conn);
        });
    });
}

// Execute query
function executeQuery(conn, sql, params = []) {
    return new Promise((resolve, reject) => {
        conn.exec(sql, params, (err, result) => {
            if (err) reject(err);
            else resolve(result);
        });
    });
}

// Search similar vectors in HANA
async function searchVectors(conn, queryEmbedding, topK = 5, moduleFilter = null) {
    const embeddingStr = `[${queryEmbedding.join(',')}]`;
    
    let sql = `
        SELECT TOP ${topK}
            ID,
            FILE_PATH,
            FILE_TYPE,
            MODULE,
            CHUNK_INDEX,
            CONTENT,
            COSINE_SIMILARITY(EMBEDDING, TO_REAL_VECTOR(?)) as SIMILARITY
        FROM TRIALBALANCE_VECTORS
    `;
    
    if (moduleFilter) {
        sql += ` WHERE MODULE = '${moduleFilter}'`;
    }
    
    sql += ` ORDER BY COSINE_SIMILARITY(EMBEDDING, TO_REAL_VECTOR(?)) DESC`;
    
    return executeQuery(conn, sql, [embeddingStr, embeddingStr]);
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

// Chat with Claude via AI Core (with Ollama fallback)
async function chatWithClaude(token, messages, systemPrompt = null) {
    // Try AI Core first
    if (token && CONFIG.aicore.deploymentId) {
        try {
            const url = `${CONFIG.aicore.baseUrl}/v2/inference/deployments/${CONFIG.aicore.deploymentId}/chat/completions`;
            
            const payload = {
                model: CONFIG.aicore.model,
                messages: messages,
                max_tokens: 2000,
                temperature: 0.3
            };

            if (systemPrompt) {
                payload.messages = [
                    { role: 'system', content: systemPrompt },
                    ...messages
                ];
            }

            const response = await httpRequest(url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                    'AI-Resource-Group': CONFIG.aicore.resourceGroup
                }
            }, JSON.stringify(payload));

            if (response.status === 200) {
                return response.data.choices[0].message.content;
            }
            log.warn('AI Core not available, using Ollama fallback');
        } catch (err) {
            log.warn(`AI Core error: ${err.message}, using Ollama fallback`);
        }
    }

    // Fallback to Ollama (glm4:9b or other available model)
    return chatWithOllama(messages, systemPrompt);
}

// Chat with Ollama (local model)
async function chatWithOllama(messages, systemPrompt = null) {
    const model = 'glm4:9b'; // Available local model
    
    let prompt = '';
    if (systemPrompt) {
        prompt += `System: ${systemPrompt}\n\n`;
    }
    for (const msg of messages) {
        prompt += `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}\n\n`;
    }
    prompt += 'Assistant: ';

    const response = await httpRequest(`${CONFIG.ollama.url}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    }, JSON.stringify({
        model: model,
        prompt: prompt,
        stream: false,
        options: {
            temperature: 0.3,
            num_predict: 2000
        }
    }));

    if (response.data.response) {
        return response.data.response;
    }
    throw new Error('Ollama chat failed');
}

// RAG Query
async function ragQuery(question, conn, token, options = {}) {
    const topK = options.topK || CONFIG.rag.topK;
    const moduleFilter = options.module || null;

    // Step 1: Get embedding for question
    log.info('Generating question embedding...');
    const questionEmbedding = await getEmbedding(question);

    // Step 2: Search HANA vector store
    log.info(`Searching HANA for top ${topK} similar documents...`);
    const results = await searchVectors(conn, questionEmbedding, topK, moduleFilter);
    
    if (results.length === 0) {
        return {
            answer: 'No relevant documents found in the knowledge base.',
            sources: []
        };
    }

    log.info(`Found ${results.length} relevant documents`);

    // Step 3: Build context
    const context = results.map((r, i) => {
        return `[Source ${i + 1}: ${r.FILE_PATH} (${r.MODULE})]
${r.CONTENT}
---`;
    }).join('\n\n');

    // Truncate if too long
    const truncatedContext = context.substring(0, CONFIG.rag.maxContextLength);

    // Step 4: Build prompt
    const systemPrompt = `You are an expert software engineer assistant helping with the TrialBalance application.
You have access to the codebase including:
- Backend (Zig): Balance calculation engines, HANA integration, API handlers
- Frontend (UI5/JS): Controllers, views, services
- ODPS: Data product specifications and validation rules
- Configuration: App settings, HANA config, AI Core config

Answer questions based ONLY on the provided context. If the answer isn't in the context, say so.
Be specific and cite file paths when relevant.`;

    const userPrompt = `Context from the codebase:

${truncatedContext}

Question: ${question}

Please provide a detailed answer based on the context above.`;

    // Step 5: Get answer from Claude
    log.info('Asking Claude 4.5 Opus via AI Core...');
    const answer = await chatWithClaude(token, [
        { role: 'user', content: userPrompt }
    ], systemPrompt);

    return {
        answer: answer,
        sources: results.map(r => ({
            file: r.FILE_PATH,
            module: r.MODULE,
            similarity: r.SIMILARITY,
            preview: r.CONTENT.substring(0, 200) + '...'
        }))
    };
}

// Interactive mode
async function interactiveMode() {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════════╗');
    console.log('║       TrialBalance RAG - Interactive Mode                         ║');
    console.log('║       (HANA Vectors + Claude 4.5 Opus)                           ║');
    console.log('╚══════════════════════════════════════════════════════════════════╝');
    console.log('');
    console.log('Commands:');
    console.log('  /quit - Exit');
    console.log('  /module <name> - Filter by module (backend, controller, odps, etc)');
    console.log('  /clear - Clear module filter');
    console.log('');

    // Initialize
    log.info('Connecting to HANA Cloud...');
    const conn = await connectHana();
    log.success('Connected to HANA');

    log.info('Getting AI Core token...');
    const token = await getAICoreToken();
    log.success('AI Core ready');
    
    console.log('');
    console.log('Ask questions about the TrialBalance codebase:');
    console.log('');

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    let moduleFilter = null;

    const askQuestion = () => {
        const prompt = moduleFilter ? `[${moduleFilter}] > ` : '> ';
        rl.question(prompt, async (input) => {
            const trimmed = input.trim();
            
            if (trimmed === '/quit' || trimmed === '/exit') {
                console.log('\nGoodbye! 👋');
                conn.disconnect();
                rl.close();
                process.exit(0);
            }
            
            if (trimmed.startsWith('/module ')) {
                moduleFilter = trimmed.substring(8).trim();
                console.log(`Filter set to: ${moduleFilter}`);
                askQuestion();
                return;
            }
            
            if (trimmed === '/clear') {
                moduleFilter = null;
                console.log('Filter cleared');
                askQuestion();
                return;
            }

            if (!trimmed) {
                askQuestion();
                return;
            }

            try {
                const result = await ragQuery(trimmed, conn, token, { module: moduleFilter });
                
                console.log('\n' + '─'.repeat(70));
                console.log('\n📝 Answer:\n');
                console.log(result.answer);
                console.log('\n' + '─'.repeat(70));
                console.log('\n📚 Sources:');
                result.sources.forEach((s, i) => {
                    console.log(`  ${i + 1}. ${s.file} (${s.module}) - ${(s.similarity * 100).toFixed(1)}%`);
                });
                console.log('');
            } catch (err) {
                log.error(`Query failed: ${err.message}`);
            }
            
            askQuestion();
        });
    };

    askQuestion();
}

// Single query mode
async function singleQuery(question) {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════════╗');
    console.log('║       TrialBalance RAG Query                                      ║');
    console.log('╚══════════════════════════════════════════════════════════════════╝');
    console.log('');

    log.info('Connecting to HANA Cloud...');
    const conn = await connectHana();
    log.success('Connected');

    log.info('Getting AI Core token...');
    const token = await getAICoreToken();
    log.success('AI Core ready');

    console.log('');
    console.log(`Question: ${question}`);
    console.log('');

    const result = await ragQuery(question, conn, token);

    console.log('─'.repeat(70));
    console.log('\n📝 Answer:\n');
    console.log(result.answer);
    console.log('\n' + '─'.repeat(70));
    console.log('\n📚 Sources:');
    result.sources.forEach((s, i) => {
        console.log(`  ${i + 1}. ${s.file} (${s.module}) - ${(s.similarity * 100).toFixed(1)}%`);
    });
    console.log('');

    conn.disconnect();
}

// Main
async function main() {
    const args = process.argv.slice(2);

    if (args.includes('--interactive') || args.includes('-i')) {
        await interactiveMode();
    } else if (args.length > 0 && !args[0].startsWith('--')) {
        await singleQuery(args.join(' '));
    } else {
        console.log('Usage:');
        console.log('  node rag_query_service.js "your question"');
        console.log('  node rag_query_service.js --interactive');
        process.exit(0);
    }
}

main().catch(err => {
    log.error(`Fatal: ${err.message}`);
    process.exit(1);
});