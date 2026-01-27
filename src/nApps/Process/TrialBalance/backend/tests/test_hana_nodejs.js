#!/usr/bin/env node
/**
 * ==============================================================================
 * SAP HANA Cloud Connection Test (Node.js)
 * Uses @sap/hana-client for native HANA protocol
 * ==============================================================================
 *
 * Usage: 
 *   npm install @sap/hana-client
 *   node test_hana_nodejs.js
 *
 * Or install globally:
 *   npm install -g @sap/hana-client
 */

const path = require('path');

// Load environment variables from .env
require('dotenv').config({ path: path.resolve(__dirname, '../../../../../.env') });

// Check if @sap/hana-client is installed
let hana;
try {
    hana = require('@sap/hana-client');
} catch (e) {
    console.log('╔══════════════════════════════════════════════════════════════════╗');
    console.log('║  @sap/hana-client not installed                                   ║');
    console.log('╚══════════════════════════════════════════════════════════════════╝');
    console.log('\nInstalling @sap/hana-client...\n');
    
    const { execSync } = require('child_process');
    try {
        execSync('npm install @sap/hana-client', { stdio: 'inherit' });
        hana = require('@sap/hana-client');
    } catch (installErr) {
        console.error('Failed to install @sap/hana-client:', installErr.message);
        console.log('\nPlease install manually: npm install @sap/hana-client');
        process.exit(1);
    }
}

console.log('');
console.log('╔══════════════════════════════════════════════════════════════════╗');
console.log('║       SAP HANA Cloud Connection Test (Node.js)                   ║');
console.log('╚══════════════════════════════════════════════════════════════════╝');
console.log('');

// Configuration from environment
const config = {
    host: process.env.HANA_HOST,
    port: parseInt(process.env.HANA_PORT || '443'),
    user: process.env.HANA_USER,
    password: process.env.HANA_PASSWORD,
    schema: process.env.HANA_SCHEMA || 'SAPABAP1',
    encrypt: process.env.HANA_ENCRYPT === 'true',
    sslValidateCertificate: process.env.HANA_VALIDATE_CERTIFICATE === 'true'
};

console.log('═══ Configuration ═══');
console.log(`Host: ${config.host}`);
console.log(`Port: ${config.port}`);
console.log(`User: ${config.user}`);
console.log(`Schema: ${config.schema}`);
console.log(`Encrypt: ${config.encrypt}`);
console.log('');

if (!config.host || !config.user || !config.password) {
    console.error('❌ Missing required environment variables:');
    console.error('   HANA_HOST, HANA_USER, HANA_PASSWORD');
    process.exit(1);
}

// Connection options
const connOptions = {
    serverNode: `${config.host}:${config.port}`,
    uid: config.user,
    pwd: config.password,
    encrypt: config.encrypt ? 'true' : 'false',
    sslValidateCertificate: config.sslValidateCertificate ? 'true' : 'false'
};

console.log('═══ Test 1: Connect to HANA ═══');
console.log('');

const conn = hana.createConnection();

conn.connect(connOptions, (err) => {
    if (err) {
        console.error('❌ Connection failed:', err.message);
        console.error('   Error code:', err.code);
        console.log('');
        console.log('Possible issues:');
        console.log('1. Database might be stopped (wake it up in BTP Cockpit)');
        console.log('2. IP might not be whitelisted');
        console.log('3. Credentials might be incorrect');
        console.log('4. SSL certificate issue - try sslValidateCertificate: false');
        process.exit(1);
    }
    
    console.log('✅ Connected to HANA Cloud!');
    console.log('');
    
    // Test 2: Simple query
    console.log('═══ Test 2: Execute Query ═══');
    console.log('SQL: SELECT 1 AS TEST, CURRENT_USER AS USERNAME FROM DUMMY');
    console.log('');
    
    conn.exec('SELECT 1 AS TEST, CURRENT_USER AS USERNAME FROM DUMMY', (err, result) => {
        if (err) {
            console.error('❌ Query failed:', err.message);
            conn.disconnect();
            process.exit(1);
        }
        
        console.log('✅ Query successful!');
        console.log('Result:', JSON.stringify(result, null, 2));
        console.log('');
        
        // Test 3: Get database info
        console.log('═══ Test 3: Database Info ═══');
        
        conn.exec(`SELECT 
            M_DATABASE.DATABASE_NAME,
            M_DATABASE.VERSION,
            M_DATABASE.USAGE
        FROM M_DATABASE`, (err, dbInfo) => {
            if (err) {
                console.log('⚠️  M_DATABASE query failed (might need privileges):', err.message);
            } else {
                console.log('Database Info:', JSON.stringify(dbInfo, null, 2));
            }
            console.log('');
            
            // Test 4: List tables
            console.log('═══ Test 4: List Available Schemas ═══');
            
            conn.exec(`SELECT SCHEMA_NAME, OWNER 
                       FROM SCHEMAS 
                       WHERE HAS_PRIVILEGES = 'TRUE' 
                       ORDER BY SCHEMA_NAME 
                       LIMIT 10`, (err, schemas) => {
                if (err) {
                    console.log('⚠️  Schemas query failed:', err.message);
                } else {
                    console.log('Available schemas:', JSON.stringify(schemas, null, 2));
                }
                console.log('');
                
                // Summary
                console.log('╔══════════════════════════════════════════════════════════════════╗');
                console.log('║                        Test Summary                              ║');
                console.log('╚══════════════════════════════════════════════════════════════════╝');
                console.log('');
                console.log('✅ HANA Cloud Connection: SUCCESS');
                console.log('✅ SQL Execution: SUCCESS');
                console.log('');
                
                conn.disconnect();
                process.exit(0);
            });
        });
    });
});