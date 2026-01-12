#!/usr/bin/env node
/**
 * TOON Format Conversion Service
 * Provides JSON ↔ TOON conversion for Shimmy-Mojo infrastructure
 * 
 * Endpoints:
 * - POST /encode - Convert JSON to TOON
 * - POST /decode - Convert TOON to JSON
 * - GET /health - Health check
 */

import { encode, decode } from '@toon-format/toon';
import express from 'express';
import cors from 'cors';

const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.text({ type: 'text/toon', limit: '50mb' }));

// ============================================================================
// Routes
// ============================================================================

/**
 * POST /encode
 * Convert JSON to TOON format
 * 
 * Body: JSON object
 * Response: TOON formatted text
 */
app.post('/encode', (req, res) => {
  try {
    const startTime = Date.now();
    
    // Encode to TOON
    const toonOutput = encode(req.body, {
      delimiter: ',',  // Default comma delimiter
      tabular: true    // Prefer tabular format
    });
    
    const duration = Date.now() - startTime;
    
    // Count tokens (rough estimate)
    const jsonTokens = JSON.stringify(req.body).length / 4;
    const toonTokens = toonOutput.length / 4;
    const savings = ((jsonTokens - toonTokens) / jsonTokens * 100).toFixed(1);
    
    console.log(`✅ Encoded to TOON in ${duration}ms (${savings}% token savings)`);
    
    res.type('text/toon').send(toonOutput);
  } catch (err) {
    console.error('❌ Encoding error:', err.message);
    res.status(400).json({ error: err.message });
  }
});

/**
 * POST /decode
 * Convert TOON to JSON format
 * 
 * Body: { toon: "toon formatted string" } or raw TOON text
 * Response: JSON object
 */
app.post('/decode', (req, res) => {
  try {
    const startTime = Date.now();
    
    // Get TOON input
    const toonInput = typeof req.body === 'string' 
      ? req.body 
      : req.body.toon;
    
    if (!toonInput) {
      return res.status(400).json({ 
        error: 'Missing TOON input. Send as { "toon": "..." } or raw text with Content-Type: text/toon' 
      });
    }
    
    // Decode from TOON
    const jsonOutput = decode(toonInput);
    
    const duration = Date.now() - startTime;
    console.log(`✅ Decoded from TOON in ${duration}ms`);
    
    res.json(jsonOutput);
  } catch (err) {
    console.error('❌ Decoding error:', err.message);
    res.status(400).json({ error: err.message });
  }
});

/**
 * POST /encode-with-stats
 * Convert JSON to TOON with detailed statistics
 * 
 * Body: JSON object
 * Response: { toon: string, stats: {...} }
 */
app.post('/encode-with-stats', (req, res) => {
  try {
    const startTime = Date.now();
    
    // Encode
    const toonOutput = encode(req.body);
    const duration = Date.now() - startTime;
    
    // Calculate stats
    const jsonString = JSON.stringify(req.body);
    const jsonTokens = Math.ceil(jsonString.length / 4);
    const toonTokens = Math.ceil(toonOutput.length / 4);
    const savings = jsonTokens - toonTokens;
    const savingsPercent = (savings / jsonTokens * 100).toFixed(1);
    
    res.json({
      toon: toonOutput,
      stats: {
        jsonTokens,
        toonTokens,
        savings,
        savingsPercent: `${savingsPercent}%`,
        encodingTimeMs: duration,
        jsonSizeBytes: jsonString.length,
        toonSizeBytes: toonOutput.length
      }
    });
  } catch (err) {
    console.error('❌ Encoding error:', err.message);
    res.status(400).json({ error: err.message });
  }
});

/**
 * GET /health
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'TOON Converter',
    version: '1.0.0',
    uptime: process.uptime(),
    endpoints: {
      encode: 'POST /encode',
      decode: 'POST /decode',
      stats: 'POST /encode-with-stats'
    }
  });
});

/**
 * GET /
 * Service info
 */
app.get('/', (req, res) => {
  res.json({
    name: 'TOON Format Conversion Service',
    description: 'Convert between JSON and TOON formats',
    version: '1.0.0',
    tokenSavings: '~40% vs JSON',
    accuracy: '74% (vs JSON 70%)',
    endpoints: {
      'POST /encode': 'Convert JSON to TOON',
      'POST /decode': 'Convert TOON to JSON',
      'POST /encode-with-stats': 'Encode with token statistics',
      'GET /health': 'Health check'
    },
    documentation: 'https://toonformat.dev'
  });
});

// ============================================================================
// Server Startup
// ============================================================================

const PORT = process.env.PORT || 8003;
const HOST = process.env.HOST || '0.0.0.0';

app.listen(PORT, HOST, () => {
  console.log('='.repeat(80));
  console.log('🎨 TOON Format Conversion Service');
  console.log('='.repeat(80));
  console.log('');
  console.log(`✅ Server listening on http://${HOST}:${PORT}`);
  console.log('');
  console.log('Endpoints:');
  console.log('  POST /encode              - Convert JSON → TOON');
