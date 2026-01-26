/**
 * Build Verification Tests
 * Ensures build outputs are correct and functional
 */

import { describe, test, expect } from 'bun:test';
import { readFileSync, statSync } from 'fs';
import { join } from 'path';

const DIST_DIR = join(__dirname, '../dist');

describe('Build Output Verification', () => {
  
  test('NetworkGraph bundle exists', () => {
    const devPath = join(DIST_DIR, 'NetworkGraph/NetworkGraph.js');
    const prodPath = join(DIST_DIR, 'NetworkGraph/NetworkGraph.min.js');
    
    expect(() => statSync(devPath)).not.toThrow();
    expect(() => statSync(prodPath)).not.toThrow();
  });
  
  test('ProcessFlow bundle exists', () => {
    const devPath = join(DIST_DIR, 'ProcessFlow/ProcessFlow.js');
    const prodPath = join(DIST_DIR, 'ProcessFlow/ProcessFlow.min.js');
    
    expect(() => statSync(devPath)).not.toThrow();
    expect(() => statSync(prodPath)).not.toThrow();
  });
  
  test('Type definitions exist', () => {
    const ngTypes = join(DIST_DIR, 'NetworkGraph/types.js');
    const pfTypes = join(DIST_DIR, 'ProcessFlow/types.js');
    
    expect(() => statSync(ngTypes)).not.toThrow();
    expect(() => statSync(pfTypes)).not.toThrow();
  });
  
  test('Production builds are smaller than development', () => {
    const ngDev = statSync(join(DIST_DIR, 'NetworkGraph/NetworkGraph.js')).size;
    const ngProd = statSync(join(DIST_DIR, 'NetworkGraph/NetworkGraph.min.js')).size;
    
    const pfDev = statSync(join(DIST_DIR, 'ProcessFlow/ProcessFlow.js')).size;
    const pfProd = statSync(join(DIST_DIR, 'ProcessFlow/ProcessFlow.min.js')).size;
    
    expect(ngProd).toBeLessThan(ngDev);
    expect(pfProd).toBeLessThan(pfDev);
    
    console.log(`NetworkGraph: ${ngDev} → ${ngProd} (${Math.round((1 - ngProd/ngDev) * 100)}% reduction)`);
    console.log(`ProcessFlow: ${pfDev} → ${pfProd} (${Math.round((1 - pfProd/pfDev) * 100)}% reduction)`);
  });
  
  test('Bundles are valid ES modules', () => {
    const ngContent = readFileSync(join(DIST_DIR, 'NetworkGraph/NetworkGraph.js'), 'utf-8');
    const pfContent = readFileSync(join(DIST_DIR, 'ProcessFlow/ProcessFlow.js'), 'utf-8');
    
    // Check for ES module exports
    expect(ngContent).toContain('export');
    expect(pfContent).toContain('export');
    
    // Check they don't have require() calls (should be ESM)
    expect(ngContent).not.toContain('require(');
    expect(pfContent).not.toContain('require(');
  });
  
  test('Bundles contain expected classes', () => {
    const ngContent = readFileSync(join(DIST_DIR, 'NetworkGraph/NetworkGraph.js'), 'utf-8');
    const pfContent = readFileSync(join(DIST_DIR, 'ProcessFlow/ProcessFlow.js'), 'utf-8');
    
    // NetworkGraph should have main class
    expect(ngContent).toContain('NetworkGraph');
    
    // ProcessFlow should have main class
    expect(pfContent).toContain('ProcessFlow');
  });
  
  test('Bundle sizes are reasonable', () => {
    const ngSize = statSync(join(DIST_DIR, 'NetworkGraph/NetworkGraph.min.js')).size;
    const pfSize = statSync(join(DIST_DIR, 'ProcessFlow/ProcessFlow.min.js')).size;
    
    // NetworkGraph should be < 50KB minified
    expect(ngSize).toBeLessThan(50 * 1024);
    
    // ProcessFlow should be < 30KB minified
    expect(pfSize).toBeLessThan(30 * 1024);
    
    // Total should be < 80KB
    expect(ngSize + pfSize).toBeLessThan(80 * 1024);
  });
  
  test('Source maps exist for development builds', () => {
    const ngMap = join(DIST_DIR, 'NetworkGraph/NetworkGraph.js.map');
    const pfMap = join(DIST_DIR, 'ProcessFlow/ProcessFlow.js.map');
    
    expect(() => statSync(ngMap)).not.toThrow();
    expect(() => statSync(pfMap)).not.toThrow();
  });
  
  test('No source maps for production builds', () => {
    const ngMapProd = join(DIST_DIR, 'NetworkGraph/NetworkGraph.min.js.map');
    const pfMapProd = join(DIST_DIR, 'ProcessFlow/ProcessFlow.min.js.map');
    
    expect(() => statSync(ngMapProd)).toThrow();
    expect(() => statSync(pfMapProd)).toThrow();
  });

});

describe('Bundle Content Validation', () => {
  
  test('No TypeScript syntax in output', () => {
    const ngContent = readFileSync(join(DIST_DIR, 'NetworkGraph/NetworkGraph.js'), 'utf-8');
    const pfContent = readFileSync(join(DIST_DIR, 'ProcessFlow/ProcessFlow.js'), 'utf-8');
    
    // Should not contain TS-specific syntax
    expect(ngContent).not.toContain('interface ');
    expect(ngContent).not.toContain('type ');
    expect(pfContent).not.toContain('interface ');
    expect(pfContent).not.toContain('type ');
  });
  
  test('Contains browser-compatible code', () => {
    const ngContent = readFileSync(join(DIST_DIR, 'NetworkGraph/NetworkGraph.js'), 'utf-8');
    
    // Should have DOM APIs
    expect(ngContent).toContain('document');
    expect(ngContent).toContain('SVG');
  });

});
