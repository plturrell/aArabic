#!/usr/bin/env bun
/**
 * Production Build Script - Minified & Optimized
 * Builds TypeScript components with maximum optimization for production
 */

console.log('🏭 Building for PRODUCTION...\n');

// Build NetworkGraph (minified)
console.log('📦 Bundling NetworkGraph (minified)...');
const networkGraphResult = await Bun.build({
  entrypoints: ['./NetworkGraph/NetworkGraph.ts'],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  minify: true,           // ✅ Enable minification
  sourcemap: 'none',      // ❌ No source maps for prod
  naming: {
    entry: 'NetworkGraph/[dir]/[name].min.[ext]'
  }
});

if (!networkGraphResult.success) {
  console.error('❌ NetworkGraph build failed:');
  for (const log of networkGraphResult.logs) {
    console.error(log);
  }
  process.exit(1);
}

console.log('✅ NetworkGraph built');

// Build ProcessFlow (minified)
console.log('📦 Bundling ProcessFlow (minified)...');
const processFlowResult = await Bun.build({
  entrypoints: ['./ProcessFlow/ProcessFlow.ts'],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  minify: true,
  sourcemap: 'none',
  naming: {
    entry: 'ProcessFlow/[dir]/[name].min.[ext]'
  }
});

if (!processFlowResult.success) {
  console.error('❌ ProcessFlow build failed:');
  for (const log of processFlowResult.logs) {
    console.error(log);
  }
  process.exit(1);
}

console.log('✅ ProcessFlow built');

// Build types (minified)
console.log('📦 Bundling types...');
const typesResult = await Bun.build({
  entrypoints: [
    './NetworkGraph/types.ts',
    './ProcessFlow/types.ts'
  ],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  minify: true,
  sourcemap: 'none'
});

if (!typesResult.success) {
  console.error('❌ Types build failed:');
  for (const log of typesResult.logs) {
    console.error(log);
  }
  process.exit(1);
}

console.log('✅ Types built\n');

// Get file sizes
const fs = require('fs');
const path = require('path');

function getFileSize(filePath: string): number {
  try {
    return fs.statSync(filePath).size;
  } catch {
    return 0;
  }
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

const ngSize = getFileSize('./dist/NetworkGraph/NetworkGraph.min.js');
const pfSize = getFileSize('./dist/ProcessFlow/ProcessFlow.min.js');
const totalSize = ngSize + pfSize;

// Summary
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('🎉 Production Build Complete!');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('📦 Minified Bundles:');
console.log(`   NetworkGraph: ${formatBytes(ngSize)}`);
console.log(`   ProcessFlow:  ${formatBytes(pfSize)}`);
console.log(`   Total:        ${formatBytes(totalSize)}`);
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('✅ Ready for production deployment');
console.log('📁 Deploy ./dist/ directory');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
