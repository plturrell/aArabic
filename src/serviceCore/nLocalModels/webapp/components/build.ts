#!/usr/bin/env bun
/**
 * Bun Build Script for NetworkGraph & ProcessFlow Components
 * Bundles TypeScript for browser with optimal settings
 */

console.log('🔨 Building NetworkGraph and ProcessFlow components...\n');

// Build NetworkGraph
console.log('📦 Bundling NetworkGraph...');
const networkGraphResult = await Bun.build({
  entrypoints: ['./NetworkGraph/NetworkGraph.ts'],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  minify: false,  // Keep readable for development
  sourcemap: 'external',
  naming: {
    entry: 'NetworkGraph/[dir]/[name].[ext]'
  }
});

if (!networkGraphResult.success) {
  console.error('❌ NetworkGraph build failed:');
  for (const log of networkGraphResult.logs) {
    console.error(log);
  }
  process.exit(1);
}

console.log('✅ NetworkGraph built successfully');
console.log(`   Output: ${networkGraphResult.outputs.length} files\n`);

// Build ProcessFlow
console.log('📦 Bundling ProcessFlow...');
const processFlowResult = await Bun.build({
  entrypoints: ['./ProcessFlow/ProcessFlow.ts'],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  minify: false,
  sourcemap: 'external',
  naming: {
    entry: 'ProcessFlow/[dir]/[name].[ext]'
  }
});

if (!processFlowResult.success) {
  console.error('❌ ProcessFlow build failed:');
  for (const log of processFlowResult.logs) {
    console.error(log);
  }
  process.exit(1);
}

console.log('✅ ProcessFlow built successfully');
console.log(`   Output: ${processFlowResult.outputs.length} files\n`);

// Build Charts
console.log('📦 Bundling Charts...');
const chartsResult = await Bun.build({
  entrypoints: ['./Charts/Charts.ts'],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  minify: false,
  sourcemap: 'external',
  naming: {
    entry: 'Charts/[dir]/[name].[ext]'
  }
});

if (!chartsResult.success) {
  console.error('❌ Charts build failed:');
  for (const log of chartsResult.logs) {
    console.error(log);
  }
  process.exit(1);
}

console.log('✅ Charts built successfully');
console.log(`   Output: ${chartsResult.outputs.length} files\n`);

// Build types separately
console.log('📦 Bundling types...');
const typesResult = await Bun.build({
  entrypoints: [
    './NetworkGraph/types.ts',
    './ProcessFlow/types.ts',
    './Charts/types.ts'
  ],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  minify: false,
  sourcemap: 'external'
});

if (!typesResult.success) {
  console.error('❌ Types build failed:');
  for (const log of typesResult.logs) {
    console.error(log);
  }
  process.exit(1);
}

console.log('✅ Types built successfully\n');

// Summary
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('✨ Build Complete!');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('📁 Output directory: ./dist/');
console.log('🌐 Ready for browser import');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
