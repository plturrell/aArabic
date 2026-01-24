//! Shimmy-Mojo LLM Server Integrations
//! 
//! This module provides a unified entry point for all server integrations:
//! - Document extraction and caching (nExtract + DragonflyDB)
//! - Vector embedding storage and similarity search
//! - Distributed rate limiting across instances
//! - Streaming extraction pipeline with backpressure
//! - Semantic document indexing (BM25 + Vector hybrid search)
//!
//! Version: 1.0.0
//! Last Updated: 2026-01-17

const std = @import("std");

/// Document cache integration (nExtract + HANA)
pub const DocumentCache = @import("document_cache/unified_doc_cache.zig");

/// HANA in-memory cache (replaces DragonflyDB)
pub const HanaCache = @import("cache/hana/hana_cache.zig");

/// Vector embedding cache with SIMD-optimized similarity search
pub const VectorCache = @import("cache/hana/hana_cache.zig");

/// Distributed rate limiter using HANA
pub const RateLimiter = @import("cache/hana/hana_cache.zig");

/// Streaming extraction pipeline (5-stage: Parse → Extract → Transform → Cache → Index)
pub const ExtractionPipeline = @import("pipeline/extraction_pipeline.zig");

/// Semantic document index (BM25 full-text + vector similarity)
pub const SemanticIndex = @import("search/semantic_index.zig");

/// AI Core integration for SAP AI Core deployments
pub const AICoreConfig = @import("aicore/aicore_config.zig");

/// Serving template generator for AI Core YAML templates
pub const ServingTemplateGenerator = @import("aicore/serving_template_generator.zig");

/// Integration version information
pub const version = struct {
    pub const major = 1;
    pub const minor = 0;
    pub const patch = 0;
    
    pub fn string() []const u8 {
        return "1.0.0";
    }
};

/// Initialize all integrations with the provided allocator
pub fn init(allocator: std.mem.Allocator) !IntegrationContext {
    return IntegrationContext{
        .allocator = allocator,
        .doc_cache = null,
        .vector_cache = null,
        .rate_limiter = null,
        .pipeline = null,
        .semantic_index = null,
    };
}

/// Integration context holding all subsystem instances
pub const IntegrationContext = struct {
    allocator: std.mem.Allocator,
    doc_cache: ?*DocumentCache.UnifiedDocCache,
    vector_cache: ?*VectorCache.VectorCache,
    rate_limiter: ?*RateLimiter.DistributedRateLimiter,
    pipeline: ?*ExtractionPipeline.Pipeline,
    semantic_index: ?*SemanticIndex.SemanticIndex,
    
    /// Initialize document cache
    pub fn initDocCache(self: *IntegrationContext, config: DocumentCache.Config) !void {
        const cache = try self.allocator.create(DocumentCache.UnifiedDocCache);
        cache.* = try DocumentCache.UnifiedDocCache.init(self.allocator, config);
        self.doc_cache = cache;
    }
    
    /// Initialize vector cache
    pub fn initVectorCache(self: *IntegrationContext, config: VectorCache.Config) !void {
        const cache = try self.allocator.create(VectorCache.VectorCache);
        cache.* = try VectorCache.VectorCache.init(self.allocator, config);
        self.vector_cache = cache;
    }
    
    /// Initialize rate limiter
    pub fn initRateLimiter(self: *IntegrationContext, config: RateLimiter.Config) !void {
        const limiter = try self.allocator.create(RateLimiter.DistributedRateLimiter);
        limiter.* = try RateLimiter.DistributedRateLimiter.init(self.allocator, config);
        self.rate_limiter = limiter;
    }
    
    /// Initialize extraction pipeline
    pub fn initPipeline(self: *IntegrationContext, config: ExtractionPipeline.PipelineConfig) !void {
        const pipeline = try self.allocator.create(ExtractionPipeline.Pipeline);
        pipeline.* = try ExtractionPipeline.Pipeline.init(self.allocator, config);
        self.pipeline = pipeline;
    }
    
    /// Initialize semantic index
    pub fn initSemanticIndex(self: *IntegrationContext, config: SemanticIndex.Config) !void {
        const index = try self.allocator.create(SemanticIndex.SemanticIndex);
        index.* = try SemanticIndex.SemanticIndex.init(self.allocator, config);
        self.semantic_index = index;
    }
    
    /// Cleanup all initialized subsystems
    pub fn deinit(self: *IntegrationContext) void {
        if (self.doc_cache) |cache| {
            cache.deinit();
            self.allocator.destroy(cache);
        }
        if (self.vector_cache) |cache| {
            cache.deinit();
            self.allocator.destroy(cache);
        }
        if (self.rate_limiter) |limiter| {
            limiter.deinit();
            self.allocator.destroy(limiter);
        }
        if (self.pipeline) |pipeline| {
            pipeline.deinit();
            self.allocator.destroy(pipeline);
        }
        if (self.semantic_index) |index| {
            index.deinit();
            self.allocator.destroy(index);
        }
    }
};

/// Quick start example configuration
pub fn defaultConfig(allocator: std.mem.Allocator) IntegrationContext {
    return IntegrationContext{
        .allocator = allocator,
        .doc_cache = null,
        .vector_cache = null,
        .rate_limiter = null,
        .pipeline = null,
        .semantic_index = null,
    };
}

test "integration module loads" {
    const testing = std.testing;
    try testing.expectEqual(@as(u32, 1), version.major);
}
