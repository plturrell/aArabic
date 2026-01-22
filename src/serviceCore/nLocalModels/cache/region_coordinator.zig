// ============================================================================
// Region Coordinator - Day 62 Implementation
// ============================================================================
// Purpose: Multi-region cache coordination with geo-aware routing
// Week: Week 13 (Days 61-65) - Advanced Scalability
// Phase: Month 4 - Advanced Features
// ============================================================================

const std = @import("std");
const Allocator = std.mem.Allocator;
const DistributedCoordinator = @import("distributed_coordinator.zig").DistributedCoordinator;
const DistributedCacheConfig = @import("distributed_coordinator.zig").DistributedCacheConfig;

// ============================================================================
// GEO LOCATION
// ============================================================================

/// Geographic location coordinates
pub const GeoLocation = struct {
    latitude: f32,
    longitude: f32,
    
    pub fn init(lat: f32, lon: f32) GeoLocation {
        return .{
            .latitude = lat,
            .longitude = lon,
        };
    }
    
    /// Calculate distance to another location (Haversine formula)
    pub fn distanceTo(self: GeoLocation, other: GeoLocation) f32 {
        const earth_radius: f32 = 6371.0; // km
        
        const lat1 = self.latitude * std.math.pi / 180.0;
        const lat2 = other.latitude * std.math.pi / 180.0;
        const delta_lat = (other.latitude - self.latitude) * std.math.pi / 180.0;
        const delta_lon = (other.longitude - self.longitude) * std.math.pi / 180.0;
        
        const a = @sin(delta_lat / 2.0) * @sin(delta_lat / 2.0) +
                 @cos(lat1) * @cos(lat2) *
                 @sin(delta_lon / 2.0) * @sin(delta_lon / 2.0);
        
        const c = 2.0 * @atan2(@sqrt(a), @sqrt(1.0 - a));
        
        return earth_radius * c;
    }
};

// ============================================================================
// REGION STATUS
// ============================================================================

/// Status of a cache region
pub const RegionStatus = enum {
    healthy,    // Operating normally
    degraded,   // Experiencing issues
    down,       // Unavailable
    recovering, // Coming back online
};

// ============================================================================
// REGION
// ============================================================================

/// A geographic region with its own cache cluster
pub const Region = struct {
    id: []const u8,
    name: []const u8,
    location: GeoLocation,
    cache_cluster: *DistributedCoordinator,
    status: RegionStatus,
    last_health_check: i64,
    avg_latency_ms: f32,
    requests_served: u64,
    cache_hit_rate: f32,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        location: GeoLocation,
        cache_config: DistributedCacheConfig,
    ) !*Region {
        const region = try allocator.create(Region);
        
        const cache_cluster = try DistributedCoordinator.init(allocator, cache_config);
        
        region.* = .{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .location = location,
            .cache_cluster = cache_cluster,
            .status = .healthy,
            .last_health_check = std.time.milliTimestamp(),
            .avg_latency_ms = 0.0,
            .requests_served = 0,
            .cache_hit_rate = 0.0,
        };
        
        return region;
    }
    
    pub fn deinit(self: *Region, allocator: Allocator) void {
        self.cache_cluster.deinit();
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.destroy(self);
    }
    
    pub fn updateHealthCheck(self: *Region) void {
        self.last_health_check = std.time.milliTimestamp();
        
        // Update status based on health check age
        const age_ms = std.time.milliTimestamp() - self.last_health_check;
        if (age_ms > 60000) { // 60 seconds
            self.status = .down;
        } else if (age_ms > 30000) { // 30 seconds
            self.status = .degraded;
        } else if (self.status == .recovering and age_ms < 10000) {
            self.status = .healthy;
        }
    }
    
    pub fn isHealthy(self: *const Region) bool {
        return self.status == .healthy;
    }
    
    pub fn markDown(self: *Region) void {
        self.status = .down;
    }
    
    pub fn markRecovering(self: *Region) void {
        self.status = .recovering;
    }
    
    pub fn recordRequest(self: *Region, latency_ms: f32, cache_hit: bool) void {
        self.requests_served += 1;
        
        // Update moving average latency
        const alpha: f32 = 0.1; // Smoothing factor
        self.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * self.avg_latency_ms;
        
        // Update cache hit rate
        if (cache_hit) {
            self.cache_hit_rate = alpha + (1.0 - alpha) * self.cache_hit_rate;
        } else {
            self.cache_hit_rate = (1.0 - alpha) * self.cache_hit_rate;
        }
    }
};

// ============================================================================
// REGION ROUTING POLICY
// ============================================================================

/// Policy for selecting regions
pub const RegionRoutingPolicy = enum {
    nearest,        // Route to geographically nearest region
    lowest_latency, // Route to region with lowest latency
    least_loaded,   // Route to region with least load
    random,         // Random region (for testing)
};

// ============================================================================
// REGION COORDINATOR CONFIG
// ============================================================================

pub const RegionCoordinatorConfig = struct {
    routing_policy: RegionRoutingPolicy = .nearest,
    health_check_interval_ms: u64 = 5000,
    failover_threshold_ms: u64 = 30000,
    max_regions: u32 = 10,
    enable_cross_region_replication: bool = true,
};

// ============================================================================
// REGION COORDINATOR
// ============================================================================

/// Coordinates multiple geographic regions
pub const RegionCoordinator = struct {
    allocator: Allocator,
    config: RegionCoordinatorConfig,
    regions: std.ArrayList(*Region),
    mutex: std.Thread.Mutex,
    
    pub fn init(allocator: Allocator, config: RegionCoordinatorConfig) !*RegionCoordinator {
        const coordinator = try allocator.create(RegionCoordinator);
        coordinator.* = .{
            .allocator = allocator,
            .config = config,
            .regions = std.ArrayList(*Region).init(allocator),
            .mutex = .{},
        };
        return coordinator;
    }
    
    pub fn deinit(self: *RegionCoordinator) void {
        for (self.regions.items) |region| {
            region.deinit(self.allocator);
        }
        self.regions.deinit();
        self.allocator.destroy(self);
    }
    
    // ========================================================================
    // REGION MANAGEMENT
    // ========================================================================
    
    /// Register a new region
    pub fn registerRegion(
        self: *RegionCoordinator,
        id: []const u8,
        name: []const u8,
        location: GeoLocation,
        cache_config: DistributedCacheConfig,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.regions.items.len >= self.config.max_regions) {
            return error.MaxRegionsReached;
        }
        
        // Check if region already exists
        for (self.regions.items) |region| {
            if (std.mem.eql(u8, region.id, id)) {
                return error.RegionAlreadyExists;
            }
        }
        
        const region = try Region.init(self.allocator, id, name, location, cache_config);
        try self.regions.append(region);
        
        std.log.info("Registered region: {s} ({s}) at {d:.2},{d:.2}", .{
            id,
            name,
            location.latitude,
            location.longitude,
        });
    }
    
    /// Remove a region
    pub fn removeRegion(self: *RegionCoordinator, id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.regions.items, 0..) |region, i| {
            if (std.mem.eql(u8, region.id, id)) {
                region.deinit(self.allocator);
                _ = self.regions.orderedRemove(i);
                std.log.info("Removed region: {s}", .{id});
                return;
            }
        }
        
        return error.RegionNotFound;
    }
    
    // ========================================================================
    // REGION SELECTION
    // ========================================================================
    
    /// Select best region for client location
    pub fn selectRegion(self: *RegionCoordinator, client_location: GeoLocation) !*Region {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.regions.items.len == 0) {
            return error.NoRegionsAvailable;
        }
        
        // Filter healthy regions
        var healthy_regions = std.ArrayList(*Region).init(self.allocator);
        defer healthy_regions.deinit();
        
        for (self.regions.items) |region| {
            if (region.isHealthy()) {
                try healthy_regions.append(region);
            }
        }
        
        if (healthy_regions.items.len == 0) {
            // Fallback to degraded regions if no healthy ones
            for (self.regions.items) |region| {
                if (region.status == .degraded or region.status == .recovering) {
                    try healthy_regions.append(region);
                }
            }
        }
        
        if (healthy_regions.items.len == 0) {
            return error.NoHealthyRegions;
        }
        
        // Select based on policy
        return switch (self.config.routing_policy) {
            .nearest => self.selectNearestRegion(client_location, healthy_regions.items),
            .lowest_latency => self.selectLowestLatencyRegion(healthy_regions.items),
            .least_loaded => self.selectLeastLoadedRegion(healthy_regions.items),
            .random => self.selectRandomRegion(healthy_regions.items),
        };
    }
    
    fn selectNearestRegion(
        self: *RegionCoordinator,
        client_location: GeoLocation,
        regions: []*Region,
    ) *Region {
        _ = self;
        var nearest = regions[0];
        var min_distance = client_location.distanceTo(nearest.location);
        
        for (regions[1..]) |region| {
            const distance = client_location.distanceTo(region.location);
            if (distance < min_distance) {
                min_distance = distance;
                nearest = region;
            }
        }
        
        return nearest;
    }
    
    fn selectLowestLatencyRegion(self: *RegionCoordinator, regions: []*Region) *Region {
        _ = self;
        var best = regions[0];
        var min_latency = best.avg_latency_ms;
        
        for (regions[1..]) |region| {
            if (region.avg_latency_ms < min_latency) {
                min_latency = region.avg_latency_ms;
                best = region;
            }
        }
        
        return best;
    }
    
    fn selectLeastLoadedRegion(self: *RegionCoordinator, regions: []*Region) *Region {
        _ = self;
        var best = regions[0];
        var min_load = best.requests_served;
        
        for (regions[1..]) |region| {
            if (region.requests_served < min_load) {
                min_load = region.requests_served;
                best = region;
            }
        }
        
        return best;
    }
    
    fn selectRandomRegion(self: *RegionCoordinator, regions: []*Region) *Region {
        _ = self;
        // Simple random: use timestamp modulo
        const now = std.time.milliTimestamp();
        const idx = @as(usize, @intCast(@mod(now, @as(i64, @intCast(regions.len)))));
        return regions[idx];
    }
    
    // ========================================================================
    // HEALTH MONITORING
    // ========================================================================
    
    /// Check health of all regions
    pub fn checkHealth(self: *RegionCoordinator) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.regions.items) |region| {
            region.updateHealthCheck();
            
            // Log unhealthy regions
            if (!region.isHealthy()) {
                std.log.warn("Region {s} is {s}", .{ region.id, @tagName(region.status) });
            }
        }
    }
    
    /// Get statistics for all regions
    pub fn getRegionStats(self: *RegionCoordinator) RegionStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var healthy: u32 = 0;
        var total_requests: u64 = 0;
        var avg_latency: f32 = 0.0;
        var avg_hit_rate: f32 = 0.0;
        
        for (self.regions.items) |region| {
            if (region.isHealthy()) healthy += 1;
            total_requests += region.requests_served;
            avg_latency += region.avg_latency_ms;
            avg_hit_rate += region.cache_hit_rate;
        }
        
        const count: f32 = @floatFromInt(self.regions.items.len);
        if (count > 0) {
            avg_latency /= count;
            avg_hit_rate /= count;
        }
        
        return .{
            .total_regions = @intCast(self.regions.items.len),
            .healthy_regions = healthy,
            .total_requests = total_requests,
            .avg_latency_ms = avg_latency,
            .avg_cache_hit_rate = avg_hit_rate,
        };
    }
};

/// Statistics across all regions
pub const RegionStats = struct {
    total_regions: u32,
    healthy_regions: u32,
    total_requests: u64,
    avg_latency_ms: f32,
    avg_cache_hit_rate: f32,
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "GeoLocation: distance calculation" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    // New York to London (approx 5570 km)
    const ny = GeoLocation.init(40.7128, -74.0060);
    const london = GeoLocation.init(51.5074, -0.1278);
    
    const distance = ny.distanceTo(london);
    
    // Allow 10% tolerance
    try std.testing.expect(distance > 5000.0 and distance < 6000.0);
}

test "Region: initialization and cleanup" {
    const allocator = std.testing.allocator;
    
    const location = GeoLocation.init(37.7749, -122.4194); // San Francisco
    const cache_config = DistributedCacheConfig{};
    
    const region = try Region.init(allocator, "us-west", "US West", location, cache_config);
    defer region.deinit(allocator);
    
    try std.testing.expectEqualStrings("us-west", region.id);
    try std.testing.expectEqualStrings("US West", region.name);
    try std.testing.expect(region.isHealthy());
}

test "RegionCoordinator: register and remove regions" {
    const allocator = std.testing.allocator;
    
    const config = RegionCoordinatorConfig{};
    const coordinator = try RegionCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    const cache_config = DistributedCacheConfig{};
    
    // Register regions
    try coordinator.registerRegion(
        "us-east",
        "US East",
        GeoLocation.init(40.7128, -74.0060),
        cache_config,
    );
    
    try coordinator.registerRegion(
        "us-west",
        "US West",
        GeoLocation.init(37.7749, -122.4194),
        cache_config,
    );
    
    const stats = coordinator.getRegionStats();
    try std.testing.expectEqual(@as(u32, 2), stats.total_regions);
    try std.testing.expectEqual(@as(u32, 2), stats.healthy_regions);
    
    // Remove region
    try coordinator.removeRegion("us-west");
    const stats2 = coordinator.getRegionStats();
    try std.testing.expectEqual(@as(u32, 1), stats2.total_regions);
}

test "RegionCoordinator: select nearest region" {
    const allocator = std.testing.allocator;
    
    const config = RegionCoordinatorConfig{
        .routing_policy = .nearest,
    };
    const coordinator = try RegionCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    const cache_config = DistributedCacheConfig{};
    
    // Register US and EU regions
    try coordinator.registerRegion(
        "us-east",
