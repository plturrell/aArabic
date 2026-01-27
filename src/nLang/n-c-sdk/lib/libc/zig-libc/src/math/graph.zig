// Graph algorithms module
// Pure Zig implementations for graph algorithms with C-compatible exports

const std = @import("std");
const math = std.math;

// ============================================================================
// Constants
// ============================================================================

pub const INF: f64 = math.inf(f64);
const MAX_GRAPH_SIZE: usize = 1024;

// ============================================================================
// Hungarian Algorithm (Kuhn-Munkres) for Assignment Problem
// ============================================================================

/// Solver for the Hungarian (Kuhn-Munkres) algorithm
pub const HungarianSolver = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) HungarianSolver {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *HungarianSolver) void {
        _ = self;
    }
};

/// Hungarian algorithm for optimal assignment problem
/// cost_matrix: n x m matrix in row-major order (cost[i*m + j] for row i, col j)
/// n: number of rows (workers)
/// m: number of columns (jobs)
/// assignments: output array of size n, assignments[i] = column assigned to row i (-1 if none)
/// Returns: total cost of optimal assignment
pub fn hungarian_solve(
    cost_matrix: [*]const f64,
    n: usize,
    m: usize,
    assignments: [*]i32,
) f64 {
    if (n == 0 or m == 0 or n > MAX_GRAPH_SIZE or m > MAX_GRAPH_SIZE) {
        return INF;
    }

    const dim = @max(n, m);

    // Use stack buffers for small problems
    var u_buf: [MAX_GRAPH_SIZE]f64 = undefined;
    var v_buf: [MAX_GRAPH_SIZE]f64 = undefined;
    var p_buf: [MAX_GRAPH_SIZE]usize = undefined;
    var way_buf: [MAX_GRAPH_SIZE]usize = undefined;
    var minv_buf: [MAX_GRAPH_SIZE]f64 = undefined;
    var used_buf: [MAX_GRAPH_SIZE]bool = undefined;

    const u = u_buf[0 .. dim + 1];
    const v = v_buf[0 .. dim + 1];
    const p = p_buf[0 .. dim + 1];
    const way = way_buf[0 .. dim + 1];

    // Initialize
    @memset(u, 0);
    @memset(v, 0);
    @memset(p, 0);
    @memset(way, 0);

    // Process each row
    for (1..dim + 1) |i| {
        p[0] = i;
        var j0: usize = 0;
        const minv = minv_buf[0 .. dim + 1];
        const used = used_buf[0 .. dim + 1];
        @memset(minv, INF);
        @memset(used, false);

        while (p[j0] != 0) {
            used[j0] = true;
            const curr_row = p[j0];
            var delta: f64 = INF;
            var j1: usize = 0;

            for (1..dim + 1) |j| {
                if (!used[j]) {
                    // Get cost, use 0 for virtual entries
                    const row_idx = curr_row - 1;
                    const col_idx = j - 1;
                    const cost = if (row_idx < n and col_idx < m)
                        cost_matrix[row_idx * m + col_idx]
                    else
                        0.0;

                    const cur = cost - u[curr_row] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            // Update potentials
            for (0..dim + 1) |j| {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        }

        // Augment path
        while (j0 != 0) {
            const j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
    }

    // Extract assignments
    for (0..n) |i| {
        assignments[i] = -1;
    }

    for (1..dim + 1) |j| {
        if (p[j] != 0 and p[j] <= n and j <= m) {
            assignments[p[j] - 1] = @intCast(j - 1);
        }
    }

    // Calculate total cost
    var total_cost: f64 = 0.0;
    for (0..n) |i| {
        if (assignments[i] >= 0) {
            const j: usize = @intCast(assignments[i]);
            total_cost += cost_matrix[i * m + j];
        }
    }

    return total_cost;
}

// ============================================================================
// PageRank Algorithm
// ============================================================================

/// Compute PageRank scores
/// adjacency: flat array of adjacency lists (use offsets array to index)
/// offsets: offsets[i] = start index of neighbors for node i, offsets[n] = total edges
/// n: number of nodes
/// damping: damping factor (default 0.85)
/// max_iter: maximum iterations (default 100)
/// result: output array of size n for PageRank scores
pub fn pagerank(
    adjacency: [*]const u32,
    offsets: [*]const u32,
    n: usize,
    damping: f64,
    max_iter: u32,
    result: [*]f64,
) void {
    if (n == 0 or n > MAX_GRAPH_SIZE) return;

    var rank_buf: [MAX_GRAPH_SIZE]f64 = undefined;
    const rank = rank_buf[0..n];
    const d = if (damping > 0.0 and damping < 1.0) damping else 0.85;

    // Initialize ranks uniformly
    const init_rank = 1.0 / @as(f64, @floatFromInt(n));
    for (0..n) |i| {
        result[i] = init_rank;
    }

    // Power iteration
    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        @memset(rank, 0);

        // Distribute rank from each node to its neighbors
        for (0..n) |i| {
            const start = offsets[i];
            const end = offsets[i + 1];
            const out_degree = end - start;
            if (out_degree > 0) {
                const contrib = result[i] / @as(f64, @floatFromInt(out_degree));
                for (start..end) |j| {
                    const neighbor = adjacency[j];
                    if (neighbor < n) {
                        rank[neighbor] += contrib;
                    }
                }
            }
        }

        // Apply damping factor
        const teleport = (1.0 - d) / @as(f64, @floatFromInt(n));
        for (0..n) |i| {
            result[i] = d * rank[i] + teleport;
        }
    }
}

// ============================================================================
// Shortest Path Algorithms
// ============================================================================

/// Dijkstra's single-source shortest path algorithm
/// adjacency: flat array of neighbor indices
/// weights: flat array of edge weights (parallel to adjacency)
/// offsets: offsets[i] = start index of neighbors for node i
/// n: number of nodes
/// source: source node index
/// distances: output array of size n for distances
/// predecessors: output array of size n for predecessors (-1 if none)
pub fn dijkstra(
    adjacency: [*]const u32,
    weights: [*]const f64,
    offsets: [*]const u32,
    n: usize,
    source: usize,
    distances: [*]f64,
    predecessors: [*]i32,
) void {
    if (n == 0 or n > MAX_GRAPH_SIZE or source >= n) return;

    var visited_buf: [MAX_GRAPH_SIZE]bool = undefined;
    const visited = visited_buf[0..n];

    // Initialize
    for (0..n) |i| {
        distances[i] = INF;
        predecessors[i] = -1;
        visited[i] = false;
    }
    distances[source] = 0;

    // Main loop - find minimum distance unvisited node
    for (0..n) |_| {
        var min_dist: f64 = INF;
        var u: usize = n; // Invalid sentinel

        for (0..n) |i| {
            if (!visited[i] and distances[i] < min_dist) {
                min_dist = distances[i];
                u = i;
            }
        }

        if (u == n) break; // No reachable unvisited nodes

        visited[u] = true;

        // Relax edges
        const start = offsets[u];
        const end = offsets[u + 1];
        for (start..end) |j| {
            const v = adjacency[j];
            if (v < n and !visited[v]) {
                const alt = distances[u] + weights[j];
                if (alt < distances[v]) {
                    distances[v] = alt;
                    predecessors[v] = @intCast(u);
                }
            }
        }
    }
}

/// Edge structure for Bellman-Ford
pub const Edge = extern struct {
    src: u32,
    dst: u32,
    weight: f64,
};

/// Bellman-Ford algorithm for single-source shortest path with negative weights
/// edges: array of edges
/// n_edges: number of edges
/// n_vertices: number of vertices
/// source: source vertex
/// distances: output array of size n_vertices
/// Returns: 0 on success, -1 if negative cycle detected
pub fn bellman_ford(
    edges: [*]const Edge,
    n_edges: usize,
    n_vertices: usize,
    source: usize,
    distances: [*]f64,
) i32 {
    if (n_vertices == 0 or n_vertices > MAX_GRAPH_SIZE or source >= n_vertices) return -1;

    // Initialize distances
    for (0..n_vertices) |i| {
        distances[i] = INF;
    }
    distances[source] = 0;

    // Relax edges n-1 times
    for (0..n_vertices - 1) |_| {
        for (0..n_edges) |i| {
            const e = edges[i];
            if (e.src < n_vertices and e.dst < n_vertices) {
                if (distances[e.src] != INF and distances[e.src] + e.weight < distances[e.dst]) {
                    distances[e.dst] = distances[e.src] + e.weight;
                }
            }
        }
    }

    // Check for negative cycles
    for (0..n_edges) |i| {
        const e = edges[i];
        if (e.src < n_vertices and e.dst < n_vertices) {
            if (distances[e.src] != INF and distances[e.src] + e.weight < distances[e.dst]) {
                return -1; // Negative cycle detected
            }
        }
    }

    return 0;
}


// ============================================================================
// Graph Traversal Algorithms
// ============================================================================

/// Breadth-first search traversal
/// adjacency: flat array of neighbor indices
/// offsets: offsets[i] = start index of neighbors for node i
/// n: number of nodes
/// start: starting node
/// visited: output array of size n (1 if visited, 0 otherwise)
/// order: output array of size n for visit order (-1 if not visited)
/// Returns: number of nodes visited
pub fn bfs(
    adjacency: [*]const u32,
    offsets: [*]const u32,
    n: usize,
    start: usize,
    visited: [*]u8,
    order: [*]i32,
) usize {
    if (n == 0 or n > MAX_GRAPH_SIZE or start >= n) return 0;

    var queue_buf: [MAX_GRAPH_SIZE]usize = undefined;
    var queue_front: usize = 0;
    var queue_back: usize = 0;

    // Initialize
    for (0..n) |i| {
        visited[i] = 0;
        order[i] = -1;
    }

    // Start BFS
    queue_buf[queue_back] = start;
    queue_back += 1;
    visited[start] = 1;
    var visit_count: usize = 0;

    while (queue_front < queue_back) {
        const u = queue_buf[queue_front];
        queue_front += 1;
        order[u] = @intCast(visit_count);
        visit_count += 1;

        // Visit neighbors
        const edge_start = offsets[u];
        const edge_end = offsets[u + 1];
        for (edge_start..edge_end) |j| {
            const v = adjacency[j];
            if (v < n and visited[v] == 0) {
                visited[v] = 1;
                queue_buf[queue_back] = v;
                queue_back += 1;
            }
        }
    }

    return visit_count;
}

/// Depth-first search traversal
/// adjacency: flat array of neighbor indices
/// offsets: offsets[i] = start index of neighbors for node i
/// n: number of nodes
/// start: starting node
/// visited: output array of size n (1 if visited, 0 otherwise)
/// order: output array of size n for visit order (-1 if not visited)
/// Returns: number of nodes visited
pub fn dfs(
    adjacency: [*]const u32,
    offsets: [*]const u32,
    n: usize,
    start: usize,
    visited: [*]u8,
    order: [*]i32,
) usize {
    if (n == 0 or n > MAX_GRAPH_SIZE or start >= n) return 0;

    var stack_buf: [MAX_GRAPH_SIZE]usize = undefined;
    var stack_top: usize = 0;

    // Initialize
    for (0..n) |i| {
        visited[i] = 0;
        order[i] = -1;
    }

    // Start DFS
    stack_buf[stack_top] = start;
    stack_top += 1;
    var visit_count: usize = 0;

    while (stack_top > 0) {
        stack_top -= 1;
        const u = stack_buf[stack_top];

        if (visited[u] == 1) continue;

        visited[u] = 1;
        order[u] = @intCast(visit_count);
        visit_count += 1;

        // Push neighbors (in reverse for consistent order)
        const edge_start = offsets[u];
        const edge_end = offsets[u + 1];
        var j = edge_end;
        while (j > edge_start) {
            j -= 1;
            const v = adjacency[j];
            if (v < n and visited[v] == 0) {
                stack_buf[stack_top] = v;
                stack_top += 1;
            }
        }
    }

    return visit_count;
}

// ============================================================================
// Community Detection
// ============================================================================

/// Label propagation algorithm for community detection
/// adjacency: flat array of neighbor indices
/// offsets: offsets[i] = start index of neighbors for node i
/// n: number of nodes
/// labels: input/output array of size n (initial labels and final community labels)
/// max_iter: maximum iterations
/// Returns: number of iterations performed
pub fn label_propagation(
    adjacency: [*]const u32,
    offsets: [*]const u32,
    n: usize,
    labels: [*]u32,
    max_iter: u32,
) u32 {
    if (n == 0 or n > MAX_GRAPH_SIZE) return 0;

    var label_count_buf: [MAX_GRAPH_SIZE]u32 = undefined;

    // Initialize each node with its own label
    for (0..n) |i| {
        labels[i] = @intCast(i);
    }

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        var changed = false;

        // Process each node
        for (0..n) |i| {
            const edge_start = offsets[i];
            const edge_end = offsets[i + 1];

            if (edge_start == edge_end) continue; // No neighbors

            // Count neighbor labels
            @memset(label_count_buf[0..n], 0);

            for (edge_start..edge_end) |j| {
                const neighbor = adjacency[j];
                if (neighbor < n) {
                    const label = labels[neighbor];
                    if (label < n) {
                        label_count_buf[label] += 1;
                    }
                }
            }

            // Find most frequent label
            var max_count: u32 = 0;
            var max_label: u32 = labels[i];
            for (0..n) |l| {
                if (label_count_buf[l] > max_count) {
                    max_count = label_count_buf[l];
                    max_label = @intCast(l);
                }
            }

            if (labels[i] != max_label) {
                labels[i] = max_label;
                changed = true;
            }
        }

        if (!changed) break;
    }

    return iter;
}


// ============================================================================
// C Export Functions with n_ prefix
// ============================================================================

pub export fn n_hungarian_solve(
    cost_matrix: [*]const f64,
    n: usize,
    m: usize,
    assignments: [*]i32,
) f64 {
    return hungarian_solve(cost_matrix, n, m, assignments);
}

pub export fn n_pagerank(
    adjacency: [*]const u32,
    offsets: [*]const u32,
    n: usize,
    damping: f64,
    max_iter: u32,
    result: [*]f64,
) void {
    pagerank(adjacency, offsets, n, damping, max_iter, result);
}

pub export fn n_dijkstra(
    adjacency: [*]const u32,
    weights: [*]const f64,
    offsets: [*]const u32,
    n: usize,
    source: usize,
    distances: [*]f64,
    predecessors: [*]i32,
) void {
    dijkstra(adjacency, weights, offsets, n, source, distances, predecessors);
}

pub export fn n_bellman_ford(
    edges: [*]const Edge,
    n_edges: usize,
    n_vertices: usize,
    source: usize,
    distances: [*]f64,
) i32 {
    return bellman_ford(edges, n_edges, n_vertices, source, distances);
}

pub export fn n_bfs(
    adjacency: [*]const u32,
    offsets: [*]const u32,
    n: usize,
    start: usize,
    visited: [*]u8,
    order: [*]i32,
) usize {
    return bfs(adjacency, offsets, n, start, visited, order);
}

pub export fn n_dfs(
    adjacency: [*]const u32,
    offsets: [*]const u32,
    n: usize,
    start: usize,
    visited: [*]u8,
    order: [*]i32,
) usize {
    return dfs(adjacency, offsets, n, start, visited, order);
}

pub export fn n_label_propagation(
    adjacency: [*]const u32,
    offsets: [*]const u32,
    n: usize,
    labels: [*]u32,
    max_iter: u32,
) u32 {
    return label_propagation(adjacency, offsets, n, labels, max_iter);
}

// ============================================================================
// Tests
// ============================================================================

test "hungarian algorithm - basic assignment" {
    // 3x3 cost matrix
    const cost = [_]f64{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    var assignments: [3]i32 = undefined;

    const total = hungarian_solve(&cost, 3, 3, &assignments);

    // Optimal assignment should assign row 0 to col 2, row 1 to col 1, row 2 to col 0
    // or some permutation with same total cost = 3 + 5 + 7 = 15
    try std.testing.expect(total <= 15.0 + 1e-9);
    try std.testing.expect(assignments[0] >= 0 and assignments[0] < 3);
    try std.testing.expect(assignments[1] >= 0 and assignments[1] < 3);
    try std.testing.expect(assignments[2] >= 0 and assignments[2] < 3);
}

test "dijkstra - simple path" {
    // Graph: 0 -> 1 (weight 1), 1 -> 2 (weight 2), 0 -> 2 (weight 4)
    const adjacency = [_]u32{ 1, 2, 2 };
    const weights = [_]f64{ 1.0, 4.0, 2.0 };
    const offsets = [_]u32{ 0, 2, 3, 3 };

    var distances: [3]f64 = undefined;
    var predecessors: [3]i32 = undefined;

    dijkstra(&adjacency, &weights, &offsets, 3, 0, &distances, &predecessors);

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), distances[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), distances[1], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), distances[2], 1e-9);
}

test "bellman_ford - with negative weight" {
    const edges = [_]Edge{
        .{ .src = 0, .dst = 1, .weight = 4.0 },
        .{ .src = 0, .dst = 2, .weight = 5.0 },
        .{ .src = 1, .dst = 2, .weight = -3.0 },
    };

    var distances: [3]f64 = undefined;
    const result = bellman_ford(&edges, 3, 3, 0, &distances);

    try std.testing.expectEqual(@as(i32, 0), result);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), distances[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), distances[1], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), distances[2], 1e-9);
}

test "bfs - traversal order" {
    // Graph: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    const adjacency = [_]u32{ 1, 2, 3, 3 };
    const offsets = [_]u32{ 0, 2, 3, 4, 4 };

    var visited: [4]u8 = undefined;
    var order: [4]i32 = undefined;

    const count = bfs(&adjacency, &offsets, 4, 0, &visited, &order);

    try std.testing.expectEqual(@as(usize, 4), count);
    try std.testing.expectEqual(@as(i32, 0), order[0]); // Start node first
    try std.testing.expect(order[1] == 1 or order[1] == 2); // Level 1
    try std.testing.expect(order[2] == 1 or order[2] == 2);
    try std.testing.expectEqual(@as(i32, 3), order[3]); // Level 2
}

test "dfs - traversal order" {
    // Graph: 0 -> 1, 0 -> 2, 1 -> 3
    const adjacency = [_]u32{ 1, 2, 3 };
    const offsets = [_]u32{ 0, 2, 3, 3, 3 };

    var visited: [4]u8 = undefined;
    var order: [4]i32 = undefined;

    const count = dfs(&adjacency, &offsets, 4, 0, &visited, &order);

    try std.testing.expectEqual(@as(usize, 4), count);
    try std.testing.expectEqual(@as(i32, 0), order[0]); // Start node first
}

test "pagerank - simple graph" {
    // Simple graph: 0 -> 1, 1 -> 2, 2 -> 0
    const adjacency = [_]u32{ 1, 2, 0 };
    const offsets = [_]u32{ 0, 1, 2, 3 };

    var result: [3]f64 = undefined;

    pagerank(&adjacency, &offsets, 3, 0.85, 100, &result);

    // All nodes should have similar PageRank in a cycle
    const total = result[0] + result[1] + result[2];
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), total, 1e-6);
}

test "label_propagation - two communities" {
    // Two disconnected components: {0, 1} and {2, 3}
    const adjacency = [_]u32{ 1, 0, 3, 2 };
    const offsets = [_]u32{ 0, 1, 2, 3, 4 };

    var labels: [4]u32 = undefined;

    _ = label_propagation(&adjacency, &offsets, 4, &labels, 10);

    // Nodes 0 and 1 should have same label, nodes 2 and 3 should have same label
    try std.testing.expectEqual(labels[0], labels[1]);
    try std.testing.expectEqual(labels[2], labels[3]);
}
