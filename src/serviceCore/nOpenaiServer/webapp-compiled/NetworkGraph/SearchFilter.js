/**
 * SearchFilter - Node search and filtering
 * Real-time search with highlighting and filtering
 */
export class SearchFilter {
    constructor() {
        this.nodes = [];
        this.edges = [];
        this.filteredNodeIds = new Set();
        this.highlightedNodeIds = new Set();
        // Callbacks
        this.onFilterChange = null;
        this.onHighlightChange = null;
    }
    // ========================================================================
    // Data Update
    // ========================================================================
    setNodes(nodes) {
        this.nodes = nodes;
    }
    setEdges(edges) {
        this.edges = edges;
    }
    // ========================================================================
    // Search
    // ========================================================================
    search(query) {
        const results = new Set();
        if (!query || query.trim() === '') {
            // Empty query - return all nodes
            this.highlightedNodeIds.clear();
            if (this.onHighlightChange) {
                this.onHighlightChange(this.highlightedNodeIds);
            }
            return results;
        }
        const lowerQuery = query.toLowerCase();
        // Search in node name, description, type, model
        for (const node of this.nodes) {
            if (node.name.toLowerCase().includes(lowerQuery) ||
                node.description.toLowerCase().includes(lowerQuery) ||
                node.type.toLowerCase().includes(lowerQuery) ||
                node.model.toLowerCase().includes(lowerQuery)) {
                results.add(node.id);
            }
        }
        // Update highlights
        this.highlightedNodeIds = results;
        if (this.onHighlightChange) {
            this.onHighlightChange(this.highlightedNodeIds);
        }
        return results;
    }
    searchByType(type) {
        const results = new Set();
        for (const node of this.nodes) {
            if (node.type === type) {
                results.add(node.id);
            }
        }
        return results;
    }
    searchByStatus(status) {
        const results = new Set();
        for (const node of this.nodes) {
            if (node.status === status) {
                results.add(node.id);
            }
        }
        return results;
    }
    // ========================================================================
    // Filtering
    // ========================================================================
    applyFilter(criteria) {
        const filtered = new Set();
        for (const node of this.nodes) {
            if (this.matchesCriteria(node, criteria)) {
                filtered.add(node.id);
            }
        }
        this.filteredNodeIds = filtered;
        if (this.onFilterChange) {
            this.onFilterChange(this.filteredNodeIds);
        }
        return filtered;
    }
    matchesCriteria(node, criteria) {
        // Search text
        if (criteria.searchText) {
            const query = criteria.searchText.toLowerCase();
            const matches = node.name.toLowerCase().includes(query) ||
                node.description.toLowerCase().includes(query) ||
                node.type.toLowerCase().includes(query) ||
                node.model.toLowerCase().includes(query);
            if (!matches)
                return false;
        }
        // Node types
        if (criteria.nodeTypes && criteria.nodeTypes.length > 0) {
            if (!criteria.nodeTypes.includes(node.type)) {
                return false;
            }
        }
        // Statuses
        if (criteria.statuses && criteria.statuses.length > 0) {
            if (!criteria.statuses.includes(node.status)) {
                return false;
            }
        }
        // Latency range
        if (criteria.minLatency !== undefined) {
            if (node.metrics.avgLatency < criteria.minLatency) {
                return false;
            }
        }
        if (criteria.maxLatency !== undefined) {
            if (node.metrics.avgLatency > criteria.maxLatency) {
                return false;
            }
        }
        // Success rate
        if (criteria.minSuccessRate !== undefined) {
            if (node.metrics.successRate < criteria.minSuccessRate) {
                return false;
            }
        }
        return true;
    }
    clearFilter() {
        this.filteredNodeIds.clear();
        if (this.onFilterChange) {
            this.onFilterChange(this.filteredNodeIds);
        }
    }
    // ========================================================================
    // Path Finding
    // ========================================================================
    findPath(fromId, toId) {
        // BFS to find shortest path
        const queue = [fromId];
        const visited = new Set([fromId]);
        const parent = new Map();
        while (queue.length > 0) {
            const current = queue.shift();
            if (current === toId) {
                // Reconstruct path
                const path = [];
                let node = toId;
                while (node !== undefined) {
                    path.unshift(node);
                    node = parent.get(node);
                }
                return path;
            }
            // Find neighbors
            for (const edge of this.edges) {
                if (edge.from === current && !visited.has(edge.to)) {
                    visited.add(edge.to);
                    parent.set(edge.to, current);
                    queue.push(edge.to);
                }
            }
        }
        return []; // No path found
    }
    highlightPath(path) {
        this.highlightedNodeIds = new Set(path);
        if (this.onHighlightChange) {
            this.onHighlightChange(this.highlightedNodeIds);
        }
    }
    // ========================================================================
    // Neighbors
    // ========================================================================
    getNeighbors(nodeId, depth = 1) {
        const neighbors = new Set();
        const queue = [{ id: nodeId, depth: 0 }];
        const visited = new Set([nodeId]);
        while (queue.length > 0) {
            const current = queue.shift();
            if (current.depth >= depth)
                continue;
            // Find connected nodes
            for (const edge of this.edges) {
                if (edge.from === current.id && !visited.has(edge.to)) {
                    neighbors.add(edge.to);
                    visited.add(edge.to);
                    queue.push({ id: edge.to, depth: current.depth + 1 });
                }
                if (edge.to === current.id && !visited.has(edge.from)) {
                    neighbors.add(edge.from);
                    visited.add(edge.from);
                    queue.push({ id: edge.from, depth: current.depth + 1 });
                }
            }
        }
        return neighbors;
    }
    focusOnNode(nodeId, includeNeighbors = true) {
        const focused = new Set([nodeId]);
        if (includeNeighbors) {
            const neighbors = this.getNeighbors(nodeId, 1);
            for (const neighbor of neighbors) {
                focused.add(neighbor);
            }
        }
        this.filteredNodeIds = focused;
        if (this.onFilterChange) {
            this.onFilterChange(this.filteredNodeIds);
        }
        return focused;
    }
    // ========================================================================
    // Statistics
    // ========================================================================
    getNodesByType() {
        const counts = new Map();
        for (const node of this.nodes) {
            counts.set(node.type, (counts.get(node.type) || 0) + 1);
        }
        return counts;
    }
    getNodesByStatus() {
        const counts = new Map();
        for (const node of this.nodes) {
            counts.set(node.status, (counts.get(node.status) || 0) + 1);
        }
        return counts;
    }
    getAverageMetrics() {
        if (this.nodes.length === 0) {
            return { avgLatency: 0, avgSuccessRate: 0, totalRequests: 0 };
        }
        let totalLatency = 0;
        let totalSuccessRate = 0;
        let totalRequests = 0;
        for (const node of this.nodes) {
            totalLatency += node.metrics.avgLatency;
            totalSuccessRate += node.metrics.successRate;
            totalRequests += node.metrics.totalRequests;
        }
        return {
            avgLatency: totalLatency / this.nodes.length,
            avgSuccessRate: totalSuccessRate / this.nodes.length,
            totalRequests
        };
    }
    // ========================================================================
    // Callbacks
    // ========================================================================
    onFilter(callback) {
        this.onFilterChange = callback;
    }
    onHighlight(callback) {
        this.onHighlightChange = callback;
    }
    // ========================================================================
    // Getters
    // ========================================================================
    getFilteredNodes() {
        return new Set(this.filteredNodeIds);
    }
    getHighlightedNodes() {
        return new Set(this.highlightedNodeIds);
    }
}
//# sourceMappingURL=SearchFilter.js.map