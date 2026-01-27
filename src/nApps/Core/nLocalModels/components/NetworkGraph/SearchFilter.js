/**
 * SearchFilter - Node search and filtering
 * Real-time search with highlighting and filtering
 */

import { GraphNode } from './GraphNode';
import { GraphEdge } from './GraphEdge';

export interface FilterCriteria {
    searchText?: string;
    nodeTypes?: string[];
    statuses?: string[];
    minLatency?: number;
    maxLatency?: number;
    minSuccessRate?: number;
}

export class SearchFilter {
    private nodes: GraphNode[] = [];
    private edges: GraphEdge[] = [];
    private filteredNodeIds: Set<string> = new Set();
    private highlightedNodeIds: Set<string> = new Set();
    
    // Callbacks
    private onFilterChange: ((filteredIds: Set<string>) => void) | null = null;
    private onHighlightChange: ((highlightedIds: Set<string>) => void) | null = null;
    
    // ========================================================================
    // Data Update
    // ========================================================================
    
    public setNodes(nodes: GraphNode[]): void {
        this.nodes = nodes;
    }
    
    public setEdges(edges: GraphEdge[]): void {
        this.edges = edges;
    }
    
    // ========================================================================
    // Search
    // ========================================================================
    
    public search(query: string): Set<string> {
        const results = new Set<string>();
        
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
            if (
                node.name.toLowerCase().includes(lowerQuery) ||
                node.description.toLowerCase().includes(lowerQuery) ||
                node.type.toLowerCase().includes(lowerQuery) ||
                node.model.toLowerCase().includes(lowerQuery)
            ) {
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
    
    public searchByType(type: string): Set<string> {
        const results = new Set<string>();
        
        for (const node of this.nodes) {
            if (node.type === type) {
                results.add(node.id);
            }
        }
        
        return results;
    }
    
    public searchByStatus(status: string): Set<string> {
        const results = new Set<string>();
        
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
    
    public applyFilter(criteria: FilterCriteria): Set<string> {
        const filtered = new Set<string>();
        
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
    
    private matchesCriteria(node: GraphNode, criteria: FilterCriteria): boolean {
        // Search text
        if (criteria.searchText) {
            const query = criteria.searchText.toLowerCase();
            const matches = 
                node.name.toLowerCase().includes(query) ||
                node.description.toLowerCase().includes(query) ||
                node.type.toLowerCase().includes(query) ||
                node.model.toLowerCase().includes(query);
            
            if (!matches) return false;
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
    
    public clearFilter(): void {
        this.filteredNodeIds.clear();
        
        if (this.onFilterChange) {
            this.onFilterChange(this.filteredNodeIds);
        }
    }
    
    // ========================================================================
    // Path Finding
    // ========================================================================
    
    public findPath(fromId: string, toId: string): string[] {
        // BFS to find shortest path
        const queue: string[] = [fromId];
        const visited = new Set<string>([fromId]);
        const parent = new Map<string, string>();
        
        while (queue.length > 0) {
            const current = queue.shift()!;
            
            if (current === toId) {
                // Reconstruct path
                const path: string[] = [];
                let node: string | undefined = toId;
                
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
        
        return [];  // No path found
    }
    
    public highlightPath(path: string[]): void {
        this.highlightedNodeIds = new Set(path);
        
        if (this.onHighlightChange) {
            this.onHighlightChange(this.highlightedNodeIds);
        }
    }
    
    // ========================================================================
    // Neighbors
    // ========================================================================
    
    public getNeighbors(nodeId: string, depth: number = 1): Set<string> {
        const neighbors = new Set<string>();
        const queue: Array<{ id: string; depth: number }> = [{ id: nodeId, depth: 0 }];
        const visited = new Set<string>([nodeId]);
        
        while (queue.length > 0) {
            const current = queue.shift()!;
            
            if (current.depth >= depth) continue;
            
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
    
    public focusOnNode(nodeId: string, includeNeighbors: boolean = true): Set<string> {
        const focused = new Set<string>([nodeId]);
        
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
    
    public getNodesByType(): Map<string, number> {
        const counts = new Map<string, number>();
        
        for (const node of this.nodes) {
            counts.set(node.type, (counts.get(node.type) || 0) + 1);
        }
        
        return counts;
    }
    
    public getNodesByStatus(): Map<string, number> {
        const counts = new Map<string, number>();
        
        for (const node of this.nodes) {
            counts.set(node.status, (counts.get(node.status) || 0) + 1);
        }
        
        return counts;
    }
    
    public getAverageMetrics(): any {
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
    
    public onFilter(callback: (filteredIds: Set<string>) => void): void {
        this.onFilterChange = callback;
    }
    
    public onHighlight(callback: (highlightedIds: Set<string>) => void): void {
        this.onHighlightChange = callback;
    }
    
    // ========================================================================
    // Getters
    // ========================================================================
    
    public getFilteredNodes(): Set<string> {
        return new Set(this.filteredNodeIds);
    }
    
    public getHighlightedNodes(): Set<string> {
        return new Set(this.highlightedNodeIds);
    }
}
