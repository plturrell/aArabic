/**
 * LayoutEngine - Graph layout algorithms
 * Force-directed, hierarchical, circular, and grid layouts
 */
import { LayoutType, DEFAULT_FORCE_CONFIG } from './types';
export class LayoutEngine {
    constructor(config) {
        this.nodes = [];
        this.edges = [];
        this.alpha = 1.0; // Simulation heat
        this.alphaDecay = 0.95;
        this.animationFrame = null;
        this.config = {
            type: LayoutType.ForceDirected,
            animate: true,
            duration: 1000,
            forces: DEFAULT_FORCE_CONFIG,
            padding: 50,
            ...config
        };
    }
    // ========================================================================
    // Public API
    // ========================================================================
    setNodes(nodes) {
        this.nodes = nodes;
    }
    setEdges(edges) {
        this.edges = edges;
    }
    setConfig(config) {
        this.config = { ...this.config, ...config };
    }
    start() {
        this.alpha = 1.0;
        switch (this.config.type) {
            case LayoutType.ForceDirected:
                this.startForceDirectedLayout();
                break;
            case LayoutType.Hierarchical:
                this.applyHierarchicalLayout();
                break;
            case LayoutType.Circular:
                this.applyCircularLayout();
                break;
            case LayoutType.Grid:
                this.applyGridLayout();
                break;
            case LayoutType.Manual:
                // User-controlled, no automatic layout
                break;
        }
    }
    stop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        this.alpha = 0;
    }
    tick() {
        if (this.alpha <= 0.01) {
            this.stop();
            return;
        }
        if (this.config.type === LayoutType.ForceDirected) {
            this.tickForceDirected();
        }
        this.alpha *= this.alphaDecay;
    }
    // ========================================================================
    // Force-Directed Layout (Fruchterman-Reingold algorithm)
    // ========================================================================
    startForceDirectedLayout() {
        if (!this.config.animate) {
            // Run simulation to completion
            while (this.alpha > 0.01) {
                this.tickForceDirected();
                this.alpha *= this.alphaDecay;
            }
            return;
        }
        // Animated simulation
        const animate = () => {
            this.tickForceDirected();
            this.alpha *= this.alphaDecay;
            if (this.alpha > 0.01) {
                this.animationFrame = requestAnimationFrame(animate);
            }
        };
        animate();
    }
    tickForceDirected() {
        const forces = this.config.forces || DEFAULT_FORCE_CONFIG;
        // Reset forces
        for (const node of this.nodes) {
            node.force.x = 0;
            node.force.y = 0;
        }
        // Apply repulsion between all node pairs (O(n²))
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                this.applyRepulsionForce(this.nodes[i], this.nodes[j], forces.repulsion);
            }
        }
        // Apply attraction along edges (O(e))
        for (const edge of this.edges) {
            const source = this.nodes.find(n => n.id === edge.from);
            const target = this.nodes.find(n => n.id === edge.to);
            if (source && target) {
                this.applyAttractionForce(source, target, forces.attraction);
            }
        }
        // Apply gravity toward center (O(n))
        const centerX = 0;
        const centerY = 0;
        for (const node of this.nodes) {
            this.applyGravityForce(node, centerX, centerY, forces.gravity);
        }
        // Update positions with damping
        for (const node of this.nodes) {
            node.updatePhysics(this.alpha);
        }
        // Update edge paths
        for (const edge of this.edges) {
            edge.updatePath();
        }
    }
    applyRepulsionForce(node1, node2, strength) {
        const dx = node2.position.x - node1.position.x;
        const dy = node2.position.y - node1.position.y;
        const distanceSquared = dx * dx + dy * dy;
        if (distanceSquared === 0)
            return;
        // Coulomb's law: F = k / r²
        const distance = Math.sqrt(distanceSquared);
        const force = strength / distanceSquared;
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        // Apply equal and opposite forces
        node1.applyForce(-fx, -fy);
        node2.applyForce(fx, fy);
    }
    applyAttractionForce(source, target, strength) {
        const dx = target.position.x - source.position.x;
        const dy = target.position.y - source.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance === 0)
            return;
        // Hooke's law: F = k * r
        const force = distance * strength;
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        source.applyForce(fx, fy);
        target.applyForce(-fx, -fy);
    }
    applyGravityForce(node, cx, cy, strength) {
        const dx = cx - node.position.x;
        const dy = cy - node.position.y;
        const fx = dx * strength;
        const fy = dy * strength;
        node.applyForce(fx, fy);
    }
    // ========================================================================
    // Hierarchical Layout (Sugiyama algorithm - simplified)
    // ========================================================================
    applyHierarchicalLayout() {
        if (this.nodes.length === 0)
            return;
        // Assign nodes to layers based on longest path
        const layers = this.assignLayers();
        // Position nodes in each layer
        const layerSpacing = 150;
        const nodeSpacing = 100;
        let y = -((layers.length - 1) * layerSpacing) / 2;
        for (const layer of layers) {
            let x = -((layer.length - 1) * nodeSpacing) / 2;
            for (const node of layer) {
                node.position.x = x;
                node.position.y = y;
                node.fixed = false; // Allow fine-tuning with physics
                x += nodeSpacing;
            }
            y += layerSpacing;
        }
        // Update edge paths
        for (const edge of this.edges) {
            edge.updatePath();
        }
        // Animate to positions
        if (this.config.animate) {
            this.animateToPositions(this.config.duration);
        }
    }
    assignLayers() {
        // Simple layer assignment: BFS from nodes with no incoming edges
        const inDegree = new Map();
        const layers = [];
        // Calculate in-degrees
        for (const node of this.nodes) {
            inDegree.set(node.id, 0);
        }
        for (const edge of this.edges) {
            const degree = inDegree.get(edge.to) || 0;
            inDegree.set(edge.to, degree + 1);
        }
        // Start with nodes that have no incoming edges
        const queue = this.nodes.filter(n => inDegree.get(n.id) === 0);
        const assigned = new Set();
        while (queue.length > 0) {
            const layer = [...queue];
            layers.push(layer);
            const nextQueue = [];
            for (const node of layer) {
                assigned.add(node.id);
                // Find outgoing edges
                for (const edge of this.edges) {
                    if (edge.from === node.id) {
                        const target = this.nodes.find(n => n.id === edge.to);
                        if (target && !assigned.has(target.id) && !nextQueue.includes(target)) {
                            nextQueue.push(target);
                        }
                    }
                }
            }
            queue.length = 0;
            queue.push(...nextQueue);
        }
        // Add any remaining nodes (cycles) to last layer
        const unassigned = this.nodes.filter(n => !assigned.has(n.id));
        if (unassigned.length > 0) {
            layers.push(unassigned);
        }
        return layers;
    }
    // ========================================================================
    // Circular Layout
    // ========================================================================
    applyCircularLayout() {
        if (this.nodes.length === 0)
            return;
        const radius = Math.max(200, this.nodes.length * 20);
        const angleStep = (2 * Math.PI) / this.nodes.length;
        this.nodes.forEach((node, i) => {
            const angle = i * angleStep;
            node.position.x = radius * Math.cos(angle);
            node.position.y = radius * Math.sin(angle);
            node.fixed = false;
        });
        // Update edge paths
        for (const edge of this.edges) {
            edge.updatePath();
        }
        if (this.config.animate) {
            this.animateToPositions(this.config.duration);
        }
    }
    // ========================================================================
    // Grid Layout
    // ========================================================================
    applyGridLayout() {
        if (this.nodes.length === 0)
            return;
        const cols = Math.ceil(Math.sqrt(this.nodes.length));
        const spacing = 150;
        this.nodes.forEach((node, i) => {
            const row = Math.floor(i / cols);
            const col = i % cols;
            node.position.x = (col - cols / 2) * spacing;
            node.position.y = (row - cols / 2) * spacing;
            node.fixed = false;
        });
        // Update edge paths
        for (const edge of this.edges) {
            edge.updatePath();
        }
        if (this.config.animate) {
            this.animateToPositions(this.config.duration);
        }
    }
    // ========================================================================
    // Animation Helpers
    // ========================================================================
    animateToPositions(duration) {
        // Store start positions
        const startPositions = this.nodes.map(n => ({ ...n.position }));
        const startTime = Date.now();
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1.0);
            const eased = this.easeInOutCubic(progress);
            this.nodes.forEach((node, i) => {
                const start = startPositions[i];
                // Lerp to current position
                node.position.x = start.x + (node.position.x - start.x) * eased;
                node.position.y = start.y + (node.position.y - start.y) * eased;
                node.updatePosition();
            });
            for (const edge of this.edges) {
                edge.updatePath();
            }
            if (progress < 1.0) {
                requestAnimationFrame(animate);
            }
        };
        animate();
    }
    easeInOutCubic(t) {
        return t < 0.5
            ? 4 * t * t * t
            : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
    // ========================================================================
    // Utilities
    // ========================================================================
    centerGraph() {
        if (this.nodes.length === 0)
            return;
        // Calculate bounding box
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        for (const node of this.nodes) {
            minX = Math.min(minX, node.position.x);
            maxX = Math.max(maxX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxY = Math.max(maxY, node.position.y);
        }
        // Calculate center offset
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        // Shift all nodes
        for (const node of this.nodes) {
            node.position.x -= centerX;
            node.position.y -= centerY;
            node.updatePosition();
        }
        // Update edges
        for (const edge of this.edges) {
            edge.updatePath();
        }
    }
    fitToViewport(width, height) {
        if (this.nodes.length === 0)
            return 1.0;
        // Calculate bounding box
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        for (const node of this.nodes) {
            minX = Math.min(minX, node.position.x - node.radius);
            maxX = Math.max(maxX, node.position.x + node.radius);
            minY = Math.min(minY, node.position.y - node.radius);
            maxY = Math.max(maxY, node.position.y + node.radius);
        }
        const graphWidth = maxX - minX;
        const graphHeight = maxY - minY;
        // Calculate scale to fit
        const padding = this.config.padding;
        const scaleX = (width - 2 * padding) / graphWidth;
        const scaleY = (height - 2 * padding) / graphHeight;
        return Math.min(scaleX, scaleY, 2.0); // Max 2x zoom
    }
}
//# sourceMappingURL=LayoutEngine.js.map