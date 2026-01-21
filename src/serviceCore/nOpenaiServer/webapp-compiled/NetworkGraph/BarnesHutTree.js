/**
 * Barnes-Hut Quadtree
 * Optimizes force calculations from O(n²) to O(n log n)
 * Essential for graphs with 100+ nodes
 */
class QuadTreeNode {
    constructor(bounds) {
        this.centerOfMass = { x: 0, y: 0 };
        this.totalMass = 0;
        this.node = null;
        // Children (NW, NE, SW, SE)
        this.children = [];
        this.bounds = bounds;
    }
    isLeaf() {
        return this.children.every(c => !c);
    }
    isExternal() {
        return this.node !== null && this.isLeaf();
    }
    containsPoint(point) {
        return Math.abs(point.x - this.bounds.x) <= this.bounds.width &&
            Math.abs(point.y - this.bounds.y) <= this.bounds.height;
    }
}
export class BarnesHutTree {
    constructor(theta = 0.5) {
        this.root = null;
        this.theta = 0.5; // Barnes-Hut approximation parameter
        this.theta = theta;
    }
    // ========================================================================
    // Build Tree
    // ========================================================================
    build(nodes) {
        if (nodes.length === 0) {
            this.root = null;
            return;
        }
        // Calculate bounding box
        const bounds = this.calculateBounds(nodes);
        // Create root
        this.root = new QuadTreeNode(bounds);
        // Insert all nodes
        for (const node of nodes) {
            this.insert(this.root, node);
        }
        // Calculate centers of mass
        this.updateCentersOfMass(this.root);
    }
    calculateBounds(nodes) {
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        for (const node of nodes) {
            minX = Math.min(minX, node.position.x);
            maxX = Math.max(maxX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxY = Math.max(maxY, node.position.y);
        }
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const width = Math.max((maxX - minX) / 2, 100); // Min size
        const height = Math.max((maxY - minY) / 2, 100);
        return { x: centerX, y: centerY, width, height };
    }
    insert(quadNode, graphNode) {
        if (!quadNode.containsPoint(graphNode.position)) {
            return false;
        }
        // If this is an external node (has one node)
        if (quadNode.isExternal()) {
            // Subdivide and insert both nodes
            this.subdivide(quadNode);
            const existingNode = quadNode.node;
            quadNode.node = null;
            this.insertIntoChild(quadNode, existingNode);
            this.insertIntoChild(quadNode, graphNode);
            return true;
        }
        // If this is an internal node
        if (!quadNode.isLeaf()) {
            return this.insertIntoChild(quadNode, graphNode);
        }
        // If this is an empty leaf
        quadNode.node = graphNode;
        return true;
    }
    subdivide(quadNode) {
        const { x, y, width, height } = quadNode.bounds;
        const halfWidth = width / 2;
        const halfHeight = height / 2;
        // NW (0), NE (1), SW (2), SE (3)
        quadNode.children[0] = new QuadTreeNode({
            x: x - halfWidth / 2,
            y: y - halfHeight / 2,
            width: halfWidth,
            height: halfHeight
        });
        quadNode.children[1] = new QuadTreeNode({
            x: x + halfWidth / 2,
            y: y - halfHeight / 2,
            width: halfWidth,
            height: halfHeight
        });
        quadNode.children[2] = new QuadTreeNode({
            x: x - halfWidth / 2,
            y: y + halfHeight / 2,
            width: halfWidth,
            height: halfHeight
        });
        quadNode.children[3] = new QuadTreeNode({
            x: x + halfWidth / 2,
            y: y + halfHeight / 2,
            width: halfWidth,
            height: halfHeight
        });
    }
    insertIntoChild(quadNode, graphNode) {
        for (const child of quadNode.children) {
            if (child && this.insert(child, graphNode)) {
                return true;
            }
        }
        return false;
    }
    updateCentersOfMass(quadNode) {
        if (!quadNode)
            return;
        // Leaf with node
        if (quadNode.isExternal() && quadNode.node) {
            quadNode.centerOfMass = { ...quadNode.node.position };
            quadNode.totalMass = quadNode.node.mass;
            return;
        }
        // Internal node - aggregate from children
        let totalMass = 0;
        let weightedX = 0;
        let weightedY = 0;
        for (const child of quadNode.children) {
            if (child) {
                this.updateCentersOfMass(child);
                totalMass += child.totalMass;
                weightedX += child.centerOfMass.x * child.totalMass;
                weightedY += child.centerOfMass.y * child.totalMass;
            }
        }
        if (totalMass > 0) {
            quadNode.centerOfMass.x = weightedX / totalMass;
            quadNode.centerOfMass.y = weightedY / totalMass;
            quadNode.totalMass = totalMass;
        }
    }
    // ========================================================================
    // Force Calculation
    // ========================================================================
    calculateForce(node, repulsionStrength) {
        const force = { x: 0, y: 0 };
        if (!this.root)
            return force;
        this.calculateForceRecursive(this.root, node, repulsionStrength, force);
        return force;
    }
    calculateForceRecursive(quadNode, graphNode, strength, force) {
        // Skip if empty
        if (quadNode.totalMass === 0)
            return;
        const dx = quadNode.centerOfMass.x - graphNode.position.x;
        const dy = quadNode.centerOfMass.y - graphNode.position.y;
        const distanceSquared = dx * dx + dy * dy;
        // Barnes-Hut approximation criterion
        const size = quadNode.bounds.width * 2;
        const distance = Math.sqrt(distanceSquared);
        if (quadNode.isExternal() || (size / distance < this.theta)) {
            // Treat as single body (approximation)
            if (distanceSquared > 0 && quadNode.node !== graphNode) {
                const repulsion = strength * quadNode.totalMass / distanceSquared;
                force.x -= (dx / distance) * repulsion;
                force.y -= (dy / distance) * repulsion;
            }
        }
        else {
            // Recurse into children
            for (const child of quadNode.children) {
                if (child) {
                    this.calculateForceRecursive(child, graphNode, strength, force);
                }
            }
        }
    }
    // ========================================================================
    // Utilities
    // ========================================================================
    getDepth() {
        return this.root ? this.getDepthRecursive(this.root) : 0;
    }
    getDepthRecursive(node) {
        if (node.isLeaf())
            return 1;
        let maxDepth = 0;
        for (const child of node.children) {
            if (child) {
                maxDepth = Math.max(maxDepth, this.getDepthRecursive(child));
            }
        }
        return maxDepth + 1;
    }
    getNodeCount() {
        return this.root ? this.getNodeCountRecursive(this.root) : 0;
    }
    getNodeCountRecursive(node) {
        if (node.isExternal())
            return 1;
        let count = 0;
        for (const child of node.children) {
            if (child) {
                count += this.getNodeCountRecursive(child);
            }
        }
        return count;
    }
}
//# sourceMappingURL=BarnesHutTree.js.map