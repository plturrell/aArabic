sap.ui.define([
    "serviceCore/nWorkflow/model/NodeTypes"
], function (NodeTypes) {
    "use strict";

    /**
     * WorkflowCanvas - Wrapper for JointJS functionality
     */
    var WorkflowCanvas = function (oConfig) {
        this._container = oConfig.container;
        this._onNodeSelect = oConfig.onNodeSelect || function () {};
        this._onNodeDeselect = oConfig.onNodeDeselect || function () {};
        this._onConnectionCreate = oConfig.onConnectionCreate || function () {};
        this._graph = null;
        this._paper = null;
        this._selectedCell = null;
        this._undoStack = [];
        this._redoStack = [];
        this._zoom = 1;
    };

    /**
     * Initialize JointJS graph and paper
     */
    WorkflowCanvas.prototype.init = function () {
        var that = this;
        
        // Create graph
        this._graph = new joint.dia.Graph();
        
        // Create paper
        this._paper = new joint.dia.Paper({
            el: this._container,
            model: this._graph,
            width: "100%",
            height: "100%",
            gridSize: 20,
            drawGrid: { name: "mesh", args: { color: "#e0e0e0" } },
            background: { color: "#fafafa" },
            interactive: { linkMove: true, elementMove: true },
            defaultLink: function () {
                return new joint.shapes.standard.Link({
                    attrs: {
                        line: {
                            stroke: "#666",
                            strokeWidth: 2,
                            targetMarker: { type: "path", d: "M 10 -5 0 0 10 5 z" }
                        }
                    },
                    router: { name: "manhattan" },
                    connector: { name: "rounded", args: { radius: 10 } }
                });
            },
            validateConnection: function (cellViewS, magnetS, cellViewT, magnetT) {
                // Prevent linking from input ports and to output ports
                if (magnetS && magnetS.getAttribute("port-group") === "in") return false;
                if (magnetT && magnetT.getAttribute("port-group") === "out") return false;
                // Prevent linking to self
                if (cellViewS === cellViewT) return false;
                return true;
            },
            snapLinks: { radius: 30 },
            markAvailable: true
        });

        // Handle cell selection
        this._paper.on("cell:pointerclick", function (cellView) {
            that._selectCell(cellView.model);
        });

        // Handle blank click (deselect)
        this._paper.on("blank:pointerclick", function () {
            that._deselectCell();
        });

        // Handle link connection
        this._graph.on("add", function (cell) {
            if (cell.isLink()) {
                that._onConnectionCreate(cell.toJSON());
            }
        });

        // Track changes for undo/redo
        this._graph.on("change", function () {
            that._saveState();
        });
    };

    /**
     * Create a node on the canvas
     */
    WorkflowCanvas.prototype.createNode = function (sType, oPosition) {
        var oNodeDef = NodeTypes.getNodeType(sType);
        if (!oNodeDef) {
            console.error("Unknown node type:", sType);
            return null;
        }

        var nodeId = "node_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
        
        // Build ports configuration
        var aPorts = [];
        (oNodeDef.inputs || []).forEach(function (port) {
            aPorts.push({
                id: port.id,
                group: "in",
                attrs: { label: { text: port.name } }
            });
        });
        (oNodeDef.outputs || []).forEach(function (port) {
            aPorts.push({
                id: port.id,
                group: "out",
                attrs: { label: { text: port.name } }
            });
        });

        // Create the node element
        var oNode = new joint.shapes.standard.Rectangle({
            id: nodeId,
            position: oPosition || { x: 100, y: 100 },
            size: { width: 140, height: 60 },
            attrs: {
                body: {
                    fill: oNodeDef.color,
                    stroke: "#333",
                    strokeWidth: 2,
                    rx: 8,
                    ry: 8
                },
                label: {
                    text: oNodeDef.name,
                    fill: "#fff",
                    fontSize: 14,
                    fontWeight: "bold"
                }
            },
            ports: {
                groups: {
                    in: {
                        position: "left",
                        attrs: {
                            circle: { magnet: "passive", r: 8, fill: "#fff", stroke: "#333" },
                            label: { fontSize: 10 }
                        }
                    },
                    out: {
                        position: "right",
                        attrs: {
                            circle: { magnet: true, r: 8, fill: "#fff", stroke: "#333" },
                            label: { fontSize: 10 }
                        }
                    }
                },
                items: aPorts
            },
            nodeType: sType,
            nodeData: this._getDefaultNodeData(oNodeDef)
        });

        this._graph.addCell(oNode);
        return oNode;
    };

    /**
     * Get default node data from definition
     */
    WorkflowCanvas.prototype._getDefaultNodeData = function (oNodeDef) {
        var oData = {};
        (oNodeDef.properties || []).forEach(function (prop) {
            oData[prop.id] = prop.default !== undefined ? prop.default : "";
        });
        return oData;
    };

    /**
     * Create a connection between two nodes
     */
    WorkflowCanvas.prototype.createConnection = function (sSourceId, sSourcePort, sTargetId, sTargetPort) {
        var oLink = new joint.shapes.standard.Link({
            source: { id: sSourceId, port: sSourcePort },
            target: { id: sTargetId, port: sTargetPort },
            attrs: {
                line: {
                    stroke: "#666",
                    strokeWidth: 2,
                    targetMarker: { type: "path", d: "M 10 -5 0 0 10 5 z" }
                }
            },
            router: { name: "manhattan" },
            connector: { name: "rounded", args: { radius: 10 } }
        });
        this._graph.addCell(oLink);
        return oLink;
    };

    /**
     * Select a cell
     */
    WorkflowCanvas.prototype._selectCell = function (oCell) {
        this._deselectCell();
        this._selectedCell = oCell;

        if (oCell.isElement()) {
            oCell.attr("body/strokeWidth", 4);
            oCell.attr("body/stroke", "#2196F3");
            this._onNodeSelect(oCell.id, oCell.get("nodeType"), oCell.get("nodeData"));
        }
    };

    /**
     * Deselect current cell
     */
    WorkflowCanvas.prototype._deselectCell = function () {
        if (this._selectedCell && this._selectedCell.isElement()) {
            this._selectedCell.attr("body/strokeWidth", 2);
            this._selectedCell.attr("body/stroke", "#333");
        }
        this._selectedCell = null;
        this._onNodeDeselect();
    };

    /**
     * Update node data
     */
    WorkflowCanvas.prototype.updateNodeData = function (sNodeId, oData) {
        var oCell = this._graph.getCell(sNodeId);
        if (oCell) {
            oCell.set("nodeData", oData);
        }
    };

    /**
     * Delete selected element
     */
    WorkflowCanvas.prototype.deleteSelected = function () {
        if (this._selectedCell) {
            this._selectedCell.remove();
            this._selectedCell = null;
            this._onNodeDeselect();
        }
    };

    /**
     * Get workflow as JSON
     */
    WorkflowCanvas.prototype.getWorkflowJSON = function () {
        var aNodes = [];
        var aConnections = [];

        this._graph.getCells().forEach(function (cell) {
            if (cell.isElement()) {
                aNodes.push({
                    id: cell.id,
                    type: cell.get("nodeType"),
                    position: cell.position(),
                    data: cell.get("nodeData")
                });
            } else if (cell.isLink()) {
                aConnections.push({
                    id: cell.id,
                    source: cell.get("source"),
                    target: cell.get("target")
                });
            }
        });

        return {
            nodes: aNodes,
            connections: aConnections
        };
    };

    /**
     * Load workflow from JSON
     */
    WorkflowCanvas.prototype.loadWorkflowJSON = function (oWorkflow) {
        this._graph.clear();

        var that = this;

        // Create nodes
        (oWorkflow.nodes || []).forEach(function (nodeData) {
            var oNode = that.createNode(nodeData.type, nodeData.position);
            if (oNode && nodeData.data) {
                oNode.set("nodeData", nodeData.data);
            }
        });

        // Create connections
        (oWorkflow.connections || []).forEach(function (conn) {
            if (conn.source && conn.target) {
                that.createConnection(
                    conn.source.id, conn.source.port,
                    conn.target.id, conn.target.port
                );
            }
        });
    };

    /**
     * Zoom in/out
     */
    WorkflowCanvas.prototype.zoom = function (fFactor) {
        this._zoom *= fFactor;
        this._zoom = Math.max(0.25, Math.min(2, this._zoom));
        this._paper.scale(this._zoom, this._zoom);
    };

    /**
     * Set zoom level
     */
    WorkflowCanvas.prototype.setZoom = function (fZoom) {
        this._zoom = Math.max(0.25, Math.min(2, fZoom));
        this._paper.scale(this._zoom, this._zoom);
    };

    /**
     * Get current zoom
     */
    WorkflowCanvas.prototype.getZoom = function () {
        return this._zoom;
    };

    /**
     * Fit content in view
     */
    WorkflowCanvas.prototype.fitContent = function () {
        this._paper.scaleContentToFit({
            padding: 50,
            maxScale: 1.5,
            minScale: 0.5
        });
        this._zoom = this._paper.scale().sx;
    };

    /**
     * Save state for undo
     */
    WorkflowCanvas.prototype._saveState = function () {
        if (this._undoStack.length > 50) {
            this._undoStack.shift();
        }
        this._undoStack.push(this._graph.toJSON());
        this._redoStack = [];
    };

    /**
     * Undo last action
     */
    WorkflowCanvas.prototype.undo = function () {
        if (this._undoStack.length > 1) {
            this._redoStack.push(this._undoStack.pop());
            var oState = this._undoStack[this._undoStack.length - 1];
            this._graph.fromJSON(oState);
        }
    };

    /**
     * Redo last undone action
     */
    WorkflowCanvas.prototype.redo = function () {
        if (this._redoStack.length > 0) {
            var oState = this._redoStack.pop();
            this._undoStack.push(oState);
            this._graph.fromJSON(oState);
        }
    };

    /**
     * Clear the canvas
     */
    WorkflowCanvas.prototype.clear = function () {
        this._graph.clear();
        this._selectedCell = null;
    };

    /**
     * Destroy the canvas
     */
    WorkflowCanvas.prototype.destroy = function () {
        if (this._paper) {
            this._paper.remove();
        }
        this._graph = null;
        this._paper = null;
    };

    return WorkflowCanvas;
});

