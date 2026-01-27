sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast"
], function (Controller, JSONModel, MessageToast) {
    "use strict";

    return Controller.extend("trialbalance.controller.LineageGraph", {

        onInit: function () {
            // Initialize view model
            const oViewModel = new JSONModel({
                vizType: "dag"
            });
            this.getView().setModel(oViewModel, "view");

            // Load lineage data
            this._loadLineageData();
        },

        /**
         * Load lineage data from ODPS API
         */
        _loadLineageData: function () {
            const sUrl = "/api/v1/lineage";
            
            fetch(sUrl)
                .then(response => response.json())
                .then(data => {
                    this._processLineageData(data);
                })
                .catch(error => {
                    // Load mock lineage data
                    this._loadMockLineageData();
                });
        },

        /**
         * Process lineage data for visualization
         */
        _processLineageData: function (data) {
            // Transform to graph format for NetworkGraphControl
            const graph = this._buildLineageGraph(data);
            
            const oModel = new JSONModel({
                graph: graph,
                entries: data.entries || []
            });
            this.getView().setModel(oModel, "lineage");
        },

        /**
         * Build graph structure from lineage entries
         */
        _buildLineageGraph: function (data) {
            const nodes = [];
            const links = [];
            const nodeMap = new Map();

            // Create nodes for each unique dataset
            if (data.entries) {
                data.entries.forEach(entry => {
                    if (!nodeMap.has(entry.source_dataset_id)) {
                        nodes.push({
                            id: entry.source_dataset_id,
                            label: entry.source_dataset_id,
                            group: this._getDatasetCategory(entry.source_dataset_id),
                            quality: entry.quality_score
                        });
                        nodeMap.set(entry.source_dataset_id, true);
                    }
                    
                    if (!nodeMap.has(entry.target_dataset_id)) {
                        nodes.push({
                            id: entry.target_dataset_id,
                            label: entry.target_dataset_id,
                            group: this._getDatasetCategory(entry.target_dataset_id),
                            quality: entry.quality_score
                        });
                        nodeMap.set(entry.target_dataset_id, true);
                    }
                    
                    // Create link
                    links.push({
                        source: entry.source_dataset_id,
                        target: entry.target_dataset_id,
                        label: entry.transformation,
                        value: entry.record_count
                    });
                });
            }

            return { nodes, links };
        },

        /**
         * Get dataset category for coloring
         */
        _getDatasetCategory: function (datasetId) {
            if (datasetId.includes("ACDOCA")) return "source";
            if (datasetId.includes("trial-balance")) return "derived";
            if (datasetId.includes("variance")) return "analytical";
            return "other";
        },

        /**
         * Load mock lineage data
         */
        _loadMockLineageData: function () {
            const aMockLineage = [
                {
                    lineage_id: "550e8400-e29b-41d4-a716-446655440001",
                    source_dataset_id: "ACDOCA_RAW",
                    target_dataset_id: "ACDOCA_TABLE",
                    source_hash: "a1b2c3d4e5f6...",
                    target_hash: "a1b2c3d4e5f6...",
                    transformation: "extract",
                    transformation_timestamp: Date.now() - 3600000,
                    quality_score: 95.0,
                    record_count: 50000
                },
                {
                    lineage_id: "550e8400-e29b-41d4-a716-446655440002",
                    source_dataset_id: "ACDOCA_TABLE",
                    target_dataset_id: "TRIAL_BALANCE_AGG",
                    source_hash: "b2c3d4e5f6g7...",
                    target_hash: "c3d4e5f6g7h8...",
                    transformation: "aggregate",
                    transformation_timestamp: Date.now() - 1800000,
                    quality_score: 92.0,
                    record_count: 5000
                },
                {
                    lineage_id: "550e8400-e29b-41d4-a716-446655440003",
                    source_dataset_id: "TRIAL_BALANCE_AGG",
                    target_dataset_id: "VARIANCES",
                    source_hash: "d4e5f6g7h8i9...",
                    target_hash: "e5f6g7h8i9j0...",
                    transformation: "calculate",
                    transformation_timestamp: Date.now() - 900000,
                    quality_score: 90.0,
                    record_count: 500
                }
            ];

            const graph = this._buildLineageGraph({ entries: aMockLineage });
            
            const oModel = new JSONModel({
                graph: graph,
                entries: aMockLineage
            });
            this.getView().setModel(oModel, "lineage");
        },

        /**
         * Change visualization type
         */
        onVizTypeChange: function (oEvent) {
            const sKey = oEvent.getParameter("item").getKey();
            MessageToast.show(`Switched to ${sKey.toUpperCase()} visualization`);
            
            // Update graph control visualization
            const oGraph = this.byId("lineageGraph");
            if (oGraph && oGraph.setVisualizationType) {
                oGraph.setVisualizationType(sKey);
            }
        },

        /**
         * Full screen mode
         */
        onFullScreen: function () {
            const oGraph = this.byId("lineageGraph");
            if (oGraph && oGraph.getDomRef()) {
                const elem = oGraph.getDomRef();
                if (elem.requestFullscreen) {
                    elem.requestFullscreen();
                }
            }
        },

        /**
         * Export SVG
         */
        onExportSVG: function () {
            const oGraph = this.byId("lineageGraph");
            if (oGraph && oGraph.exportSVG) {
                oGraph.exportSVG("lineage-graph.svg");
                MessageToast.show("Lineage graph exported");
            } else {
                MessageToast.show("Export not yet implemented");
            }
        },

        /**
         * Lineage item pressed
         */
        onLineageItemPress: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("lineage");
            const oItem = oContext.getObject();
            
            MessageBox.information(
                `Lineage Entry\n\n` +
                `ID: ${oItem.lineage_id}\n` +
                `Source: ${oItem.source_dataset_id}\n` +
                `Target: ${oItem.target_dataset_id}\n` +
                `Transformation: ${oItem.transformation}\n` +
                `Quality: ${oItem.quality_score}%\n` +
                `Records: ${oItem.record_count.toLocaleString()}`,
                {
                    title: "Lineage Details"
                }
            );
        },

        /**
         * Navigate to catalog
         */
        onViewCatalog: function () {
            this.getOwnerComponent().getRouter().navTo("odpsCatalog");
        },

        /**
         * Navigate to quality dashboard
         */
        onViewQuality: function () {
            this.getOwnerComponent().getRouter().navTo("qualityDashboard");
        },

        /**
         * Navigate back
         */
        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        }
    });
});