/**
 * ============================================================================
 * Lineage Graph Controller
 * SCIP-based code-to-data lineage visualization
 * ============================================================================
 *
 * [CODE:file=LineageGraph.controller.js]
 * [CODE:module=controller]
 * [CODE:language=javascript]
 *
 * [ODPS:product=data-lineage]
 *
 * [VIEW:binding=LineageGraph.view.xml]
 *
 * [API:consumes=/api/v1/lineage]
 * [API:consumes=/api/v1/lineage/symbol/{id}]
 *
 * [RELATION:uses=CODE:ApiService.js]
 * [RELATION:uses=CODE:NetworkGraphControl.js]
 * [RELATION:displays=SCIP:lineage-graph]
 *
 * This controller displays the SCIP-based code lineage graph showing
 * relationships between ODPS products, Zig code, SQL tables, and APIs.
 */
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "trialbalance/service/ApiService"
], function (Controller, JSONModel, MessageToast, ApiService) {
    "use strict";

    return Controller.extend("trialbalance.controller.LineageGraph", {

        /**
         * Controller initialization
         */
        onInit: function () {
            // Initialize API service
            this._oApiService = new ApiService();
            
            // Create view model
            var oViewModel = new JSONModel({
                busy: true,
                graph: {
                    nodes: [],
                    edges: []
                },
                selectedNode: null,
                nodeDetails: null,
                filters: {
                    showODPS: true,
                    showCode: true,
                    showTables: true,
                    showAPIs: true
                },
                nodeTypes: [
                    { key: "odps", text: "ODPS Products", color: "#1a73e8" },
                    { key: "zig", text: "Zig Code", color: "#f4b400" },
                    { key: "table", text: "SQL Tables", color: "#0f9d58" },
                    { key: "api", text: "API Endpoints", color: "#db4437" }
                ],
                stats: {
                    totalNodes: 0,
                    totalEdges: 0,
                    odpsCount: 0,
                    codeCount: 0,
                    tableCount: 0,
                    apiCount: 0
                },
                lastUpdated: null
            });
            this.getView().setModel(oViewModel, "view");

            // Create lineage model for view bindings
            var oLineageModel = new JSONModel({
                graph: {
                    nodes: [],
                    edges: []
                },
                entries: [
                    { lineage_id: "LIN-001", source_dataset_id: "ACDOCA", source_hash: "abc123", target_dataset_id: "TB_TRIAL_BALANCE", target_hash: "def456", transformation: "Aggregation", quality_score: 95, record_count: 25000, transformation_timestamp: "2026-01-27 10:00:00" },
                    { lineage_id: "LIN-002", source_dataset_id: "TB_EXCHANGE_RATES", source_hash: "ghi789", target_dataset_id: "TB_TRIAL_BALANCE", target_hash: "def456", transformation: "FX Conversion", quality_score: 98, record_count: 500, transformation_timestamp: "2026-01-27 10:05:00" },
                    { lineage_id: "LIN-003", source_dataset_id: "TB_TRIAL_BALANCE", source_hash: "def456", target_dataset_id: "TB_VARIANCE_DETAILS", target_hash: "jkl012", transformation: "Variance Calc", quality_score: 92, record_count: 1200, transformation_timestamp: "2026-01-27 10:10:00" }
                ]
            });
            this.getView().setModel(oLineageModel, "lineage");

            // Load data when route is matched
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("lineageGraph").attachPatternMatched(this._onRouteMatched, this);
        },

        /**
         * Route matched handler
         * @private
         */
        _onRouteMatched: function () {
            this._loadLineageGraph();
        },

        /**
         * Load lineage graph from API
         * @private
         */
        _loadLineageGraph: function () {
            var that = this;
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/busy", true);
            
            this._oApiService.getLineageGraph()
                .then(function (oData) {
                    var aNodes = (oData.nodes || []).map(function (n) {
                        return {
                            id: n.id,
                            label: n.label || n.name,
                            type: n.type,
                            module: n.module || "",
                            file: n.file || "",
                            line: n.line || 0,
                            description: n.description || "",
                            color: that._getNodeColor(n.type)
                        };
                    });
                    
                    var aEdges = (oData.edges || []).map(function (e) {
                        return {
                            source: e.source,
                            target: e.target,
                            type: e.type || "relates_to",
                            label: e.label || ""
                        };
                    });
                    
                    oViewModel.setProperty("/graph", {
                        nodes: aNodes,
                        edges: aEdges
                    });
                    
                    // Calculate stats
                    oViewModel.setProperty("/stats", {
                        totalNodes: aNodes.length,
                        totalEdges: aEdges.length,
                        odpsCount: aNodes.filter(function(n) { return n.type === "odps"; }).length,
                        codeCount: aNodes.filter(function(n) { return n.type === "zig"; }).length,
                        tableCount: aNodes.filter(function(n) { return n.type === "table"; }).length,
                        apiCount: aNodes.filter(function(n) { return n.type === "api"; }).length
                    });
                    
                    oViewModel.setProperty("/lastUpdated", new Date());
                    oViewModel.setProperty("/busy", false);
                })
                .catch(function (oError) {
                    // Use static data
                    var oStaticGraph = that._getStaticGraph();
                    oViewModel.setProperty("/graph", oStaticGraph);
                    oViewModel.setProperty("/stats", {
                        totalNodes: oStaticGraph.nodes.length,
                        totalEdges: oStaticGraph.edges.length,
                        odpsCount: oStaticGraph.nodes.filter(function(n) { return n.type === "odps"; }).length,
                        codeCount: oStaticGraph.nodes.filter(function(n) { return n.type === "zig"; }).length,
                        tableCount: oStaticGraph.nodes.filter(function(n) { return n.type === "table"; }).length,
                        apiCount: oStaticGraph.nodes.filter(function(n) { return n.type === "api"; }).length
                    });
                    oViewModel.setProperty("/busy", false);
                });
        },

        /**
         * Get node color by type
         * @private
         */
        _getNodeColor: function (sType) {
            var oColors = {
                "odps": "#1a73e8",
                "zig": "#f4b400",
                "table": "#0f9d58",
                "api": "#db4437"
            };
            return oColors[sType] || "#666";
        },

        /**
         * Get static graph for offline mode
         * @private
         */
        _getStaticGraph: function () {
            return {
                nodes: [
                    // ODPS Products
                    { id: "odps:trial-balance-aggregated", label: "Trial Balance", type: "odps", color: "#1a73e8" },
                    { id: "odps:variances", label: "Variances", type: "odps", color: "#1a73e8" },
                    { id: "odps:exchange-rates", label: "Exchange Rates", type: "odps", color: "#1a73e8" },
                    
                    // Zig Code
                    { id: "zig:balance_engine", label: "balance_engine.zig", type: "zig", color: "#f4b400", file: "balance_engine.zig" },
                    { id: "zig:fx_converter", label: "fx_converter.zig", type: "zig", color: "#f4b400", file: "fx_converter.zig" },
                    { id: "zig:odps_api", label: "odps_api.zig", type: "zig", color: "#f4b400", file: "odps_api.zig" },
                    { id: "zig:trial_balance", label: "trial_balance.zig", type: "zig", color: "#f4b400", file: "trial_balance.zig" },
                    
                    // SQL Tables
                    { id: "table:TB_TRIAL_BALANCE", label: "TB_TRIAL_BALANCE", type: "table", color: "#0f9d58" },
                    { id: "table:TB_VARIANCE_DETAILS", label: "TB_VARIANCE_DETAILS", type: "table", color: "#0f9d58" },
                    { id: "table:TB_EXCHANGE_RATES", label: "TB_EXCHANGE_RATES", type: "table", color: "#0f9d58" },
                    
                    // API Endpoints
                    { id: "api:/trial-balance", label: "/api/v1/trial-balance", type: "api", color: "#db4437" },
                    { id: "api:/variances", label: "/api/v1/variances", type: "api", color: "#db4437" },
                    { id: "api:/exchange-rates", label: "/api/v1/exchange-rates", type: "api", color: "#db4437" }
                ],
                edges: [
                    // ODPS -> Code
                    { source: "odps:trial-balance-aggregated", target: "zig:balance_engine", type: "implemented_by" },
                    { source: "odps:variances", target: "zig:balance_engine", type: "implemented_by" },
                    { source: "odps:exchange-rates", target: "zig:fx_converter", type: "implemented_by" },
                    
                    // Code -> Tables
                    { source: "zig:balance_engine", target: "table:TB_TRIAL_BALANCE", type: "reads_writes" },
                    { source: "zig:balance_engine", target: "table:TB_VARIANCE_DETAILS", type: "writes" },
                    { source: "zig:fx_converter", target: "table:TB_EXCHANGE_RATES", type: "reads" },
                    
                    // Code -> API
                    { source: "zig:trial_balance", target: "api:/trial-balance", type: "exposes" },
                    { source: "zig:balance_engine", target: "api:/variances", type: "exposes" },
                    { source: "zig:fx_converter", target: "api:/exchange-rates", type: "exposes" },
                    
                    // Code -> Code
                    { source: "zig:trial_balance", target: "zig:balance_engine", type: "calls" },
                    { source: "zig:balance_engine", target: "zig:fx_converter", type: "calls" },
                    { source: "zig:odps_api", target: "zig:balance_engine", type: "calls" }
                ]
            };
        },

        /**
         * Handle node selection
         */
        onNodeSelect: function (oEvent) {
            var that = this;
            var oNode = oEvent.getParameter("node");
            
            if (!oNode) return;
            
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/selectedNode", oNode);
            
            // Load node details
            this._oApiService.getSymbolDetails(oNode.id)
                .then(function (oDetails) {
                    oViewModel.setProperty("/nodeDetails", oDetails);
                })
                .catch(function () {
                    oViewModel.setProperty("/nodeDetails", {
                        id: oNode.id,
                        type: oNode.type,
                        label: oNode.label,
                        description: "Details not available"
                    });
                });
        },

        /**
         * Toggle filter
         */
        onToggleFilter: function (oEvent) {
            var sType = oEvent.getSource().data("type");
            var oViewModel = this.getView().getModel("view");
            var bCurrent = oViewModel.getProperty("/filters/show" + sType.charAt(0).toUpperCase() + sType.slice(1));
            oViewModel.setProperty("/filters/show" + sType.charAt(0).toUpperCase() + sType.slice(1), !bCurrent);
            // Trigger graph re-render
            this._applyFilters();
        },

        /**
         * Apply filters to graph
         * @private
         */
        _applyFilters: function () {
            // Implementation would filter the graph based on selected types
            MessageToast.show("Filters applied");
        },

        /**
         * Zoom in
         */
        onZoomIn: function () {
            // Would interact with NetworkGraph control
            MessageToast.show("Zoom in");
        },

        /**
         * Zoom out
         */
        onZoomOut: function () {
            // Would interact with NetworkGraph control
            MessageToast.show("Zoom out");
        },

        /**
         * Fit to screen
         */
        onFitToScreen: function () {
            // Would interact with NetworkGraph control
            MessageToast.show("Fit to screen");
        },

        /**
         * Refresh graph
         */
        onRefresh: function () {
            this._loadLineageGraph();
            MessageToast.show("Graph refreshed");
        },

        /**
         * Navigate back
         */
        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        },

        /**
         * Handle visualization type change
         */
        onVizTypeChange: function (oEvent) {
            var sSelectedKey = oEvent.getParameter("item").getKey();
            MessageToast.show("Visualization type changed to: " + sSelectedKey);
        },

        /**
         * Toggle full screen mode
         */
        onFullScreen: function () {
            MessageToast.show("Full screen mode toggled");
        },

        /**
         * Export graph as SVG
         */
        onExportSVG: function () {
            MessageToast.show("Exporting lineage graph as SVG");
        },

        /**
         * Handle lineage item press
         */
        onLineageItemPress: function (oEvent) {
            var oItem = oEvent.getSource();
            var oContext = oItem.getBindingContext("lineage");
            var sLineageId = oContext.getProperty("lineage_id");
            MessageToast.show("Lineage item selected: " + sLineageId);
        },

        /**
         * Navigate to catalog view
         */
        onViewCatalog: function () {
            this.getOwnerComponent().getRouter().navTo("odpsCatalog");
        },

        /**
         * Navigate to quality view
         */
        onViewQuality: function () {
            this.getOwnerComponent().getRouter().navTo("qualityDashboard");
        }

    });
});