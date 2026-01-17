sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/format/DateFormat"
], function (Controller, MessageBox, MessageToast, DateFormat) {
    "use strict";

    return Controller.extend("hypershimmy.controller.Mindmap", {
        
        /**
         * Layout algorithm descriptions
         * @private
         */
        _layoutDescriptions: {
            tree: "Traditional hierarchical tree structure. Root at top, children spread horizontally per level.",
            radial: "Concentric circles around root. Children distributed in circles with increasing radius per level."
        },

        /**
         * SVG configuration
         * @private
         */
        _svgConfig: {
            nodeRadius: 30,
            nodeStrokeWidth: 2,
            edgeStrokeWidth: 2,
            fontSize: 12,
            labelOffset: 5,
            nodeColors: {
                root: "#0070f2",
                branch: "#1db954",
                leaf: "#ff9800"
            },
            edgeStyles: {
                solid: "none",
                dashed: "5,5",
                dotted: "2,2"
            }
        },

        /**
         * Pan/drag state
         * @private
         */
        _panState: {
            isPanning: false,
            startX: 0,
            startY: 0,
            offsetX: 0,
            offsetY: 0
        },

        /**
         * Collapsed nodes set
         * @private
         */
        _collapsedNodes: new Set(),

        /**
         * Called when the controller is instantiated
         */
        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("mindmap").attachPatternMatched(this._onRouteMatched, this);
            
            // Initialize mindmap settings
            this._initializeMindmapSettings();
            
            // Load saved settings
            this._loadMindmapSettings();
        },

        /**
         * Initialize default mindmap settings
         * @private
         */
        _initializeMindmapSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Set defaults if not already set
            if (!oAppStateModel.getProperty("/mindmapLayout")) {
                oAppStateModel.setProperty("/mindmapLayout", "tree");
            }
            if (!oAppStateModel.getProperty("/mindmapMaxDepth")) {
                oAppStateModel.setProperty("/mindmapMaxDepth", 5);
            }
            if (!oAppStateModel.getProperty("/mindmapMaxChildren")) {
                oAppStateModel.setProperty("/mindmapMaxChildren", 10);
            }
            if (!oAppStateModel.getProperty("/mindmapCanvasWidth")) {
                oAppStateModel.setProperty("/mindmapCanvasWidth", 1200);
            }
            if (!oAppStateModel.getProperty("/mindmapCanvasHeight")) {
                oAppStateModel.setProperty("/mindmapCanvasHeight", 800);
            }
            if (oAppStateModel.getProperty("/mindmapAutoSelectRoot") === undefined) {
                oAppStateModel.setProperty("/mindmapAutoSelectRoot", true);
            }
            if (oAppStateModel.getProperty("/mindmapConfigExpanded") === undefined) {
                oAppStateModel.setProperty("/mindmapConfigExpanded", true);
            }
            
            oAppStateModel.setProperty("/mindmapGenerated", false);
            oAppStateModel.setProperty("/mindmapZoom", 100);
            oAppStateModel.setProperty("/selectedNode", null);
            oAppStateModel.setProperty("/mindmapSearchQuery", "");
            oAppStateModel.setProperty("/mindmapFilteredNodes", []);
            
            // Set initial description
            this._updateLayoutDescription();
            
            // Initialize collapsed nodes
            this._collapsedNodes = new Set();
        },

        /**
         * Load mindmap settings from localStorage
         * @private
         */
        _loadMindmapSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            try {
                var sSettings = localStorage.getItem("hypershimmy.mindmapSettings");
                if (sSettings) {
                    var oSettings = JSON.parse(sSettings);
                    oAppStateModel.setProperty("/mindmapLayout", oSettings.layout || "tree");
                    oAppStateModel.setProperty("/mindmapMaxDepth", oSettings.maxDepth || 5);
                    oAppStateModel.setProperty("/mindmapMaxChildren", oSettings.maxChildren || 10);
                    oAppStateModel.setProperty("/mindmapCanvasWidth", oSettings.canvasWidth || 1200);
                    oAppStateModel.setProperty("/mindmapCanvasHeight", oSettings.canvasHeight || 800);
                    oAppStateModel.setProperty("/mindmapAutoSelectRoot", oSettings.autoSelectRoot !== false);
                    
                    this._updateLayoutDescription();
                }
            } catch (e) {
                console.error("Failed to load mindmap settings:", e);
            }
        },

        /**
         * Save mindmap settings to localStorage
         * @private
         */
        _saveMindmapSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            var oSettings = {
                layout: oAppStateModel.getProperty("/mindmapLayout") || "tree",
                maxDepth: oAppStateModel.getProperty("/mindmapMaxDepth") || 5,
                maxChildren: oAppStateModel.getProperty("/mindmapMaxChildren") || 10,
                canvasWidth: oAppStateModel.getProperty("/mindmapCanvasWidth") || 1200,
                canvasHeight: oAppStateModel.getProperty("/mindmapCanvasHeight") || 800,
                autoSelectRoot: oAppStateModel.getProperty("/mindmapAutoSelectRoot") !== false
            };
            
            try {
                localStorage.setItem("hypershimmy.mindmapSettings", JSON.stringify(oSettings));
            } catch (e) {
                console.error("Failed to save mindmap settings:", e);
            }
        },

        /**
         * Route matched handler
         * @param {sap.ui.base.Event} oEvent the route matched event
         * @private
         */
        _onRouteMatched: function (oEvent) {
            var sSourceId = oEvent.getParameter("arguments").sourceId;
            
            // Store current source ID
            this._currentSourceId = sSourceId;
            
            // Update app state
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            oAppStateModel.setProperty("/selectedSourceId", sSourceId);
            
            // Bind the view to the selected source
            var oView = this.getView();
            oView.bindElement({
                path: "/Sources('" + sSourceId + "')",
                parameters: {
                    $expand: "Mindmaps"
                }
            });
        },

        /**
         * Handler for layout change
         * @param {sap.ui.base.Event} oEvent the change event
         */
        onLayoutChange: function (oEvent) {
            this._updateLayoutDescription();
            this._saveMindmapSettings();
        },

        /**
         * Update layout description
         * @private
         */
        _updateLayoutDescription: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var sLayout = oAppStateModel.getProperty("/mindmapLayout");
            var sDescription = this._layoutDescriptions[sLayout] || "";
            
            oAppStateModel.setProperty("/mindmapLayoutDescription", sDescription);
        },

        /**
         * Handler for generate mindmap button
         */
        onGenerateMindmap: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Get source IDs (for now, just use current source)
            var aSourceIds = [this._currentSourceId];
            
            // Get configuration
            var sLayout = oAppStateModel.getProperty("/mindmapLayout");
            var iMaxDepth = oAppStateModel.getProperty("/mindmapMaxDepth");
            var iMaxChildren = oAppStateModel.getProperty("/mindmapMaxChildren");
            var iCanvasWidth = oAppStateModel.getProperty("/mindmapCanvasWidth");
            var iCanvasHeight = oAppStateModel.getProperty("/mindmapCanvasHeight");
            var bAutoSelectRoot = oAppStateModel.getProperty("/mindmapAutoSelectRoot");
            
            // Set busy state
            oAppStateModel.setProperty("/busy", true);
            oAppStateModel.setProperty("/mindmapGenerated", false);
            oAppStateModel.setProperty("/mindmapConfigExpanded", false);
            oAppStateModel.setProperty("/selectedNode", null);
            
            // Call OData Mindmap action
            this._callMindmapAction(
                aSourceIds,
                sLayout,
                iMaxDepth,
                iMaxChildren,
                iCanvasWidth,
                iCanvasHeight,
                bAutoSelectRoot
            )
                .then(function(oResponse) {
                    // Process and display mindmap
                    this._displayMindmap(oResponse);
                    
                    oAppStateModel.setProperty("/busy", false);
                    oAppStateModel.setProperty("/mindmapGenerated", true);
                    
                    // Save settings
                    this._saveMindmapSettings();
                    
                    MessageToast.show("Mindmap generated successfully");
                }.bind(this))
                .catch(function(oError) {
                    // Handle error
                    oAppStateModel.setProperty("/busy", false);
                    
                    var sErrorMessage = "Failed to generate mindmap. Please try again.";
                    if (oError.responseText) {
                        try {
                            var oErrorData = JSON.parse(oError.responseText);
                            if (oErrorData.error && oErrorData.error.message) {
                                sErrorMessage = oErrorData.error.message;
                            }
                        } catch (e) {
                            // Ignore JSON parse error
                        }
                    }
                    
                    MessageBox.error(sErrorMessage);
                }.bind(this));
        },

        /**
         * Call OData GenerateMindmap action
         * @param {array} aSourceIds array of source IDs
         * @param {string} sLayout layout algorithm
         * @param {number} iMaxDepth max depth
         * @param {number} iMaxChildren max children per node
         * @param {number} iCanvasWidth canvas width
         * @param {number} iCanvasHeight canvas height
         * @param {boolean} bAutoSelectRoot auto select root flag
         * @returns {Promise} promise that resolves with mindmap response
         * @private
         */
        _callMindmapAction: function(
            aSourceIds,
            sLayout,
            iMaxDepth,
            iMaxChildren,
            iCanvasWidth,
            iCanvasHeight,
            bAutoSelectRoot
        ) {
            return new Promise(function(resolve, reject) {
                // Prepare request payload
                var oPayload = {
                    SourceIds: aSourceIds,
                    LayoutAlgorithm: sLayout,
                    MaxDepth: iMaxDepth,
                    MaxChildrenPerNode: iMaxChildren,
                    CanvasWidth: iCanvasWidth,
                    CanvasHeight: iCanvasHeight,
                    AutoSelectRoot: bAutoSelectRoot
                };
                
                // Call OData action
                jQuery.ajax({
                    url: "/odata/v4/research/GenerateMindmap",
                    method: "POST",
                    contentType: "application/json",
                    data: JSON.stringify(oPayload),
                    success: function(oData) {
                        resolve(oData);
                    },
                    error: function(oError) {
                        reject(oError);
                    }
                });
            });
        },

        /**
         * Display mindmap in the UI
         * @param {object} oMindmap the mindmap response
         * @private
         */
        _displayMindmap: function (oMindmap) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Store current mindmap
            oAppStateModel.setProperty("/currentMindmap", oMindmap);
            
            // Set generated time
            var oDateFormat = DateFormat.getDateTimeInstance({
                pattern: "MMM dd, yyyy HH:mm:ss"
            });
            oAppStateModel.setProperty("/mindmapGeneratedTime", oDateFormat.format(new Date()));
            
            // Render mindmap SVG
            this._renderMindmapSVG(oMindmap);
            
            // Enable pan/drag
            this._enablePanDrag();
            
            // Render minimap
            this._renderMinimap(oMindmap);
        },

        /**
         * Render mindmap as SVG
         * @param {object} oMindmap the mindmap data
         * @private
         */
        _renderMindmapSVG: function (oMindmap) {
            var oContainer = document.getElementById("mindmapSvgContainer");
            if (!oContainer) {
                console.error("SVG container not found");
                return;
            }
            
            // Clear previous content
            oContainer.innerHTML = "";
            
            // Create SVG element
            var iWidth = oMindmap.Nodes.reduce(function(max, node) {
                return Math.max(max, node.X + 100);
            }, 0);
            var iHeight = oMindmap.Nodes.reduce(function(max, node) {
                return Math.max(max, node.Y + 100);
            }, 0);
            
            var sSvg = '<svg width="' + iWidth + '" height="' + iHeight + '" xmlns="http://www.w3.org/2000/svg">';
            
            // Add definitions for markers (arrows)
            sSvg += '<defs>';
            sSvg += '<marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">';
            sSvg += '<polygon points="0 0, 10 3, 0 6" fill="#666" />';
            sSvg += '</marker>';
            sSvg += '</defs>';
            
            // Render edges first (so they appear behind nodes)
            sSvg += this._renderEdges(oMindmap.Edges, oMindmap.Nodes);
            
            // Render nodes
            sSvg += this._renderNodes(oMindmap.Nodes);
            
            sSvg += '</svg>';
            
            // Inject SVG
            oContainer.innerHTML = sSvg;
            
            // Attach event listeners to nodes
            this._attachNodeEventListeners(oMindmap.Nodes);
            
            // Add animation
            this._animateNodes();
        },

        /**
         * Render edges as SVG
         * @param {array} aEdges array of edges
         * @param {array} aNodes array of nodes (for position lookup)
         * @returns {string} SVG markup for edges
         * @private
         */
        _renderEdges: function (aEdges, aNodes) {
            var sSvg = '';
            var oConfig = this._svgConfig;
            
            // Create node lookup map
            var mNodes = {};
            aNodes.forEach(function(oNode) {
                mNodes[oNode.Id] = oNode;
            });
            
            aEdges.forEach(function(oEdge) {
                var oFromNode = mNodes[oEdge.FromNodeId];
                var oToNode = mNodes[oEdge.ToNodeId];
                
                if (!oFromNode || !oToNode) {
                    return;
                }
                
                // Skip edges to collapsed nodes
                if (this._collapsedNodes.has(oToNode.Id)) {
                    return;
                }
                
                var iX1 = oFromNode.X;
                var iY1 = oFromNode.Y;
                var iX2 = oToNode.X;
                var iY2 = oToNode.Y;
                
                // Get edge style
                var sStrokeDasharray = oConfig.edgeStyles[oEdge.Style] || oConfig.edgeStyles.solid;
                
                sSvg += '<line ';
                sSvg += 'x1="' + iX1 + '" ';
                sSvg += 'y1="' + iY1 + '" ';
                sSvg += 'x2="' + iX2 + '" ';
                sSvg += 'y2="' + iY2 + '" ';
                sSvg += 'stroke="#666" ';
                sSvg += 'stroke-width="' + oConfig.edgeStrokeWidth + '" ';
                if (sStrokeDasharray !== "none") {
                    sSvg += 'stroke-dasharray="' + sStrokeDasharray + '" ';
                }
                sSvg += 'marker-end="url(#arrowhead)" ';
                sSvg += 'class="mindmap-edge" ';
                sSvg += '/>';
                
                // Add edge label if weight exists
                if (oEdge.Weight !== undefined && oEdge.Weight !== null) {
                    var iMidX = (iX1 + iX2) / 2;
                    var iMidY = (iY1 + iY2) / 2;
                    
                    sSvg += '<text ';
                    sSvg += 'x="' + iMidX + '" ';
                    sSvg += 'y="' + (iMidY - 5) + '" ';
                    sSvg += 'text-anchor="middle" ';
                    sSvg += 'font-size="10" ';
                    sSvg += 'fill="#999" ';
                    sSvg += 'class="edge-label" ';
                    sSvg += '>' + oEdge.Weight.toFixed(2) + '</text>';
                }
            }.bind(this));
            
            return sSvg;
        },

        /**
         * Render nodes as SVG
         * @param {array} aNodes array of nodes
         * @returns {string} SVG markup for nodes
         * @private
         */
        _renderNodes: function (aNodes) {
            var sSvg = '';
            var oConfig = this._svgConfig;
            
            aNodes.forEach(function(oNode) {
                // Skip collapsed nodes
                if (this._collapsedNodes.has(oNode.Id)) {
                    return;
                }
                
                var iX = oNode.X;
                var iY = oNode.Y;
                var sNodeType = oNode.NodeType || "leaf";
                var sColor = oConfig.nodeColors[sNodeType] || oConfig.nodeColors.leaf;
                
                // Check if node has children
                var bHasChildren = oNode.ChildCount > 0;
                
                // Render node circle
                sSvg += '<circle ';
                sSvg += 'cx="' + iX + '" ';
                sSvg += 'cy="' + iY + '" ';
                sSvg += 'r="' + oConfig.nodeRadius + '" ';
                sSvg += 'fill="' + sColor + '" ';
                sSvg += 'stroke="#fff" ';
                sSvg += 'stroke-width="' + oConfig.nodeStrokeWidth + '" ';
                sSvg += 'class="mindmap-node" ';
                sSvg += 'data-node-id="' + oNode.Id + '" ';
                sSvg += 'data-has-children="' + bHasChildren + '" ';
                sSvg += 'style="cursor:pointer;opacity:0;transition:opacity 0.3s ease;" ';
                sSvg += '/>';
                
                // Render expand/collapse indicator if has children
                if (bHasChildren) {
                    sSvg += '<circle ';
                    sSvg += 'cx="' + (iX + oConfig.nodeRadius - 5) + '" ';
                    sSvg += 'cy="' + (iY - oConfig.nodeRadius + 5) + '" ';
                    sSvg += 'r="8" ';
                    sSvg += 'fill="#fff" ';
                    sSvg += 'stroke="' + sColor + '" ';
                    sSvg += 'stroke-width="2" ';
                    sSvg += 'class="expand-indicator" ';
                    sSvg += 'data-node-id="' + oNode.Id + '" ';
                    sSvg += 'style="cursor:pointer;" ';
                    sSvg += '/>';
                    
                    // Plus/minus sign
                    var sSign = this._isNodeExpanded(oNode.Id) ? '-' : '+';
                    sSvg += '<text ';
                    sSvg += 'x="' + (iX + oConfig.nodeRadius - 5) + '" ';
                    sSvg += 'y="' + (iY - oConfig.nodeRadius + 9) + '" ';
                    sSvg += 'text-anchor="middle" ';
                    sSvg += 'font-size="12" ';
                    sSvg += 'font-weight="bold" ';
                    sSvg += 'fill="' + sColor + '" ';
                    sSvg += 'class="expand-sign" ';
                    sSvg += 'data-node-id="' + oNode.Id + '" ';
                    sSvg += 'style="pointer-events:none;" ';
                    sSvg += '>' + sSign + '</text>';
                }
                
                // Render node label
                var sLabel = oNode.Label;
                if (sLabel.length > 20) {
                    sLabel = sLabel.substring(0, 17) + "...";
                }
                
                sSvg += '<text ';
                sSvg += 'x="' + iX + '" ';
                sSvg += 'y="' + (iY + oConfig.nodeRadius + oConfig.fontSize + oConfig.labelOffset) + '" ';
                sSvg += 'text-anchor="middle" ';
                sSvg += 'font-size="' + oConfig.fontSize + '" ';
                sSvg += 'fill="#333" ';
                sSvg += 'font-family="Arial, sans-serif" ';
                sSvg += 'class="mindmap-label" ';
                sSvg += 'data-node-id="' + oNode.Id + '" ';
                sSvg += 'style="cursor:pointer;pointer-events:none;opacity:0;transition:opacity 0.3s ease;" ';
                sSvg += '>' + this._escapeHtml(sLabel) + '</text>';
            }.bind(this));
            
            return sSvg;
        },

        /**
         * Attach event listeners to node elements
         * @param {array} aNodes array of nodes
         * @private
         */
        _attachNodeEventListeners: function (aNodes) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Create node lookup map
            var mNodes = {};
            aNodes.forEach(function(oNode) {
                mNodes[oNode.Id] = oNode;
            });
            
            // Attach click handlers to all node circles
            var aNodeElements = document.querySelectorAll('.mindmap-node');
            aNodeElements.forEach(function(oElement) {
                var sNodeId = oElement.getAttribute('data-node-id');
                var oNode = mNodes[sNodeId];
                
                if (oNode) {
                    oElement.addEventListener('click', function() {
                        // Select node
                        oAppStateModel.setProperty("/selectedNode", oNode);
                        
                        // Highlight selected node
                        this._highlightNode(sNodeId);
                        
                        MessageToast.show("Node selected: " + oNode.Label);
                    }.bind(this));
                    
                    // Add hover effect
                    oElement.addEventListener('mouseenter', function(e) {
                        oElement.style.opacity = '0.8';
                        this._showTooltip(e, oNode);
                    }.bind(this));
                    
                    oElement.addEventListener('mouseleave', function() {
                        oElement.style.opacity = '1';
                        this._hideTooltip();
                    }.bind(this));
                }
            }.bind(this));
            
            // Attach expand/collapse handlers
            var aExpandIndicators = document.querySelectorAll('.expand-indicator');
            aExpandIndicators.forEach(function(oElement) {
                var sNodeId = oElement.getAttribute('data-node-id');
                
                oElement.addEventListener('click', function(e) {
                    e.stopPropagation();
                    this._toggleNodeExpansion(sNodeId);
                }.bind(this));
            }.bind(this));
        },

        /**
         * Highlight selected node
         * @param {string} sNodeId the node ID to highlight
         * @private
         */
        _highlightNode: function (sNodeId) {
            // Remove previous highlights
            var aNodes = document.querySelectorAll('.mindmap-node');
            aNodes.forEach(function(oNode) {
                oNode.style.strokeWidth = this._svgConfig.nodeStrokeWidth + 'px';
                oNode.style.stroke = '#fff';
            }.bind(this));
            
            // Highlight selected node
            var oSelectedNode = document.querySelector('.mindmap-node[data-node-id="' + sNodeId + '"]');
            if (oSelectedNode) {
                oSelectedNode.style.strokeWidth = '4px';
                oSelectedNode.style.stroke = '#ff0000';
            }
        },

        /**
         * Escape HTML characters
         * @param {string} sText the text to escape
         * @returns {string} escaped text
         * @private
         */
        _escapeHtml: function (sText) {
            var oDiv = document.createElement('div');
            oDiv.textContent = sText;
            return oDiv.innerHTML;
        },

        /**
         * Handler for zoom in button
         */
        onZoomIn: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var iZoom = oAppStateModel.getProperty("/mindmapZoom") || 100;
            
            iZoom = Math.min(iZoom + 10, 200);
            oAppStateModel.setProperty("/mindmapZoom", iZoom);
            
            this._applyZoom(iZoom);
        },

        /**
         * Handler for zoom out button
         */
        onZoomOut: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var iZoom = oAppStateModel.getProperty("/mindmapZoom") || 100;
            
            iZoom = Math.max(iZoom - 10, 50);
            oAppStateModel.setProperty("/mindmapZoom", iZoom);
            
            this._applyZoom(iZoom);
        },

        /**
         * Handler for reset zoom button
         */
        onResetZoom: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            oAppStateModel.setProperty("/mindmapZoom", 100);
            
            this._applyZoom(100);
        },

        /**
         * Apply zoom to SVG
         * @param {number} iZoom zoom percentage
         * @private
         */
        _applyZoom: function (iZoom) {
            var oContainer = document.getElementById("mindmapSvgContainer");
            if (oContainer) {
                var fScale = iZoom / 100;
                oContainer.style.transform = 'scale(' + fScale + ')';
                oContainer.style.transformOrigin = 'top left';
            }
        },

        /**
         * Handler for export mindmap button
         */
        onExportMindmap: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oMindmap = oAppStateModel.getProperty("/currentMindmap");
            
            if (!oMindmap) {
                MessageToast.show("No mindmap to export");
                return;
            }
            
            // Export as JSON
            var sExportData = JSON.stringify(oMindmap, null, 2);
            
            // Create download link
            var oBlob = new Blob([sExportData], { type: "application/json;charset=utf-8" });
            var sUrl = URL.createObjectURL(oBlob);
            var sFilename = "mindmap-" + oMindmap.LayoutAlgorithm + "-" + new Date().toISOString().split('T')[0] + ".json";
            
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = sFilename;
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            URL.revokeObjectURL(sUrl);
            
            MessageToast.show("Mindmap exported successfully");
        },

        /**
         * Handler for toggle fullscreen button
         */
        onToggleFullscreen: function () {
            var oScrollContainer = this.byId("mindmapScroll");
            if (!oScrollContainer) {
                return;
            }
            
            var oDomRef = oScrollContainer.getDomRef();
            
            if (!document.fullscreenElement) {
                if (oDomRef.requestFullscreen) {
                    oDomRef.requestFullscreen();
                    MessageToast.show("Entered fullscreen mode");
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                    MessageToast.show("Exited fullscreen mode");
                }
            }
        },

        /**
         * Handler for reset view button
         */
        onResetView: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Reset zoom
            oAppStateModel.setProperty("/mindmapZoom", 100);
            this._applyZoom(100);
            
            // Reset selected node
            oAppStateModel.setProperty("/selectedNode", null);
            
            // Remove highlights
            var aNodes = document.querySelectorAll('.mindmap-node');
            aNodes.forEach(function(oNode) {
                oNode.style.strokeWidth = this._svgConfig.nodeStrokeWidth + 'px';
                oNode.style.stroke = '#fff';
            }.bind(this));
            
            MessageToast.show("View reset");
        },

        /**
         * Handler for navigation back button
         */
        onNavBack: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var sSourceId = oAppStateModel.getProperty("/selectedSourceId");
            
            // Navigate back to detail view
            oRouter.navTo("detail", {
                sourceId: sSourceId
            });
        },

        /**
         * Check if node is expanded
         * @param {string} sNodeId the node ID
         * @returns {boolean} true if expanded
         * @private
         */
        _isNodeExpanded: function (sNodeId) {
            return !this._collapsedNodes.has(sNodeId);
        },

        /**
         * Toggle node expansion
         * @param {string} sNodeId the node ID
         * @private
         */
        _toggleNodeExpansion: function (sNodeId) {
            if (this._collapsedNodes.has(sNodeId)) {
                // Expand node
                this._collapsedNodes.delete(sNodeId);
                MessageToast.show("Node expanded");
            } else {
                // Collapse node
                this._collapsedNodes.add(sNodeId);
                MessageToast.show("Node collapsed");
            }
            
            // Re-render mindmap
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oMindmap = oAppStateModel.getProperty("/currentMindmap");
            if (oMindmap) {
                this._renderMindmapSVG(oMindmap);
                this._enablePanDrag();
            }
        },

        /**
         * Animate nodes on render
         * @private
         */
        _animateNodes: function () {
            // Fade in nodes with staggered delay
            var aNodes = document.querySelectorAll('.mindmap-node');
            var aLabels = document.querySelectorAll('.mindmap-label');
            
            aNodes.forEach(function(oNode, iIndex) {
                setTimeout(function() {
                    oNode.style.opacity = '1';
                }, iIndex * 30);
            });
            
            aLabels.forEach(function(oLabel, iIndex) {
                setTimeout(function() {
                    oLabel.style.opacity = '1';
                }, iIndex * 30);
            });
        },

        /**
         * Enable pan/drag functionality
         * @private
         */
        _enablePanDrag: function () {
            var oScrollContainer = document.getElementById("mindmapScroll");
            var oSvgContainer = document.getElementById("mindmapSvgContainer");
            
            if (!oScrollContainer || !oSvgContainer) {
                return;
            }
            
            var that = this;
            
            oScrollContainer.addEventListener('mousedown', function(e) {
                // Only pan with middle mouse button or ctrl+left button
                if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
                    e.preventDefault();
                    that._panState.isPanning = true;
                    that._panState.startX = e.clientX - that._panState.offsetX;
                    that._panState.startY = e.clientY - that._panState.offsetY;
                    oScrollContainer.style.cursor = 'grabbing';
                }
            });
            
            oScrollContainer.addEventListener('mousemove', function(e) {
                if (that._panState.isPanning) {
                    e.preventDefault();
                    that._panState.offsetX = e.clientX - that._panState.startX;
                    that._panState.offsetY = e.clientY - that._panState.startY;
                    
                    oScrollContainer.scrollLeft = -that._panState.offsetX;
                    oScrollContainer.scrollTop = -that._panState.offsetY;
                }
            });
            
            oScrollContainer.addEventListener('mouseup', function(e) {
                if (that._panState.isPanning) {
                    that._panState.isPanning = false;
                    oScrollContainer.style.cursor = 'default';
                }
            });
            
            oScrollContainer.addEventListener('mouseleave', function(e) {
                if (that._panState.isPanning) {
                    that._panState.isPanning = false;
                    oScrollContainer.style.cursor = 'default';
                }
            });
        },

        /**
         * Show tooltip for node
         * @param {Event} oEvent the mouse event
         * @param {object} oNode the node data
         * @private
         */
        _showTooltip: function (oEvent, oNode) {
            // Create tooltip if doesn't exist
            var oTooltip = document.getElementById("mindmapTooltip");
            if (!oTooltip) {
                oTooltip = document.createElement("div");
                oTooltip.id = "mindmapTooltip";
                oTooltip.className = "mindmap-tooltip";
                document.body.appendChild(oTooltip);
            }
            
            // Build tooltip content
            var sContent = '<div class="tooltip-header">' + this._escapeHtml(oNode.Label) + '</div>';
            sContent += '<div class="tooltip-body">';
            sContent += '<div><strong>Type:</strong> ' + (oNode.NodeType || 'N/A') + '</div>';
            sContent += '<div><strong>Entity:</strong> ' + (oNode.EntityType || 'N/A') + '</div>';
            sContent += '<div><strong>Level:</strong> ' + (oNode.Level || 0) + '</div>';
            sContent += '<div><strong>Children:</strong> ' + (oNode.ChildCount || 0) + '</div>';
            if (oNode.Confidence !== undefined) {
                sContent += '<div><strong>Confidence:</strong> ' + (oNode.Confidence * 100).toFixed(1) + '%</div>';
            }
            sContent += '</div>';
            
            oTooltip.innerHTML = sContent;
            
            // Position tooltip
            var iLeft = oEvent.pageX + 15;
            var iTop = oEvent.pageY + 15;
            
            oTooltip.style.left = iLeft + 'px';
            oTooltip.style.top = iTop + 'px';
            oTooltip.style.display = 'block';
        },

        /**
         * Hide tooltip
         * @private
         */
        _hideTooltip: function () {
            var oTooltip = document.getElementById("mindmapTooltip");
            if (oTooltip) {
                oTooltip.style.display = 'none';
            }
        },

        /**
         * Render minimap
         * @param {object} oMindmap the mindmap data
         * @private
         */
        _renderMinimap: function (oMindmap) {
            var oMinimapContainer = document.getElementById("mindmapMinimap");
            if (!oMinimapContainer) {
                return;
            }
            
            // Clear previous content
            oMinimapContainer.innerHTML = "";
            
            // Calculate minimap dimensions (1/10 scale)
            var iScale = 0.1;
            var iWidth = oMindmap.Nodes.reduce(function(max, node) {
                return Math.max(max, node.X * iScale + 10);
            }, 0);
            var iHeight = oMindmap.Nodes.reduce(function(max, node) {
                return Math.max(max, node.Y * iScale + 10);
            }, 0);
            
            var sSvg = '<svg width="' + iWidth + '" height="' + iHeight + '" xmlns="http://www.w3.org/2000/svg">';
            
            // Render minimap edges
            oMindmap.Edges.forEach(function(oEdge) {
                var oFromNode = oMindmap.Nodes.find(function(n) { return n.Id === oEdge.FromNodeId; });
                var oToNode = oMindmap.Nodes.find(function(n) { return n.Id === oEdge.ToNodeId; });
                
                if (oFromNode && oToNode && !this._collapsedNodes.has(oToNode.Id)) {
                    sSvg += '<line ';
                    sSvg += 'x1="' + (oFromNode.X * iScale) + '" ';
                    sSvg += 'y1="' + (oFromNode.Y * iScale) + '" ';
                    sSvg += 'x2="' + (oToNode.X * iScale) + '" ';
                    sSvg += 'y2="' + (oToNode.Y * iScale) + '" ';
                    sSvg += 'stroke="#ccc" stroke-width="1" />';
                }
            }.bind(this));
            
            // Render minimap nodes
            oMindmap.Nodes.forEach(function(oNode) {
                if (!this._collapsedNodes.has(oNode.Id)) {
                    var sNodeType = oNode.NodeType || "leaf";
                    var sColor = this._svgConfig.nodeColors[sNodeType];
                    
                    sSvg += '<circle ';
                    sSvg += 'cx="' + (oNode.X * iScale) + '" ';
                    sSvg += 'cy="' + (oNode.Y * iScale) + '" ';
                    sSvg += 'r="3" ';
                    sSvg += 'fill="' + sColor + '" />';
                }
            }.bind(this));
            
            sSvg += '</svg>';
            
            oMinimapContainer.innerHTML = sSvg;
        },

        /**
         * Handler for search input
         * @param {sap.ui.base.Event} oEvent the search event
         */
        onSearchNodes: function (oEvent) {
            var sQuery = oEvent.getParameter("query") || oEvent.getParameter("newValue") || "";
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oMindmap = oAppStateModel.getProperty("/currentMindmap");
            
            if (!oMindmap) {
                return;
            }
            
            oAppStateModel.setProperty("/mindmapSearchQuery", sQuery);
            
            if (!sQuery) {
                // Clear filter
                oAppStateModel.setProperty("/mindmapFilteredNodes", []);
                this._clearSearchHighlight();
                return;
            }
            
            // Filter nodes by label
            var aFilteredNodes = oMindmap.Nodes.filter(function(oNode) {
                return oNode.Label.toLowerCase().indexOf(sQuery.toLowerCase()) >= 0;
            });
            
            oAppStateModel.setProperty("/mindmapFilteredNodes", aFilteredNodes);
            
            // Highlight matching nodes
            this._highlightSearchResults(aFilteredNodes);
            
            MessageToast.show("Found " + aFilteredNodes.length + " matching nodes");
        },

        /**
         * Highlight search results
         * @param {array} aNodes array of matching nodes
         * @private
         */
        _highlightSearchResults: function (aNodes) {
            // Clear previous highlights
            this._clearSearchHighlight();
            
            // Highlight matching nodes
            aNodes.forEach(function(oNode) {
                var oElement = document.querySelector('.mindmap-node[data-node-id="' + oNode.Id + '"]');
                if (oElement) {
                    oElement.classList.add('search-highlight');
                }
            });
        },

        /**
         * Clear search highlight
         * @private
         */
        _clearSearchHighlight: function () {
            var aHighlighted = document.querySelectorAll('.search-highlight');
            aHighlighted.forEach(function(oElement) {
                oElement.classList.remove('search-highlight');
            });
        },

        /**
         * Handler for export to PNG
         */
        onExportPNG: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oMindmap = oAppStateModel.getProperty("/currentMindmap");
            
            if (!oMindmap) {
                MessageToast.show("No mindmap to export");
                return;
            }
            
            MessageBox.information("PNG export requires html2canvas library. This is a placeholder implementation.");
        },

        /**
         * Handler for export to SVG
         */
        onExportSVG: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oMindmap = oAppStateModel.getProperty("/currentMindmap");
            
            if (!oMindmap) {
                MessageToast.show("No mindmap to export");
                return;
            }
            
            // Get SVG element
            var oSvgContainer = document.getElementById("mindmapSvgContainer");
            if (!oSvgContainer) {
                return;
            }
            
            var oSvgElement = oSvgContainer.querySelector('svg');
            if (!oSvgElement) {
                return;
            }
            
            // Serialize SVG
            var oSerializer = new XMLSerializer();
            var sSvgString = oSerializer.serializeToString(oSvgElement);
            
            // Add XML declaration and styling
            var sSvgData = '<?xml version="1.0" encoding="UTF-8"?>\n';
            sSvgData += '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n';
            sSvgData += sSvgString;
            
            // Create download link
            var oBlob = new Blob([sSvgData], { type: "image/svg+xml;charset=utf-8" });
            var sUrl = URL.createObjectURL(oBlob);
            var sFilename = "mindmap-" + oMindmap.LayoutAlgorithm + "-" + new Date().toISOString().split('T')[0] + ".svg";
            
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = sFilename;
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            URL.revokeObjectURL(sUrl);
            
            MessageToast.show("Mindmap exported as SVG successfully");
        },

        /**
         * Handler for expand all nodes
         */
        onExpandAll: function () {
            this._collapsedNodes.clear();
            
            // Re-render mindmap
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oMindmap = oAppStateModel.getProperty("/currentMindmap");
            if (oMindmap) {
                this._renderMindmapSVG(oMindmap);
                this._enablePanDrag();
                this._renderMinimap(oMindmap);
            }
            
            MessageToast.show("All nodes expanded");
        },

        /**
         * Handler for collapse all nodes
         */
        onCollapseAll: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oMindmap = oAppStateModel.getProperty("/currentMindmap");
            
            if (!oMindmap) {
                return;
            }
            
            // Collapse all non-root nodes
            oMindmap.Nodes.forEach(function(oNode) {
                if (oNode.NodeType !== 'root' && oNode.Level > 1) {
                    this._collapsedNodes.add(oNode.Id);
                }
            }.bind(this));
            
            // Re-render mindmap
            this._renderMindmapSVG(oMindmap);
            this._enablePanDrag();
            this._renderMinimap(oMindmap);
            
            MessageToast.show("All nodes collapsed");
        }
    });
});
