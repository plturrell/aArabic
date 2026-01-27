sap.ui.define([
    "trialbalance/extensions/ComponentExtension",
    "sap/base/Log"
], function(ComponentExtension, Log) {
    "use strict";

    /**
     * Enhanced Network Graph Extension
     * Example extension that enhances the NetworkGraphControl with additional features
     * 
     * Features demonstrated:
     * - Data transformation (adding calculated metrics)
     * - Custom rendering enhancements
     * - User interaction handling
     * - Backend integration
     * 
     * @class
     * @extends trialbalance.extensions.ComponentExtension
     */
    return ComponentExtension.extend("trialbalance.extensions.examples.EnhancedNetworkGraph", {
        
        constructor: function(mSettings) {
            ComponentExtension.call(this, mSettings);
            
            // Extension-specific configuration
            this.setId("enhanced-network-graph");
            this.setName("Enhanced Network Graph");
            this.setVersion("1.0.0");
            this.setTargetComponents(["trialbalance.control.NetworkGraphControl"]);
            this.setPriority(10);
            
            this._metricsCache = new Map();
        },

        /**
         * Initialize the extension
         * Load any required resources or perform setup
         */
        init: function() {
            Log.info("Initializing EnhancedNetworkGraph extension", "trialbalance.extensions.examples");
            
            // Example: Load configuration from backend
            return this._loadConfiguration().then(function(config) {
                this._config = config;
                Log.debug("Configuration loaded", "trialbalance.extensions.examples");
            }.bind(this));
        },

        /**
         * Called before extending a component
         * Set up any component-specific resources
         */
        onBeforeExtend: function(oComponent) {
            Log.debug("Extending NetworkGraph component: " + oComponent.getId(), 
                     "trialbalance.extensions.examples");
            
            // Store reference to component
            this._component = oComponent;
            
            // Add custom CSS class for styling
            oComponent.addStyleClass("enhanced-network-graph");
        },

        /**
         * Called after extending a component
         * Finalize setup and enable features
         */
        onAfterExtend: function(oComponent) {
            Log.debug("NetworkGraph component extended successfully", 
                     "trialbalance.extensions.examples");
            
            // Example: Register custom event handlers
            this._registerCustomHandlers(oComponent);
        },

        /**
         * Transform data received by the component
         * Add calculated metrics and enhancements
         */
        onDataReceived: function(vData, mContext) {
            if (!vData || mContext.property !== "nodes") {
                return vData;
            }

            Log.debug("Transforming network graph data", "trialbalance.extensions.examples");

            // Example: Add calculated metrics to each node
            if (Array.isArray(vData)) {
                return vData.map(function(node) {
                    return Object.assign({}, node, {
                        // Add calculated metric
                        centrality: this._calculateCentrality(node, vData),
                        // Add risk indicator
                        riskLevel: this._calculateRiskLevel(node),
                        // Add visual enhancement flag
                        enhanced: true
                    });
                }.bind(this));
            }

            return vData;
        },

        /**
         * Transform data before rendering
         * Apply visual enhancements
         */
        onBeforeRender: function(vData, mContext) {
            Log.debug("Applying pre-render transformations", "trialbalance.extensions.examples");
            
            // Example: Apply layout algorithm
            if (vData && this._config && this._config.autoLayout) {
                // Would apply force-directed layout, etc.
            }
            
            return vData;
        },

        /**
         * Called after component rendering
         * Add interactive features
         */
        onAfterRender: function(oComponent) {
            Log.debug("Applying post-render enhancements", "trialbalance.extensions.examples");
            
            // Example: Add tooltips, animations, etc.
            this._enhanceVisuals(oComponent);
        },

        /**
         * Handle user actions
         * Intercept clicks and add custom behavior
         */
        onUserAction: function(sAction, mParams) {
            Log.debug("User action: " + sAction, "trialbalance.extensions.examples");
            
            if (sAction === "nodeClick") {
                // Example: Load detailed data from backend
                this._loadNodeDetails(mParams.nodeId).then(function(details) {
                    // Show enhanced details dialog
                    this._showEnhancedDetails(details);
                }.bind(this));
                
                // Allow default action to proceed
                return true;
            }
            
            return true;
        },

        /**
         * Handle errors
         * Provide graceful degradation
         */
        onError: function(oError, mContext) {
            Log.error("Error in EnhancedNetworkGraph: " + oError.message, 
                     oError.stack, "trialbalance.extensions.examples");
            
            // Example: Fallback to basic rendering
            if (this._component) {
                this._component.removeStyleClass("enhanced-network-graph");
            }
        },

        // ========== Private Helper Methods ==========

        /**
         * Load configuration from backend
         * @private
         */
        _loadConfiguration: function() {
            // Example: Fetch configuration from backend extension API
            return fetch('/api/v1/extensions/enhanced-network-graph/config')
                .then(function(response) { return response.json(); })
                .catch(function() {
                    // Return default configuration
                    return {
                        autoLayout: true,
                        showMetrics: true,
                        animationEnabled: true
                    };
                });
        },

        /**
         * Calculate node centrality metric
         * @private
         */
        _calculateCentrality: function(node, allNodes) {
            // Simple degree centrality calculation
            // In real implementation, would use proper graph algorithms
            const nodeId = node.id;
            let connections = 0;
            
            allNodes.forEach(function(n) {
                if (n.connections && n.connections.includes(nodeId)) {
                    connections++;
                }
            });
            
            return connections / Math.max(allNodes.length - 1, 1);
        },

        /**
         * Calculate risk level for a node
         * @private
         */
        _calculateRiskLevel: function(node) {
            // Example: Calculate risk based on balance variance
            const balance = parseFloat(node.balance) || 0;
            const variance = parseFloat(node.variance) || 0;
            
            if (variance === 0) return "low";
            
            const variancePercent = Math.abs(variance / balance) * 100;
            
            if (variancePercent > 20) return "high";
            if (variancePercent > 10) return "medium";
            return "low";
        },

        /**
         * Register custom event handlers
         * @private
         */
        _registerCustomHandlers: function(oComponent) {
            // Example: Add custom click handler
            const domRef = oComponent.getDomRef();
            if (domRef) {
                domRef.addEventListener('click', function(event) {
                    // Handle click events
                    this.onUserAction('nodeClick', {
                        nodeId: event.target.dataset.nodeId
                    });
                }.bind(this));
            }
        },

        /**
         * Enhance visuals after rendering
         * @private
         */
        _enhanceVisuals: function(oComponent) {
            // Example: Add CSS animations, transitions
            const domRef = oComponent.getDomRef();
            if (domRef) {
                // Add animation class
                domRef.classList.add("animated");
                
                // Example: Add risk indicators
                const nodes = domRef.querySelectorAll('.graph-node');
                nodes.forEach(function(node) {
                    const riskLevel = node.dataset.riskLevel;
                    if (riskLevel) {
                        node.classList.add('risk-' + riskLevel);
                    }
                });
            }
        },

        /**
         * Load node details from backend
         * @private
         */
        _loadNodeDetails: function(nodeId) {
            // Check cache first
            if (this._metricsCache.has(nodeId)) {
                return Promise.resolve(this._metricsCache.get(nodeId));
            }
            
            // Fetch from backend
            return fetch('/api/v1/extensions/enhanced-network-graph/node/' + nodeId)
                .then(function(response) { return response.json(); })
                .then(function(details) {
                    this._metricsCache.set(nodeId, details);
                    return details;
                }.bind(this))
                .catch(function(error) {
                    Log.error("Error loading node details", error.message, "trialbalance.extensions.examples");
                    return { nodeId: nodeId, error: "Failed to load" };
                });
        },

        /**
         * Show enhanced details dialog
         * @private
         */
        _showEnhancedDetails: function(details) {
            // Example: Create and show a dialog with enhanced information
            Log.info("Showing enhanced details for node: " + details.nodeId, 
                    "trialbalance.extensions.examples");
            
            // In real implementation, would create sap.m.Dialog with rich content
        },

        /**
         * Cleanup on destroy
         */
        destroy: function() {
            this._metricsCache.clear();
            this._metricsCache = null;
            this._component = null;
            this._config = null;
            
            ComponentExtension.prototype.destroy.call(this);
        }
    });
});