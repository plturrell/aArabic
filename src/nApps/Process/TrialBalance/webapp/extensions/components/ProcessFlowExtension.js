sap.ui.define([
    "trialbalance/extensions/ComponentExtension",
    "sap/base/Log"
], function(ComponentExtension, Log) {
    "use strict";

    /**
     * ProcessFlow Extension
     * Wraps the TypeScript ProcessFlow component and integrates it with the extension framework.
     * Provides extension points for customizing behavior and connecting to the backend.
     * 
     * Features:
     * - Wraps existing ProcessFlow TypeScript component
     * - Provides extension hooks for data transformation
     * - Integrates with backend via extension API
     * - Supports semantic zoom levels (SAP Fiori standard)
     * - Path highlighting for search/filter
     * - Hot-reload and dynamic configuration
     * 
     * @class
     * @extends trialbalance.extensions.ComponentExtension
     */
    return ComponentExtension.extend("trialbalance.extensions.components.ProcessFlowExtension", {
        
        constructor: function(mSettings) {
            ComponentExtension.call(this, mSettings);
            
            // Extension metadata
            this.setId("process-flow-core");
            this.setName("Process Flow Core");
            this.setVersion("1.0.0");
            this.setTargetComponents(["trialbalance.control.ProcessFlowControl"]);
            this.setPriority(100); // High priority - core extension
            
            // Internal state
            this._processFlow = null;
            this._config = null;
            this._dataCache = new Map();
            this._eventHandlers = new Map();
            this._extensionManager = null;
            this._currentZoomLevel = "Two"; // Default SAP zoom level
        },

        /**
         * Initialize the extension
         * Load the ProcessFlow TypeScript component and configuration
         */
        init: function() {
            Log.info("Initializing ProcessFlow Extension", "trialbalance.extensions.components.ProcessFlowExtension");
            
            return Promise.all([
                this._loadProcessFlowScript(),
                this._loadConfiguration()
            ]).then(function() {
                Log.info("ProcessFlow Extension initialized successfully", 
                        "trialbalance.extensions.components.ProcessFlowExtension");
            }.bind(this));
        },

        /**
         * Set the extension manager for hook execution
         * @param {Object} oManager - Extension manager instance
         */
        setExtensionManager: function(oManager) {
            this._extensionManager = oManager;
        },

        /**
         * Hook called before extending a component
         */
        onBeforeExtend: function(oComponent) {
            Log.debug("Preparing to extend ProcessFlowControl: " + oComponent.getId(), 
                     "trialbalance.extensions.components.ProcessFlowExtension");
            
            // Store reference to UI5 control
            this._ui5Control = oComponent;
            
            // Add extension marker class
            oComponent.addStyleClass("process-flow-extended");
        },

        /**
         * Hook called after extending a component
         * Initialize the actual ProcessFlow instance
         */
        onAfterExtend: function(oComponent) {
            Log.debug("ProcessFlowControl extended, initializing process flow instance", 
                     "trialbalance.extensions.components.ProcessFlowExtension");
            
            // Wait for DOM to be ready
            setTimeout(function() {
                this._initializeProcessFlow(oComponent);
            }.bind(this), 100);
        },

        /**
         * Transform data received by the component
         * Apply extension hooks and transformations
         */
        onDataReceived: function(vData, mContext) {
            if (!vData) return vData;
            
            Log.debug("Transforming data for ProcessFlow", 
                     "trialbalance.extensions.components.ProcessFlowExtension");
            
            // Execute data transformation hooks from other extensions
            if (this._extensionManager) {
                vData = this._extensionManager.executeHook("processflow.data.transform", {
                    data: vData,
                    metadata: mContext
                });
            }
            
            // Cache the data for later use
            if (mContext && mContext.property) {
                this._dataCache.set(mContext.property, vData);
            }
            
            return vData;
        },

        /**
         * Hook called after component rendering
         * Sync data with the process flow instance
         */
        onAfterRender: function(oComponent) {
            if (this._processFlow) {
                // Sync any cached data
                this._syncCachedData();
                
                // Execute post-render hooks
                if (this._extensionManager) {
                    this._extensionManager.executeHook("processflow.rendered", {
                        processFlow: this._processFlow,
                        component: oComponent
                    });
                }
            }
        },

        /**
         * Handle user actions on the process flow
         */
        onUserAction: function(sAction, mParams) {
            Log.debug("User action: " + sAction, "trialbalance.extensions.components.ProcessFlowExtension");
            
            // Execute pre-action hooks
            if (this._extensionManager) {
                const bContinue = this._extensionManager.executeHook("processflow.action.before", {
                    action: sAction,
                    params: mParams
                });
                
                if (bContinue === false) {
                    return false;
                }
            }
            
            // Handle specific actions
            switch (sAction) {
                case "nodeClick":
                    this._handleNodeClick(mParams);
                    break;
                case "zoomChange":
                    this._handleZoomChange(mParams);
                    break;
            }
            
            // Execute post-action hooks
            if (this._extensionManager) {
                this._extensionManager.executeHook("processflow.action.after", {
                    action: sAction,
                    params: mParams
                });
            }
            
            return true;
        },

        /**
         * Handle errors
         */
        onError: function(oError, mContext) {
            Log.error("ProcessFlow error: " + oError.message, 
                     oError.stack, 
                     "trialbalance.extensions.components.ProcessFlowExtension");
            
            // Execute error hooks
            if (this._extensionManager) {
                this._extensionManager.executeHook("processflow.error", {
                    error: oError,
                    context: mContext
                });
            }
        },

        // ========== Public API ==========

        /**
         * Get the underlying ProcessFlow instance
         * @returns {Object} ProcessFlow instance
         */
        getProcessFlow: function() {
            return this._processFlow;
        },

        /**
         * Load data into the process flow
         * @param {Object} oData - Data object with lanes, nodes, connections
         * @returns {Promise} Promise resolving when data is loaded
         */
        loadData: function(oData) {
            if (!this._processFlow) {
                // Cache for later
                this._dataCache.set("lanes", oData.lanes);
                this._dataCache.set("nodes", oData.nodes);
                this._dataCache.set("connections", oData.connections);
                return Promise.resolve();
            }
            
            // Transform data via hooks
            if (this._extensionManager) {
                oData = this._extensionManager.executeHook("processflow.data.transform", {
                    data: oData,
                    metadata: { source: "loadData" }
                });
            }
            
            this._processFlow.loadData(oData);
            return Promise.resolve();
        },

        /**
         * Load data from the backend
         * @param {string} sApiUrl - API URL
         * @returns {Promise} Promise resolving when data is loaded
         */
        loadFromBackend: function(sApiUrl) {
            return fetch(sApiUrl)
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    return this.loadData(data);
                }.bind(this));
        },

        /**
         * Highlight a path of nodes
         * @param {Array<string>} aNodeIds - Array of node IDs to highlight
         */
        highlightPath: function(aNodeIds) {
            if (!this._processFlow) {
                Log.warning("ProcessFlow not initialized", 
                           "trialbalance.extensions.components.ProcessFlowExtension");
                return;
            }
            
            // Execute hook before highlighting
            if (this._extensionManager) {
                aNodeIds = this._extensionManager.executeHook("processflow.pathHighlight.before", {
                    data: aNodeIds,
                    metadata: {}
                }) || aNodeIds;
            }
            
            this._processFlow.highlightPath(aNodeIds);
            
            // Execute hook after highlighting
            if (this._extensionManager) {
                this._extensionManager.executeHook("processflow.pathHighlight", {
                    nodeIds: aNodeIds
                });
            }
        },

        /**
         * Clear all highlighting
         */
        clearHighlight: function() {
            if (this._processFlow) {
                this._processFlow.clearHighlight();
            }
        },

        /**
         * Set zoom level
         * @param {string} sLevel - Zoom level (One, Two, Three, Four)
         */
        setZoomLevel: function(sLevel) {
            if (!this._processFlow) {
                this._currentZoomLevel = sLevel;
                return;
            }
            
            this._processFlow.setZoomLevel(sLevel);
            this._currentZoomLevel = sLevel;
        },

        /**
         * Get current zoom level
         * @returns {string} Current zoom level
         */
        getZoomLevel: function() {
            return this._processFlow ? this._processFlow.getZoomLevel() : this._currentZoomLevel;
        },

        /**
         * Enable auto zoom detection
         */
        enableAutoZoom: function() {
            if (this._processFlow) {
                this._processFlow.enableAutoZoom();
            }
        },

        /**
         * Select a node programmatically
         * @param {string} sNodeId - Node ID to select (null to deselect)
         */
        selectNode: function(sNodeId) {
            if (this._processFlow) {
                this._processFlow.selectNode(sNodeId);
            }
        },

        /**
         * Get a node by ID
         * @param {string} sNodeId - Node ID
         * @returns {Object} Node object
         */
        getNode: function(sNodeId) {
            return this._processFlow ? this._processFlow.getNode(sNodeId) : null;
        },

        /**
         * Export process flow data
         * @returns {Object} Process flow data
         */
        exportData: function() {
            return this._processFlow ? this._processFlow.exportData() : null;
        },

        /**
         * Register event handler on the process flow
         * @param {string} sEvent - Event name
         * @param {Function} fnHandler - Handler function
         */
        on: function(sEvent, fnHandler) {
            if (this._processFlow) {
                this._processFlow.on(sEvent, fnHandler);
            }
            
            // Store for later if process flow not ready
            if (!this._eventHandlers.has(sEvent)) {
                this._eventHandlers.set(sEvent, []);
            }
            this._eventHandlers.get(sEvent).push(fnHandler);
        },

        /**
         * Remove event handler
         * @param {string} sEvent - Event name
         * @param {Function} fnHandler - Handler function (optional)
         */
        off: function(sEvent, fnHandler) {
            if (this._processFlow) {
                this._processFlow.off(sEvent, fnHandler);
            }
            
            if (!fnHandler) {
                this._eventHandlers.delete(sEvent);
            } else {
                const handlers = this._eventHandlers.get(sEvent);
                if (handlers) {
                    const idx = handlers.indexOf(fnHandler);
                    if (idx !== -1) handlers.splice(idx, 1);
                }
            }
        },

        // ========== Private Methods ==========

        /**
         * Load the ProcessFlow TypeScript component
         * @private
         */
        _loadProcessFlowScript: function() {
            return new Promise(function(resolve, reject) {
                // Check if already loaded
                if (window.ProcessFlow) {
                    resolve();
                    return;
                }
                
                // Path to compiled ProcessFlow
                const sScriptPath = sap.ui.require.toUrl("trialbalance/components/dist/ProcessFlow/ProcessFlow.min.js");
                const sCssPath = sap.ui.require.toUrl("trialbalance/components/ProcessFlow/processflow.css");
                
                // Load CSS
                const link = document.createElement("link");
                link.rel = "stylesheet";
                link.href = sCssPath;
                document.head.appendChild(link);
                
                // Load JS as ES6 module
                const script = document.createElement("script");
                script.type = "module";
                script.textContent = `
                    import { ProcessFlow } from '${sScriptPath}';
                    window.ProcessFlow = ProcessFlow;
                    window.dispatchEvent(new CustomEvent('processflow-loaded'));
                `;
                document.head.appendChild(script);
                
                // Wait for load
                window.addEventListener('processflow-loaded', function() {
                    Log.info("ProcessFlow script loaded", "trialbalance.extensions.components.ProcessFlowExtension");
                    resolve();
                }, { once: true });
                
                // Timeout fallback
                setTimeout(function() {
                    if (!window.ProcessFlow) {
                        reject(new Error("ProcessFlow script load timeout"));
                    }
                }, 10000);
            });
        },

        /**
         * Load configuration from backend
         * @private
         */
        _loadConfiguration: function() {
            return fetch('/api/v1/extensions/process-flow-core/config')
                .then(function(response) { return response.json(); })
                .then(function(config) {
                    this._config = config;
                    Log.debug("Configuration loaded", "trialbalance.extensions.components.ProcessFlowExtension");
                }.bind(this))
                .catch(function() {
                    // Use default configuration
                    this._config = {
                        zoomLevel: null, // Auto-detect
                        wheelZoomable: true
                    };
                }.bind(this));
        },

        /**
         * Initialize the ProcessFlow instance
         * @private
         */
        _initializeProcessFlow: function(oComponent) {
            if (!window.ProcessFlow) {
                Log.error("ProcessFlow not loaded", "trialbalance.extensions.components.ProcessFlowExtension");
                return;
            }
            
            // Get the container element
            const domRef = oComponent.getDomRef();
            if (!domRef) {
                Log.error("Component DOM not ready", "trialbalance.extensions.components.ProcessFlowExtension");
                return;
            }
            
            const canvas = domRef.querySelector(".processflow-canvas");
            if (!canvas) {
                Log.error("Canvas element not found", "trialbalance.extensions.components.ProcessFlowExtension");
                return;
            }
            
            // Create ProcessFlow instance
            try {
                this._processFlow = new window.ProcessFlow(canvas, this._config);
                
                // Apply zoom level if configured
                if (this._currentZoomLevel) {
                    this._processFlow.setZoomLevel(this._currentZoomLevel);
                }
                
                // Wire up event handlers
                this._wireEventHandlers();
                
                // Sync cached data
                this._syncCachedData();
                
                Log.info("ProcessFlow instance created successfully", 
                        "trialbalance.extensions.components.ProcessFlowExtension");
                
            } catch (e) {
                this.onError(e, { action: "initializeProcessFlow" });
            }
        },

        /**
         * Wire up stored event handlers
         * @private
         */
        _wireEventHandlers: function() {
            if (!this._processFlow) return;
            
            // Register stored handlers
            this._eventHandlers.forEach(function(handlers, event) {
                handlers.forEach(function(handler) {
                    this._processFlow.on(event, handler);
                }.bind(this));
            }.bind(this));
            
            // Register extension hooks for standard events
            const standardEvents = ['nodeClick', 'zoomChange'];
            
            standardEvents.forEach(function(event) {
                this._processFlow.on(event, function(data) {
                    this.onUserAction(event, data);
                }.bind(this));
            }.bind(this));
        },

        /**
         * Sync cached data to the process flow
         * @private
         */
        _syncCachedData: function() {
            if (!this._processFlow) return;
            
            const lanes = this._dataCache.get("lanes");
            const nodes = this._dataCache.get("nodes");
            const connections = this._dataCache.get("connections");
            
            if (lanes || nodes || connections) {
                this._processFlow.loadData({
                    lanes: lanes || [],
                    nodes: nodes || [],
                    connections: connections || []
                });
            }
        },

        /**
         * Handle node click
         * @private
         */
        _handleNodeClick: function(mParams) {
            if (mParams && mParams.node) {
                // Load additional data from backend
                this._loadNodeDetails(mParams.node.id);
            }
        },

        /**
         * Handle zoom change
         * @private
         */
        _handleZoomChange: function(mParams) {
            if (mParams && mParams.level) {
                this._currentZoomLevel = mParams.level;
                
                // Execute zoom change hook
                if (this._extensionManager) {
                    this._extensionManager.executeHook("processflow.zoomChange", mParams);
                }
            }
        },

        /**
         * Load node details from backend
         * @private
         */
        _loadNodeDetails: function(sNodeId) {
            return fetch('/api/v1/extensions/process-flow-core/node/' + sNodeId)
                .then(function(response) { return response.json(); })
                .then(function(details) {
                    // Execute hook with details
                    if (this._extensionManager) {
                        this._extensionManager.executeHook("processflow.nodeDetails", {
                            nodeId: sNodeId,
                            details: details
                        });
                    }
                    return details;
                }.bind(this))
                .catch(function(error) {
                    Log.error("Failed to load node details", error.message, 
                             "trialbalance.extensions.components.ProcessFlowExtension");
                    return null;
                });
        },

        /**
         * Cleanup on destroy
         */
        destroy: function() {
            Log.info("Destroying ProcessFlow Extension", 
                    "trialbalance.extensions.components.ProcessFlowExtension");
            
            // Destroy process flow instance
            if (this._processFlow) {
                this._processFlow.destroy();
                this._processFlow = null;
            }
            
            // Clear caches
            this._dataCache.clear();
            this._eventHandlers.clear();
            this._config = null;
            this._ui5Control = null;
            this._extensionManager = null;
            
            ComponentExtension.prototype.destroy.call(this);
        }
    });
});