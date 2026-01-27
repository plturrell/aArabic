sap.ui.define([
    "trialbalance/extensions/ComponentExtension",
    "sap/base/Log"
], function(ComponentExtension, Log) {
    "use strict";

    /**
     * NetworkGraph Extension
     * Wraps the TypeScript NetworkGraph component and integrates it with the extension framework.
     * Provides extension points for customizing behavior and connecting to the backend.
     * 
     * Features:
     * - Wraps existing NetworkGraph TypeScript component
     * - Provides extension hooks for data transformation
     * - Integrates with backend via extension API
     * - Supports hot-reload and dynamic configuration
     * 
     * @class
     * @extends trialbalance.extensions.ComponentExtension
     */
    return ComponentExtension.extend("trialbalance.extensions.components.NetworkGraphExtension", {
        
        constructor: function(mSettings) {
            ComponentExtension.call(this, mSettings);
            
            // Extension metadata
            this.setId("network-graph-core");
            this.setName("Network Graph Core");
            this.setVersion("1.0.0");
            this.setTargetComponents(["trialbalance.control.NetworkGraphControl"]);
            this.setPriority(100); // High priority - core extension
            
            // Internal state
            this._graph = null;
            this._config = null;
            this._dataCache = new Map();
            this._eventHandlers = new Map();
            this._backendConnected = false;
            this._extensionManager = null;
        },

        /**
         * Initialize the extension
         * Load the NetworkGraph TypeScript component and configuration
         */
        init: function() {
            Log.info("Initializing NetworkGraph Extension", "trialbalance.extensions.components.NetworkGraphExtension");
            
            return Promise.all([
                this._loadNetworkGraphScript(),
                this._loadConfiguration()
            ]).then(function() {
                Log.info("NetworkGraph Extension initialized successfully", 
                        "trialbalance.extensions.components.NetworkGraphExtension");
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
            Log.debug("Preparing to extend NetworkGraphControl: " + oComponent.getId(), 
                     "trialbalance.extensions.components.NetworkGraphExtension");
            
            // Store reference to UI5 control
            this._ui5Control = oComponent;
            
            // Add extension marker class
            oComponent.addStyleClass("network-graph-extended");
        },

        /**
         * Hook called after extending a component
         * Initialize the actual NetworkGraph instance
         */
        onAfterExtend: function(oComponent) {
            Log.debug("NetworkGraphControl extended, initializing graph instance", 
                     "trialbalance.extensions.components.NetworkGraphExtension");
            
            // Wait for DOM to be ready
            setTimeout(function() {
                this._initializeGraph(oComponent);
            }.bind(this), 100);
        },

        /**
         * Transform data received by the component
         * Apply extension hooks and transformations
         */
        onDataReceived: function(vData, mContext) {
            if (!vData) return vData;
            
            Log.debug("Transforming data for NetworkGraph", 
                     "trialbalance.extensions.components.NetworkGraphExtension");
            
            // Execute data transformation hooks from other extensions
            if (this._extensionManager) {
                vData = this._extensionManager.executeHook("networkgraph.data.transform", {
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
         * Sync data with the graph instance
         */
        onAfterRender: function(oComponent) {
            if (this._graph) {
                // Sync any cached data
                this._syncCachedData();
                
                // Execute post-render hooks
                if (this._extensionManager) {
                    this._extensionManager.executeHook("networkgraph.rendered", {
                        graph: this._graph,
                        component: oComponent
                    });
                }
            }
        },

        /**
         * Handle user actions on the graph
         */
        onUserAction: function(sAction, mParams) {
            Log.debug("User action: " + sAction, "trialbalance.extensions.components.NetworkGraphExtension");
            
            // Execute pre-action hooks
            if (this._extensionManager) {
                const bContinue = this._extensionManager.executeHook("networkgraph.action.before", {
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
                case "edgeClick":
                    this._handleEdgeClick(mParams);
                    break;
                case "viewportChange":
                    this._handleViewportChange(mParams);
                    break;
            }
            
            // Execute post-action hooks
            if (this._extensionManager) {
                this._extensionManager.executeHook("networkgraph.action.after", {
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
            Log.error("NetworkGraph error: " + oError.message, 
                     oError.stack, 
                     "trialbalance.extensions.components.NetworkGraphExtension");
            
            // Execute error hooks
            if (this._extensionManager) {
                this._extensionManager.executeHook("networkgraph.error", {
                    error: oError,
                    context: mContext
                });
            }
        },

        // ========== Public API ==========

        /**
         * Get the underlying NetworkGraph instance
         * @returns {Object} NetworkGraph instance
         */
        getGraph: function() {
            return this._graph;
        },

        /**
         * Load data from the backend
         * @param {string} sApiUrl - API URL
         * @returns {Promise} Promise resolving when data is loaded
         */
        loadFromBackend: function(sApiUrl) {
            if (!this._graph) {
                return Promise.reject(new Error("Graph not initialized"));
            }
            
            return this._graph.loadFromAPI(sApiUrl);
        },

        /**
         * Connect to WebSocket for real-time updates
         * @param {string} sWsUrl - WebSocket URL
         */
        connectRealTime: function(sWsUrl) {
            if (!this._graph) {
                Log.error("Cannot connect WebSocket - graph not initialized", 
                         "trialbalance.extensions.components.NetworkGraphExtension");
                return;
            }
            
            this._graph.connectWebSocket(sWsUrl);
            this._backendConnected = true;
        },

        /**
         * Disconnect WebSocket
         */
        disconnectRealTime: function() {
            if (this._graph) {
                this._graph.disconnectWebSocket();
                this._backendConnected = false;
            }
        },

        /**
         * Apply a layout to the graph
         * @param {string} sLayoutType - Layout type (ForceDirected, Hierarchical, etc.)
         */
        setLayout: function(sLayoutType) {
            if (this._graph) {
                this._graph.setLayout(sLayoutType);
            }
        },

        /**
         * Export graph data
         * @returns {Object} Graph data
         */
        exportData: function() {
            return this._graph ? this._graph.exportData() : null;
        },

        /**
         * Export graph as image
         * @returns {string} Data URL
         */
        exportImage: function() {
            return this._graph ? this._graph.exportImage() : null;
        },

        /**
         * Get graph statistics
         * @returns {Object} Statistics
         */
        getStats: function() {
            return this._graph ? this._graph.getStats() : null;
        },

        /**
         * Register event handler on the graph
         * @param {string} sEvent - Event name
         * @param {Function} fnHandler - Handler function
         */
        on: function(sEvent, fnHandler) {
            if (this._graph) {
                this._graph.on(sEvent, fnHandler);
            }
            
            // Store for later if graph not ready
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
            if (this._graph) {
                this._graph.off(sEvent, fnHandler);
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
         * Load the NetworkGraph TypeScript component
         * @private
         */
        _loadNetworkGraphScript: function() {
            return new Promise(function(resolve, reject) {
                // Check if already loaded
                if (window.NetworkGraph) {
                    resolve();
                    return;
                }
                
                // Path to compiled NetworkGraph
                const sScriptPath = sap.ui.require.toUrl("trialbalance/components/dist/NetworkGraph/NetworkGraph.min.js");
                const sCssPath = sap.ui.require.toUrl("trialbalance/components/NetworkGraph/styles.css");
                
                // Load CSS
                const link = document.createElement("link");
                link.rel = "stylesheet";
                link.href = sCssPath;
                document.head.appendChild(link);
                
                // Load JS as ES6 module
                const script = document.createElement("script");
                script.type = "module";
                script.textContent = `
                    import { NetworkGraph } from '${sScriptPath}';
                    window.NetworkGraph = NetworkGraph;
                    window.dispatchEvent(new CustomEvent('networkgraph-loaded'));
                `;
                document.head.appendChild(script);
                
                // Wait for load
                window.addEventListener('networkgraph-loaded', function() {
                    Log.info("NetworkGraph script loaded", "trialbalance.extensions.components.NetworkGraphExtension");
                    resolve();
                }, { once: true });
                
                // Timeout fallback
                setTimeout(function() {
                    if (!window.NetworkGraph) {
                        reject(new Error("NetworkGraph script load timeout"));
                    }
                }, 10000);
            });
        },

        /**
         * Load configuration from backend
         * @private
         */
        _loadConfiguration: function() {
            return fetch('/api/v1/extensions/network-graph-core/config')
                .then(function(response) { return response.json(); })
                .then(function(config) {
                    this._config = config;
                    Log.debug("Configuration loaded", "trialbalance.extensions.components.NetworkGraphExtension");
                }.bind(this))
                .catch(function() {
                    // Use default configuration
                    this._config = {
                        layout: "ForceDirected",
                        minimap: true,
                        animation: true,
                        websocket: null
                    };
                }.bind(this));
        },

        /**
         * Initialize the NetworkGraph instance
         * @private
         */
        _initializeGraph: function(oComponent) {
            if (!window.NetworkGraph) {
                Log.error("NetworkGraph not loaded", "trialbalance.extensions.components.NetworkGraphExtension");
                return;
            }
            
            // Get the container element
            const domRef = oComponent.getDomRef();
            if (!domRef) {
                Log.error("Component DOM not ready", "trialbalance.extensions.components.NetworkGraphExtension");
                return;
            }
            
            const canvas = domRef.querySelector(".networkgraph-canvas");
            if (!canvas) {
                Log.error("Canvas element not found", "trialbalance.extensions.components.NetworkGraphExtension");
                return;
            }
            
            // Create NetworkGraph instance
            try {
                this._graph = new window.NetworkGraph(canvas);
                
                // Apply configuration
                if (this._config) {
                    if (this._config.layout) {
                        this._graph.setLayout(this._config.layout);
                    }
                    if (this._config.minimap) {
                        this._graph.enableMinimap();
                    }
                    if (this._config.websocket) {
                        this.connectRealTime(this._config.websocket);
                    }
                }
                
                // Wire up event handlers
                this._wireEventHandlers();
                
                // Sync cached data
                this._syncCachedData();
                
                Log.info("NetworkGraph instance created successfully", 
                        "trialbalance.extensions.components.NetworkGraphExtension");
                
            } catch (e) {
                this.onError(e, { action: "initializeGraph" });
            }
        },

        /**
         * Wire up stored event handlers
         * @private
         */
        _wireEventHandlers: function() {
            if (!this._graph) return;
            
            // Register stored handlers
            this._eventHandlers.forEach(function(handlers, event) {
                handlers.forEach(function(handler) {
                    this._graph.on(event, handler);
                }.bind(this));
            }.bind(this));
            
            // Register extension hooks for standard events
            const standardEvents = [
                'nodeClick', 'nodeDrag', 'edgeClick', 
                'selectionChanged', 'viewportChange', 
                'dataLoaded', 'layoutApplied'
            ];
            
            standardEvents.forEach(function(event) {
                this._graph.on(event, function(data) {
                    this.onUserAction(event, data);
                }.bind(this));
            }.bind(this));
        },

        /**
         * Sync cached data to the graph
         * @private
         */
        _syncCachedData: function() {
            if (!this._graph) return;
            
            // Sync nodes
            const nodes = this._dataCache.get("nodes");
            if (nodes && Array.isArray(nodes)) {
                nodes.forEach(function(node) {
                    this._graph.addNode(node);
                }.bind(this));
            }
            
            // Sync edges
            const edges = this._dataCache.get("edges");
            if (edges && Array.isArray(edges)) {
                edges.forEach(function(edge) {
                    this._graph.addEdge(edge);
                }.bind(this));
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
         * Handle edge click
         * @private
         */
        _handleEdgeClick: function(mParams) {
            Log.debug("Edge clicked: " + JSON.stringify(mParams), 
                     "trialbalance.extensions.components.NetworkGraphExtension");
        },

        /**
         * Handle viewport change
         * @private
         */
        _handleViewportChange: function(mParams) {
            // Update minimap if enabled
            if (this._config && this._config.minimap && this._graph) {
                // Minimap updates are handled by NetworkGraph internally
            }
        },

        /**
         * Load node details from backend
         * @private
         */
        _loadNodeDetails: function(sNodeId) {
            return fetch('/api/v1/extensions/network-graph-core/node/' + sNodeId)
                .then(function(response) { return response.json(); })
                .then(function(details) {
                    // Execute hook with details
                    if (this._extensionManager) {
                        this._extensionManager.executeHook("networkgraph.nodeDetails", {
                            nodeId: sNodeId,
                            details: details
                        });
                    }
                    return details;
                }.bind(this))
                .catch(function(error) {
                    Log.error("Failed to load node details", error.message, 
                             "trialbalance.extensions.components.NetworkGraphExtension");
                    return null;
                });
        },

        /**
         * Cleanup on destroy
         */
        destroy: function() {
            Log.info("Destroying NetworkGraph Extension", 
                    "trialbalance.extensions.components.NetworkGraphExtension");
            
            // Disconnect WebSocket
            this.disconnectRealTime();
            
            // Destroy graph instance
            if (this._graph) {
                this._graph.destroy();
                this._graph = null;
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