sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log"
], function(BaseObject, Log) {
    "use strict";

    /**
     * Component Bridge
     * Provides seamless communication between UI5 components and the Zig backend.
     * Handles data synchronization, WebSocket connections, and extension API calls.
     * 
     * Features:
     * - REST API integration
     * - WebSocket real-time updates
     * - Request/response transformation via extensions
     * - Error handling and retry logic
     * - Caching and offline support
     * 
     * @class
     * @extends sap.ui.base.Object
     */
    return BaseObject.extend("trialbalance.extensions.ComponentBridge", {
        
        constructor: function(mSettings) {
            BaseObject.call(this);
            
            this._baseUrl = mSettings && mSettings.baseUrl 
                ? mSettings.baseUrl 
                : "/api/v1";
            
            this._wsUrl = mSettings && mSettings.wsUrl 
                ? mSettings.wsUrl 
                : null;
            
            this._extensionManager = mSettings && mSettings.extensionManager 
                ? mSettings.extensionManager 
                : null;
            
            this._ws = null;
            this._wsReconnectAttempts = 0;
            this._maxReconnectAttempts = 5;
            this._reconnectDelay = 3000;
            
            this._cache = new Map();
            this._cacheTTL = 30000; // 30 seconds default
            this._pendingRequests = new Map();
            
            // Event handlers
            this._eventHandlers = new Map();
            
            Log.info("ComponentBridge initialized", "trialbalance.extensions.ComponentBridge");
        },

        /**
         * Set the extension manager for hook execution
         * @param {Object} oManager - Extension manager instance
         */
        setExtensionManager: function(oManager) {
            this._extensionManager = oManager;
        },

        // ========== REST API Methods ==========

        /**
         * Make a GET request
         * @param {string} sPath - API path
         * @param {Object} mOptions - Request options
         * @returns {Promise} Promise resolving to response data
         */
        get: function(sPath, mOptions) {
            return this._request("GET", sPath, null, mOptions);
        },

        /**
         * Make a POST request
         * @param {string} sPath - API path
         * @param {Object} oData - Request body
         * @param {Object} mOptions - Request options
         * @returns {Promise} Promise resolving to response data
         */
        post: function(sPath, oData, mOptions) {
            return this._request("POST", sPath, oData, mOptions);
        },

        /**
         * Make a PUT request
         * @param {string} sPath - API path
         * @param {Object} oData - Request body
         * @param {Object} mOptions - Request options
         * @returns {Promise} Promise resolving to response data
         */
        put: function(sPath, oData, mOptions) {
            return this._request("PUT", sPath, oData, mOptions);
        },

        /**
         * Make a DELETE request
         * @param {string} sPath - API path
         * @param {Object} mOptions - Request options
         * @returns {Promise} Promise resolving to response data
         */
        delete: function(sPath, mOptions) {
            return this._request("DELETE", sPath, null, mOptions);
        },

        /**
         * Make a request to an extension endpoint
         * @param {string} sExtensionId - Extension ID
         * @param {string} sPath - Path within extension
         * @param {string} sMethod - HTTP method
         * @param {Object} oData - Request body
         * @returns {Promise} Promise resolving to response data
         */
        extensionRequest: function(sExtensionId, sPath, sMethod, oData) {
            const fullPath = "/extensions/" + sExtensionId + sPath;
            return this._request(sMethod || "GET", fullPath, oData);
        },

        // ========== WebSocket Methods ==========

        /**
         * Connect to WebSocket for real-time updates
         * @param {string} sUrl - WebSocket URL (optional, uses configured URL if not provided)
         */
        connectWebSocket: function(sUrl) {
            const wsUrl = sUrl || this._wsUrl;
            
            if (!wsUrl) {
                Log.warning("No WebSocket URL configured", "trialbalance.extensions.ComponentBridge");
                return;
            }
            
            if (this._ws) {
                this.disconnectWebSocket();
            }
            
            Log.info("Connecting to WebSocket: " + wsUrl, "trialbalance.extensions.ComponentBridge");
            
            this._ws = new WebSocket(wsUrl);
            
            this._ws.onopen = function() {
                Log.info("WebSocket connected", "trialbalance.extensions.ComponentBridge");
                this._wsReconnectAttempts = 0;
                this._emit("wsConnected", {});
            }.bind(this);
            
            this._ws.onmessage = function(event) {
                this._handleWebSocketMessage(event);
            }.bind(this);
            
            this._ws.onerror = function(error) {
                Log.error("WebSocket error", error.message, "trialbalance.extensions.ComponentBridge");
                this._emit("wsError", { error: error });
            }.bind(this);
            
            this._ws.onclose = function() {
                Log.info("WebSocket disconnected", "trialbalance.extensions.ComponentBridge");
                this._emit("wsDisconnected", {});
                
                // Auto-reconnect
                this._scheduleReconnect();
            }.bind(this);
        },

        /**
         * Disconnect WebSocket
         */
        disconnectWebSocket: function() {
            if (this._ws) {
                this._ws.close();
                this._ws = null;
            }
        },

        /**
         * Send message via WebSocket
         * @param {Object} oMessage - Message to send
         */
        sendWebSocketMessage: function(oMessage) {
            if (!this._ws || this._ws.readyState !== WebSocket.OPEN) {
                Log.warning("WebSocket not connected", "trialbalance.extensions.ComponentBridge");
                return false;
            }
            
            this._ws.send(JSON.stringify(oMessage));
            return true;
        },

        /**
         * Check if WebSocket is connected
         * @returns {boolean} True if connected
         */
        isWebSocketConnected: function() {
            return this._ws && this._ws.readyState === WebSocket.OPEN;
        },

        // ========== Caching Methods ==========

        /**
         * Get cached data
         * @param {string} sKey - Cache key
         * @returns {*} Cached data or undefined
         */
        getFromCache: function(sKey) {
            const entry = this._cache.get(sKey);
            
            if (!entry) return undefined;
            
            // Check if expired
            if (Date.now() > entry.expires) {
                this._cache.delete(sKey);
                return undefined;
            }
            
            return entry.data;
        },

        /**
         * Set cached data
         * @param {string} sKey - Cache key
         * @param {*} vData - Data to cache
         * @param {number} nTTL - Time to live in ms (optional)
         */
        setCache: function(sKey, vData, nTTL) {
            this._cache.set(sKey, {
                data: vData,
                expires: Date.now() + (nTTL || this._cacheTTL)
            });
        },

        /**
         * Clear cache
         * @param {string} sKey - Specific key to clear (optional, clears all if not provided)
         */
        clearCache: function(sKey) {
            if (sKey) {
                this._cache.delete(sKey);
            } else {
                this._cache.clear();
            }
        },

        // ========== Event Methods ==========

        /**
         * Register event handler
         * @param {string} sEvent - Event name
         * @param {Function} fnHandler - Handler function
         */
        on: function(sEvent, fnHandler) {
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

        // ========== Specialized Methods ==========

        /**
         * Load trial balance data
         * @param {Object} mParams - Parameters (period, entity, etc.)
         * @returns {Promise} Promise resolving to trial balance data
         */
        loadTrialBalance: function(mParams) {
            return this.get("/trial-balance", { params: mParams });
        },

        /**
         * Calculate trial balance
         * @param {Object} oInput - Calculation input
         * @returns {Promise} Promise resolving to calculation result
         */
        calculateTrialBalance: function(oInput) {
            return this.post("/trial-balance/calculate", oInput);
        },

        /**
         * Get account details
         * @param {string} sAccountId - Account ID
         * @returns {Promise} Promise resolving to account details
         */
        getAccountDetails: function(sAccountId) {
            return this.get("/accounts/" + sAccountId);
        },

        /**
         * Get network graph data for lineage view
         * @param {string} sEntityId - Entity ID
         * @returns {Promise} Promise resolving to graph data
         */
        getLineageGraph: function(sEntityId) {
            return this.get("/lineage/" + sEntityId);
        },

        /**
         * Get extension configuration
         * @param {string} sExtensionId - Extension ID
         * @returns {Promise} Promise resolving to configuration
         */
        getExtensionConfig: function(sExtensionId) {
            return this.extensionRequest(sExtensionId, "/config", "GET");
        },

        // ========== Private Methods ==========

        /**
         * Make an HTTP request
         * @private
         */
        _request: function(sMethod, sPath, oData, mOptions) {
            const options = mOptions || {};
            const cacheKey = sMethod + ":" + sPath;
            
            // Check cache for GET requests
            if (sMethod === "GET" && !options.noCache) {
                const cached = this.getFromCache(cacheKey);
                if (cached) {
                    Log.debug("Cache hit: " + cacheKey, "trialbalance.extensions.ComponentBridge");
                    return Promise.resolve(cached);
                }
            }
            
            // Check for pending request (deduplicate)
            if (this._pendingRequests.has(cacheKey)) {
                return this._pendingRequests.get(cacheKey);
            }
            
            // Transform request via extensions
            let requestData = oData;
            if (this._extensionManager) {
                requestData = this._extensionManager.executeHook("bridge.request", {
                    data: oData,
                    metadata: { method: sMethod, path: sPath }
                });
            }
            
            // Build URL
            let url = this._baseUrl + sPath;
            if (options.params) {
                const params = new URLSearchParams(options.params);
                url += "?" + params.toString();
            }
            
            // Make request
            const fetchOptions = {
                method: sMethod,
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            };
            
            if (requestData && sMethod !== "GET") {
                fetchOptions.body = JSON.stringify(requestData);
            }
            
            const requestPromise = fetch(url, fetchOptions)
                .then(function(response) {
                    if (!response.ok) {
                        throw new Error("HTTP " + response.status + ": " + response.statusText);
                    }
                    return response.json();
                })
                .then(function(data) {
                    // Transform response via extensions
                    if (this._extensionManager) {
                        data = this._extensionManager.executeHook("bridge.response", {
                            data: data,
                            metadata: { method: sMethod, path: sPath }
                        });
                    }
                    
                    // Cache GET responses
                    if (sMethod === "GET" && !options.noCache) {
                        this.setCache(cacheKey, data, options.cacheTTL);
                    }
                    
                    return data;
                }.bind(this))
                .catch(function(error) {
                    Log.error("Request failed: " + sMethod + " " + sPath, 
                             error.message, "trialbalance.extensions.ComponentBridge");
                    
                    // Execute error hook
                    if (this._extensionManager) {
                        this._extensionManager.executeHook("bridge.error", {
                            error: error,
                            metadata: { method: sMethod, path: sPath }
                        });
                    }
                    
                    throw error;
                }.bind(this))
                .finally(function() {
                    this._pendingRequests.delete(cacheKey);
                }.bind(this));
            
            this._pendingRequests.set(cacheKey, requestPromise);
            
            return requestPromise;
        },

        /**
         * Handle WebSocket message
         * @private
         */
        _handleWebSocketMessage: function(event) {
            try {
                const message = JSON.parse(event.data);
                
                Log.debug("WebSocket message received: " + message.type, 
                         "trialbalance.extensions.ComponentBridge");
                
                // Transform message via extensions
                let transformedMessage = message;
                if (this._extensionManager) {
                    transformedMessage = this._extensionManager.executeHook("bridge.wsMessage", {
                        data: message,
                        metadata: { raw: event.data }
                    });
                }
                
                // Emit message event
                this._emit("wsMessage", transformedMessage);
                
                // Emit type-specific event
                if (transformedMessage.type) {
                    this._emit("ws:" + transformedMessage.type, transformedMessage);
                }
                
            } catch (e) {
                Log.error("Failed to parse WebSocket message", e.message, 
                         "trialbalance.extensions.ComponentBridge");
            }
        },

        /**
         * Schedule WebSocket reconnection
         * @private
         */
        _scheduleReconnect: function() {
            if (this._wsReconnectAttempts >= this._maxReconnectAttempts) {
                Log.warning("Max reconnection attempts reached", 
                          "trialbalance.extensions.ComponentBridge");
                return;
            }
            
            this._wsReconnectAttempts++;
            const delay = this._reconnectDelay * this._wsReconnectAttempts;
            
            Log.info("Scheduling reconnection in " + delay + "ms (attempt " + 
                    this._wsReconnectAttempts + "/" + this._maxReconnectAttempts + ")", 
                    "trialbalance.extensions.ComponentBridge");
            
            setTimeout(function() {
                this.connectWebSocket();
            }.bind(this), delay);
        },

        /**
         * Emit event
         * @private
         */
        _emit: function(sEvent, oData) {
            const handlers = this._eventHandlers.get(sEvent);
            if (handlers) {
                handlers.forEach(function(handler) {
                    try {
                        handler(oData);
                    } catch (e) {
                        Log.error("Event handler error: " + sEvent, e.message, 
                                 "trialbalance.extensions.ComponentBridge");
                    }
                });
            }
        },

        /**
         * Destroy the bridge
         */
        destroy: function() {
            Log.info("Destroying ComponentBridge", "trialbalance.extensions.ComponentBridge");
            
            this.disconnectWebSocket();
            this.clearCache();
            this._pendingRequests.clear();
            this._eventHandlers.clear();
            this._extensionManager = null;
            
            BaseObject.prototype.destroy.call(this);
        }
    });
});