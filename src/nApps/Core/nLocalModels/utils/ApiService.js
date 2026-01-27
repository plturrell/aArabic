sap.ui.define([
    "llm/server/dashboard/utils/RuntimeConfig"
], function (RuntimeConfig) {
    "use strict";

    /**
     * Unified API Service for all Dashboard pages
     * Handles: REST API calls, WebSocket, Keycloak auth, error handling
     */

    var runtimeConfig = RuntimeConfig.get();

    const ApiService = {
        
        // Configuration
        config: {
            baseUrl: runtimeConfig.apiBaseUrl,
            wsUrl: runtimeConfig.wsUrl,
            keycloakUrl: runtimeConfig.keycloak.url,
            keycloakRealm: runtimeConfig.keycloak.realm,
            keycloakClientId: runtimeConfig.keycloak.clientId
        },
        
        // State
        _token: null,
        _keycloak: null,
        _wsConnection: null,
        _wsCallbacks: [],
        _shouldReconnect: true,
        _reconnectDelayMs: 5000,
        
        // ========================================================================
        // AUTHENTICATION (Keycloak OAuth2)
        // ========================================================================
        
        initKeycloak: function() {
            return new Promise((resolve, reject) => {
                if (typeof Keycloak === 'undefined') {
                    console.warn("Keycloak not loaded, using mock auth");
                    this._token = "mock-token-for-development";
                    resolve(true);
                    return;
                }
                
                this._keycloak = new Keycloak({
                    url: this.config.keycloakUrl,
                    realm: this.config.keycloakRealm,
                    clientId: this.config.keycloakClientId
                });
                
                this._keycloak.init({ onLoad: 'login-required' })
                    .then(authenticated => {
                        if (authenticated) {
                            this._token = this._keycloak.token;
                            
                            // Auto-refresh token
                            setInterval(() => {
                                this._keycloak.updateToken(70).then(refreshed => {
                                    if (refreshed) {
                                        this._token = this._keycloak.token;
                                        console.log('Token refreshed');
                                    }
                                });
                            }, 60000);
                            
                            resolve(true);
                        } else {
                            reject("Not authenticated");
                        }
                    })
                    .catch(reject);
            });
        },
        
        getToken: function() {
            return this._token || "mock-token";
        },
        
        getUserProfile: function() {
            if (this._keycloak && this._keycloak.tokenParsed) {
                return {
                    username: this._keycloak.tokenParsed.preferred_username,
                    email: this._keycloak.tokenParsed.email,
                    name: this._keycloak.tokenParsed.name
                };
            }
            return { username: "demo-user", email: "demo@example.com", name: "Demo User" };
        },
        
        logout: function() {
            if (this._keycloak) {
                this._keycloak.logout();
            }
        },
        
        // ========================================================================
        // HTTP REQUEST HELPER
        // ========================================================================
        
        _request: function(method, endpoint, data = null) {
            const url = this.config.baseUrl + endpoint;
            const headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + this.getToken()
            };
            
            const options = {
                method: method,
                headers: headers
            };
            
            if (data && (method === 'POST' || method === 'PUT')) {
                options.body = JSON.stringify(data);
            }
            
            return fetch(url, options)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .catch(error => {
                    console.error(`API Error [${method} ${endpoint}]:`, error);
                    throw error;
                });
        },
        
        // ========================================================================
        // MODEL MANAGEMENT API
        // ========================================================================
        
        getModels: function() {
            return this._request('GET', '/api/v1/models');
        },
        
        getModel: function(modelId) {
            return this._request('GET', `/api/v1/models/${modelId}`);
        },
        
        loadModel: function(modelId) {
            return this._request('POST', `/api/v1/models/${modelId}/load`);
        },
        
        getModelStatus: function(modelId) {
            return this._request('GET', `/api/v1/models/${modelId}/status`);
        },
        
        // ========================================================================
        // METRICS API
        // ========================================================================
        
        getCurrentMetrics: function(modelId) {
            const endpoint = modelId 
                ? `/api/v1/metrics/current?model=${modelId}`
                : '/api/v1/metrics/current';
            return this._request('GET', endpoint);
        },
        
        getMetricsHistory: function(modelId, range = '1h') {
            return this._request('GET', `/api/v1/metrics/history?model=${modelId}&range=${range}`);
        },
        
        getTierStats: function() {
            return this._request('GET', '/api/v1/tiers/stats');
        },
        
        // ========================================================================
        // CHAT / PROMPTS API
        // ========================================================================
        
        sendChatCompletion: function(request) {
            // OpenAI-compatible endpoint
            return this._request('POST', '/v1/chat/completions', request);
        },
        
        savePrompt: function(promptData) {
            return this._request('POST', '/api/v1/prompts', promptData);
        },
        
        getPromptHistory: function(userId, limit = 50) {
            return this._request('GET', `/api/v1/prompts/history?user=${userId}&limit=${limit}`);
        },
        
        getSavedPrompts: function(userId) {
            return this._request('GET', `/api/v1/prompts/saved?user=${userId}`);
        },
        
        // ========================================================================
        // MODE MANAGEMENT API
        // ========================================================================
        
        getModes: function() {
            return this._request('GET', '/api/v1/modes');
        },
        
        getMode: function(modeName) {
            return this._request('GET', `/api/v1/modes/${modeName}`);
        },
        
        activateMode: function(modeName, modelId) {
            return this._request('POST', `/api/v1/modes/${modeName}/activate`, { model_id: modelId });
        },
        
        createCustomMode: function(modeConfig) {
            return this._request('POST', '/api/v1/modes/custom', modeConfig);
        },
        
        // ========================================================================
        // MHC FINE-TUNING API
        // ========================================================================
        
        getMHCConfig: function(modelId) {
            return this._request('GET', `/api/v1/mhc/config?model=${modelId}`);
        },
        
        updateMHCConfig: function(modelId, config) {
            return this._request('POST', '/api/v1/mhc/config', { model_id: modelId, config: config });
        },
        
        startMHCTraining: function(trainingRequest) {
            return this._request('POST', '/api/v1/mhc/train', trainingRequest);
        },
        
        getMHCJobs: function() {
            return this._request('GET', '/api/v1/mhc/jobs');
        },
        
        getMHCJob: function(jobId) {
            return this._request('GET', `/api/v1/mhc/jobs/${jobId}`);
        },
        
        // ========================================================================
        // ORCHESTRATION API
        // ========================================================================

        getAgents: function() {
            return this._request('GET', '/api/v1/agents');
        },

        createAgent: function(agentConfig) {
            return this._request('POST', '/api/v1/agents', agentConfig);
        },

        createWorkflow: function(workflowConfig) {
            return this._request('POST', '/api/v1/workflows', workflowConfig);
        },

        executeWorkflow: function(workflowId, input) {
            return this._request('POST', `/api/v1/workflows/${workflowId}/execute`, input);
        },

        // ========================================================================
        // INTELLIGENT MODEL ROUTING API
        // ========================================================================

        /**
         * Get all agent-model assignments
         */
        getAgentModelAssignments: function() {
            return this._request('GET', '/api/v1/model-router/assignments');
        },

        /**
         * Update agent-model assignment
         */
        updateAgentModelAssignment: function(agentId, modelId, isAutoAssigned) {
            return this._request('PUT', `/api/v1/model-router/assignments/${agentId}`, {
                model_id: modelId,
                auto_assigned: isAutoAssigned
            });
        },

        /**
         * Auto-assign models to all agents
         */
        autoAssignAllModels: function(strategy) {
            return this._request('POST', '/api/v1/model-router/auto-assign', {
                strategy: strategy || 'balanced'
            });
        },

        /**
         * Get routing decision for a task
         */
        getRoutingDecision: function(taskInput, agentType) {
            return this._request('POST', '/api/v1/model-router/route', {
                input: taskInput,
                agent_type: agentType
            });
        },

        /**
         * Record routing outcome for RL learning
         */
        recordRoutingOutcome: function(decisionId, success, latencyMs) {
            return this._request('POST', '/api/v1/model-router/outcome', {
                decision_id: decisionId,
                success: success,
                latency_ms: latencyMs
            });
        },

        /**
         * Get model routing statistics
         */
        getRoutingStats: function() {
            return this._request('GET', '/api/v1/model-router/stats');
        },

        /**
         * Get model router configuration
         */
        getRouterConfig: function() {
            return this._request('GET', '/api/v1/model-router/config');
        },

        /**
         * Update model router configuration
         */
        updateRouterConfig: function(config) {
            return this._request('PUT', '/api/v1/model-router/config', config);
        },

        /**
         * Register a new model in the router
         */
        registerModelInRouter: function(modelProfile) {
            return this._request('POST', '/api/v1/model-router/models', modelProfile);
        },
        
        // ========================================================================
        // WEBSOCKET REAL-TIME UPDATES
        // ========================================================================
        
        connectWebSocket: function() {
            this._shouldReconnect = true;
            if (this._wsConnection && this._wsConnection.readyState === WebSocket.OPEN) {
                console.log("WebSocket already connected");
                return;
            }
            
            try {
                this._wsConnection = new WebSocket(this.config.wsUrl);
                
                this._wsConnection.onopen = () => {
                    console.log("WebSocket connected");
                    this._reconnectDelayMs = 5000;
                    // Send auth token
                    this._wsConnection.send(JSON.stringify({
                        type: 'auth',
                        token: this.getToken()
                    }));
                };
                
                this._wsConnection.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        this._notifyCallbacks(message);
                    } catch (e) {
                        console.error("WebSocket message parse error:", e);
                    }
                };
                
                this._wsConnection.onerror = (error) => {
                    console.error("WebSocket error:", error);
                };
                
                this._wsConnection.onclose = () => {
                    if (!this._shouldReconnect) {
                        return;
                    }
                    console.log("WebSocket disconnected, reconnecting in " + this._reconnectDelayMs + "ms...");
                    var delay = this._reconnectDelayMs;
                    this._reconnectDelayMs = Math.min(30000, Math.round(this._reconnectDelayMs * 1.5));
                    setTimeout(() => this.connectWebSocket(), delay);
                };
                
            } catch (error) {
                console.error("WebSocket connection failed:", error);
            }
        },
        
        onMetricsUpdate: function(callback) {
            this._wsCallbacks.push(callback);
        },
        
        _notifyCallbacks: function(message) {
            if (message.type === 'metrics_update') {
                this._wsCallbacks.forEach(cb => {
                    try {
                        cb(message);
                    } catch (e) {
                        console.error("Callback error:", e);
                    }
                });
            }
        },
        
        disconnectWebSocket: function() {
            if (this._wsConnection) {
                this._shouldReconnect = false;
                this._wsConnection.close();
                this._wsConnection = null;
            }
        },
        
        // ========================================================================
        // UTILITY FUNCTIONS
        // ========================================================================
        
        exportData: function(data, filename, format = 'json') {
            const blob = format === 'json'
                ? new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
                : new Blob([this._convertToCSV(data)], { type: 'text/csv' });
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${filename}.${format}`;
            a.click();
            window.URL.revokeObjectURL(url);
        },
        
        _convertToCSV: function(data) {
            if (!Array.isArray(data) || data.length === 0) return '';
            
            const headers = Object.keys(data[0]).join(',');
            const rows = data.map(row => 
                Object.values(row).map(val => 
                    typeof val === 'string' ? `"${val}"` : val
                ).join(',')
            );
            
            return [headers, ...rows].join('\n');
        }
    };
    
    return ApiService;
});
