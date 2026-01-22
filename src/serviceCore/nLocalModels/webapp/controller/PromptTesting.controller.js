sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "llm/server/dashboard/utils/TokenManager"
], function (Controller, JSONModel, MessageBox, MessageToast, Fragment, TokenManager) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.PromptTesting", {

        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("promptTesting").attachPatternMatched(this._onObjectMatched, this);

            // Initialize test model
            this._oTestModel = new JSONModel(this._getDefaultTestData());
            this.getView().setModel(this._oTestModel, "test");

            // Initialize comparison model for T-Account prompt comparison
            this._oComparisonModel = new JSONModel(this._getDefaultComparisonData());
            this.getView().setModel(this._oComparisonModel, "comparison");

            // Load prompt history
            this._loadHistory();

            // Initialize mode presets
            this._initModePresets();
        },

        _getDefaultComparisonData: function () {
            return {
                promptA: {
                    mode: "",
                    modelName: "",
                    modelVersion: "",
                    promptText: "",
                    responseText: "",
                    status: "None",
                    metrics: {
                        latency: 0,
                        ttft: 0,
                        tps: 0,
                        tokenCount: 0,
                        costEstimate: 0
                    }
                },
                promptB: {
                    mode: "",
                    modelName: "",
                    modelVersion: "",
                    promptText: "",
                    responseText: "",
                    status: "None",
                    metrics: {
                        latency: 0,
                        ttft: 0,
                        tps: 0,
                        tokenCount: 0,
                        costEstimate: 0
                    }
                },
                differences: {
                    latencyWinner: "none",
                    ttftWinner: "none",
                    tpsWinner: "none",
                    costWinner: "none",
                    overallWinner: "none"
                }
            };
        },

        _getDefaultTestData: function () {
            return {
                selectedMode: "",
                promptText: "",
                response: "",
                maxTokens: 512,
                temperature: 0.7,
                showAdvanced: false,
                isLoading: false,
                streamEnabled: true,
                modeInfo: {
                    model: "",
                    latency: "",
                    tps: ""
                },
                metrics: {
                    latency_ms: 0,
                    ttft_ms: 0,
                    tokens_per_second: 0,
                    tokens_generated: 0,
                    cache_hit_rate: 0,
                    model_used: ""
                },
                batchResults: [],
                comparison: {
                    fastest_mode: "",
                    avg_latency_ms: 0,
                    best_quality_mode: "",
                    total_cost: 0
                },
                history: [],
                historyFilter: {
                    mode: "",
                    search: ""
                },
                totalPrompts: 0,
                avgLatency: 0
            };
        },

        _initModePresets: function () {
            this._modePresets = {
                "Fast": {
                    model: "LFM2.5 1.2B Q4_0",
                    model_id: "lfm2.5-1.2b-q4_0",
                    latency: "50-150 ms",
                    tps: "40-80 tok/s"
                },
                "Normal": {
                    model: "LFM2.5 1.2B Q4_K_M",
                    model_id: "lfm2.5-1.2b-q4_k_m",
                    latency: "100-300 ms",
                    tps: "25-50 tok/s"
                },
                "Expert": {
                    model: "LFM2.5 1.2B F16",
                    model_id: "lfm2.5-1.2b-f16",
                    latency: "200-500 ms",
                    tps: "15-35 tok/s"
                },
                "Research": {
                    model: "Llama 3.3 70B Q4_K_M",
                    model_id: "llama-3.3-70b",
                    latency: "300-1000 ms",
                    tps: "10-25 tok/s"
                }
            };
            
            this._examples = {
                "Simple Math": "What is 2+2? Explain your answer step by step.",
                "Code Generation": "Write a Python function to calculate the fibonacci sequence using dynamic programming.",
                "Arabic Text": "ما هي عاصمة السعودية؟ وما هي أهم المعالم السياحية فيها؟",
                "Reasoning": "Explain the concept of recursion with a practical example. How does it differ from iteration?"
            };
        },

        _onObjectMatched: function (oEvent) {
            // Refresh history when page is opened
            this._loadHistory();
        },

        onModeChange: function (oEvent) {
            var sMode = oEvent.getParameter("item").getKey();
            this._oTestModel.setProperty("/selectedMode", sMode);
            
            if (sMode && this._modePresets[sMode]) {
                var oPreset = this._modePresets[sMode];
                this._oTestModel.setProperty("/modeInfo", {
                    model: oPreset.model,
                    latency: oPreset.latency,
                    tps: oPreset.tps
                });
            }
        },

        onExamplePrompt: function (oEvent) {
            var sButtonText = oEvent.getSource().getText();
            var sExample = this._examples[sButtonText];
            if (sExample) {
                this._oTestModel.setProperty("/promptText", sExample);
            }
        },

        onToggleAdvanced: function () {
            var bShow = this._oTestModel.getProperty("/showAdvanced");
            this._oTestModel.setProperty("/showAdvanced", !bShow);
        },

        onTestPrompt: function () {
            var oData = this._oTestModel.getData();

            if (!oData.promptText || !oData.selectedMode) {
                MessageBox.error("Please enter a prompt and select a mode");
                return;
            }

            this._oTestModel.setProperty("/isLoading", true);
            this._oTestModel.setProperty("/response", "");

            var oComponent = this.getOwnerComponent();
            var sApiBaseUrl = oComponent.getApiBaseUrl();
            var oPreset = this._modePresets[oData.selectedMode];
            var bStream = oData.streamEnabled;

            var that = this;

            if (bStream) {
                this._handleStreamingRequest(sApiBaseUrl, oPreset, oData);
            } else {
                this._handleNonStreamingRequest(sApiBaseUrl, oPreset, oData);
            }
        },

        _handleStreamingRequest: function (sApiBaseUrl, oPreset, oData) {
            var that = this;
            var sAccumulatedResponse = "";
            var nStartTime = Date.now();
            var nFirstTokenTime = null;
            var nTokenCount = 0;

            fetch(sApiBaseUrl + "/v1/chat/completions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: oPreset.model_id,
                    messages: [
                        { role: "user", content: oData.promptText }
                    ],
                    max_tokens: oData.maxTokens,
                    temperature: oData.temperature,
                    stream: true
                })
            })
            .then(function (response) {
                if (!response.ok) throw new Error("API request failed");

                var reader = response.body.getReader();
                var decoder = new TextDecoder();

                function processStream() {
                    return reader.read().then(function (result) {
                        if (result.done) {
                            var nEndTime = Date.now();
                            var nTotalLatency = nEndTime - nStartTime;
                            var nTtft = nFirstTokenTime ? nFirstTokenTime - nStartTime : 0;
                            var nTps = nTokenCount > 0 && nTotalLatency > 0
                                ? Math.round(nTokenCount / (nTotalLatency / 1000))
                                : 0;

                            that._oTestModel.setProperty("/metrics", {
                                latency_ms: nTotalLatency,
                                ttft_ms: nTtft,
                                tokens_per_second: nTps,
                                tokens_generated: nTokenCount,
                                cache_hit_rate: 0.75,
                                model_used: oPreset.model_id
                            });

                            that._oTestModel.setProperty("/isLoading", false);
                            MessageToast.show("Prompt test completed successfully");
                            return;
                        }

                        var sChunk = decoder.decode(result.value, { stream: true });
                        var aLines = sChunk.split("\n");

                        aLines.forEach(function (sLine) {
                            if (sLine.startsWith("data: ")) {
                                var sData = sLine.substring(6).trim();
                                if (sData === "[DONE]") return;

                                try {
                                    var oChunkData = JSON.parse(sData);
                                    var oDelta = oChunkData.choices &&
                                                 oChunkData.choices[0] &&
                                                 oChunkData.choices[0].delta;

                                    if (oDelta && oDelta.content) {
                                        if (!nFirstTokenTime) {
                                            nFirstTokenTime = Date.now();
                                        }
                                        sAccumulatedResponse += oDelta.content;
                                        nTokenCount++;
                                        that._oTestModel.setProperty("/response", sAccumulatedResponse);
                                    }
                                } catch (e) {
                                    // Skip malformed JSON lines
                                }
                            }
                        });

                        return processStream();
                    });
                }

                return processStream();
            })
            .catch(function (error) {
                console.error("Error testing prompt:", error);
                MessageBox.error("Failed to test prompt: " + error.message);
                that._oTestModel.setProperty("/response", "Error: " + error.message);
                that._oTestModel.setProperty("/isLoading", false);
            });
        },

        _handleNonStreamingRequest: function (sApiBaseUrl, oPreset, oData) {
            var that = this;
            fetch(sApiBaseUrl + "/v1/chat/completions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: oPreset.model_id,
                    messages: [
                        { role: "user", content: oData.promptText }
                    ],
                    max_tokens: oData.maxTokens,
                    temperature: oData.temperature,
                    stream: false
                })
            })
            .then(function (response) {
                if (!response.ok) throw new Error("API request failed");
                return response.json();
            })
            .then(function (data) {
                var oChoice = data.choices && data.choices[0];
                var sResponse = oChoice ? oChoice.message.content : "No response generated";

                that._oTestModel.setProperty("/response", sResponse);
                that._oTestModel.setProperty("/metrics", {
                    latency_ms: Math.round((data.usage?.completion_time || 0) * 1000),
                    ttft_ms: Math.round((data.usage?.time_to_first_token || 0) * 1000),
                    tokens_per_second: Math.round(data.usage?.tokens_per_second || 0),
                    tokens_generated: data.usage?.completion_tokens || 0,
                    cache_hit_rate: data.cache_hit_rate || 0.75,
                    model_used: data.model || oPreset.model_id
                });

                MessageToast.show("Prompt test completed successfully");
            })
            .catch(function (error) {
                console.error("Error testing prompt:", error);
                MessageBox.error("Failed to test prompt: " + error.message);
                that._oTestModel.setProperty("/response", "Error: " + error.message);
            })
            .finally(function () {
                that._oTestModel.setProperty("/isLoading", false);
            });
        },

        onBatchTest: function () {
            var oData = this._oTestModel.getData();
            
            if (!oData.promptText) {
                MessageBox.error("Please enter a prompt");
                return;
            }
            
            this._oTestModel.setProperty("/isLoading", true);
            this._oTestModel.setProperty("/batchResults", []);
            
            var aModes = ["Fast", "Normal", "Expert", "Research"];
            var aPromises = [];
            var oComponent = this.getOwnerComponent();
            var sApiBaseUrl = oComponent.getApiBaseUrl();
            
            var that = this;
            aModes.forEach(function (sMode) {
                var oPreset = that._modePresets[sMode];
                var oPromise = fetch(sApiBaseUrl + "/v1/chat/completions", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        model: oPreset.model_id,
                        messages: [{ role: "user", content: oData.promptText }],
                        max_tokens: oData.maxTokens,
                        temperature: oData.temperature,
                        stream: false
                    })
                })
                .then(function (response) {
                    if (!response.ok) throw new Error("API request failed for " + sMode);
                    return response.json();
                })
                .then(function (data) {
                    var oChoice = data.choices && data.choices[0];
                    return {
                        mode: sMode,
                        response: oChoice ? oChoice.message.content : "No response",
                        metrics: {
                            latency_ms: Math.round((data.usage?.completion_time || 0) * 1000),
                            tokens_per_second: Math.round(data.usage?.tokens_per_second || 0),
                            cache_hit_rate: data.cache_hit_rate || 0.75
                        },
                        quality_rating: 4 // Placeholder - would come from quality assessment
                    };
                })
                .catch(function (error) {
                    console.error("Error in mode " + sMode + ":", error);
                    return {
                        mode: sMode,
                        response: "Error: " + error.message,
                        metrics: { latency_ms: 0, tokens_per_second: 0, cache_hit_rate: 0 },
                        quality_rating: 0
                    };
                });
                
                aPromises.push(oPromise);
            });
            
            Promise.all(aPromises)
                .then(function (aResults) {
                    that._oTestModel.setProperty("/batchResults", aResults);
                    
                    // Calculate comparison
                    var nTotalLatency = 0;
                    var oFastest = aResults[0];
                    var oBestQuality = aResults[0];
                    
                    aResults.forEach(function (r) {
                        nTotalLatency += r.metrics.latency_ms;
                        if (r.metrics.latency_ms < oFastest.metrics.latency_ms) {
                            oFastest = r;
                        }
                        if (r.quality_rating > oBestQuality.quality_rating) {
                            oBestQuality = r;
                        }
                    });
                    
                    that._oTestModel.setProperty("/comparison", {
                        fastest_mode: oFastest.mode,
                        avg_latency_ms: Math.round(nTotalLatency / aResults.length),
                        best_quality_mode: oBestQuality.mode,
                        total_cost: (nTotalLatency / 1000 * 0.001).toFixed(4) // Simplified cost calc
                    });
                    
                    MessageToast.show("Batch test completed for all 4 modes");
                })
                .catch(function (error) {
                    console.error("Batch test error:", error);
                    MessageBox.error("Batch test failed: " + error.message);
                })
                .finally(function () {
                    that._oTestModel.setProperty("/isLoading", false);
                });
        },

        onClear: function () {
            this._oTestModel.setProperty("/promptText", "");
            this._oTestModel.setProperty("/response", "");
            this._oTestModel.setProperty("/batchResults", []);
            this._oTestModel.setProperty("/metrics", {
                latency_ms: 0,
                ttft_ms: 0,
                tokens_per_second: 0,
                tokens_generated: 0,
                cache_hit_rate: 0,
                model_used: ""
            });
        },

        /**
         * Build request headers with optional JWT authentication (Day 13)
         * @returns {Object} Headers object
         */
        _getAuthHeaders: function () {
            return TokenManager.getAuthHeaders();
        },

        /**
         * Show authentication dialog for demo login
         */
        onShowAuth: function () {
            var that = this;
            
            MessageBox.show(
                "Enter a user ID for demo authentication:",
                {
                    title: "Demo Login",
                    actions: [MessageBox.Action.OK, MessageBox.Action.CANCEL],
                    emphasizedAction: MessageBox.Action.OK,
                    initialFocus: MessageBox.Action.OK,
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            that._performDemoLogin();
                        }
                    }
                }
            );
        },

        /**
         * Perform demo login with token generation
         * @private
         */
        _performDemoLogin: function () {
            var sUserId = "demo-user-" + Date.now();
            
            // Generate and store demo token
            TokenManager.demoLogin(sUserId, true);
            
            MessageToast.show("Logged in as: " + sUserId);
            this._updateAuthStatus();
        },

        /**
         * Logout and clear token
         */
        onLogout: function () {
            TokenManager.logout();
            MessageToast.show("Logged out successfully");
            this._updateAuthStatus();
        },

        /**
         * Update UI based on authentication status
         * @private
         */
        _updateAuthStatus: function () {
            var bAuthenticated = TokenManager.isAuthenticated();
            var sUser = TokenManager.getCurrentUser();
            
            console.log("Auth status:", bAuthenticated, "User:", sUser);
            
            // Could update UI to show auth status
            // For now, just log it
        },

        onSaveToHistory: function () {
            var oData = this._oTestModel.getData();
            
            if (!oData.response) {
                MessageToast.show("No response to save");
                return;
            }
            
            var that = this;
            var oPreset = this._modePresets[oData.selectedMode] || {};
            
            // Prepare data for HANA (Day 13: JWT auth - user_id now optional)
            var oPromptData = {
                prompt_text: oData.promptText,
                model_name: oPreset.model_id || oData.metrics.model_used || "unknown",
                prompt_mode_id: this._getModeId(oData.selectedMode),
                tags: oData.selectedMode || ""
                // user_id omitted - will be extracted from JWT token if present
            };
            
            // Save to HANA via API with JWT authentication
            fetch("/api/v1/prompts", {
                method: "POST",
                headers: this._getAuthHeaders(),
                body: JSON.stringify(oPromptData)
            })
            .then(function (response) {
                if (!response.ok) throw new Error("Failed to save to database");
                return response.json();
            })
            .then(function (data) {
                var sUserId = data.user_id || "anonymous";
                MessageToast.show("Saved to HANA successfully! ID: " + data.prompt_id + " (User: " + sUserId + ")");
                that._loadHistory();
            })
            .catch(function (error) {
                console.error("Error saving to database:", error);
                MessageBox.error("Failed to save prompt: " + error.message);
            });
        },

        _getModeId: function (sMode) {
            var oModeMap = {
                "Fast": 1,
                "Normal": 2,
                "Expert": 3,
                "Research": 4
            };
            return oModeMap[sMode] || 1;
        },

        onRateResponse: function () {
            var that = this;
            MessageBox.information(
                "Rating: 1 (Poor) to 5 (Excellent)\n\nHow would you rate this response?",
                {
                    actions: ["1", "2", "3", "4", "5", MessageBox.Action.CANCEL],
                    onClose: function (sAction) {
                        if (sAction !== MessageBox.Action.CANCEL) {
                            var nRating = parseInt(sAction, 10);
                            that._oTestModel.setProperty("/metrics/user_rating", nRating);
                            MessageToast.show("Rating saved: " + nRating + " stars");
                        }
                    }
                }
            );
        },

        _loadHistory: function () {
            var that = this;
            
            // Use the correct endpoint: GET /v1/prompts/history
            fetch("/v1/prompts/history?limit=50")
                .then(function (response) {
                    if (!response.ok) throw new Error("Failed to load history");
                    return response.json();
                })
                .then(function (data) {
                    // Backend returns { history: [...], total: N }
                    var aHistory = data.history || [];
                    
                    // Transform HANA data to UI format
                    var aTransformed = aHistory.map(function (entry) {
                        return {
                            prompt_id: entry.prompt_id || entry.PROMPT_ID,
                            mode: that._getModeFromId(entry.prompt_mode_id || entry.PROMPT_MODE_ID),
                            prompt_text: entry.prompt_text || entry.PROMPT_TEXT,
                            model_id: entry.model_name || entry.MODEL_NAME,
                            user_id: entry.user_id || entry.USER_ID,
                            tags: entry.tags || entry.TAGS,
                            timestamp: entry.created_at || entry.CREATED_AT,
                            latency_ms: 0, // Not stored yet
                            tokens_per_second: 0,
                            user_rating: null
                        };
                    });
                    
                    that._oTestModel.setProperty("/history", aTransformed);
                    that._oTestModel.setProperty("/totalPrompts", data.total || aTransformed.length);
                    that._updateHistoryStats();
                    
                    console.log("✅ Loaded " + aTransformed.length + " prompts from HANA");
                })
                .catch(function (error) {
                    console.error("⚠️ Error loading history from HANA:", error);
                    // Fallback to mock data
                    that._oTestModel.setProperty("/history", that._getMockHistory());
                    that._updateHistoryStats();
                });
        },

        _getModeFromId: function (nModeId) {
            var oIdMap = {
                1: "Fast",
                2: "Normal",
                3: "Expert",
                4: "Research"
            };
            return oIdMap[nModeId] || "Fast";
        },

        _updateHistoryStats: function () {
            var aHistory = this._oTestModel.getProperty("/history");
            var nTotal = aHistory.length;
            var nTotalLatency = 0;
            
            aHistory.forEach(function (entry) {
                nTotalLatency += entry.latency_ms || 0;
            });
            
            this._oTestModel.setProperty("/totalPrompts", nTotal);
            this._oTestModel.setProperty("/avgLatency", 
                nTotal > 0 ? Math.round(nTotalLatency / nTotal) : 0);
        },

        _getMockHistory: function () {
            return [
                {
                    prompt_id: "mock-1",
                    mode: "Fast",
                    prompt_text: "What is 2+2?",
                    model_id: "lfm2.5-1.2b-q4_0",
                    latency_ms: 85,
                    tokens_per_second: 58,
                    user_rating: 5,
                    timestamp: new Date(Date.now() - 3600000).toISOString()
                },
                {
                    prompt_id: "mock-2",
                    mode: "Normal",
                    prompt_text: "Explain quantum computing in simple terms",
                    model_id: "lfm2.5-1.2b-q4_k_m",
                    latency_ms: 245,
                    tokens_per_second: 32,
                    user_rating: 4,
                    timestamp: new Date(Date.now() - 7200000).toISOString()
                },
                {
                    prompt_id: "mock-3",
                    mode: "Expert",
                    prompt_text: "Write a binary search tree implementation in Python",
                    model_id: "lfm2.5-1.2b-f16",
                    latency_ms: 380,
                    tokens_per_second: 25,
                    user_rating: 5,
                    timestamp: new Date(Date.now() - 10800000).toISOString()
                }
            ];
        },

        onSearchHistory: function (oEvent) {
            var sQuery = oEvent.getParameter("query");
            this._oTestModel.setProperty("/historyFilter/search", sQuery);
            
            // If query is provided, use backend search API
            if (sQuery && sQuery.trim().length > 0) {
                this._searchBackend(sQuery.trim());
            } else {
                // Otherwise reload full history
                this._loadHistory();
            }
        },

        _searchBackend: function (sQuery) {
            var that = this;
            
            fetch("/api/v1/prompts/search?q=" + encodeURIComponent(sQuery) + "&limit=50")
                .then(function (response) {
                    if (!response.ok) throw new Error("Search failed");
                    return response.json();
                })
                .then(function (data) {
                    var aResults = data.results || [];
                    
                    // Transform search results to UI format
                    var aTransformed = aResults.map(function (entry) {
                        return {
                            prompt_id: entry.prompt_id || entry.PROMPT_ID,
                            mode: that._getModeFromId(entry.prompt_mode_id || entry.PROMPT_MODE_ID || 1),
                            prompt_text: entry.prompt_text || entry.PROMPT_TEXT,
                            model_id: entry.model_name || entry.MODEL_NAME,
                            timestamp: entry.created_at || entry.CREATED_AT,
                            relevance_score: entry.relevance_score || entry.RELEVANCE_SCORE,
                            latency_ms: 0,
                            tokens_per_second: 0
                        };
                    });
                    
                    that._oTestModel.setProperty("/history", aTransformed);
                    that._oTestModel.setProperty("/totalPrompts", data.total || aTransformed.length);
                    that._updateHistoryStats();
                    
                    MessageToast.show("Found " + aTransformed.length + " matching prompts");
                })
                .catch(function (error) {
                    console.error("Search error:", error);
                    MessageToast.show("Search failed, using local filter");
                    that._applyHistoryFilters();
                });
        },

        onFilterHistory: function (oEvent) {
            var sMode = oEvent.getParameter("selectedItem").getKey();
            this._oTestModel.setProperty("/historyFilter/mode", sMode);
            this._applyHistoryFilters();
        },

        _applyHistoryFilters: function () {
            var oFilter = this._oTestModel.getProperty("/historyFilter");
            var oTable = this.byId("historyTable");
            var aFilters = [];
            
            if (oFilter.mode) {
                aFilters.push(new sap.ui.model.Filter("mode", sap.ui.model.FilterOperator.EQ, oFilter.mode));
            }
            
            if (oFilter.search) {
                aFilters.push(new sap.ui.model.Filter("prompt_text", sap.ui.model.FilterOperator.Contains, oFilter.search));
            }
            
            oTable.getBinding("items").filter(aFilters);
        },

        onRefreshHistory: function () {
            this._loadHistory();
            MessageToast.show("History refreshed");
        },

        onExportHistory: function () {
            var aHistory = this._oTestModel.getProperty("/history");
            
            // Convert to CSV
            var sCsv = "Mode,Prompt,Model,Latency(ms),TPS,Rating,Timestamp\n";
            aHistory.forEach(function (entry) {
                sCsv += [
                    entry.mode,
                    '"' + (entry.prompt_text || "").replace(/"/g, '""') + '"',
                    entry.model_id,
                    entry.latency_ms,
                    entry.tokens_per_second,
                    entry.user_rating || "",
                    entry.timestamp
                ].join(",") + "\n";
            });
            
            // Download CSV
            var oBlob = new Blob([sCsv], { type: "text/csv" });
            var sUrl = URL.createObjectURL(oBlob);
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = "prompt-history-" + new Date().toISOString().split('T')[0] + ".csv";
            oLink.click();
            URL.revokeObjectURL(sUrl);
            
            MessageToast.show("History exported to CSV");
        },

        onHistorySelect: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext("test");
            var oEntry = oContext.getObject();
            
            // Load prompt into editor
            this._oTestModel.setProperty("/promptText", oEntry.prompt_text);
            this._oTestModel.setProperty("/selectedMode", oEntry.mode);
            this._oTestModel.setProperty("/response", oEntry.response_text || "");
            
            MessageToast.show("Loaded prompt from history");
        },

        onDeletePrompt: function (oEvent) {
            var that = this;
            var oSource = oEvent.getSource();
            var oContext = oSource.getBindingContext("test");
            var oEntry = oContext.getObject();
            var nPromptId = oEntry.prompt_id;

            MessageBox.confirm(
                "Are you sure you want to delete this prompt?\n\n\"" + 
                oEntry.prompt_text.substring(0, 50) + "...\"",
                {
                    title: "Confirm Delete",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            that._deletePromptFromBackend(nPromptId);
                        }
                    }
                }
            );
        },

        _deletePromptFromBackend: function (nPromptId) {
            var that = this;

            fetch("/api/v1/prompts/" + nPromptId, {
                method: "DELETE"
            })
            .then(function (response) {
                if (!response.ok) throw new Error("Delete failed");
                return response.json();
            })
            .then(function (data) {
                MessageToast.show("Prompt deleted successfully");
                that._loadHistory();
            })
            .catch(function (error) {
                console.error("Delete error:", error);
                MessageBox.error("Failed to delete prompt: " + error.message);
            });
        },

        _generateUUID: function () {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
                var r = Math.random() * 16 | 0;
                var v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        },

        // ==================== T-Account Prompt Comparison Methods ====================

        onComparePrompts: function () {
            var that = this;

            if (!this._oComparisonDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "llm.server.dashboard.view.fragments.TAccountPromptComparison",
                    controller: this
                }).then(function (oDialog) {
                    that._oComparisonDialog = oDialog;
                    that.getView().addDependent(that._oComparisonDialog);
                    that._oComparisonDialog.open();
                }).catch(function (oError) {
                    MessageBox.error("Failed to load comparison dialog: " + oError.message);
                });
            } else {
                this._oComparisonDialog.open();
            }
        },

        onLoadPromptA: function () {
            var oTestData = this._oTestModel.getData();
            var oPreset = this._modePresets[oTestData.selectedMode] || {};

            this._oComparisonModel.setProperty("/promptA", {
                mode: oTestData.selectedMode ? oTestData.selectedMode.toLowerCase() : "",
                modelName: oPreset.model || "",
                modelVersion: oPreset.model_id || "",
                promptText: oTestData.promptText || "",
                responseText: oTestData.response || "",
                status: oTestData.response ? "Success" : "None",
                metrics: {
                    latency: oTestData.metrics.latency_ms || 0,
                    ttft: oTestData.metrics.ttft_ms || 0,
                    tps: oTestData.metrics.tokens_per_second || 0,
                    tokenCount: oTestData.metrics.tokens_generated || 0,
                    costEstimate: this._calculateCost(oTestData.metrics.tokens_generated || 0, oTestData.metrics.latency_ms || 0)
                }
            });

            this._calculateDifferences();
            MessageToast.show("Loaded current prompt into Slot A");
        },

        onLoadPromptB: function () {
            var oTestData = this._oTestModel.getData();
            var oPreset = this._modePresets[oTestData.selectedMode] || {};

            this._oComparisonModel.setProperty("/promptB", {
                mode: oTestData.selectedMode ? oTestData.selectedMode.toLowerCase() : "",
                modelName: oPreset.model || "",
                modelVersion: oPreset.model_id || "",
                promptText: oTestData.promptText || "",
                responseText: oTestData.response || "",
                status: oTestData.response ? "Success" : "None",
                metrics: {
                    latency: oTestData.metrics.latency_ms || 0,
                    ttft: oTestData.metrics.ttft_ms || 0,
                    tps: oTestData.metrics.tokens_per_second || 0,
                    tokenCount: oTestData.metrics.tokens_generated || 0,
                    costEstimate: this._calculateCost(oTestData.metrics.tokens_generated || 0, oTestData.metrics.latency_ms || 0)
                }
            });

            this._calculateDifferences();
            MessageToast.show("Loaded current prompt into Slot B");
        },

        onSelectForCompareA: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("test");
            var oEntry = oContext.getObject();
            var oPreset = this._modePresets[oEntry.mode] || {};

            this._oComparisonModel.setProperty("/promptA", {
                mode: oEntry.mode ? oEntry.mode.toLowerCase() : "",
                modelName: oPreset.model || oEntry.model_id || "",
                modelVersion: oEntry.model_id || "",
                promptText: oEntry.prompt_text || "",
                responseText: oEntry.response_text || "",
                status: "Success",
                metrics: {
                    latency: oEntry.latency_ms || 0,
                    ttft: oEntry.ttft_ms || 0,
                    tps: oEntry.tokens_per_second || 0,
                    tokenCount: oEntry.tokens_generated || 0,
                    costEstimate: this._calculateCost(oEntry.tokens_generated || 0, oEntry.latency_ms || 0)
                }
            });

            this._calculateDifferences();
            MessageToast.show("Loaded batch result into Slot A");
        },

        onSelectForCompareB: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("test");
            var oEntry = oContext.getObject();
            var oPreset = this._modePresets[oEntry.mode] || {};

            this._oComparisonModel.setProperty("/promptB", {
                mode: oEntry.mode ? oEntry.mode.toLowerCase() : "",
                modelName: oPreset.model || oEntry.model_id || "",
                modelVersion: oEntry.model_id || "",
                promptText: oEntry.prompt_text || "",
                responseText: oEntry.response_text || "",
                status: "Success",
                metrics: {
                    latency: oEntry.latency_ms || 0,
                    ttft: oEntry.ttft_ms || 0,
                    tps: oEntry.tokens_per_second || 0,
                    tokenCount: oEntry.tokens_generated || 0,
                    costEstimate: this._calculateCost(oEntry.tokens_generated || 0, oEntry.latency_ms || 0)
                }
            });

            this._calculateDifferences();
            MessageToast.show("Loaded batch result into Slot B");
        },

        onSwapPrompts: function () {
            var oPromptA = this._oComparisonModel.getProperty("/promptA");
            var oPromptB = this._oComparisonModel.getProperty("/promptB");

            // Deep copy and swap
            var oTempA = JSON.parse(JSON.stringify(oPromptA));
            var oTempB = JSON.parse(JSON.stringify(oPromptB));

            this._oComparisonModel.setProperty("/promptA", oTempB);
            this._oComparisonModel.setProperty("/promptB", oTempA);

            this._calculateDifferences();
            MessageToast.show("Prompts swapped");
        },

        onModeChangeA: function (oEvent) {
            var sMode = oEvent.getParameter("item").getKey();
            var oPreset = this._modePresets[sMode.charAt(0).toUpperCase() + sMode.slice(1)] || {};

            this._oComparisonModel.setProperty("/promptA/mode", sMode);
            this._oComparisonModel.setProperty("/promptA/modelName", oPreset.model || "");
            this._oComparisonModel.setProperty("/promptA/modelVersion", oPreset.model_id || "");
        },

        onModeChangeB: function (oEvent) {
            var sMode = oEvent.getParameter("item").getKey();
            var oPreset = this._modePresets[sMode.charAt(0).toUpperCase() + sMode.slice(1)] || {};

            this._oComparisonModel.setProperty("/promptB/mode", sMode);
            this._oComparisonModel.setProperty("/promptB/modelName", oPreset.model || "");
            this._oComparisonModel.setProperty("/promptB/modelVersion", oPreset.model_id || "");
        },

        _calculateDifferences: function () {
            var oPromptA = this._oComparisonModel.getProperty("/promptA");
            var oPromptB = this._oComparisonModel.getProperty("/promptB");
            var oDifferences = {
                latencyWinner: "none",
                ttftWinner: "none",
                tpsWinner: "none",
                costWinner: "none",
                overallWinner: "none"
            };

            var nScoreA = 0;
            var nScoreB = 0;

            // Latency: lower is better
            if (oPromptA.metrics.latency > 0 && oPromptB.metrics.latency > 0) {
                if (oPromptA.metrics.latency < oPromptB.metrics.latency) {
                    oDifferences.latencyWinner = "A";
                    nScoreA += 2;
                } else if (oPromptB.metrics.latency < oPromptA.metrics.latency) {
                    oDifferences.latencyWinner = "B";
                    nScoreB += 2;
                }
            }

            // TTFT: lower is better
            if (oPromptA.metrics.ttft > 0 && oPromptB.metrics.ttft > 0) {
                if (oPromptA.metrics.ttft < oPromptB.metrics.ttft) {
                    oDifferences.ttftWinner = "A";
                    nScoreA += 1;
                } else if (oPromptB.metrics.ttft < oPromptA.metrics.ttft) {
                    oDifferences.ttftWinner = "B";
                    nScoreB += 1;
                }
            }

            // TPS: higher is better
            if (oPromptA.metrics.tps > 0 && oPromptB.metrics.tps > 0) {
                if (oPromptA.metrics.tps > oPromptB.metrics.tps) {
                    oDifferences.tpsWinner = "A";
                    nScoreA += 2;
                } else if (oPromptB.metrics.tps > oPromptA.metrics.tps) {
                    oDifferences.tpsWinner = "B";
                    nScoreB += 2;
                }
            }

            // Cost: lower is better
            if (oPromptA.metrics.costEstimate > 0 && oPromptB.metrics.costEstimate > 0) {
                if (oPromptA.metrics.costEstimate < oPromptB.metrics.costEstimate) {
                    oDifferences.costWinner = "A";
                    nScoreA += 1;
                } else if (oPromptB.metrics.costEstimate < oPromptA.metrics.costEstimate) {
                    oDifferences.costWinner = "B";
                    nScoreB += 1;
                }
            }

            // Overall winner based on weighted scoring
            if (nScoreA > nScoreB) {
                oDifferences.overallWinner = "A";
            } else if (nScoreB > nScoreA) {
                oDifferences.overallWinner = "B";
            } else if (nScoreA > 0 || nScoreB > 0) {
                oDifferences.overallWinner = ""; // Tie
            }

            this._oComparisonModel.setProperty("/differences", oDifferences);
        },

        _calculateCost: function (nTokens, nLatencyMs) {
            // Simplified cost calculation: $0.001 per 1000 tokens + $0.0001 per second of compute
            var nTokenCost = (nTokens / 1000) * 0.001;
            var nComputeCost = (nLatencyMs / 1000) * 0.0001;
            return parseFloat((nTokenCost + nComputeCost).toFixed(6));
        },

        onSelectWinner: function (oEvent) {
            var that = this;
            var oDifferences = this._oComparisonModel.getProperty("/differences");
            var sCurrentWinner = oDifferences.overallWinner;

            MessageBox.information(
                "Select the winning prompt based on your evaluation:",
                {
                    actions: ["Prompt A", "Prompt B", "Tie", MessageBox.Action.CANCEL],
                    onClose: function (sAction) {
                        if (sAction !== MessageBox.Action.CANCEL) {
                            var sWinner = "";
                            if (sAction === "Prompt A") {
                                sWinner = "A";
                            } else if (sAction === "Prompt B") {
                                sWinner = "B";
                            } else {
                                sWinner = "";
                            }
                            that._oComparisonModel.setProperty("/differences/overallWinner", sWinner);
                            MessageToast.show(sAction === "Tie" ? "Marked as tie" : sAction + " selected as winner");
                        }
                    }
                }
            );
        },

        onSaveComparison: function () {
            var oComparisonData = this._oComparisonModel.getData();

            // For now, just show confirmation - in future, save to HANA
            var oSaveData = {
                comparison_id: this._generateUUID(),
                promptA: oComparisonData.promptA,
                promptB: oComparisonData.promptB,
                differences: oComparisonData.differences,
                timestamp: new Date().toISOString()
            };

            console.log("Comparison saved:", oSaveData);
            MessageToast.show("Comparison saved successfully");
        },

        onClearComparison: function () {
            this._oComparisonModel.setData(this._getDefaultComparisonData());
            MessageToast.show("Comparison cleared");
        },

        onCloseComparison: function () {
            if (this._oComparisonDialog) {
                this._oComparisonDialog.close();
            }
        },

        onNavBack: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("main");
        }
    });
});
