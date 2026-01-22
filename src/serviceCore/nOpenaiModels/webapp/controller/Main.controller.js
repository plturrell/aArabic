sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "llm/server/dashboard/utils/ApiService"
], function (Controller, JSONModel, ApiService) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.Main", {

        onInit: function () {
            // Initialize Keycloak authentication
            ApiService.initKeycloak()
                .then(() => {
                    console.log("Authentication successful");
                    this._initializeDashboard();
                })
                .catch(error => {
                    console.error("Authentication failed:", error);
                    // Fallback to mock mode for development
                    this._initializeDashboard();
                });
        },
        
        _initializeDashboard: function() {
            // Initialize metrics model
            var oMetricsModel = this.getOwnerComponent().getModel("metrics");
            
            if (!oMetricsModel) {
                oMetricsModel = new JSONModel();
                this.getOwnerComponent().setModel(oMetricsModel, "metrics");
            }
            
            // Initialize chat model for quick prompts
            var oChatModel = new JSONModel({
                quickPrompt: "",
                quickResponse: "",
                lastMetrics: {
                    latency: 0,
                    ttft: 0,
                    tps: 0,
                    tokens: 0
                },
                historicalAvg: {
                    latency: 67,
                    ttft: 35,
                    tps: 54
                }
            });
            this.getView().setModel(oChatModel, "chat");
            
            // Load initial data from API
            this._loadModels();
            this._loadCurrentMetrics();
            this._loadTierStats();
            
            // Connect WebSocket for real-time updates
            ApiService.connectWebSocket();
            ApiService.onMetricsUpdate(this._onMetricsUpdate.bind(this));
            
            // Fallback: Poll metrics every 10 seconds if WebSocket fails
            this._pollingInterval = setInterval(() => {
                this._loadCurrentMetrics();
            }, 10000);
        },
        
        _loadModels: function() {
            ApiService.getModels()
                .then(response => {
                    var oModel = this.getOwnerComponent().getModel("metrics");
                    
                    // Transform API response to match UI model
                    var models = response.models || response.available_models || [];
                    var formattedModels = models.map(model => ({
                        id: model.id,
                        display_name: model.name || model.display_name,
                        quantization: model.quantization,
                        architecture: model.architecture,
                        parameter_count: model.size_mb ? (model.size_mb / 1000).toFixed(1) + "GB" : "N/A",
                        health: model.status === "active" || model.status === "ready" ? "healthy" : "error",
                        avgLatency: model.avg_latency || 0,
                        avgThroughput: model.avg_throughput || 0,
                        requests: model.total_requests || 0,
                        lastUsed: model.last_used || "Never"
                    }));
                    
                    oModel.setProperty("/availableModels", formattedModels);
                    
                    // Set selected model (first active model)
                    var activeModel = formattedModels.find(m => m.health === "healthy");
                    if (activeModel) {
                        oModel.setProperty("/selectedModel", activeModel.id);
                    }
                    
                    console.log("Loaded", formattedModels.length, "models from API");
                })
                .catch(error => {
                    console.error("Failed to load models:", error);
                    // Fallback to mock data
                    this._initializeMockData(this.getOwnerComponent().getModel("metrics"));
                });
        },
        
        _loadCurrentMetrics: function() {
            var oModel = this.getOwnerComponent().getModel("metrics");
            var selectedModel = oModel.getProperty("/selectedModel");
            
            if (!selectedModel) return;
            
            ApiService.getCurrentMetrics(selectedModel)
                .then(metrics => {
                    this._updateMetricsModel(metrics);
                })
                .catch(error => {
                    console.error("Failed to load metrics:", error);
                });
        },
        
        _loadTierStats: function() {
            ApiService.getTierStats()
                .then(tiers => {
                    var oModel = this.getOwnerComponent().getModel("metrics");
                    oModel.setProperty("/tiers", tiers);
                })
                .catch(error => {
                    console.error("Failed to load tier stats:", error);
                });
        },
        
        _onMetricsUpdate: function(message) {
            // WebSocket real-time update
            if (message && message.metrics) {
                this._updateMetricsModel(message.metrics);
            }
            
            if (message && message.tiers) {
                var oModel = this.getOwnerComponent().getModel("metrics");
                oModel.setProperty("/tiers", message.tiers);
            }
        },
        
        _updateMetricsModel: function(metrics) {
            var oModel = this.getOwnerComponent().getModel("metrics");
            
            oModel.setProperty("/selectedModelData/current", {
                p50: metrics.latency_p50 || metrics.p50 || 0,
                p95: metrics.latency_p95 || metrics.p95 || 0,
                p99: metrics.latency_p99 || metrics.p99 || 0,
                tps: metrics.throughput || metrics.tps || 0,
                ttft: metrics.ttft || 0,
                cacheHit: metrics.cache_hit_rate || 0,
                queueDepth: metrics.queue_depth || 0,
                totalTokens: metrics.tokens_total || (metrics.tokens_input + metrics.tokens_output) || 0
            });
            
            oModel.setProperty("/connected", true);
        },
        
        onModelChange: function (oEvent) {
            var sSelectedModel = oEvent.getParameter("selectedItem").getKey();
            console.log("Model changed to:", sSelectedModel);
            
            // Load model via API
            ApiService.loadModel(sSelectedModel)
                .then(() => {
                    sap.m.MessageToast.show("Switched to " + oEvent.getParameter("selectedItem").getText());
                    
                    // Reload metrics for new model
                    this._loadCurrentMetrics();
                    
                    // Load history for new model
                    ApiService.getMetricsHistory(sSelectedModel, '1h')
                        .then(history => {
                            this._updateHistoryData(history);
                        });
                })
                .catch(error => {
                    sap.m.MessageBox.error("Failed to switch model: " + error.message);
                });
        },
        
        _updateHistoryData: function(history) {
            var oModel = this.getOwnerComponent().getModel("metrics");
            
            // Transform history data for charts
            if (history && history.data_points) {
                oModel.setProperty("/selectedModelData/latencyHistory", history.data_points.map(dp => ({
                    timestamp: dp.timestamp,
                    p50: dp.latency_p50,
                    p95: dp.latency_p95,
                    p99: dp.latency_p99
                })));
                
                oModel.setProperty("/selectedModelData/throughputHistory", history.data_points.map(dp => ({
                    timestamp: dp.timestamp,
                    tps: dp.throughput
                })));
                
                // Add other history arrays as needed
            }
        },
        
        onModelSelectFromTable: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext("metrics");
            var sModelId = oContext.getProperty("id");
            
            // Update selected model
            var oModel = this.getOwnerComponent().getModel("metrics");
            oModel.setProperty("/selectedModel", sModelId);
            
            // Trigger model change
            this._loadCurrentMetrics();
            
            ApiService.getMetricsHistory(sModelId, '1h')
                .then(history => {
                    this._updateHistoryData(history);
                });
        },
        
        onQuickPrompt: function () {
            var oChatModel = this.getView().getModel("chat");
            var sPrompt = oChatModel.getProperty("/quickPrompt");
            
            if (!sPrompt) {
                sap.m.MessageToast.show("Please enter a prompt");
                return;
            }
            
            var oMetricsModel = this.getOwnerComponent().getModel("metrics");
            var sModel = oMetricsModel.getProperty("/selectedModel");
            
            var startTime = Date.now();
            
            // Call real API
            ApiService.sendChatCompletion({
                model: sModel,
                messages: [{ role: "user", content: sPrompt }],
                max_tokens: 512,
                temperature: 0.7,
                stream: false
            })
            .then(response => {
                var endTime = Date.now();
                var latency = endTime - startTime;
                
                // Extract response
                var content = response.choices && response.choices[0] 
                    ? response.choices[0].message.content 
                    : "No response";
                
                // Extract metrics
                var usage = response.usage || {};
                
                oChatModel.setProperty("/quickResponse", content);
                oChatModel.setProperty("/lastMetrics", {
                    latency: latency,
                    ttft: response.ttft_ms || 0,
                    tps: usage.completion_tokens && response.generation_time 
                        ? (usage.completion_tokens / (response.generation_time / 1000)).toFixed(1)
                        : 0,
                    tokens: usage.total_tokens || 0
                });
                
                // Save to history
                var userProfile = ApiService.getUserProfile();
                ApiService.savePrompt({
                    user_id: userProfile.username,
                    model_id: sModel,
                    prompt_text: sPrompt,
                    response_text: content,
                    latency_ms: latency,
                    ttft_ms: response.ttft_ms || 0,
                    tokens_generated: usage.completion_tokens || 0,
                    tokens_per_second: usage.completion_tokens && response.generation_time 
                        ? usage.completion_tokens / (response.generation_time / 1000)
                        : 0,
                    prompt_tokens: usage.prompt_tokens || 0
                }).catch(err => console.warn("Failed to save prompt:", err));
                
                sap.m.MessageToast.show("Response received in " + latency + "ms");
            })
            .catch(error => {
                sap.m.MessageBox.error("Prompt execution failed: " + error.message);
                console.error("Prompt error:", error);
            });
        },
        
        onClearQuickPrompt: function () {
            var oChatModel = this.getView().getModel("chat");
            oChatModel.setProperty("/quickPrompt", "");
            oChatModel.setProperty("/quickResponse", "");
        },
        
        onBreadcrumbHome: function () {
            // Already on Dashboard (it's now the home page)
            var oNavContainer = this.getView().getParent().getParent().byId("navContainer");
            if (oNavContainer) {
                oNavContainer.to(oNavContainer.getPages()[0]);
            }
        },
        
        onOpenModelConfigurator: function () {
            // Initialize model config model if not exists
            if (!this._modelConfigDialog) {
                this._initializeModelConfig();
            }
            
            // Load current model's configuration
            var oMetricsModel = this.getOwnerComponent().getModel("metrics");
            var sSelectedModel = oMetricsModel.getProperty("/selectedModel");
            
            if (sSelectedModel) {
                this._loadModelConfig(sSelectedModel);
            }
            
            // Open dialog
            this._modelConfigDialog.open();
        },
        
        _initializeModelConfig: function() {
            // Create model config model
            var oConfigModel = new JSONModel({
                selectedModelId: "",
                availableModels: [],
                config: this._getDefaultConfig()
            });
            this.getView().setModel(oConfigModel, "modelConfig");
            
            // Load fragment
            if (!this._modelConfigDialog) {
                this._modelConfigDialog = sap.ui.xmlfragment(
                    this.getView().getId(),
                    "llm.server.dashboard.view.fragments.ModelConfiguratorDialog",
                    this
                );
                this.getView().addDependent(this._modelConfigDialog);
            }
        },
        
        _getDefaultConfig: function() {
            return {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 40,
                max_tokens: 2048,
                context_length: 4096,
                repeat_penalty: 1.1,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                stream: true,
                enable_cache: true,
                logprobs: false,
                seed: null,
                stop_sequences: ""
            };
        },
        
        _loadModelConfig: function(modelId) {
            var oConfigModel = this.getView().getModel("modelConfig");
            var oMetricsModel = this.getOwnerComponent().getModel("metrics");
            
            // Set available models
            var availableModels = oMetricsModel.getProperty("/availableModels") || [];
            oConfigModel.setProperty("/availableModels", availableModels);
            oConfigModel.setProperty("/selectedModelId", modelId);
            
            // Try to load config from localStorage
            var savedConfig = localStorage.getItem("modelConfig_" + modelId);
            if (savedConfig) {
                try {
                    var config = JSON.parse(savedConfig);
                    oConfigModel.setProperty("/config", config);
                } catch (e) {
                    console.warn("Failed to parse saved config, using defaults");
                    oConfigModel.setProperty("/config", this._getDefaultConfig());
                }
            } else {
                // Use defaults
                oConfigModel.setProperty("/config", this._getDefaultConfig());
            }
        },
        
        onConfigModelChange: function(oEvent) {
            var sNewModelId = oEvent.getParameter("selectedItem").getKey();
            this._loadModelConfig(sNewModelId);
        },
        
        onParameterChange: function() {
            // Live update - could add validation here
            console.log("Parameter changed");
        },
        
        onResetConfig: function() {
            var oConfigModel = this.getView().getModel("modelConfig");
            oConfigModel.setProperty("/config", this._getDefaultConfig());
            sap.m.MessageToast.show("Configuration reset to defaults");
        },
        
        onSaveConfig: function() {
            var oConfigModel = this.getView().getModel("modelConfig");
            var sModelId = oConfigModel.getProperty("/selectedModelId");
            var oConfig = oConfigModel.getProperty("/config");
            
            // Save to localStorage
            try {
                localStorage.setItem("modelConfig_" + sModelId, JSON.stringify(oConfig));
                sap.m.MessageToast.show("Configuration saved for " + sModelId);
                this._modelConfigDialog.close();
            } catch (e) {
                sap.m.MessageBox.error("Failed to save configuration: " + e.message);
            }
        },
        
        onCloseConfigurator: function() {
            this._modelConfigDialog.close();
        },
        
        onConfiguratorDialogClose: function() {
            // Cleanup if needed
        },
        
        // ==================== NOTIFICATIONS ====================
        
        onOpenNotifications: function(oEvent) {
            // Initialize notifications if not exists
            if (!this._notificationsPopover) {
                this._initializeNotifications();
            }
            
            // Open popover
            this._notificationsPopover.openBy(oEvent.getSource());
        },
        
        _initializeNotifications: function() {
            // Create notifications model
            var oNotificationsModel = new JSONModel({
                items: this._getMockNotifications(),
                unreadCount: 3,
                hasMore: false
            });
            this.getView().setModel(oNotificationsModel, "notifications");
            
            // Load fragment
            if (!this._notificationsPopover) {
                this._notificationsPopover = sap.ui.xmlfragment(
                    this.getView().getId(),
                    "llm.server.dashboard.view.fragments.NotificationsPopover",
                    this
                );
                this.getView().addDependent(this._notificationsPopover);
            }
        },
        
        _getMockNotifications: function() {
            var now = new Date();
            return [
                {
                    id: "notif_1",
                    type: "warning",
                    category: "Performance",
                    title: "High Latency Detected",
                    message: "Model lfm2.5-1.2b-q4_0 P95 latency exceeded 500ms threshold",
                    timestamp: this._formatTimeAgo(new Date(now.getTime() - 5 * 60000)),
                    read: false,
                    action: "viewMetrics",
                    actionText: "View Metrics"
                },
                {
                    id: "notif_2",
                    type: "info",
                    category: "System",
                    title: "Model Update Available",
                    message: "Llama 3.3 70B v2.1 is now available for download",
                    timestamp: this._formatTimeAgo(new Date(now.getTime() - 30 * 60000)),
                    read: false,
                    action: "viewModels",
                    actionText: "View Models"
                },
                {
                    id: "notif_3",
                    type: "error",
                    category: "Training",
                    title: "Training Job Failed",
                    message: "Job mhc_ft_20260121_001 failed due to GPU memory error",
                    timestamp: this._formatTimeAgo(new Date(now.getTime() - 120 * 60000)),
                    read: false,
                    action: "viewTraining",
                    actionText: "View Details"
                }
            ];
        },
        
        _formatTimeAgo: function(date) {
            var seconds = Math.floor((new Date() - date) / 1000);
            if (seconds < 60) return seconds + "s ago";
            var minutes = Math.floor(seconds / 60);
            if (minutes < 60) return minutes + "m ago";
            var hours = Math.floor(minutes / 60);
            if (hours < 24) return hours + "h ago";
            var days = Math.floor(hours / 24);
            return days + "d ago";
        },
        
        onRefreshNotifications: function() {
            var oModel = this.getView().getModel("notifications");
            oModel.setProperty("/items", this._getMockNotifications());
            sap.m.MessageToast.show("Notifications refreshed");
        },
        
        onMarkAllRead: function() {
            var oModel = this.getView().getModel("notifications");
            var items = oModel.getProperty("/items");
            items.forEach(item => item.read = true);
            oModel.setProperty("/items", items);
            oModel.setProperty("/unreadCount", 0);
            sap.m.MessageToast.show("All notifications marked as read");
        },
        
        onClearAllNotifications: function() {
            sap.m.MessageBox.confirm("Clear all notifications?", {
                onClose: function(oAction) {
                    if (oAction === sap.m.MessageBox.Action.OK) {
                        var oModel = this.getView().getModel("notifications");
                        oModel.setProperty("/items", []);
                        oModel.setProperty("/unreadCount", 0);
                        sap.m.MessageToast.show("All notifications cleared");
                    }
                }.bind(this)
            });
        },
        
        onDeleteNotification: function(oEvent) {
            var oModel = this.getView().getModel("notifications");
            var oItem = oEvent.getParameter("listItem");
            var sPath = oItem.getBindingContext("notifications").getPath();
            var index = parseInt(sPath.split("/")[2]);
            
            var items = oModel.getProperty("/items");
            items.splice(index, 1);
            oModel.setProperty("/items", items);
            
            // Update unread count
            var unread = items.filter(i => !i.read).length;
            oModel.setProperty("/unreadCount", unread);
        },
        
        onNotificationPress: function(oEvent) {
            var oItem = oEvent.getSource();
            var oContext = oItem.getBindingContext("notifications");
            var notification = oContext.getObject();
            
            console.log("Notification pressed:", notification);
            // Mark as read
            oContext.getModel().setProperty(oContext.getPath() + "/read", true);
            this._updateUnreadCount();
        },
        
        onNotificationAction: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("notifications");
            var notification = oContext.getObject();
            
            console.log("Notification action:", notification.action);
            sap.m.MessageToast.show("Action: " + notification.action);
            // Navigate based on action
        },
        
        onMarkAsRead: function(oEvent) {
            var oItem = oEvent.getSource().getParent().getParent();
            var oContext = oItem.getBindingContext("notifications");
            oContext.getModel().setProperty(oContext.getPath() + "/read", true);
            this._updateUnreadCount();
        },
        
        _updateUnreadCount: function() {
            var oModel = this.getView().getModel("notifications");
            var items = oModel.getProperty("/items");
            var unread = items.filter(i => !i.read).length;
            oModel.setProperty("/unreadCount", unread);
        },
        
        onLoadMoreNotifications: function() {
            sap.m.MessageToast.show("Loading more notifications...");
        },
        
        onNotificationsPopoverClose: function() {
            // Cleanup if needed
        },
        
        // ==================== SETTINGS ====================
        
        onOpenSettings: function() {
            // Initialize settings if not exists
            if (!this._settingsDialog) {
                this._initializeSettings();
            }
            
            // Load current settings
            this._loadSettings();
            
            // Open dialog
            this._settingsDialog.open();
        },
        
        _initializeSettings: function() {
            // Create settings model
            var oSettingsModel = new JSONModel(this._getDefaultSettings());
            this.getView().setModel(oSettingsModel, "settings");
            
            // Load fragment
            if (!this._settingsDialog) {
                this._settingsDialog = sap.ui.xmlfragment(
                    this.getView().getId(),
                    "llm.server.dashboard.view.fragments.SettingsDialog",
                    this
                );
                this.getView().addDependent(this._settingsDialog);
            }
        },
        
        _getDefaultSettings: function() {
            return {
                // General
                theme: "sap_horizon",
                language: "en",
                dateFormat: "MM/DD/YYYY",
                timeFormat: "12h",
                
                // API
                apiBaseUrl: window.location.origin || "http://localhost:8080",
                websocketUrl: "ws://localhost:8080/ws",
                apiKey: "",
                requestTimeout: 30,
                enableApiCache: true,
                
                // Dashboard
                autoRefresh: true,
                refreshInterval: 10,
                showAdvancedMetrics: false,
                enableChartAnimation: true,
                compactMode: false,
                defaultChartRange: "1h",
                
                // Notifications
                enableDesktopNotifications: false,
                enableNotificationSound: false,
                notificationTypes: {
                    system: true,
                    model: true,
                    training: true,
                    performance: true
                },
                autoDismissTimeout: 10,
                notificationPermission: Notification.permission,
                
                // Privacy
                savePromptHistory: true,
                enableAnalytics: false,
                enableErrorReporting: true,
                storageUsage: this._calculateStorageUsage()
            };
        },
        
        _loadSettings: function() {
            var oModel = this.getView().getModel("settings");
            var savedSettings = localStorage.getItem("appSettings");
            
            if (savedSettings) {
                try {
                    var settings = JSON.parse(savedSettings);
                    // Merge with defaults
                    var defaults = this._getDefaultSettings();
                    oModel.setData(Object.assign(defaults, settings));
                } catch (e) {
                    console.warn("Failed to load settings:", e);
                    oModel.setData(this._getDefaultSettings());
                }
            }
        },
        
        _calculateStorageUsage: function() {
            var total = 0;
            for (var key in localStorage) {
                if (localStorage.hasOwnProperty(key)) {
                    total += localStorage[key].length + key.length;
                }
            }
            return (total / 1024).toFixed(2);
        },
        
        onThemeChange: function(oEvent) {
            var sTheme = oEvent.getParameter("selectedItem").getKey();
            sap.ui.getCore().applyTheme(sTheme);
            sap.m.MessageToast.show("Theme changed to " + sTheme);
        },
        
        onLanguageChange: function() {
            sap.m.MessageToast.show("Language change requires page reload");
        },
        
        onDateFormatChange: function() {
            // Update date formatting
        },
        
        onTimeFormatChange: function() {
            // Update time formatting
        },
        
        onApiSettingChange: function() {
            // API settings changed
        },
        
        onTestApiConnection: function() {
            var oModel = this.getView().getModel("settings");
            var apiBaseUrl = oModel.getProperty("/apiBaseUrl");
            
            sap.m.MessageToast.show("Testing connection to " + apiBaseUrl);
            
            ApiService.getModels()
                .then(() => {
                    sap.m.MessageBox.success("API connection successful!");
                })
                .catch(error => {
                    sap.m.MessageBox.error("API connection failed: " + error.message);
                });
        },
        
        onDashboardSettingChange: function() {
            // Dashboard settings changed
        },
        
        onNotificationSettingChange: function() {
            // Notification settings changed
        },
        
        onRequestNotificationPermission: function() {
            if ("Notification" in window) {
                Notification.requestPermission().then(permission => {
                    var oModel = this.getView().getModel("settings");
                    oModel.setProperty("/notificationPermission", permission);
                    
                    if (permission === "granted") {
                        sap.m.MessageToast.show("Notification permission granted");
                    } else {
                        sap.m.MessageBox.warning("Notification permission denied");
                    }
                });
            } else {
                sap.m.MessageBox.error("Notifications not supported in this browser");
            }
        },
        
        onPrivacySettingChange: function() {
            // Privacy settings changed
        },
        
        onClearLocalStorage: function() {
            sap.m.MessageBox.confirm(
                "This will delete all locally stored data including configurations and history. Continue?",
                {
                    title: "Clear Local Storage",
                    onClose: function(oAction) {
                        if (oAction === sap.m.MessageBox.Action.OK) {
                            localStorage.clear();
                            sap.m.MessageToast.show("Local storage cleared");
                            // Reload settings
                            this._loadSettings();
                        }
                    }.bind(this)
                }
            );
        },
        
        onExportUserData: function() {
            var data = {};
            for (var key in localStorage) {
                if (localStorage.hasOwnProperty(key)) {
                    data[key] = localStorage[key];
                }
            }
            
            var blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
            var url = URL.createObjectURL(blob);
            var a = document.createElement("a");
            a.href = url;
            a.download = "user_data_export_" + new Date().toISOString().split('T')[0] + ".json";
            a.click();
            URL.revokeObjectURL(url);
            
            sap.m.MessageToast.show("User data exported");
        },
        
        onResetAllSettings: function() {
            sap.m.MessageBox.confirm("Reset all settings to defaults?", {
                onClose: function(oAction) {
                    if (oAction === sap.m.MessageBox.Action.OK) {
                        var oModel = this.getView().getModel("settings");
                        oModel.setData(this._getDefaultSettings());
                        sap.m.MessageToast.show("Settings reset to defaults");
                    }
                }.bind(this)
            });
        },
        
        onSaveSettings: function() {
            var oModel = this.getView().getModel("settings");
            var settings = oModel.getData();
            
            try {
                localStorage.setItem("appSettings", JSON.stringify(settings));
                sap.m.MessageToast.show("Settings saved successfully");
                this._settingsDialog.close();
                
                // Apply settings
                if (settings.theme) {
                    sap.ui.getCore().applyTheme(settings.theme);
                }
            } catch (e) {
                sap.m.MessageBox.error("Failed to save settings: " + e.message);
            }
        },
        
        onCloseSettings: function() {
            this._settingsDialog.close();
        },
        
        onSettingsDialogClose: function() {
            // Cleanup if needed
        },
        
        onExportMetrics: function() {
            var oModel = this.getOwnerComponent().getModel("metrics");
            var selectedModel = oModel.getProperty("/selectedModel");
            
            ApiService.getMetricsHistory(selectedModel, '24h')
                .then(history => {
                    ApiService.exportData(history.data_points, `metrics_${selectedModel}_24h`, 'csv');
                    sap.m.MessageToast.show("Metrics exported");
                })
                .catch(error => {
                    sap.m.MessageBox.error("Export failed: " + error.message);
                });
        },
        
        _initializeMockData: function(oModel) {
            // Fallback mock data if API fails
            var now = new Date();
            
            var oData = {
                connected: false,
                selectedModel: "lfm2.5-1.2b-q4_0",
                availableModels: [
                    {
                        id: "lfm2.5-1.2b-q4_0",
                        display_name: "LFM2.5 1.2B",
                        quantization: "Q4_0",
                        architecture: "lfm2",
                        parameter_count: "1.2B",
                        health: "healthy",
                        avgLatency: 52,
                        avgThroughput: 65,
                        requests: 1247,
                        lastUsed: "2 min ago"
                    }
                ],
                selectedModelData: {
                    current: {
                        p50: 52,
                        p95: 89,
                        p99: 156,
                        tps: 65,
                        ttft: 35,
                        cacheHit: 0.82,
                        queueDepth: 3,
                        totalTokens: 245
                    }
                },
                tiers: {
                    gpu: { used: 0, total: 0, hitRate: 0 },
                    ram: { used: 2.2, total: 16, hitRate: 0.15 },
                    dragonfly: { used: 0.5, total: 2, hitRate: 0.82 },
                    postgres: { used: 1.2, total: 10, hitRate: 0.45 },
                    ssd: { used: 5.6, total: 50, hitRate: 0.12 }
                }
            };
            
            oModel.setData(oData);
        },
        
        onExit: function () {
            // Clean up
            if (this._pollingInterval) {
                clearInterval(this._pollingInterval);
            }
            ApiService.disconnectWebSocket();
        }
    });
});
