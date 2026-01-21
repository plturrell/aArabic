sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], function (Controller, JSONModel, MessageBox, MessageToast) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.ModelConfigurator", {

        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("modelConfigurator").attachPatternMatched(this._onObjectMatched, this);
            
            // Initialize configuration model
            this._oConfigModel = new JSONModel(this._getDefaultConfig());
            this.getView().setModel(this._oConfigModel, "config");
            
            // Load prompt mode presets
            this._initPromptModePresets();
            
            // Load available models
            this._loadAvailableModels();
        },
        
        _initPromptModePresets: function () {
            // Define the 4 prompt mode presets
            this._modePresets = {
                "Fast": {
                    name: "Fast",
                    description: "Optimized for lowest latency and quick responses. Uses highly compressed models.",
                    gpuPercent: 65,
                    ramPercent: 25,
                    ssdPercent: 10,
                    expectedLatency: "50-150",
                    expectedTPS: "40-80",
                    useCases: "development, testing, real-time chat",
                    compatibleModels: ["lfm2.5-1.2b-q4_0", "hymt-1.5-7b-q4_k_m"],
                    recommendedModels: ["lfm2.5-1.2b-q4_0"],
                    excludedModels: ["llama-3.3-70b", "deepseek-coder-33b"],
                    tierConfig: {
                        gpu: { enabled: true, memoryLimitGB: 52 },
                        ram: { memoryLimitGB: 20, evictionPolicy: "lru" },
                        dragonfly: { enabled: false, memoryLimitGB: 0 },
                        ssd: { storageLimitGB: 100, compressionEnabled: false }
                    }
                },
                "Normal": {
                    name: "Normal",
                    description: "Balanced performance and quality for production workloads.",
                    gpuPercent: 45,
                    ramPercent: 35,
                    ssdPercent: 20,
                    expectedLatency: "100-300",
                    expectedTPS: "25-50",
                    useCases: "production, general purpose",
                    compatibleModels: ["lfm2.5-1.2b-q4_k_m", "hymt-1.5-7b-q6_k", "deepseek-coder-33b"],
                    recommendedModels: ["lfm2.5-1.2b-q4_k_m", "hymt-1.5-7b-q6_k"],
                    excludedModels: ["llama-3.3-70b"],
                    tierConfig: {
                        gpu: { enabled: true, memoryLimitGB: 36 },
                        ram: { memoryLimitGB: 28, evictionPolicy: "adaptive" },
                        dragonfly: { enabled: true, memoryLimitGB: 8 },
                        ssd: { storageLimitGB: 200, compressionEnabled: true }
                    }
                },
                "Expert": {
                    name: "Expert",
                    description: "High quality responses with optimized tiering. Uses higher precision quantization.",
                    gpuPercent: 35,
                    ramPercent: 45,
                    ssdPercent: 20,
                    expectedLatency: "200-500",
                    expectedTPS: "15-35",
                    useCases: "code generation, complex reasoning",
                    compatibleModels: ["lfm2.5-1.2b-f16", "hymt-1.5-7b-q8_0", "deepseek-coder-33b"],
                    recommendedModels: ["lfm2.5-1.2b-f16", "deepseek-coder-33b"],
                    excludedModels: [],
                    tierConfig: {
                        gpu: { enabled: true, memoryLimitGB: 28 },
                        ram: { memoryLimitGB: 36, evictionPolicy: "adaptive" },
                        dragonfly: { enabled: true, memoryLimitGB: 12 },
                        ssd: { storageLimitGB: 300, compressionEnabled: true }
                    }
                },
                "Research": {
                    name: "Research",
                    description: "Maximum quality with full tiering support. Suitable for large models and benchmarking.",
                    gpuPercent: 25,
                    ramPercent: 35,
                    ssdPercent: 40,
                    expectedLatency: "300-1000",
                    expectedTPS: "10-25",
                    useCases: "research, benchmarking, quality validation",
                    compatibleModels: ["llama-3.3-70b", "deepseek-coder-33b", "lfm2.5-1.2b-f16", "microsoft-phi-2"],
                    recommendedModels: ["llama-3.3-70b"],
                    excludedModels: [],
                    tierConfig: {
                        gpu: { enabled: true, memoryLimitGB: 20 },
                        ram: { memoryLimitGB: 28, evictionPolicy: "adaptive" },
                        dragonfly: { enabled: true, memoryLimitGB: 16 },
                        ssd: { storageLimitGB: 500, compressionEnabled: true }
                    }
                }
            };
        },
        
        onPromptModeChange: function (oEvent) {
            var sMode = oEvent.getParameter("item").getKey();
            this._oConfigModel.setProperty("/selectedPromptMode", sMode);
            
            if (sMode && this._modePresets[sMode]) {
                var oPreset = this._modePresets[sMode];
                
                // Update mode info display
                this._oConfigModel.setProperty("/modeInfo", {
                    description: oPreset.description,
                    gpuPercent: oPreset.gpuPercent,
                    ramPercent: oPreset.ramPercent,
                    ssdPercent: oPreset.ssdPercent,
                    expectedLatency: oPreset.expectedLatency,
                    expectedTPS: oPreset.expectedTPS,
                    useCases: oPreset.useCases
                });
                
                // Apply tier configuration
                this._applyPromptModeTiers(oPreset.tierConfig);
                
                // Filter and grey out incompatible models
                this._filterModelsByPromptMode(oPreset);
                
                // Auto-select recommended model if available
                this._autoSelectRecommendedModel(oPreset);
                
                // Update preview
                this._updateResourcePreview();
            }
        },
        
        _applyPromptModeTiers: function (oTierConfig) {
            // Apply the tier configuration from the preset
            var oCurrentTiers = this._oConfigModel.getProperty("/tiers");
            
            // Merge with current config
            oCurrentTiers.gpu.enabled = oTierConfig.gpu.enabled;
            oCurrentTiers.gpu.memoryLimitGB = oTierConfig.gpu.memoryLimitGB;
            oCurrentTiers.ram.memoryLimitGB = oTierConfig.ram.memoryLimitGB;
            oCurrentTiers.ram.evictionPolicy = oTierConfig.ram.evictionPolicy;
            oCurrentTiers.dragonfly.enabled = oTierConfig.dragonfly.enabled;
            oCurrentTiers.dragonfly.memoryLimitGB = oTierConfig.dragonfly.memoryLimitGB;
            oCurrentTiers.ssd.storageLimitGB = oTierConfig.ssd.storageLimitGB;
            oCurrentTiers.ssd.compressionEnabled = oTierConfig.ssd.compressionEnabled;
            
            this._oConfigModel.setProperty("/tiers", oCurrentTiers);
        },
        
        _filterModelsByPromptMode: function (oPreset) {
            var aAllModels = this._oConfigModel.getProperty("/availableModels");
            var sCurrentMode = this._oConfigModel.getProperty("/selectedMode");
            
            // First apply format filter
            var aFilteredModels = aAllModels.filter(function (m) {
                if (sCurrentMode === "both") return true;
                if (sCurrentMode === "production") return m.format === "gguf";
                if (sCurrentMode === "training") return m.format === "safetensors";
                return true;
            });
            
            // Then apply prompt mode compatibility filter
            aFilteredModels = aFilteredModels.map(function (m) {
                var bIsExcluded = oPreset.excludedModels.some(function (excluded) {
                    return m.id.indexOf(excluded) >= 0;
                });
                
                var bIsCompatible = oPreset.compatibleModels.some(function (compatible) {
                    return m.id.indexOf(compatible) >= 0;
                });
                
                var bIsRecommended = oPreset.recommendedModels.some(function (recommended) {
                    return m.id.indexOf(recommended) >= 0;
                });
                
                return Object.assign({}, m, {
                    enabled: !bIsExcluded && bIsCompatible,
                    isRecommended: bIsRecommended
                });
            });
            
            // Update model families with enabled flag
            this._updateModelFamiliesWithFilter(aFilteredModels);
        },
        
        _updateModelFamiliesWithFilter: function (aFilteredModels) {
            // Extract unique families
            var oFamilies = {};
            aFilteredModels.forEach(function (model) {
                var family = model.id.split("-")[0] || model.id;
                if (!oFamilies[family]) {
                    oFamilies[family] = {
                        family: family,
                        displayName: model.name.split(/Q\d|F\d/)[0].trim() || model.name,
                        models: [],
                        enabled: false
                    };
                }
                oFamilies[family].models.push(model);
                // Family is enabled if at least one model is enabled
                if (model.enabled) {
                    oFamilies[family].enabled = true;
                }
            });
            
            // Convert to array
            var aFamilies = Object.keys(oFamilies).map(function (key) {
                return oFamilies[key];
            });
            
            this._oConfigModel.setProperty("/modelFamilies", aFamilies);
        },
        
        _autoSelectRecommendedModel: function (oPreset) {
            var aAllModels = this._oConfigModel.getProperty("/availableModels");
            
            // Find first recommended model that's available
            for (var i = 0; i < oPreset.recommendedModels.length; i++) {
                var sRecommended = oPreset.recommendedModels[i];
                var oModel = aAllModels.find(function (m) {
                    return m.id.indexOf(sRecommended) >= 0;
                });
                
                if (oModel) {
                    // Extract family
                    var sFamily = oModel.id.split("-")[0] || oModel.id;
                    this._oConfigModel.setProperty("/selectedFamily", sFamily);
                    this._updateVariants(sFamily);
                    this._oConfigModel.setProperty("/selectedModelId", oModel.id);
                    this._oConfigModel.setProperty("/modelRecommendation", 
                        "✓ Recommended: " + oModel.name + " is optimal for " + oPreset.name + " mode");
                    return;
                }
            }
            
            // If no recommended model found, show general message
            this._oConfigModel.setProperty("/modelRecommendation", "");
        },
        
        onModeFilterChange: function (oEvent) {
            var sSelectedMode = oEvent.getParameter("item").getKey();
            this._oConfigModel.setProperty("/selectedMode", sSelectedMode);
            this._updateModelFamilies();
        },
        
        onFamilyChange: function (oEvent) {
            var sFamily = oEvent.getParameter("selectedItem").getKey();
            this._oConfigModel.setProperty("/selectedFamily", sFamily);
            this._updateVariants(sFamily);
        },
        
        _updateModelFamilies: function () {
            var sMode = this._oConfigModel.getProperty("/selectedMode");
            var aAllModels = this._oConfigModel.getProperty("/availableModels");
            
            // Filter models by mode
            var aFilteredModels = aAllModels.filter(function (m) {
                if (sMode === "both") return true;
                if (sMode === "production") return m.format === "gguf";
                if (sMode === "training") return m.format === "safetensors";
                return true;
            });
            
            // Extract unique families
            var oFamilies = {};
            aFilteredModels.forEach(function (model) {
                var family = model.id.split("-")[0] || model.id;
                if (!oFamilies[family]) {
                    oFamilies[family] = {
                        family: family,
                        displayName: model.name.split(/Q\d|F\d/)[0].trim() || model.name,
                        models: []
                    };
                }
                oFamilies[family].models.push(model);
            });
            
            // Convert to array
            var aFamilies = Object.keys(oFamilies).map(function (key) {
                return oFamilies[key];
            });
            
            this._oConfigModel.setProperty("/modelFamilies", aFamilies);
            this._oConfigModel.setProperty("/selectedFamily", "");
            this._oConfigModel.setProperty("/availableVariants", []);
        },
        
        _updateVariants: function (sFamily) {
            var aFamilies = this._oConfigModel.getProperty("/modelFamilies");
            var oFamily = aFamilies.find(function (f) { return f.family === sFamily; });
            
            if (oFamily && oFamily.models) {
                // Add variant display text
                var aVariants = oFamily.models.map(function (m) {
                    return {
                        id: m.id,
                        name: m.name,
                        variantDisplay: m.quantization + " (" + 
                            (m.size_mb >= 1024 ? (m.size_mb / 1024).toFixed(1) + "GB" : m.size_mb + "MB") + ")",
                        architecture: m.architecture,
                        quantization: m.quantization,
                        format: m.format,
                        size_mb: m.size_mb
                    };
                });
                
                this._oConfigModel.setProperty("/availableVariants", aVariants);
                
                // Auto-select first variant
                if (aVariants.length > 0) {
                    this._oConfigModel.setProperty("/selectedModelId", aVariants[0].id);
                    this._updateSelectedModelInfo(aVariants[0]);
                }
            }
        },
        
        _updateSelectedModelInfo: function (variant) {
            this._oConfigModel.setProperty("/currentModel", {
                name: variant.name,
                version: "1.0.0",
                architecture: variant.architecture,
                quantization: variant.quantization,
                format: variant.format,
                size_mb: variant.size_mb
            });
        },

        _onObjectMatched: function (oEvent) {
            var sModelId = oEvent.getParameter("arguments").modelId;
            if (sModelId && sModelId !== "new") {
                this._loadModelConfiguration(sModelId);
            } else {
                this._oConfigModel.setData(this._getDefaultConfig());
            }
        },

        _getDefaultConfig: function () {
            return {
                selectedMode: "both",
                selectedFamily: "",
                selectedModelId: "",
                currentModel: {
                    name: "",
                    version: "",
                    architecture: "",
                    quantization: "",
                    format: "",
                    size_mb: 0
                },
                availableModels: [],
                modelFamilies: [],
                availableVariants: [],
                tiers: {
                    gpu: {
                        enabled: true,
                        memoryLimitGB: 40,
                    },
                    ram: {
                        memoryLimitGB: 64,
                        evictionPolicy: "adaptive"
                    },
                    dragonfly: {
                        enabled: true,
                        memoryLimitGB: 32
                    },
                    postgresql: {
                        enabled: true,
                        connectionPoolSize: 20
                    },
                    ssd: {
                        storageLimitGB: 500,
                        compressionEnabled: true,
                        compressionAlgorithm: "int8_symmetric"
                    }
                },
                quotas: {
                    maxConcurrentRequests: 100,
                    maxTokensPerHour: 1000000,
                    maxRequestsPerMinute: 60,
                    burstMultiplier: 2.0
                },
                cacheSharing: {
                    enabled: true,
                    minPrefixLength: 10,
                    maxSharedEntries: 1000
                },
                advanced: {
                    simdEnabled: true,
                    batchProcessing: true,
                    optimalBatchSize: 32,
                    prefetchingEnabled: true,
                    prefetchWindowSize: 16
                },
                validation: {
                    isValid: true,
                    message: "",
                    type: "Success"
                },
                preview: {
                    totalMemoryGB: 0,
                    estimatedCostPerMonth: 0,
                    expectedThroughput: 0,
                    expectedLatencyP99: 0
                }
            };
        },

        _loadAvailableModels: function () {
            // Fetch from Model Registry API
            var that = this;
            var oComponent = this.getOwnerComponent();
            var sApiBaseUrl = oComponent.getApiBaseUrl();
            
            fetch(sApiBaseUrl + "/v1/models")
                .then(function (response) {
                    if (!response.ok) throw new Error("Failed to fetch models");
                    return response.json();
                })
                .then(function (data) {
                    // Parse the OpenAI-format response
                    var aModels = (data.data || []).map(function (m) {
                        return {
                            id: m.id,
                            name: m.display_name || m.id,
                            version: "1.0.0",
                            architecture: m.architecture || "unknown",
                            quantization: m.quantization || "",
                            parameter_count: m.parameter_count || "",
                            format: m.format || "gguf",
                            size_mb: m.size_mb || 0
                        };
                    });
                    that._oConfigModel.setProperty("/availableModels", aModels);
                    
                    // Initialize with default mode filter
                    that._oConfigModel.setProperty("/selectedMode", "both");
                    that._updateModelFamilies();
                })
                .catch(function (error) {
                    console.error("Error loading models:", error);
                    // Show error - no fallback to mock data
                    that._oConfigModel.setProperty("/availableModels", []);
                });
        },

        _loadModelConfiguration: function (sModelId) {
            // Fetch existing configuration
            var that = this;
            fetch("/api/v1/models/" + sModelId + "/config")
                .then(function (response) {
                    if (!response.ok) throw new Error("Failed to fetch model config");
                    return response.json();
                })
                .then(function (data) {
                    that._oConfigModel.setData(data);
                    that._updateResourcePreview();
                })
                .catch(function (error) {
                    console.error("Error loading model config:", error);
                    MessageToast.show("Failed to load configuration. Using defaults.");
                });
        },

        onModelChange: function (oEvent) {
            var sModelId = oEvent.getParameter("selectedItem").getKey();
            var aModels = this._oConfigModel.getProperty("/availableModels");
            var oSelectedModel = aModels.find(function (m) { return m.id === sModelId; });
            
            if (oSelectedModel) {
                this._oConfigModel.setProperty("/currentModel", {
                    name: oSelectedModel.name,
                    version: oSelectedModel.version,
                    architecture: oSelectedModel.architecture,
                    quantization: oSelectedModel.quantization
                });
                this._updateResourcePreview();
            }
        },

        onTierParamChange: function () {
            this._validateConfiguration();
            this._updateResourcePreview();
        },

        onQuotaChange: function () {
            this._validateConfiguration();
            this._updateResourcePreview();
        },

        onCacheSharingChange: function () {
            this._validateConfiguration();
            this._updateResourcePreview();
        },

        onAdvancedChange: function () {
            this._validateConfiguration();
            this._updateResourcePreview();
        },

        _validateConfiguration: function () {
            var oConfig = this._oConfigModel.getData();
            var aErrors = [];
            
            // Validate model selection
            if (!oConfig.selectedModelId) {
                aErrors.push("Please select a model");
            }
            
            // Validate tier limits
            var nTotalMemory = 0;
            if (oConfig.tiers.gpu.enabled) {
                nTotalMemory += oConfig.tiers.gpu.memoryLimitGB;
            }
            nTotalMemory += oConfig.tiers.ram.memoryLimitGB;
            if (oConfig.tiers.dragonfly.enabled) {
                nTotalMemory += oConfig.tiers.dragonfly.memoryLimitGB;
            }
            
            // Check system limits (example: 256 GB total)
            if (nTotalMemory > 256) {
                aErrors.push("Total memory allocation (" + nTotalMemory + " GB) exceeds system limit (256 GB)");
            }
            
            // Validate quotas
            if (oConfig.quotas.maxConcurrentRequests < 1) {
                aErrors.push("Max concurrent requests must be at least 1");
            }
            
            if (oConfig.quotas.maxRequestsPerMinute > oConfig.quotas.maxConcurrentRequests * 60) {
                aErrors.push("Max requests per minute exceeds concurrent capacity");
            }
            
            // Validate cache sharing
            if (oConfig.cacheSharing.enabled && oConfig.cacheSharing.minPrefixLength < 1) {
                aErrors.push("Minimum prefix length must be at least 1");
            }
            
            // Update validation status
            var bIsValid = aErrors.length === 0;
            this._oConfigModel.setProperty("/validation/isValid", bIsValid);
            this._oConfigModel.setProperty("/validation/message", 
                bIsValid ? "Configuration is valid" : aErrors.join("; "));
            this._oConfigModel.setProperty("/validation/type", 
                bIsValid ? "Success" : "Error");
        },

        _updateResourcePreview: function () {
            var oConfig = this._oConfigModel.getData();
            
            // Calculate total memory
            var nTotalMemory = 0;
            if (oConfig.tiers.gpu.enabled) {
                nTotalMemory += oConfig.tiers.gpu.memoryLimitGB;
            }
            nTotalMemory += oConfig.tiers.ram.memoryLimitGB;
            if (oConfig.tiers.dragonfly.enabled) {
                nTotalMemory += oConfig.tiers.dragonfly.memoryLimitGB;
            }
            nTotalMemory += 5; // PostgreSQL overhead
            
            // Estimate cost (simplified)
            // GPU: $2.50/GB/month, RAM: $0.50/GB/month, Dragonfly: $1.00/GB/month, SSD: $0.10/GB/month
            var nCost = 0;
            if (oConfig.tiers.gpu.enabled) {
                nCost += oConfig.tiers.gpu.memoryLimitGB * 2.50;
            }
            nCost += oConfig.tiers.ram.memoryLimitGB * 0.50;
            if (oConfig.tiers.dragonfly.enabled) {
                nCost += oConfig.tiers.dragonfly.memoryLimitGB * 1.00;
            }
            nCost += oConfig.tiers.ssd.storageLimitGB * 0.10;
            
            // Estimate throughput (tokens/sec)
            var nBaseThroughput = 5000; // Baseline
            if (oConfig.advanced.simdEnabled) nBaseThroughput *= 1.5;
            if (oConfig.advanced.batchProcessing) nBaseThroughput *= 1.3;
            if (oConfig.tiers.gpu.enabled) nBaseThroughput *= 2.5;
            if (oConfig.cacheSharing.enabled) nBaseThroughput *= 1.2;
            
            var nCompressionMultiplier = 1.0;
            if (oConfig.tiers.ssd.compressionEnabled) {
                if (oConfig.tiers.ssd.compressionAlgorithm === "fp16") {
                    nCompressionMultiplier = 1.1;
                } else if (oConfig.tiers.ssd.compressionAlgorithm.startsWith("int8")) {
                    nCompressionMultiplier = 1.05;
                }
            }
            nBaseThroughput *= nCompressionMultiplier;
            
            // Estimate latency P99 (ms)
            var nBaseLatency = 150; // Baseline
            if (oConfig.tiers.gpu.enabled) nBaseLatency *= 0.4; // GPU speedup
            if (oConfig.advanced.prefetchingEnabled) nBaseLatency *= 0.9;
            if (oConfig.tiers.ram.evictionPolicy === "adaptive") nBaseLatency *= 0.95;
            
            // Adjust for cache hit rates
            var nCacheHitRate = 0.70; // Baseline
            if (oConfig.cacheSharing.enabled) nCacheHitRate += 0.10;
            if (oConfig.tiers.gpu.enabled) nCacheHitRate += 0.05;
            nBaseLatency *= (1 - nCacheHitRate * 0.5); // Better cache = lower latency
            
            // Update preview
            this._oConfigModel.setProperty("/preview/totalMemoryGB", Math.round(nTotalMemory));
            this._oConfigModel.setProperty("/preview/estimatedCostPerMonth", Math.round(nCost));
            this._oConfigModel.setProperty("/preview/expectedThroughput", Math.round(nBaseThroughput));
            this._oConfigModel.setProperty("/preview/expectedLatencyP99", Math.round(nBaseLatency));
        },

        onSaveConfig: function () {
            var oConfig = this._oConfigModel.getData();
            var sModelId = oConfig.selectedModelId;
            
            if (!sModelId) {
                MessageBox.error("Please select a model first");
                return;
            }
            
            // Save to server
            var that = this;
            fetch("/api/v1/models/" + sModelId + "/config", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(oConfig)
            })
                .then(function (response) {
                    if (!response.ok) throw new Error("Failed to save configuration");
                    return response.json();
                })
                .then(function () {
                    MessageToast.show("Configuration saved successfully");
                })
                .catch(function (error) {
                    console.error("Error saving config:", error);
                    MessageBox.error("Failed to save configuration: " + error.message);
                });
        },

        onExportConfig: function () {
            var oConfig = this._oConfigModel.getData();
            var sJson = JSON.stringify(oConfig, null, 2);
            var oBlob = new Blob([sJson], { type: "application/json" });
            var sUrl = URL.createObjectURL(oBlob);
            
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = "model-config-" + (oConfig.selectedModelId || "new") + ".json";
            oLink.click();
            
            URL.revokeObjectURL(sUrl);
            MessageToast.show("Configuration exported");
        },

        onImportConfig: function () {
            var that = this;
            var oFileUpload = document.createElement("input");
            oFileUpload.type = "file";
            oFileUpload.accept = ".json";
            oFileUpload.onchange = function (oEvent) {
                var oFile = oEvent.target.files[0];
                if (oFile) {
                    var oReader = new FileReader();
                    oReader.onload = function (e) {
                        try {
                            var oConfig = JSON.parse(e.target.result);
                            that._oConfigModel.setData(oConfig);
                            that._validateConfiguration();
                            that._updateResourcePreview();
                            MessageToast.show("Configuration imported successfully");
                        } catch (error) {
                            MessageBox.error("Invalid configuration file: " + error.message);
                        }
                    };
                    oReader.readAsText(oFile);
                }
            };
            oFileUpload.click();
        },

        onResetDefaults: function () {
            var that = this;
            MessageBox.confirm("Are you sure you want to reset to default configuration?", {
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        that._oConfigModel.setData(that._getDefaultConfig());
                        that._validateConfiguration();
                        that._updateResourcePreview();
                        MessageToast.show("Configuration reset to defaults");
                    }
                }
            });
        },

        onApplyConfiguration: function () {
            var oConfig = this._oConfigModel.getData();
            var sModelId = oConfig.selectedModelId;
            
            if (!sModelId) {
                MessageBox.error("Please select a model first");
                return;
            }
            
            var that = this;
            MessageBox.confirm(
                "Are you sure you want to apply this configuration? This will restart the model with new settings.",
                {
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            that._applyConfiguration(sModelId, oConfig);
                        }
                    }
                }
            );
        },

        _applyConfiguration: function (sModelId, oConfig) {
            var that = this;
            
            // Apply configuration via API
            fetch("/api/v1/models/" + sModelId + "/apply", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(oConfig)
            })
                .then(function (response) {
                    if (!response.ok) throw new Error("Failed to apply configuration");
                    return response.json();
                })
                .then(function (data) {
                    MessageBox.success("Configuration applied successfully. Model is restarting...", {
                        onClose: function () {
                            that.onNavBack();
                        }
                    });
                })
                .catch(function (error) {
                    console.error("Error applying config:", error);
                    MessageBox.error("Failed to apply configuration: " + error.message);
                });
        },

        onCancel: function () {
            this.onNavBack();
        },

        onNavBack: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("main");
        }
    });
});
