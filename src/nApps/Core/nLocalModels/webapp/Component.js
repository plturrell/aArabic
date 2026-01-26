sap.ui.define([
	"sap/ui/core/UIComponent",
	"sap/ui/model/json/JSONModel",
	"sap/ui/Device",
	"llm/server/dashboard/utils/RuntimeConfig"
], function (UIComponent, JSONModel, Device, RuntimeConfig) {
	"use strict";

	return UIComponent.extend("llm.server.dashboard.Component", {

		metadata: {
			manifest: "json"
		},

		init: function () {
			// Call the base component's init function
			UIComponent.prototype.init.apply(this, arguments);
			
			// DO NOT initialize router - manual view management is used across controllers
			
			this.setModel(new JSONModel(Device), "device");

			var oRuntimeConfig = RuntimeConfig.get();
			var oMetricsModel = new JSONModel({
				selectedModel: "",
				selectedModelData: {
					current: {
						p50: 0,
						p95: 0,
						p99: 0,
						tps: 0,
						ttft: 0,
						cacheHit: 0,
						queueDepth: 0,
						totalTokens: 0
					},
					latencyHistory: [],
					throughputHistory: []
				},
				models: [],
				tiers: {
					gpu: { used: 0, total: 0, hitRate: 0 },
					ram: { used: 0, total: 0, hitRate: 0 },
					hana: { used: 0, total: 0, hitRate: 0 },
					ssd: { used: 0, total: 0, hitRate: 0 }
				},
				cache: { totalHitRate: 0, sharingRatio: 0, compressionRatio: 0, evictions: 0 },
				llm: { ttft: 0, tps: 0, generationTime: 0, queueDepth: 0, promptTokens: 0, completionTokens: 0 },
				system: { cpuUsage: 0, memoryUsage: 0, networkIn: 0, networkOut: 0 },
				latency: { histogram: [], p50: 0, p95: 0, p99: 0 },
				lastUpdate: null,
				connected: false
			});
			this.setModel(oMetricsModel, "metrics");

			var oChatModel = new JSONModel({
				selectedModel: "",
				availableModels: [],
				maxTokens: 512,
				temperature: 0.7,
				systemPrompt: "",
				prompt: "",
				response: "",
				lastTTFT: 0,
				lastTPS: 0,
				lastTokens: 0,
				isLoading: false
			});
			this.setModel(oChatModel, "chat");

				this._config = oRuntimeConfig;
				this._sApiBaseUrl = oRuntimeConfig.apiBaseUrl;
				this._sWsUrl = oRuntimeConfig.wsUrl;

				// Metrics are updated via ApiService in controllers; no duplicate polling here.
				this._fetchAvailableModels();

				// Enable routing for deep links/history
				this.getRouter().initialize();
			},


		getApiBaseUrl: function () {
			return this._sApiBaseUrl;
		},

			_fetchAvailableModels: function () {
				var oChatModel = this.getModel("chat");
				var that = this;
				
				fetch(this._sApiBaseUrl + "/v1/models")
					.then(function (r) { return r.ok ? r.json() : Promise.reject("HTTP " + r.status); })
					.then(function (oData) {
						// Parse models with full metadata
						var aModels = (oData.data || []).map(function (m) {
							return {
								id: m.id,
								name: m.display_name || m.id,
								display_name: m.display_name || m.id,
								architecture: m.architecture || "unknown",
								quantization: m.quantization || "",
								parameter_count: m.parameter_count || "",
								format: m.format || "gguf",
								size_mb: m.size_mb || 0,
								size_bytes: m.size_bytes || 0,
								enabled: m.enabled !== false,
								health_status: m.health_status || "unknown",
								use_count: m.use_count || 0,
								preload: m.preload || false,
								// Format display string
								displayText: that._formatModelDisplayText(m)
							};
						});
						
						oChatModel.setProperty("/availableModels", aModels);
						
						if (aModels.length > 0) {
							oChatModel.setProperty("/selectedModel", aModels[0].id);
							that._updateSelectedModelInfo(aModels[0]);
						}
					})
					.catch(function (e) { 
						console.error("Models fetch error:", e);
						// Set empty array on error
						oChatModel.setProperty("/availableModels", []);
					});
			},
			
			_formatModelDisplayText: function (model) {
				var parts = [];
				
				// Display name
				parts.push(model.display_name || model.id);
				
				// Add size if available
				if (model.size_mb) {
					if (model.size_mb >= 1024) {
						parts.push("(" + (model.size_mb / 1024).toFixed(1) + "GB");
					} else {
						parts.push("(" + model.size_mb + "MB");
					}
					parts.push("-");
					parts.push(model.architecture || "unknown");
					parts.push(")");
				}
				
				return parts.join(" ");
			},
			
			_updateSelectedModelInfo: function (model) {
				var oChatModel = this.getModel("chat");
				oChatModel.setProperty("/selectedModelInfo", model);
			},

		exit: function () {
			if (this._metricsInterval) clearInterval(this._metricsInterval);
		},

		/**
		 * Get the content density class based on device type
		 * @public
		 * @returns {string} Content density class for styling
		 */
		getContentDensityClass: function () {
			if (this._sContentDensityClass === undefined) {
				// Check for touch support
				if (Device.support.touch) {
					this._sContentDensityClass = "sapUiSizeCozy";
				} else {
					this._sContentDensityClass = "sapUiSizeCompact";
				}
			}
			return this._sContentDensityClass;
		}

	});
});
