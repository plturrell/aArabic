sap.ui.define([
	"sap/ui/base/ManagedObject"
], function (ManagedObject) {
	"use strict";

	/**
	 * HANA Cloud Service for interacting with SAP HANA via Zig backend API
	 * @extends sap.ui.base.ManagedObject
	 */
	var HanaService = ManagedObject.extend("llm.server.dashboard.service.HanaService", {

		metadata: {
			events: {
				connectionStatusChange: {
					parameters: {
						connected: { type: "boolean" },
						status: { type: "string" }
					}
				},
				error: {
					parameters: {
						error: { type: "object" },
						operation: { type: "string" }
					}
				}
			}
		},

		constructor: function (mSettings) {
			ManagedObject.apply(this, arguments);

			mSettings = mSettings || {};
			this._sBaseUrl = mSettings.baseUrl || "/api/v1/hana";
			this._bConnected = false;
			this._sConnectionStatus = "disconnected";
			this._iHealthCheckInterval = mSettings.healthCheckInterval || 30000;
			this._oHealthCheckTimer = null;
		},

		/**
		 * Initialize the service and start health checks
		 * @public
		 * @returns {Promise} Resolves when initial health check completes
		 */
		initialize: function () {
			return this.checkHealth().then(function () {
				this._startHealthCheck();
			}.bind(this));
		},

		/**
		 * Check HANA connection health
		 * @public
		 * @returns {Promise<object>} Health status
		 */
		checkHealth: function () {
			return this._executeRequest("/health", "GET").then(function (oResponse) {
				this._bConnected = oResponse.connected === true;
				this._sConnectionStatus = oResponse.status || (this._bConnected ? "connected" : "disconnected");
				this.fireEvent("connectionStatusChange", {
					connected: this._bConnected,
					status: this._sConnectionStatus
				});
				return oResponse;
			}.bind(this)).catch(function (oError) {
				this._bConnected = false;
				this._sConnectionStatus = "error";
				this.fireEvent("connectionStatusChange", {
					connected: false,
					status: "error"
				});
				throw oError;
			}.bind(this));
		},

		/**
		 * Check if HANA is connected
		 * @public
		 * @returns {boolean} Connection status
		 */
		isConnected: function () {
			return this._bConnected;
		},

		/**
		 * Get connection status string
		 * @public
		 * @returns {string} Connection status
		 */
		getConnectionStatus: function () {
			return this._sConnectionStatus;
		},

		/**
		 * Query AI_TRAINING.MODEL_VERSIONS table
		 * @public
		 * @param {object} [mFilters] - Optional filters
		 * @returns {Promise<Array>} Model versions
		 */
		getModelVersions: function (mFilters) {
			var oPayload = {
				table: "AI_TRAINING.MODEL_VERSIONS",
				filters: mFilters || {}
			};
			return this._executeRequest("/query", "POST", oPayload).then(function (oResponse) {
				return oResponse.data || [];
			});
		},

		/**
		 * Query AI_TRAINING.TRAINING_EXPERIMENTS table
		 * @public
		 * @param {object} [mFilters] - Optional filters
		 * @returns {Promise<Array>} Training experiments
		 */
		getTrainingExperiments: function (mFilters) {
			var oPayload = {
				table: "AI_TRAINING.TRAINING_EXPERIMENTS",
				filters: mFilters || {}
			};
			return this._executeRequest("/query", "POST", oPayload).then(function (oResponse) {
				return oResponse.data || [];
			});
		},

		/**
		 * Query AI_TRAINING.TRAINING_METRICS for a specific experiment
		 * @public
		 * @param {string} sExperimentId - Experiment ID
		 * @returns {Promise<Array>} Training metrics
		 */
		getTrainingMetrics: function (sExperimentId) {
			var oPayload = {
				table: "AI_TRAINING.TRAINING_METRICS",
				filters: {
					experiment_id: sExperimentId
				}
			};
			return this._executeRequest("/query", "POST", oPayload).then(function (oResponse) {
				return oResponse.data || [];
			});
		},

		/**
		 * Query AI_TRAINING.INFERENCE_METRICS with partition awareness
		 * @public
		 * @param {string} sVersionId - Model version ID
		 * @param {object} oTimeRange - Time range with start and end properties
		 * @returns {Promise<Array>} Inference metrics
		 */
		getInferenceMetrics: function (sVersionId, oTimeRange) {
			var oPayload = {
				table: "AI_TRAINING.INFERENCE_METRICS",
				filters: {
					version_id: sVersionId
				},
				partitionAware: true
			};
			if (oTimeRange) {
				oPayload.timeRange = {
					start: oTimeRange.start,
					end: oTimeRange.end
				};
			}
			return this._executeRequest("/query", "POST", oPayload).then(function (oResponse) {
				return oResponse.data || [];
			});
		},

		/**
		 * Query AI_TRAINING.AUDIT_LOG for specific entity
		 * @public
		 * @param {string} sEntityType - Entity type
		 * @param {string} sEntityId - Entity ID
		 * @returns {Promise<Array>} Audit log entries
		 */
		getAuditLog: function (sEntityType, sEntityId) {
			var oPayload = {
				table: "AI_TRAINING.AUDIT_LOG",
				filters: {
					entity_type: sEntityType,
					entity_id: sEntityId
				}
			};
			return this._executeRequest("/query", "POST", oPayload).then(function (oResponse) {
				return oResponse.data || [];
			});
		},

		/**
		 * Promote a model version to a new status with audit logging
		 * @public
		 * @param {string} sId - Model version ID
		 * @param {string} sTargetStatus - Target status (e.g., "production", "staging")
		 * @param {string} sUserId - User performing the promotion
		 * @returns {Promise<object>} Promotion result
		 */
		promoteModelVersion: function (sId, sTargetStatus, sUserId) {
			var oPayload = {
				operation: "promote",
				table: "AI_TRAINING.MODEL_VERSIONS",
				id: sId,
				targetStatus: sTargetStatus,
				userId: sUserId,
				auditLog: {
					table: "AI_TRAINING.AUDIT_LOG",
					action: "PROMOTE",
					entityType: "MODEL_VERSION",
					entityId: sId
				}
			};
			return this._executeRequest("/execute", "POST", oPayload);
		},

		/**
		 * Create a new deployment record
		 * @public
		 * @param {string} sVersionId - Model version ID to deploy
		 * @param {string} sEnvironment - Target environment
		 * @param {number} iTraffic - Traffic percentage (0-100)
		 * @param {string} [sRollbackFromId] - Optional rollback source deployment ID
		 * @returns {Promise<object>} Created deployment record
		 */
		createDeployment: function (sVersionId, sEnvironment, iTraffic, sRollbackFromId) {
			var oPayload = {
				operation: "insert",
				table: "AI_TRAINING.MODEL_DEPLOYMENTS",
				data: {
					version_id: sVersionId,
					environment: sEnvironment,
					traffic_percentage: iTraffic,
					rollback_from_id: sRollbackFromId || null,
					created_at: new Date().toISOString()
				}
			};
			return this._executeRequest("/execute", "POST", oPayload);
		},

		/**
		 * Execute HTTP request to Zig backend API
		 * @private
		 * @param {string} sEndpoint - API endpoint
		 * @param {string} sMethod - HTTP method
		 * @param {object} [oPayload] - Request payload
		 * @returns {Promise<object>} Response data
		 */
		_executeRequest: function (sEndpoint, sMethod, oPayload) {
			var that = this;
			var sUrl = this._sBaseUrl + sEndpoint;

			return new Promise(function (resolve, reject) {
				var oXhr = new XMLHttpRequest();
				oXhr.open(sMethod, sUrl, true);
				oXhr.setRequestHeader("Content-Type", "application/json");

				oXhr.onreadystatechange = function () {
					if (oXhr.readyState === 4) {
						if (oXhr.status >= 200 && oXhr.status < 300) {
							try {
								var oResponse = JSON.parse(oXhr.responseText);
								resolve(oResponse);
							} catch (e) {
								resolve({ data: oXhr.responseText });
							}
						} else {
							var oError = {
								status: oXhr.status,
								statusText: oXhr.statusText,
								message: oXhr.responseText
							};
							that.fireEvent("error", {
								error: oError,
								operation: sEndpoint
							});
							reject(oError);
						}
					}
				};

				oXhr.onerror = function () {
					var oError = {
						status: 0,
						statusText: "Network Error",
						message: "Failed to connect to HANA backend"
					};
					that.fireEvent("error", {
						error: oError,
						operation: sEndpoint
					});
					reject(oError);
				};

				if (oPayload) {
					oXhr.send(JSON.stringify(oPayload));
				} else {
					oXhr.send();
				}
			});
		},

		/**
		 * Start periodic health check
		 * @private
		 */
		_startHealthCheck: function () {
			if (this._oHealthCheckTimer) {
				clearInterval(this._oHealthCheckTimer);
			}
			this._oHealthCheckTimer = setInterval(function () {
				this.checkHealth().catch(function (oError) {
					console.error("HANA health check failed:", oError);
				});
			}.bind(this), this._iHealthCheckInterval);
		},

		/**
		 * Stop periodic health check
		 * @private
		 */
		_stopHealthCheck: function () {
			if (this._oHealthCheckTimer) {
				clearInterval(this._oHealthCheckTimer);
				this._oHealthCheckTimer = null;
			}
		},

		/**
		 * Cleanup on service destruction
		 * @public
		 */
		destroy: function () {
			this._stopHealthCheck();
			ManagedObject.prototype.destroy.apply(this, arguments);
		}

	});

	return HanaService;
});
