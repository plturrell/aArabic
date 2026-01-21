sap.ui.define([
	"sap/ui/base/ManagedObject"
], function (ManagedObject) {
	"use strict";

	/**
	 * WebSocket Service for real-time training progress communication
	 * @extends sap.ui.base.ManagedObject
	 */
	var TrainingWebSocketService = ManagedObject.extend("llm.server.dashboard.service.TrainingWebSocketService", {

		metadata: {
			events: {
				connectionChange: {
					parameters: {
						connected: { type: "boolean" }
					}
				},
				trainingProgress: {
					parameters: {
						experimentId: { type: "string" },
						step: { type: "int" },
						epoch: { type: "int" },
						metrics: { type: "object" },
						timestamp: { type: "string" }
					}
				},
				trainingStarted: {
					parameters: {
						experimentId: { type: "string" },
						modelId: { type: "string" },
						algorithm: { type: "string" }
					}
				},
				trainingCompleted: {
					parameters: {
						experimentId: { type: "string" },
						finalMetrics: { type: "object" },
						modelVersionId: { type: "string" }
					}
				},
				trainingFailed: {
					parameters: {
						experimentId: { type: "string" },
						error: { type: "object" }
					}
				},
				metricsUpdate: {
					parameters: {
						experimentId: { type: "string" },
						metricName: { type: "string" },
						value: { type: "any" }
					}
				},
				checkpointSaved: {
					parameters: {
						experimentId: { type: "string" },
						checkpointPath: { type: "string" },
						step: { type: "int" }
					}
				},
				error: {
					parameters: {
						error: { type: "object" }
					}
				},
				maxReconnectAttemptsReached: {}
			}
		},

		constructor: function (mSettings) {
			ManagedObject.apply(this, arguments);

			mSettings = mSettings || {};
			this._sUrl = mSettings.url || "ws://localhost:8080/ws/training";
			this._iReconnectInterval = mSettings.reconnectInterval || 3000;
			this._iMaxReconnectAttempts = mSettings.maxReconnectAttempts || 10;
			this._iReconnectAttempts = 0;
			this._oSocket = null;
			this._bConnected = false;
			this._oReconnectTimer = null;
			this._aMessageQueue = [];
			this._aSubscribedExperiments = [];
			this._bSubscribedToAll = false;
			this._aActiveExperiments = [];
		},

		/**
		 * Connect to training WebSocket server
		 * @public
		 */
		connect: function () {
			if (this._oSocket && this._oSocket.readyState === WebSocket.OPEN) {
				console.log("Training WebSocket already connected");
				return;
			}

			try {
				console.log("Connecting to Training WebSocket:", this._sUrl);
				this._oSocket = new WebSocket(this._sUrl);

				this._oSocket.onopen = this._onOpen.bind(this);
				this._oSocket.onmessage = this._onMessage.bind(this);
				this._oSocket.onerror = this._onError.bind(this);
				this._oSocket.onclose = this._onClose.bind(this);
			} catch (e) {
				console.error("Failed to create Training WebSocket:", e);
				this._scheduleReconnect();
			}
		},

		/**
		 * Disconnect from training WebSocket server and cleanup
		 * @public
		 */
		disconnect: function () {
			if (this._oReconnectTimer) {
				clearTimeout(this._oReconnectTimer);
				this._oReconnectTimer = null;
			}

			if (this._oSocket) {
				this._oSocket.close(1000, "Client disconnect");
				this._oSocket = null;
			}

			this._bConnected = false;
			this._iReconnectAttempts = 0;
			this._aSubscribedExperiments = [];
			this._bSubscribedToAll = false;
		},

		/**
		 * Subscribe to specific experiment updates
		 * @public
		 * @param {string} sExperimentId - Experiment ID to subscribe to
		 */
		subscribeToExperiment: function (sExperimentId) {
			if (!sExperimentId) {
				console.warn("No experiment ID provided for subscription");
				return;
			}

			if (this._aSubscribedExperiments.indexOf(sExperimentId) === -1) {
				this._aSubscribedExperiments.push(sExperimentId);
			}

			this._send({
				type: "SUBSCRIBE",
				experimentId: sExperimentId
			});
		},

		/**
		 * Unsubscribe from specific experiment updates
		 * @public
		 * @param {string} sExperimentId - Experiment ID to unsubscribe from
		 */
		unsubscribeFromExperiment: function (sExperimentId) {
			var iIndex = this._aSubscribedExperiments.indexOf(sExperimentId);
			if (iIndex > -1) {
				this._aSubscribedExperiments.splice(iIndex, 1);
			}

			this._send({
				type: "UNSUBSCRIBE",
				experimentId: sExperimentId
			});
		},

		/**
		 * Subscribe to all training updates
		 * @public
		 */
		subscribeToAll: function () {
			this._bSubscribedToAll = true;
			this._send({
				type: "SUBSCRIBE_ALL"
			});
		},

		/**
		 * Get list of currently running experiments
		 * @public
		 * @returns {Array} Array of active experiment IDs
		 */
		getActiveExperiments: function () {
			return this._aActiveExperiments.slice();
		},

		/**
		 * Check if WebSocket is connected
		 * @public
		 * @returns {boolean} Connection status
		 */
		isConnected: function () {
			return this._bConnected;
		},

		/**
		 * Send message to WebSocket server
		 * @private
		 * @param {object} oMessage - Message to send
		 */
		_send: function (oMessage) {
			if (this._bConnected && this._oSocket) {
				try {
					this._oSocket.send(JSON.stringify(oMessage));
				} catch (e) {
					console.error("Failed to send message:", e);
					this._aMessageQueue.push(oMessage);
				}
			} else {
				this._aMessageQueue.push(oMessage);
			}
		},

		/**
		 * Handle WebSocket open event
		 * @private
		 */
		_onOpen: function () {
			console.log("Training WebSocket connected");
			this._bConnected = true;
			this._iReconnectAttempts = 0;

			this.fireEvent("connectionChange", {
				connected: true
			});

			// Restore subscriptions after reconnect
			this._restoreSubscriptions();

			// Send queued messages
			this._flushMessageQueue();
		},

		/**
		 * Restore subscriptions after reconnection
		 * @private
		 */
		_restoreSubscriptions: function () {
			if (this._bSubscribedToAll) {
				this._send({ type: "SUBSCRIBE_ALL" });
			} else {
				this._aSubscribedExperiments.forEach(function (sExperimentId) {
					this._send({
						type: "SUBSCRIBE",
						experimentId: sExperimentId
					});
				}.bind(this));
			}
		},

		/**
		 * Handle WebSocket message event
		 * @private
		 * @param {MessageEvent} oEvent - Message event
		 */
		_onMessage: function (oEvent) {
			try {
				var oMessage = JSON.parse(oEvent.data);
				this._handleMessage(oMessage);
			} catch (e) {
				console.error("Failed to parse training message:", e);
			}
		},

		/**
		 * Handle parsed message and fire appropriate events
		 * @private
		 * @param {object} oMessage - Parsed message object
		 */
		_handleMessage: function (oMessage) {
			var sType = oMessage.type;
			var oData = oMessage.data || oMessage;

			switch (sType) {
				case "TRAINING_PROGRESS":
					this.fireEvent("trainingProgress", {
						experimentId: oData.experimentId,
						step: oData.step,
						epoch: oData.epoch,
						metrics: {
							loss: oData.loss,
							accuracy: oData.accuracy,
							learningRate: oData.learningRate
						},
						timestamp: oData.timestamp || new Date().toISOString()
					});
					break;

				case "TRAINING_METRICS":
					this.fireEvent("metricsUpdate", {
						experimentId: oData.experimentId,
						metricName: oData.metricName,
						value: oData.value
					});
					break;

				case "TRAINING_STATUS":
					this._handleStatusChange(oData);
					break;

				case "CHECKPOINT_SAVED":
					this.fireEvent("checkpointSaved", {
						experimentId: oData.experimentId,
						checkpointPath: oData.checkpointPath,
						step: oData.step
					});
					break;

				case "ACTIVE_EXPERIMENTS":
					this._aActiveExperiments = oData.experiments || [];
					break;

				default:
					console.log("Unknown training message type:", sType);
			}
		},

		/**
		 * Handle training status changes
		 * @private
		 * @param {object} oData - Status change data
		 */
		_handleStatusChange: function (oData) {
			var sStatus = oData.status;
			var sExperimentId = oData.experimentId;

			switch (sStatus) {
				case "RUNNING":
					if (this._aActiveExperiments.indexOf(sExperimentId) === -1) {
						this._aActiveExperiments.push(sExperimentId);
					}
					this.fireEvent("trainingStarted", {
						experimentId: sExperimentId,
						modelId: oData.modelId,
						algorithm: oData.algorithm
					});
					break;

				case "COMPLETED":
					this._removeActiveExperiment(sExperimentId);
					this.fireEvent("trainingCompleted", {
						experimentId: sExperimentId,
						finalMetrics: oData.finalMetrics,
						modelVersionId: oData.modelVersionId
					});
					break;

				case "FAILED":
					this._removeActiveExperiment(sExperimentId);
					this.fireEvent("trainingFailed", {
						experimentId: sExperimentId,
						error: oData.error
					});
					break;

				case "PAUSED":
					// Keep in active experiments but could fire a paused event if needed
					break;
			}
		},

		/**
		 * Remove experiment from active experiments list
		 * @private
		 * @param {string} sExperimentId - Experiment ID to remove
		 */
		_removeActiveExperiment: function (sExperimentId) {
			var iIndex = this._aActiveExperiments.indexOf(sExperimentId);
			if (iIndex > -1) {
				this._aActiveExperiments.splice(iIndex, 1);
			}
		},

		/**
		 * Handle WebSocket error event
		 * @private
		 * @param {Event} oEvent - Error event
		 */
		_onError: function (oEvent) {
			console.error("Training WebSocket error:", oEvent);
			this.fireEvent("error", {
				error: oEvent
			});
		},

		/**
		 * Handle WebSocket close event
		 * @private
		 * @param {CloseEvent} oEvent - Close event
		 */
		_onClose: function (oEvent) {
			console.log("Training WebSocket closed:", oEvent.code, oEvent.reason);
			this._bConnected = false;

			this.fireEvent("connectionChange", {
				connected: false
			});

			// Attempt reconnection for abnormal closure
			if (oEvent.code !== 1000) {
				this._scheduleReconnect();
			}
		},

		/**
		 * Schedule reconnection attempt with exponential backoff
		 * @private
		 */
		_scheduleReconnect: function () {
			if (this._iReconnectAttempts >= this._iMaxReconnectAttempts) {
				console.error("Max reconnection attempts reached for training WebSocket");
				this.fireEvent("maxReconnectAttemptsReached");
				return;
			}

			this._iReconnectAttempts++;

			// Exponential backoff with jitter
			var iDelay = Math.min(
				this._iReconnectInterval * Math.pow(2, this._iReconnectAttempts - 1),
				30000 // Max 30 seconds
			);
			iDelay += Math.random() * 1000; // Add jitter

			console.log("Reconnecting training WebSocket in " + Math.round(iDelay / 1000) +
				" seconds (attempt " + this._iReconnectAttempts + "/" + this._iMaxReconnectAttempts + ")");

			this._oReconnectTimer = setTimeout(function () {
				this._oReconnectTimer = null;
				this.connect();
			}.bind(this), iDelay);
		},

		/**
		 * Send all queued messages
		 * @private
		 */
		_flushMessageQueue: function () {
			while (this._aMessageQueue.length > 0 && this._bConnected) {
				var oMessage = this._aMessageQueue.shift();
				this._send(oMessage);
			}
		},

		/**
		 * Cleanup on service destruction
		 * @public
		 */
		destroy: function () {
			this.disconnect();
			ManagedObject.prototype.destroy.apply(this, arguments);
		}

	});

	return TrainingWebSocketService;
});
