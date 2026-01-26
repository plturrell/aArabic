sap.ui.define([
	"sap/ui/base/ManagedObject"
], function (ManagedObject) {
	"use strict";

	/**
	 * WebSocket Service for real-time communication with LLM server
	 * @extends sap.ui.base.ManagedObject
	 */
	var WebSocketService = ManagedObject.extend("llm.server.dashboard.service.WebSocketService", {

		metadata: {
			events: {
				connectionChange: {
					parameters: {
						connected: { type: "boolean" }
					}
				},
				message: {
					parameters: {
						message: { type: "object" }
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
			this._sUrl = mSettings.url || "ws://localhost:8080/ws";
			this._iReconnectInterval = mSettings.reconnectInterval || 3000;
			this._iMaxReconnectAttempts = mSettings.maxReconnectAttempts || 10;
			this._iReconnectAttempts = 0;
			this._oSocket = null;
			this._bConnected = false;
			this._oReconnectTimer = null;
			this._aMessageQueue = [];
		},

		/**
		 * Connect to WebSocket server
		 * @public
		 */
		connect: function () {
			if (this._oSocket && this._oSocket.readyState === WebSocket.OPEN) {
				console.log("WebSocket already connected");
				return;
			}

			try {
				console.log("Connecting to WebSocket:", this._sUrl);
				this._oSocket = new WebSocket(this._sUrl);

				this._oSocket.onopen = this._onOpen.bind(this);
				this._oSocket.onmessage = this._onMessage.bind(this);
				this._oSocket.onerror = this._onError.bind(this);
				this._oSocket.onclose = this._onClose.bind(this);
			} catch (e) {
				console.error("Failed to create WebSocket:", e);
				this._scheduleReconnect();
			}
		},

		/**
		 * Disconnect from WebSocket server
		 * @public
		 */
		disconnect: function () {
			if (this._oReconnectTimer) {
				clearTimeout(this._oReconnectTimer);
				this._oReconnectTimer = null;
			}

			if (this._oSocket) {
				this._oSocket.close();
				this._oSocket = null;
			}

			this._bConnected = false;
			this._iReconnectAttempts = 0;
		},

		/**
		 * Send message to WebSocket server
		 * @public
		 * @param {object} oMessage - Message to send
		 */
		send: function (oMessage) {
			if (this._bConnected && this._oSocket) {
				try {
					this._oSocket.send(JSON.stringify(oMessage));
				} catch (e) {
					console.error("Failed to send message:", e);
					this._aMessageQueue.push(oMessage);
				}
			} else {
				// Queue message for sending when connected
				this._aMessageQueue.push(oMessage);
			}
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
		 * Handle WebSocket open event
		 * @private
		 */
		_onOpen: function () {
			console.log("WebSocket connected");
			this._bConnected = true;
			this._iReconnectAttempts = 0;

			// Fire connection change event
			this.fireEvent("connectionChange", {
				connected: true
			});

			// Send queued messages
			this._flushMessageQueue();
		},

		/**
		 * Handle WebSocket message event
		 * @private
		 * @param {MessageEvent} oEvent - Message event
		 */
		_onMessage: function (oEvent) {
			try {
				var oMessage = JSON.parse(oEvent.data);
				
				// Fire message event
				this.fireEvent("message", {
					message: oMessage
				});
			} catch (e) {
				console.error("Failed to parse message:", e);
			}
		},

		/**
		 * Handle WebSocket error event
		 * @private
		 * @param {Event} oEvent - Error event
		 */
		_onError: function (oEvent) {
			console.error("WebSocket error:", oEvent);
			
			// Fire error event
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
			console.log("WebSocket closed:", oEvent.code, oEvent.reason);
			this._bConnected = false;

			// Fire connection change event
			this.fireEvent("connectionChange", {
				connected: false
			});

			// Attempt reconnection
			if (oEvent.code !== 1000) { // Not normal closure
				this._scheduleReconnect();
			}
		},

		/**
		 * Schedule reconnection attempt
		 * @private
		 */
		_scheduleReconnect: function () {
			if (this._iReconnectAttempts >= this._iMaxReconnectAttempts) {
				console.error("Max reconnection attempts reached");
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

			console.log("Reconnecting in " + Math.round(iDelay / 1000) + " seconds (attempt " + 
				this._iReconnectAttempts + "/" + this._iMaxReconnectAttempts + ")");

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
				this.send(oMessage);
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

	return WebSocketService;
});
