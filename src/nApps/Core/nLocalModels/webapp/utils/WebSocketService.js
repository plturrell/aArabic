sap.ui.define([], function () {
    "use strict";

    /**
     * WebSocket Service for real-time updates
     * Handles: WebSocket connection, auto-reconnect, channel subscriptions, events
     */
    const WebSocketService = {

        // Configuration
        config: {
            wsUrl: "ws://localhost:8080/ws",
            reconnectBaseDelay: 1000,      // Base delay for exponential backoff (1s)
            reconnectMaxDelay: 30000,      // Maximum delay (30s)
            reconnectMultiplier: 2         // Exponential multiplier
        },

        // Connection status constants
        Status: {
            DISCONNECTED: "disconnected",
            CONNECTING: "connecting",
            CONNECTED: "connected",
            ERROR: "error"
        },

        // Default channels
        Channels: {
            AGENTS: "agents",
            MODELS: "models",
            WORKFLOWS: "workflows",
            METRICS: "metrics"
        },

        // State
        _connection: null,
        _status: "disconnected",
        _token: null,
        _reconnectAttempts: 0,
        _reconnectTimer: null,
        _subscriptions: {},              // { channel: [callback1, callback2, ...] }
        _statusListeners: [],            // Callbacks for status changes

        // ========================================================================
        // CONNECTION MANAGEMENT
        // ========================================================================

        /**
         * Set authentication token for WebSocket connection
         * @param {string} token - Bearer token for authentication
         */
        setToken: function(token) {
            this._token = token;
        },

        /**
         * Get current connection status
         * @returns {string} Current status (connected, connecting, disconnected, error)
         */
        getStatus: function() {
            return this._status;
        },

        /**
         * Check if currently connected
         * @returns {boolean} True if connected
         */
        isConnected: function() {
            return this._status === this.Status.CONNECTED;
        },

        /**
         * Connect to WebSocket server
         * @returns {Promise} Resolves when connected, rejects on error
         */
        connect: function() {
            return new Promise((resolve, reject) => {
                if (this._connection && this._connection.readyState === WebSocket.OPEN) {
                    console.log("WebSocketService: Already connected");
                    resolve();
                    return;
                }

                this._setStatus(this.Status.CONNECTING);
                
                try {
                    this._connection = new WebSocket(this.config.wsUrl);

                    this._connection.onopen = () => {
                        console.log("WebSocketService: Connected");
                        this._reconnectAttempts = 0;
                        this._setStatus(this.Status.CONNECTED);

                        // Send authentication
                        if (this._token) {
                            this._sendAuth();
                        }

                        // Resubscribe to all channels
                        this._resubscribeAll();

                        resolve();
                    };

                    this._connection.onmessage = (event) => {
                        this._handleMessage(event);
                    };

                    this._connection.onerror = (error) => {
                        console.error("WebSocketService: Connection error", error);
                        this._setStatus(this.Status.ERROR);
                        reject(error);
                    };

                    this._connection.onclose = (event) => {
                        console.log("WebSocketService: Connection closed", event.code, event.reason);
                        this._setStatus(this.Status.DISCONNECTED);
                        this._scheduleReconnect();
                    };

                } catch (error) {
                    console.error("WebSocketService: Failed to create connection", error);
                    this._setStatus(this.Status.ERROR);
                    reject(error);
                }
            });
        },

        /**
         * Disconnect from WebSocket server
         */
        disconnect: function() {
            this._clearReconnectTimer();
            
            if (this._connection) {
                // Prevent auto-reconnect on intentional close
                this._connection.onclose = null;
                this._connection.close(1000, "Client disconnect");
                this._connection = null;
            }
            
            this._setStatus(this.Status.DISCONNECTED);
            console.log("WebSocketService: Disconnected");
        },

        // ========================================================================
        // SUBSCRIPTION MANAGEMENT
        // ========================================================================

        /**
         * Subscribe to a channel
         * @param {string} channel - Channel name (agents, models, workflows, metrics)
         * @param {function} callback - Callback function(data) for channel updates
         * @returns {function} Unsubscribe function
         */
        subscribe: function(channel, callback) {
            if (!this._subscriptions[channel]) {
                this._subscriptions[channel] = [];
            }

            this._subscriptions[channel].push(callback);

            // Send subscription message if connected
            if (this.isConnected()) {
                this._sendSubscription(channel);
            }

            // Return unsubscribe function
            return () => this._removeCallback(channel, callback);
        },

        /**
         * Unsubscribe from a channel (removes all callbacks for that channel)
         * @param {string} channel - Channel name to unsubscribe from
         */
        unsubscribe: function(channel) {
            if (this._subscriptions[channel]) {
                delete this._subscriptions[channel];

                // Send unsubscription message if connected
                if (this.isConnected()) {
                    this._sendUnsubscription(channel);
                }

                console.log("WebSocketService: Unsubscribed from", channel);
            }
        },

        /**
         * Listen for connection status changes
         * @param {function} callback - Callback function(status, previousStatus)
         * @returns {function} Unsubscribe function
         */
        onStatusChange: function(callback) {
            this._statusListeners.push(callback);
            return () => {
                const idx = this._statusListeners.indexOf(callback);
                if (idx > -1) {
                    this._statusListeners.splice(idx, 1);
                }
            };
        },

        // ========================================================================
        // MESSAGING
        // ========================================================================

        /**
         * Send a message through WebSocket
         * @param {object|string} message - Message to send (will be JSON stringified if object)
         * @returns {boolean} True if message was sent successfully
         */
        send: function(message) {
            if (!this.isConnected()) {
                console.warn("WebSocketService: Cannot send, not connected");
                return false;
            }

            try {
                const payload = typeof message === "string"
                    ? message
                    : JSON.stringify(message);
                this._connection.send(payload);
                return true;
            } catch (error) {
                console.error("WebSocketService: Send error", error);
                return false;
            }
        },

        // ========================================================================
        // PRIVATE METHODS
        // ========================================================================

        _setStatus: function(newStatus) {
            const previousStatus = this._status;
            this._status = newStatus;

            if (previousStatus !== newStatus) {
                this._statusListeners.forEach(callback => {
                    try {
                        callback(newStatus, previousStatus);
                    } catch (e) {
                        console.error("WebSocketService: Status listener error", e);
                    }
                });
            }
        },

        _sendAuth: function() {
            this.send({
                type: "auth",
                token: "Bearer " + this._token
            });
        },

        _sendSubscription: function(channel) {
            this.send({
                type: "subscribe",
                channel: channel
            });
        },

        _sendUnsubscription: function(channel) {
            this.send({
                type: "unsubscribe",
                channel: channel
            });
        },

        _resubscribeAll: function() {
            Object.keys(this._subscriptions).forEach(channel => {
                if (this._subscriptions[channel].length > 0) {
                    this._sendSubscription(channel);
                }
            });
        },

        _handleMessage: function(event) {
            try {
                const message = JSON.parse(event.data);
                const channel = message.channel || message.type;

                // Notify subscribers for this channel
                if (channel && this._subscriptions[channel]) {
                    this._subscriptions[channel].forEach(callback => {
                        try {
                            callback(message.data || message);
                        } catch (e) {
                            console.error("WebSocketService: Callback error for", channel, e);
                        }
                    });
                }

            } catch (e) {
                console.error("WebSocketService: Message parse error", e);
            }
        },

        _removeCallback: function(channel, callback) {
            if (this._subscriptions[channel]) {
                const idx = this._subscriptions[channel].indexOf(callback);
                if (idx > -1) {
                    this._subscriptions[channel].splice(idx, 1);
                }

                // Unsubscribe from channel if no more callbacks
                if (this._subscriptions[channel].length === 0) {
                    this.unsubscribe(channel);
                }
            }
        },

        _scheduleReconnect: function() {
            this._clearReconnectTimer();

            // Calculate delay with exponential backoff
            const delay = Math.min(
                this.config.reconnectBaseDelay * Math.pow(this.config.reconnectMultiplier, this._reconnectAttempts),
                this.config.reconnectMaxDelay
            );

            console.log(`WebSocketService: Reconnecting in ${delay}ms (attempt ${this._reconnectAttempts + 1})`);

            this._reconnectTimer = setTimeout(() => {
                this._reconnectAttempts++;
                this.connect().catch(() => {
                    // Connection failed, will trigger onclose and reschedule
                });
            }, delay);
        },

        _clearReconnectTimer: function() {
            if (this._reconnectTimer) {
                clearTimeout(this._reconnectTimer);
                this._reconnectTimer = null;
            }
        }
    };

    return WebSocketService;
});

