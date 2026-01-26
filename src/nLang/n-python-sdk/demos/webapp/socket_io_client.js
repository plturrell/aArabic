// HPC WebSocket Client for n-c-sdk Real-Time Metrics Streaming
(function(global) {
    'use strict';

    console.log('[HPC-WS] Client library loaded');

    function Socket(url) {
        this.url = url || ('ws://' + window.location.host + '/socket.io/');
        this.ws = null;
        this.connected = false;
        this.handlers = {};
        this.reconnectAttempts = 0;
        this.maxReconnects = 10;
        this.reconnectDelay = 2000;
    }

    Socket.prototype.connect = function() {
        var self = this;

        try {
            console.log('[HPC-WS] Connecting to', this.url);
            this.ws = new WebSocket(this.url);

            this.ws.onopen = function() {
                self.connected = true;
                self.reconnectAttempts = 0;
                self._emit('connect');
                console.log('[HPC-WS] ✅ Connected to HPC metrics stream');
            };

            this.ws.onclose = function() {
                var wasConnected = self.connected;
                self.connected = false;
                self._emit('disconnect');

                // Auto-reconnect
                if (self.reconnectAttempts < self.maxReconnects) {
                    self.reconnectAttempts++;
                    console.log('[HPC-WS] Reconnecting... attempt', self.reconnectAttempts);
                    setTimeout(function() {
                        self.connect();
                    }, self.reconnectDelay);
                } else if (wasConnected) {
                    console.log('[HPC-WS] Max reconnect attempts reached');
                }
            };

            this.ws.onerror = function(err) {
                // Silent - WebSocket errors are expected during development
            };

            this.ws.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    if (data.type) {
                        // Emit the specific event type
                        self._emit(data.type, data.payload || data);
                    }
                    self._emit('message', data);
                } catch (e) {
                    self._emit('message', event.data);
                }
            };
        } catch (e) {
            console.log('[HPC-WS] Connection failed:', e.message);
        }
        return this;
    };

    Socket.prototype.on = function(event, callback) {
        if (!this.handlers[event]) {
            this.handlers[event] = [];
        }
        this.handlers[event].push(callback);
        return this;
    };

    Socket.prototype.off = function(event, callback) {
        if (this.handlers[event]) {
            if (callback) {
                this.handlers[event] = this.handlers[event].filter(function(h) {
                    return h !== callback;
                });
            } else {
                this.handlers[event] = [];
            }
        }
        return this;
    };

    Socket.prototype._emit = function(event, data) {
        var handlers = this.handlers[event];
        if (handlers) {
            for (var i = 0; i < handlers.length; i++) {
                try {
                    handlers[i](data);
                } catch (e) {
                    console.error('[HPC-WS] Handler error:', e);
                }
            }
        }
    };

    Socket.prototype.emit = function(type, payload) {
        if (this.ws && this.connected && this.ws.readyState === WebSocket.OPEN) {
            var msg = JSON.stringify({ type: type, payload: payload });
            this.ws.send(msg);
        }
        return this;
    };

    Socket.prototype.disconnect = function() {
        this.maxReconnects = 0; // Prevent auto-reconnect
        if (this.ws) {
            this.ws.close();
        }
        return this;
    };

    // io() factory function - compatible with Socket.io API
    function io(url) {
        var socket = new Socket(url);
        // Delay connection slightly to not block page load
        setTimeout(function() {
            socket.connect();
        }, 100);
        return socket;
    }

    io.Socket = Socket;
    io.connect = io;

    // Export
    global.io = io;

    console.log('[HPC-WS] Ready for real-time HPC metrics');

})(typeof window !== 'undefined' ? window : this);

