/**
 * JWT Token Management Utility (Day 13)
 * 
 * Handles JWT token storage, retrieval, and generation for testing.
 * In production, tokens would be obtained from Keycloak or other OAuth provider.
 */
sap.ui.define([], function () {
    "use strict";

    var TokenManager = {
        
        /**
         * Storage keys
         */
        _STORAGE_KEY: "jwt_token",
        _SESSION_KEY: "jwt_session_token",
        _USER_KEY: "current_user",

        /**
         * Get the current JWT token
         * @returns {string|null} JWT token or null
         */
        getToken: function () {
            // Check localStorage first (persistent)
            var sToken = localStorage.getItem(this._STORAGE_KEY);
            if (sToken && this._isTokenValid(sToken)) {
                return sToken;
            }

            // Check sessionStorage (session-only)
            sToken = sessionStorage.getItem(this._SESSION_KEY);
            if (sToken && this._isTokenValid(sToken)) {
                return sToken;
            }

            return null;
        },

        /**
         * Set the JWT token
         * @param {string} sToken - JWT token
         * @param {boolean} bPersistent - Store in localStorage (true) or sessionStorage (false)
         */
        setToken: function (sToken, bPersistent) {
            if (bPersistent) {
                localStorage.setItem(this._STORAGE_KEY, sToken);
            } else {
                sessionStorage.setItem(this._SESSION_KEY, sToken);
            }

            // Extract and store user info
            var oClaims = this._decodeToken(sToken);
            if (oClaims && oClaims.user_id) {
                localStorage.setItem(this._USER_KEY, oClaims.user_id);
            }
        },

        /**
         * Remove the JWT token (logout)
         */
        clearToken: function () {
            localStorage.removeItem(this._STORAGE_KEY);
            sessionStorage.removeItem(this._SESSION_KEY);
            localStorage.removeItem(this._USER_KEY);
        },

        /**
         * Get current user ID
         * @returns {string|null} User ID or null
         */
        getCurrentUser: function () {
            var sToken = this.getToken();
            if (sToken) {
                var oClaims = this._decodeToken(sToken);
                return oClaims ? oClaims.user_id : null;
            }

            // Fallback to stored user
            return localStorage.getItem(this._USER_KEY);
        },

        /**
         * Check if user is authenticated
         * @returns {boolean} True if valid token exists
         */
        isAuthenticated: function () {
            return this.getToken() !== null;
        },

        /**
         * Decode JWT token payload (no verification, just parsing)
         * @param {string} sToken - JWT token
         * @returns {Object|null} Decoded payload or null
         * @private
         */
        _decodeToken: function (sToken) {
            try {
                var aParts = sToken.split('.');
                if (aParts.length !== 3) return null;

                // Decode payload (second part)
                var sPayload = aParts[1];
                // Add padding if needed
                var sPadded = sPayload + '='.repeat((4 - sPayload.length % 4) % 4);
                var sDecoded = atob(sPadded.replace(/-/g, '+').replace(/_/g, '/'));
                
                return JSON.parse(sDecoded);
            } catch (e) {
                console.error("Failed to decode token:", e);
                return null;
            }
        },

        /**
         * Check if token is valid (not expired)
         * @param {string} sToken - JWT token
         * @returns {boolean} True if valid
         * @private
         */
        _isTokenValid: function (sToken) {
            var oClaims = this._decodeToken(sToken);
            if (!oClaims || !oClaims.exp) return false;

            // Check expiration
            var nNow = Math.floor(Date.now() / 1000);
            return oClaims.exp > nNow;
        },

        /**
         * Generate a demo JWT token for testing (DEMO ONLY)
         * In production, tokens should come from Keycloak/OAuth provider
         * 
         * @param {string} sUserId - User identifier
         * @param {number} nExpirationHours - Token lifetime in hours
         * @returns {string} JWT token
         */
        generateDemoToken: function (sUserId, nExpirationHours) {
            nExpirationHours = nExpirationHours || 24;

            // Create header
            var oHeader = {
                alg: "HS256",
                typ: "JWT"
            };

            // Create payload
            var nNow = Math.floor(Date.now() / 1000);
            var oPayload = {
                user_id: sUserId,
                iat: nNow,
                exp: nNow + (nExpirationHours * 3600)
            };

            // Encode parts (base64url)
            var sHeader = this._base64urlEncode(JSON.stringify(oHeader));
            var sPayload = this._base64urlEncode(JSON.stringify(oPayload));

            // For demo purposes, use a simple signature placeholder
            // In production, this would be HMAC-SHA256 signed by the server
            var sSignature = this._base64urlEncode("demo-signature-" + sUserId + "-" + nNow);

            return sHeader + "." + sPayload + "." + sSignature;
        },

        /**
         * Base64 URL-safe encoding
         * @param {string} sInput - String to encode
         * @returns {string} Base64 URL-safe encoded string
         * @private
         */
        _base64urlEncode: function (sInput) {
            var sEncoded = btoa(sInput);
            return sEncoded
                .replace(/\+/g, '-')
                .replace(/\//g, '_')
                .replace(/=/g, '');
        },

        /**
         * Get authorization headers for API requests
         * @returns {Object} Headers object with Authorization if token exists
         */
        getAuthHeaders: function () {
            var oHeaders = {
                "Content-Type": "application/json"
            };

            var sToken = this.getToken();
            if (sToken) {
                oHeaders["Authorization"] = "Bearer " + sToken;
            }

            return oHeaders;
        },

        /**
         * Create a demo login for testing
         * @param {string} sUserId - User ID for demo
         * @param {boolean} bRememberMe - Store persistently
         */
        demoLogin: function (sUserId, bRememberMe) {
            var sToken = this.generateDemoToken(sUserId, 24);
            this.setToken(sToken, bRememberMe);
            return sToken;
        },

        /**
         * Logout (clear token)
         */
        logout: function () {
            this.clearToken();
        }
    };

    return TokenManager;
});
