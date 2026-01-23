sap.ui.define([], function () {
	"use strict";

	// Resolve runtime configuration from window overrides or sensible defaults.
	function resolveConfig() {
		var userConfig = window.__LLM_DASHBOARD_CONFIG__ || {};
		var origin = window.location && window.location.origin ? window.location.origin : "http://localhost:8080";
		var apiBaseUrl = userConfig.apiBaseUrl || origin;
		var wsPath = userConfig.wsPath || "/ws";
		var wsUrl = userConfig.wsUrl || apiBaseUrl.replace(/^http/i, "ws").replace(/\/+$/, "") + wsPath;

		return {
			apiBaseUrl: apiBaseUrl,
			wsUrl: wsUrl,
			keycloak: {
				url: userConfig.keycloakUrl || (apiBaseUrl + "/auth"),
				realm: userConfig.keycloakRealm || "nucleus",
				clientId: userConfig.keycloakClientId || "dashboard-client"
			}
		};
	}

	var _resolved = resolveConfig();

	return {
		get: function () {
			return _resolved;
		},
		refresh: function () {
			_resolved = resolveConfig();
			return _resolved;
		}
	};
});
