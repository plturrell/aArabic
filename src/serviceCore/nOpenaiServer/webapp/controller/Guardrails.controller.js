sap.ui.define([
	"sap/ui/core/mvc/Controller",
	"sap/ui/model/json/JSONModel",
	"llm/server/dashboard/utils/ApiService"
], function (Controller, JSONModel, ApiService) {
	"use strict";

	return Controller.extend("llm.server.dashboard.controller.Guardrails", {

		onInit: function () {
			// Initialize guardrails data model
			const oModel = new JSONModel({
				total_validations: 0,
				violations: 0,
				violation_rate: 0.0,
				violations_by_type: {
					toxicity: 0,
					pii_detected: 0,
					jailbreak_attempt: 0,
					size_limit: 0,
					blocked_pattern: 0
				},
				config: {
					max_tokens: 4096,
					pii_detection: true,
					jailbreak_enabled: true,
					mask_pii: true,
					toxicity_threshold: 0.7,
					sexual_threshold: 0.8,
					violence_threshold: 0.8
				},
				recent_violations: [],
				blocked_patterns: [],
				test_content: "",
				test_type: "input",
				test_result: {
					tested: false,
					passed: false,
					reason: "",
					score: 0,
					masked_content: ""
				}
			});
			this.getView().setModel(oModel, "guardrails");

			// Load initial data
			this._loadMetrics();
			this._loadPolicies();
			this._loadViolations();

			// Poll for updates every 5 seconds
			this._metricsInterval = setInterval(() => {
				this._loadMetrics();
				this._loadViolations();
			}, 5000);
		},

		onExit: function () {
			if (this._metricsInterval) {
				clearInterval(this._metricsInterval);
			}
		},

		_loadMetrics: async function () {
			try {
				const metrics = await ApiService.get("/v1/guardrails/metrics");
				const oModel = this.getView().getModel("guardrails");
				oModel.setProperty("/total_validations", metrics.total_validations || 0);
				oModel.setProperty("/violations", metrics.violations || 0);
				oModel.setProperty("/violation_rate", metrics.violation_rate || 0.0);
				oModel.setProperty("/violations_by_type", metrics.violations_by_type || {});
			} catch (err) {
				console.warn("Failed to load guardrails metrics:", err);
			}
		},

		_loadPolicies: async function () {
			try {
				const policies = await ApiService.get("/v1/guardrails/policies");
				const oModel = this.getView().getModel("guardrails");
				oModel.setProperty("/config", policies.config || oModel.getProperty("/config"));
				oModel.setProperty("/blocked_patterns", policies.blocked_patterns || []);
			} catch (err) {
				console.warn("Failed to load policies:", err);
			}
		},

		_loadViolations: async function () {
			try {
				const violations = await ApiService.get("/v1/guardrails/violations");
				const oModel = this.getView().getModel("guardrails");
				oModel.setProperty("/recent_violations", violations.violations || []);
			} catch (err) {
				console.warn("Failed to load violations:", err);
			}
		},

		onTogglePolicy: function (oEvent) {
			const oSwitch = oEvent.getSource();
			const sPath = oSwitch.getBinding("state").getPath();
			const bEnabled = oEvent.getParameter("state");
			
			console.log("Policy toggled:", sPath, "=>", bEnabled);
			
			// Save policy change
			this._savePolicy();
		},

		onThresholdChange: function (oEvent) {
			// Save policy change when slider moves
			this._savePolicy();
		},

		onUpdatePolicy: function () {
			this._savePolicy();
		},

		_savePolicy: async function () {
			try {
				const oModel = this.getView().getModel("guardrails");
				const config = oModel.getProperty("/config");
				
				await ApiService.post("/v1/guardrails/policies", { config });
				
				sap.m.MessageToast.show("Policy updated successfully");
			} catch (err) {
				sap.m.MessageBox.error("Failed to save policy: " + err.message);
			}
		},

		onTestValidation: async function () {
			const oModel = this.getView().getModel("guardrails");
			const content = oModel.getProperty("/test_content");
			const type = oModel.getProperty("/test_type");

			if (!content || content.trim() === "") {
				sap.m.MessageBox.warning("Please enter content to validate");
				return;
			}

			try {
				const result = await ApiService.post("/v1/guardrails/validate", {
					content,
					type
				});

				oModel.setProperty("/test_result", {
					tested: true,
					passed: result.passed,
					reason: result.reason || "Content is safe",
					score: result.score || 0,
					masked_content: result.masked_content || ""
				});
			} catch (err) {
				sap.m.MessageBox.error("Validation failed: " + err.message);
			}
		},

		onAddBlockedPattern: function () {
			const oInput = this.byId("newPattern");
			const sPattern = oInput.getValue();

			if (!sPattern || sPattern.trim() === "") {
				sap.m.MessageBox.warning("Please enter a pattern");
				return;
			}

			const oModel = this.getView().getModel("guardrails");
			const aPatterns = oModel.getProperty("/blocked_patterns");
			
			aPatterns.push({
				pattern: sPattern,
				match_count: 0
			});

			oModel.setProperty("/blocked_patterns", aPatterns);
			oInput.setValue("");

			// Save to backend
			this._savePolicy();
		},

		onDeletePattern: function (oEvent) {
			const oList = oEvent.getSource();
			const oItem = oEvent.getParameter("listItem");
			const sPath = oItem.getBindingContext("guardrails").getPath();
			const iIndex = parseInt(sPath.split("/").pop());

			const oModel = this.getView().getModel("guardrails");
			const aPatterns = oModel.getProperty("/blocked_patterns");
			aPatterns.splice(iIndex, 1);
			oModel.setProperty("/blocked_patterns", aPatterns);

			// Save to backend
			this._savePolicy();
		},

		onShowViolationDetails: function (oEvent) {
			const oButton = oEvent.getSource();
			const oContext = oButton.getBindingContext("guardrails");
			const oViolation = oContext.getObject();

			sap.m.MessageBox.information(
				`Type: ${oViolation.type}\n` +
				`Score: ${oViolation.score}\n` +
				`Action: ${oViolation.action}\n\n` +
				`Content:\n${oViolation.content}`,
				{
					title: "Violation Details"
				}
			);
		}

	});
});
