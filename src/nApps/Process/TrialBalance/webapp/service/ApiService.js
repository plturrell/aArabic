/**
 * ============================================================================
 * Trial Balance API Service Layer
 * Central service for backend API communication
 * ============================================================================
 *
 * [CODE:file=ApiService.js]
 * [CODE:module=service]
 * [CODE:language=javascript]
 *
 * [ODPS:product=trial-balance-aggregated]
 * [ODPS:product=variances]
 * [ODPS:product=exchange-rates]
 *
 * [API:consumes=/api/v1/trial-balance]
 * [API:consumes=/api/v1/variances]
 * [API:consumes=/api/v1/exchange-rates]
 * [API:consumes=/api/v1/odps]
 * [API:consumes=/api/v1/quality]
 * [API:consumes=/api/v1/lineage]
 *
 * [RELATION:calls=CODE:main.zig]
 * [RELATION:calls=CODE:trial_balance.zig]
 * [RELATION:calls=CODE:odps_api.zig]
 *
 * This service provides a centralized interface for all backend API calls,
 * implementing proper error handling and response transformation.
 */
sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log"
], function (BaseObject, Log) {
    "use strict";

    /**
     * ODPS Validation Rule IDs
     * Reference: backend/models/calculation/balance_engine.zig
     */
    var ODPSRuleID = {
        // Trial Balance Rules
        TB001_BALANCE_EQUATION: "TB001",
        TB002_DEBIT_CREDIT_BALANCE: "TB002",
        TB003_IFRS_CLASSIFICATION: "TB003",
        TB004_PERIOD_DATA_ACCURACY: "TB004",
        TB005_GCOA_MAPPING_COMPLETENESS: "TB005",
        TB006_GLOBAL_MAPPING_CURRENCY: "TB006",
        // Variance Rules
        VAR001_VARIANCE_CALCULATION: "VAR001",
        VAR002_VARIANCE_PERCENT: "VAR002",
        VAR003_MATERIALITY_THRESHOLD_BS: "VAR003",
        VAR004_MATERIALITY_THRESHOLD_PL: "VAR004",
        VAR005_COMMENTARY_REQUIRED: "VAR005",
        VAR006_COMMENTARY_COVERAGE_90: "VAR006",
        VAR007_EXCEPTION_FLAGGING: "VAR007",
        VAR008_MAJOR_DRIVER_IDENTIFICATION: "VAR008",
        // Exchange Rate Rules
        FX001_FROM_CURRENCY_MANDATORY: "FX001",
        FX002_TO_CURRENCY_MANDATORY: "FX002",
        FX003_RATE_POSITIVE: "FX003",
        FX004_RATIO_POSITIVE: "FX004",
        FX005_EXCHANGE_RATE_VERIFICATION: "FX005",
        FX006_PERIOD_SPECIFIC_RATE: "FX006",
        FX007_GROUP_RATE_SOURCE: "FX007"
    };

    /**
     * DOI Thresholds - Must match backend/models/calculation/balance_engine.zig
     */
    var DOIThresholds = {
        BALANCE_SHEET_AMOUNT: 100000000.0,    // $100M
        PROFIT_LOSS_AMOUNT: 3000000.0,        // $3M
        VARIANCE_PERCENTAGE: 0.10,            // 10%
        COMMENTARY_COVERAGE: 0.90,            // 90%
        BALANCE_TOLERANCE: 0.01               // 1 cent
    };

    /**
     * Driver Categories - Must match backend/models/calculation/balance_engine.zig
     */
    var DriverCategory = {
        VOLUME: "VOLUME",
        PRICE: "PRICE",
        MIX: "MIX",
        FX: "FX",
        ONE_TIME: "ONE_TIME",
        TIMING: "TIMING",
        ACQUISITION_DISPOSAL: "ACQUISITION_DISPOSAL",
        POLICY_CHANGE: "POLICY_CHANGE",
        OTHER: "OTHER"
    };

    /**
     * Rate Sources - Must match backend/models/calculation/fx_converter.zig
     */
    var RateSource = {
        GROUP_TREASURY: "GROUP_TREASURY",
        ECB: "ECB",
        FED: "FED",
        MANUAL: "MANUAL"
    };

    var ApiService = BaseObject.extend("trialbalance.service.ApiService", {
        
        /**
         * Constructor
         * @param {string} sBaseUrl - Base URL for API calls (default: /api/v1)
         */
        constructor: function (sBaseUrl) {
            BaseObject.call(this);
            this._sBaseUrl = sBaseUrl || "/api/v1";
            this._oCache = {};
            Log.info("[ApiService] Initialized with base URL: " + this._sBaseUrl);
        },

        // ========================================================================
        // Trial Balance APIs
        // ========================================================================

        /**
         * Fetch trial balance data
         * [API:endpoint=GET /api/v1/trial-balance]
         * @param {object} oParams - Query parameters (company_code, fiscal_year, period)
         * @returns {Promise<object>} Trial balance result
         */
        getTrialBalance: function (oParams) {
            return this._fetch("/trial-balance", oParams);
        },

        /**
         * Validate trial balance (runs TB001-TB006)
         * [API:endpoint=POST /api/v1/trial-balance/validate]
         * @param {object} oData - Trial balance data to validate
         * @returns {Promise<object>} Validation result with rule statuses
         */
        validateTrialBalance: function (oData) {
            return this._post("/trial-balance/validate", oData);
        },

        /**
         * Get YTD analysis
         * [API:endpoint=GET /api/v1/trial-balance/ytd]
         * @param {object} oParams - Query parameters
         * @returns {Promise<object>} YTD analysis result
         */
        getYTDAnalysis: function (oParams) {
            return this._fetch("/trial-balance/ytd", oParams);
        },

        // ========================================================================
        // Variance APIs
        // ========================================================================

        /**
         * Fetch variance analysis
         * [API:endpoint=GET /api/v1/variances]
         * @param {object} oParams - Query parameters
         * @returns {Promise<array>} Array of variance records
         */
        getVariances: function (oParams) {
            return this._fetch("/variances", oParams);
        },

        /**
         * Get material variances only (VAR003/VAR004 filtered)
         * [API:endpoint=GET /api/v1/variances/material]
         * @param {object} oParams - Query parameters
         * @returns {Promise<array>} Material variances
         */
        getMaterialVariances: function (oParams) {
            return this._fetch("/variances/material", oParams);
        },

        /**
         * Update variance commentary (VAR005)
         * [API:endpoint=PUT /api/v1/variances/{id}/commentary]
         * @param {string} sId - Variance ID
         * @param {object} oData - Commentary data
         * @returns {Promise<object>} Updated variance
         */
        updateVarianceCommentary: function (sId, oData) {
            return this._put("/variances/" + sId + "/commentary", oData);
        },

        /**
         * Get commentary coverage (VAR006)
         * [API:endpoint=GET /api/v1/variances/coverage]
         * @returns {Promise<object>} Coverage statistics
         */
        getCommentaryCoverage: function () {
            return this._fetch("/variances/coverage");
        },

        // ========================================================================
        // Exchange Rate APIs
        // ========================================================================

        /**
         * Fetch exchange rates
         * [API:endpoint=GET /api/v1/exchange-rates]
         * @param {object} oParams - Query parameters (from_currency, to_currency, date)
         * @returns {Promise<array>} Exchange rates
         */
        getExchangeRates: function (oParams) {
            return this._fetch("/exchange-rates", oParams);
        },

        /**
         * Convert amount using exchange rate
         * [API:endpoint=POST /api/v1/exchange-rates/convert]
         * @param {object} oData - Conversion request
         * @returns {Promise<object>} Converted amount
         */
        convertAmount: function (oData) {
            return this._post("/exchange-rates/convert", oData);
        },

        /**
         * Validate exchange rates (FX001-FX007)
         * [API:endpoint=POST /api/v1/exchange-rates/validate]
         * @param {object} oData - Rates to validate
         * @returns {Promise<object>} Validation result
         */
        validateExchangeRates: function (oData) {
            return this._post("/exchange-rates/validate", oData);
        },

        // ========================================================================
        // ODPS Catalog APIs
        // ========================================================================

        /**
         * Get ODPS products
         * [API:endpoint=GET /api/v1/odps/products]
         * @returns {Promise<array>} ODPS product catalog
         */
        getODPSProducts: function () {
            return this._fetch("/odps/products");
        },

        /**
         * Get ODPS rules for a product
         * [API:endpoint=GET /api/v1/odps/products/{id}/rules]
         * @param {string} sProductId - Product ID
         * @returns {Promise<array>} ODPS rules
         */
        getODPSRules: function (sProductId) {
            return this._fetch("/odps/products/" + sProductId + "/rules");
        },

        /**
         * Get ODPS lineage for a product
         * [API:endpoint=GET /api/v1/odps/products/{id}/lineage]
         * @param {string} sProductId - Product ID
         * @returns {Promise<object>} SCIP lineage graph
         */
        getODPSLineage: function (sProductId) {
            return this._fetch("/odps/products/" + sProductId + "/lineage");
        },

        // ========================================================================
        // Data Quality APIs
        // ========================================================================

        /**
         * Get data quality metrics
         * [API:endpoint=GET /api/v1/quality]
         * @returns {Promise<object>} Quality dashboard data
         */
        getQualityMetrics: function () {
            return this._fetch("/quality");
        },

        /**
         * Get rule validation status
         * [API:endpoint=GET /api/v1/quality/rules]
         * @returns {Promise<array>} Rule status array
         */
        getRuleValidationStatus: function () {
            return this._fetch("/quality/rules");
        },

        /**
         * Get quality trends
         * [API:endpoint=GET /api/v1/quality/trends]
         * @param {object} oParams - Date range parameters
         * @returns {Promise<array>} Historical quality metrics
         */
        getQualityTrends: function (oParams) {
            return this._fetch("/quality/trends", oParams);
        },

        // ========================================================================
        // Lineage APIs
        // ========================================================================

        /**
         * Get code lineage graph
         * [API:endpoint=GET /api/v1/lineage]
         * @returns {Promise<object>} SCIP lineage data
         */
        getLineageGraph: function () {
            return this._fetch("/lineage");
        },

        /**
         * Get symbol details
         * [API:endpoint=GET /api/v1/lineage/symbol/{id}]
         * @param {string} sSymbolId - SCIP symbol ID
         * @returns {Promise<object>} Symbol details
         */
        getSymbolDetails: function (sSymbolId) {
            return this._fetch("/lineage/symbol/" + encodeURIComponent(sSymbolId));
        },

        // ========================================================================
        // Workflow APIs
        // ========================================================================

        /**
         * Get workflow status
         * [API:endpoint=GET /api/v1/workflow]
         * @returns {Promise<object>} Current workflow state
         */
        getWorkflowStatus: function () {
            return this._fetch("/workflow");
        },

        /**
         * Get checklist items
         * [API:endpoint=GET /api/v1/workflow/checklist]
         * @returns {Promise<array>} Checklist items
         */
        getChecklistItems: function () {
            return this._fetch("/workflow/checklist");
        },

        /**
         * Update checklist item
         * [API:endpoint=PUT /api/v1/workflow/checklist/{id}]
         * @param {string} sId - Checklist item ID
         * @param {object} oData - Update data
         * @returns {Promise<object>} Updated item
         */
        updateChecklistItem: function (sId, oData) {
            return this._put("/workflow/checklist/" + sId, oData);
        },

        /**
         * Submit for review (maker -> checker)
         * [API:endpoint=POST /api/v1/workflow/submit]
         * @param {object} oData - Submission data
         * @returns {Promise<object>} Workflow state
         */
        submitForReview: function (oData) {
            return this._post("/workflow/submit", oData);
        },

        /**
         * Approve workflow (checker approval)
         * [API:endpoint=POST /api/v1/workflow/approve]
         * @param {object} oData - Approval data
         * @returns {Promise<object>} Workflow state
         */
        approveWorkflow: function (oData) {
            return this._post("/workflow/approve", oData);
        },

        // ========================================================================
        // Internal Methods
        // ========================================================================

        /**
         * Perform GET request
         * @private
         */
        _fetch: function (sPath, oParams) {
            var sUrl = this._sBaseUrl + sPath;
            
            if (oParams) {
                var aParams = Object.keys(oParams).map(function (key) {
                    return encodeURIComponent(key) + "=" + encodeURIComponent(oParams[key]);
                });
                if (aParams.length > 0) {
                    sUrl += "?" + aParams.join("&");
                }
            }

            return this._request("GET", sUrl);
        },

        /**
         * Perform POST request
         * @private
         */
        _post: function (sPath, oData) {
            return this._request("POST", this._sBaseUrl + sPath, oData);
        },

        /**
         * Perform PUT request
         * @private
         */
        _put: function (sPath, oData) {
            return this._request("PUT", this._sBaseUrl + sPath, oData);
        },

        /**
         * Generic request handler
         * @private
         */
        _request: function (sMethod, sUrl, oData) {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                var oXhr = new XMLHttpRequest();
                oXhr.open(sMethod, sUrl, true);
                oXhr.setRequestHeader("Content-Type", "application/json");
                oXhr.setRequestHeader("Accept", "application/json");

                oXhr.onload = function () {
                    if (oXhr.status >= 200 && oXhr.status < 300) {
                        try {
                            var oResponse = JSON.parse(oXhr.responseText);
                            Log.info("[ApiService] " + sMethod + " " + sUrl + " succeeded");
                            resolve(oResponse);
                        } catch (e) {
                            Log.error("[ApiService] Failed to parse response", e);
                            reject(new Error("Invalid JSON response"));
                        }
                    } else {
                        Log.error("[ApiService] " + sMethod + " " + sUrl + " failed: " + oXhr.status);
                        reject(new Error("Request failed: " + oXhr.status + " " + oXhr.statusText));
                    }
                };

                oXhr.onerror = function () {
                    Log.error("[ApiService] Network error for " + sUrl);
                    reject(new Error("Network error"));
                };

                if (oData) {
                    oXhr.send(JSON.stringify(oData));
                } else {
                    oXhr.send();
                }
            });
        },

        /**
         * Clear cache
         */
        clearCache: function () {
            this._oCache = {};
        }
    });

    // Expose constants
    ApiService.ODPSRuleID = ODPSRuleID;
    ApiService.DOIThresholds = DOIThresholds;
    ApiService.DriverCategory = DriverCategory;
    ApiService.RateSource = RateSource;

    return ApiService;
});