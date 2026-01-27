/**
 * ============================================================================
 * SAP HANA Cloud Service
 * Frontend wrapper for HANA Cloud data access via backend API
 * ============================================================================
 *
 * [CODE:file=HanaService.js]
 * [CODE:module=service]
 * [CODE:language=javascript]
 *
 * [CONFIG:requires=HANA_HOST]
 * [CONFIG:requires=HANA_USER]
 *
 * [ODPS:product=acdoca-journal-entries]
 * [ODPS:product=trial-balance-aggregated]
 * [ODPS:product=account-master]
 * [ODPS:product=exchange-rates]
 *
 * [TABLE:reads=ACDOCA]
 * [TABLE:reads=SKA1]
 * [TABLE:reads=TCURR]
 *
 * [API:consumes=/api/v1/hana]
 *
 * [RELATION:uses=CODE:ApiService.js]
 * [RELATION:calls=CODE:hana_client.zig]
 *
 * This service provides HANA Cloud data access for the Trial Balance application,
 * fetching journal entries, exchange rates, and account master data.
 */
sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log"
], function (BaseObject, Log) {
    "use strict";

    /**
     * HANA Configuration
     */
    var HanaConfig = {
        baseApiUrl: "/api/v1/hana",
        defaultSchema: "SAPABAP1",
        defaultClient: "100"
    };

    /**
     * Common SAP table mappings
     */
    var SAPTables = {
        ACDOCA: {
            name: "ACDOCA",
            description: "Universal Journal Entry Line Items",
            keyFields: ["RBUKRS", "GJAHR", "BELNR", "BUZEI", "RLDNR"]
        },
        SKA1: {
            name: "SKA1",
            description: "G/L Account Master (Chart of Accounts)",
            keyFields: ["KTOPL", "SAKNR"]
        },
        SKAT: {
            name: "SKAT",
            description: "G/L Account Master Record (Description)",
            keyFields: ["SPRAS", "KTOPL", "SAKNR"]
        },
        TCURR: {
            name: "TCURR",
            description: "Exchange Rates",
            keyFields: ["KURST", "FCURR", "TCURR", "GDATU"]
        },
        T001: {
            name: "T001",
            description: "Company Codes",
            keyFields: ["BUKRS"]
        }
    };

    var HanaService = BaseObject.extend("trialbalance.service.HanaService", {
        
        /**
         * Constructor
         * @param {object} oConfig - Configuration override
         */
        constructor: function (oConfig) {
            BaseObject.call(this);
            
            if (oConfig) {
                Object.assign(HanaConfig, oConfig);
            }
            
            this._connectionStatus = "unknown";
            Log.info("[HanaService] Initialized with base URL: " + HanaConfig.baseApiUrl);
        },

        // ========================================================================
        // Connection Management
        // ========================================================================

        /**
         * Test HANA connection
         * @returns {Promise<object>} Connection status
         */
        testConnection: function () {
            var that = this;
            return this._request("GET", "/test-connection")
                .then(function (oResult) {
                    that._connectionStatus = oResult.connected ? "connected" : "disconnected";
                    Log.info("[HanaService] Connection test: " + that._connectionStatus);
                    return {
                        connected: oResult.connected,
                        host: oResult.host,
                        schema: oResult.schema,
                        message: oResult.message
                    };
                })
                .catch(function (oError) {
                    that._connectionStatus = "error";
                    return {
                        connected: false,
                        message: oError.message
                    };
                });
        },

        /**
         * Get connection status
         * @returns {string} Connection status
         */
        getConnectionStatus: function () {
            return this._connectionStatus;
        },

        // ========================================================================
        // Journal Entries (ACDOCA)
        // ========================================================================

        /**
         * Get journal entries from ACDOCA
         * @param {object} oParams - Query parameters
         * @returns {Promise<array>} Journal entries
         */
        getJournalEntries: function (oParams) {
            return this._request("GET", "/journal-entries", {
                company_code: oParams.companyCode || oParams.company_code,
                fiscal_year: oParams.fiscalYear || oParams.fiscal_year,
                period: oParams.period,
                account: oParams.account,
                limit: oParams.limit || 1000
            });
        },

        /**
         * Get journal entry details
         * @param {string} sDocNumber - Document number
         * @param {string} sFiscalYear - Fiscal year
         * @param {string} sCompanyCode - Company code
         * @returns {Promise<object>} Journal entry details
         */
        getJournalEntryDetails: function (sDocNumber, sFiscalYear, sCompanyCode) {
            return this._request("GET", "/journal-entries/" + sDocNumber, {
                fiscal_year: sFiscalYear,
                company_code: sCompanyCode
            });
        },

        // ========================================================================
        // Trial Balance
        // ========================================================================

        /**
         * Get trial balance summary from ACDOCA
         * @param {object} oParams - Query parameters
         * @returns {Promise<object>} Trial balance data
         */
        getTrialBalance: function (oParams) {
            return this._request("GET", "/trial-balance", {
                company_code: oParams.companyCode || oParams.company_code,
                fiscal_year: oParams.fiscalYear || oParams.fiscal_year,
                period: oParams.period,
                ledger: oParams.ledger || "0L"
            });
        },

        /**
         * Get trial balance by account
         * @param {string} sAccount - Account number
         * @param {object} oParams - Query parameters
         * @returns {Promise<object>} Account trial balance
         */
        getAccountBalance: function (sAccount, oParams) {
            return this._request("GET", "/trial-balance/account/" + sAccount, {
                company_code: oParams.companyCode || oParams.company_code,
                fiscal_year: oParams.fiscalYear || oParams.fiscal_year,
                period: oParams.period
            });
        },

        // ========================================================================
        // Account Master (SKA1/SKAT)
        // ========================================================================

        /**
         * Get account master data
         * @param {string} sChartOfAccounts - Chart of accounts
         * @returns {Promise<array>} Account master records
         */
        getAccountMaster: function (sChartOfAccounts) {
            return this._request("GET", "/accounts", {
                chart_of_accounts: sChartOfAccounts || "YCOA"
            });
        },

        /**
         * Get account details
         * @param {string} sAccount - Account number
         * @param {string} sChartOfAccounts - Chart of accounts
         * @returns {Promise<object>} Account details
         */
        getAccountDetails: function (sAccount, sChartOfAccounts) {
            return this._request("GET", "/accounts/" + sAccount, {
                chart_of_accounts: sChartOfAccounts || "YCOA"
            });
        },

        /**
         * Search accounts by description
         * @param {string} sQuery - Search query
         * @param {string} sChartOfAccounts - Chart of accounts
         * @returns {Promise<array>} Matching accounts
         */
        searchAccounts: function (sQuery, sChartOfAccounts) {
            return this._request("GET", "/accounts/search", {
                query: sQuery,
                chart_of_accounts: sChartOfAccounts || "YCOA",
                limit: 50
            });
        },

        // ========================================================================
        // Exchange Rates (TCURR)
        // ========================================================================

        /**
         * Get exchange rates
         * @param {object} oParams - Query parameters
         * @returns {Promise<array>} Exchange rates
         */
        getExchangeRates: function (oParams) {
            return this._request("GET", "/exchange-rates", {
                from_currency: oParams.fromCurrency || oParams.from_currency,
                to_currency: oParams.toCurrency || oParams.to_currency,
                date: oParams.date,
                rate_type: oParams.rateType || oParams.rate_type || "M"
            });
        },

        /**
         * Get latest exchange rate
         * @param {string} sFromCurrency - From currency
         * @param {string} sToCurrency - To currency
         * @returns {Promise<object>} Exchange rate
         */
        getLatestExchangeRate: function (sFromCurrency, sToCurrency) {
            return this._request("GET", "/exchange-rates/latest", {
                from_currency: sFromCurrency,
                to_currency: sToCurrency
            });
        },

        /**
         * Convert amount using HANA exchange rates
         * @param {number} fAmount - Amount to convert
         * @param {string} sFromCurrency - From currency
         * @param {string} sToCurrency - To currency
         * @param {string} sDate - Date for rate lookup
         * @returns {Promise<object>} Converted amount
         */
        convertAmount: function (fAmount, sFromCurrency, sToCurrency, sDate) {
            return this._request("POST", "/exchange-rates/convert", {
                amount: fAmount,
                from_currency: sFromCurrency,
                to_currency: sToCurrency,
                date: sDate
            });
        },

        // ========================================================================
        // Company Data (T001)
        // ========================================================================

        /**
         * Get company codes
         * @returns {Promise<array>} Company codes
         */
        getCompanyCodes: function () {
            return this._request("GET", "/companies");
        },

        /**
         * Get company details
         * @param {string} sCompanyCode - Company code
         * @returns {Promise<object>} Company details
         */
        getCompanyDetails: function (sCompanyCode) {
            return this._request("GET", "/companies/" + sCompanyCode);
        },

        // ========================================================================
        // Custom SQL Query (for advanced use)
        // ========================================================================

        /**
         * Execute custom SQL query
         * Note: Should be protected by backend authorization
         * @param {string} sSql - SQL query
         * @returns {Promise<object>} Query result
         */
        executeQuery: function (sSql) {
            return this._request("POST", "/query", {
                sql: sSql
            });
        },

        // ========================================================================
        // Bulk Operations
        // ========================================================================

        /**
         * Load all data for trial balance view
         * @param {object} oParams - Query parameters
         * @returns {Promise<object>} Combined data
         */
        loadTrialBalanceData: function (oParams) {
            var that = this;
            
            return Promise.all([
                this.getTrialBalance(oParams),
                this.getAccountMaster(oParams.chartOfAccounts),
                this.getCompanyDetails(oParams.companyCode)
            ]).then(function (aResults) {
                return {
                    trialBalance: aResults[0],
                    accounts: aResults[1],
                    company: aResults[2],
                    loadedAt: new Date()
                };
            });
        },

        /**
         * Load data for variance analysis
         * @param {object} oParams - Query parameters
         * @returns {Promise<object>} Variance data
         */
        loadVarianceData: function (oParams) {
            var that = this;
            var sPreviousPeriod = this._getPreviousPeriod(oParams.period);
            
            return Promise.all([
                this.getTrialBalance({
                    companyCode: oParams.companyCode,
                    fiscalYear: oParams.fiscalYear,
                    period: oParams.period
                }),
                this.getTrialBalance({
                    companyCode: oParams.companyCode,
                    fiscalYear: oParams.fiscalYear,
                    period: sPreviousPeriod
                })
            ]).then(function (aResults) {
                return {
                    currentPeriod: aResults[0],
                    previousPeriod: aResults[1],
                    loadedAt: new Date()
                };
            });
        },

        // ========================================================================
        // Utility Methods
        // ========================================================================

        /**
         * Get previous period
         * @private
         */
        _getPreviousPeriod: function (sPeriod) {
            var iPeriod = parseInt(sPeriod, 10);
            if (iPeriod <= 1) {
                return "012"; // December of previous year
            }
            return String(iPeriod - 1).padStart(3, "0");
        },

        /**
         * Make HTTP request to backend HANA API
         * @private
         */
        _request: function (sMethod, sPath, oParams) {
            var sUrl = HanaConfig.baseApiUrl + sPath;
            
            return new Promise(function (resolve, reject) {
                var oXhr = new XMLHttpRequest();
                
                if (sMethod === "GET" && oParams) {
                    var aParams = Object.keys(oParams)
                        .filter(function (key) { return oParams[key] !== undefined && oParams[key] !== null; })
                        .map(function (key) {
                            return encodeURIComponent(key) + "=" + encodeURIComponent(oParams[key]);
                        });
                    if (aParams.length > 0) {
                        sUrl += "?" + aParams.join("&");
                    }
                }
                
                oXhr.open(sMethod, sUrl, true);
                oXhr.setRequestHeader("Content-Type", "application/json");
                oXhr.setRequestHeader("Accept", "application/json");
                
                oXhr.onload = function () {
                    if (oXhr.status >= 200 && oXhr.status < 300) {
                        try {
                            var oResponse = JSON.parse(oXhr.responseText);
                            Log.info("[HanaService] " + sMethod + " " + sPath + " succeeded");
                            resolve(oResponse);
                        } catch (e) {
                            Log.error("[HanaService] Invalid JSON response", e);
                            reject(new Error("Invalid response format"));
                        }
                    } else {
                        Log.error("[HanaService] Request failed: " + oXhr.status);
                        var sMessage = "Request failed: " + oXhr.status;
                        try {
                            var oError = JSON.parse(oXhr.responseText);
                            sMessage = oError.error || oError.message || sMessage;
                        } catch (e) {
                            // Use default message
                        }
                        reject(new Error(sMessage));
                    }
                };
                
                oXhr.onerror = function () {
                    Log.error("[HanaService] Network error for " + sUrl);
                    reject(new Error("Network error - HANA service unavailable"));
                };
                
                if (sMethod === "POST" && oParams) {
                    oXhr.send(JSON.stringify(oParams));
                } else {
                    oXhr.send();
                }
            });
        }

    });

    // Expose constants
    HanaService.SAPTables = SAPTables;
    HanaService.HanaConfig = HanaConfig;

    // Singleton instance
    var _instance = null;
    HanaService.getInstance = function (oConfig) {
        if (!_instance) {
            _instance = new HanaService(oConfig);
        }
        return _instance;
    };

    return HanaService;
});