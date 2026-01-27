/**
 * ============================================================================
 * SAP AI Core Client for Anthropic Claude 4.5 Opus
 * Provides AI-powered analysis and insights for Trial Balance
 * ============================================================================
 *
 * [CODE:file=AICoreClient.js]
 * [CODE:module=service]
 * [CODE:language=javascript]
 *
 * [CONFIG:requires=AICORE_CLIENT_ID]
 * [CONFIG:requires=AICORE_CLIENT_SECRET]
 * [CONFIG:requires=AICORE_AUTH_URL]
 * [CONFIG:requires=AICORE_BASE_URL]
 *
 * [API:provider=SAP AI Core]
 * [API:model=anthropic--claude-4-5-opus]
 *
 * [RELATION:uses=CODE:ApiService.js]
 *
 * This client provides AI-powered features using Anthropic Claude 4.5 Opus
 * via SAP AI Core for variance analysis, commentary suggestions, and
 * intelligent data quality insights.
 */
sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log"
], function (BaseObject, Log) {
    "use strict";

    /**
     * AI Core Configuration
     * Values loaded from environment or config
     */
    var AICoreConfig = {
        // Default values - should be overridden by environment
        clientId: null,
        clientSecret: null,
        authUrl: null,
        baseUrl: null,
        resourceGroup: "default",
        
        // Model configuration
        modelName: "anthropic--claude-4-5-opus",
        modelVersion: "latest",
        
        // Request defaults
        maxTokens: 4096,
        temperature: 0.3,
        topP: 0.9
    };

    /**
     * Token cache for OAuth
     */
    var TokenCache = {
        accessToken: null,
        expiresAt: null,
        
        isValid: function () {
            return this.accessToken && this.expiresAt && Date.now() < this.expiresAt;
        },
        
        set: function (token, expiresIn) {
            this.accessToken = token;
            // Expire 5 minutes early to be safe
            this.expiresAt = Date.now() + (expiresIn - 300) * 1000;
        },
        
        clear: function () {
            this.accessToken = null;
            this.expiresAt = null;
        }
    };

    var AICoreClient = BaseObject.extend("trialbalance.service.AICoreClient", {
        
        /**
         * Constructor
         * @param {object} oConfig - Configuration override
         */
        constructor: function (oConfig) {
            BaseObject.call(this);
            
            // Merge configuration
            if (oConfig) {
                Object.assign(AICoreConfig, oConfig);
            }
            
            Log.info("[AICoreClient] Initialized with model: " + AICoreConfig.modelName);
        },

        // ========================================================================
        // Configuration
        // ========================================================================

        /**
         * Configure the client with credentials
         * @param {object} oCredentials - AI Core credentials
         */
        configure: function (oCredentials) {
            AICoreConfig.clientId = oCredentials.clientId;
            AICoreConfig.clientSecret = oCredentials.clientSecret;
            AICoreConfig.authUrl = oCredentials.authUrl;
            AICoreConfig.baseUrl = oCredentials.baseUrl;
            AICoreConfig.resourceGroup = oCredentials.resourceGroup || "default";
            
            Log.info("[AICoreClient] Configuration updated");
        },

        /**
         * Load configuration from backend API
         * @returns {Promise<boolean>} Success status
         */
        loadConfigFromBackend: function () {
            var that = this;
            return new Promise(function (resolve, reject) {
                var oXhr = new XMLHttpRequest();
                oXhr.open("GET", "/api/v1/config/aicore", true);
                oXhr.setRequestHeader("Accept", "application/json");
                
                oXhr.onload = function () {
                    if (oXhr.status === 200) {
                        try {
                            var oConfig = JSON.parse(oXhr.responseText);
                            that.configure({
                                clientId: oConfig.client_id,
                                clientSecret: oConfig.client_secret,
                                authUrl: oConfig.auth_url,
                                baseUrl: oConfig.base_url,
                                resourceGroup: oConfig.resource_group
                            });
                            resolve(true);
                        } catch (e) {
                            reject(new Error("Invalid config response"));
                        }
                    } else {
                        reject(new Error("Failed to load config: " + oXhr.status));
                    }
                };
                
                oXhr.onerror = function () {
                    reject(new Error("Network error loading config"));
                };
                
                oXhr.send();
            });
        },

        // ========================================================================
        // Authentication
        // ========================================================================

        /**
         * Get OAuth access token
         * @returns {Promise<string>} Access token
         */
        _getAccessToken: function () {
            var that = this;
            
            // Return cached token if valid
            if (TokenCache.isValid()) {
                return Promise.resolve(TokenCache.accessToken);
            }
            
            return new Promise(function (resolve, reject) {
                if (!AICoreConfig.clientId || !AICoreConfig.authUrl) {
                    reject(new Error("AI Core not configured. Call configure() or loadConfigFromBackend() first."));
                    return;
                }
                
                var oXhr = new XMLHttpRequest();
                oXhr.open("POST", AICoreConfig.authUrl, true);
                oXhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
                
                // Basic auth header
                var sAuth = btoa(AICoreConfig.clientId + ":" + AICoreConfig.clientSecret);
                oXhr.setRequestHeader("Authorization", "Basic " + sAuth);
                
                oXhr.onload = function () {
                    if (oXhr.status === 200) {
                        try {
                            var oResponse = JSON.parse(oXhr.responseText);
                            TokenCache.set(oResponse.access_token, oResponse.expires_in || 3600);
                            Log.info("[AICoreClient] Token obtained, expires in: " + oResponse.expires_in + "s");
                            resolve(TokenCache.accessToken);
                        } catch (e) {
                            reject(new Error("Invalid token response"));
                        }
                    } else {
                        Log.error("[AICoreClient] Token request failed: " + oXhr.status);
                        reject(new Error("Authentication failed: " + oXhr.status));
                    }
                };
                
                oXhr.onerror = function () {
                    reject(new Error("Network error during authentication"));
                };
                
                oXhr.send("grant_type=client_credentials");
            });
        },

        // ========================================================================
        // Core AI Methods
        // ========================================================================

        /**
         * Send a completion request to Claude 4.5 Opus
         * @param {string} sPrompt - The prompt text
         * @param {object} oOptions - Request options
         * @returns {Promise<string>} AI response
         */
        complete: function (sPrompt, oOptions) {
            var that = this;
            oOptions = oOptions || {};
            
            return this._getAccessToken().then(function (sToken) {
                return that._sendCompletionRequest(sToken, sPrompt, oOptions);
            });
        },

        /**
         * Send completion request with token
         * @private
         */
        _sendCompletionRequest: function (sToken, sPrompt, oOptions) {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                var oXhr = new XMLHttpRequest();
                var sUrl = AICoreConfig.baseUrl + "/v2/inference/deployments/" + 
                           AICoreConfig.modelName + "/chat/completions";
                
                oXhr.open("POST", sUrl, true);
                oXhr.setRequestHeader("Content-Type", "application/json");
                oXhr.setRequestHeader("Authorization", "Bearer " + sToken);
                oXhr.setRequestHeader("AI-Resource-Group", AICoreConfig.resourceGroup);
                
                var oRequestBody = {
                    model: AICoreConfig.modelName,
                    messages: [
                        {
                            role: "user",
                            content: sPrompt
                        }
                    ],
                    max_tokens: oOptions.maxTokens || AICoreConfig.maxTokens,
                    temperature: oOptions.temperature !== undefined ? oOptions.temperature : AICoreConfig.temperature,
                    top_p: oOptions.topP || AICoreConfig.topP
                };
                
                // Add system message if provided
                if (oOptions.systemPrompt) {
                    oRequestBody.messages.unshift({
                        role: "system",
                        content: oOptions.systemPrompt
                    });
                }
                
                oXhr.onload = function () {
                    if (oXhr.status === 200) {
                        try {
                            var oResponse = JSON.parse(oXhr.responseText);
                            var sContent = oResponse.choices[0].message.content;
                            Log.info("[AICoreClient] Completion received, tokens: " + 
                                    (oResponse.usage ? oResponse.usage.total_tokens : "unknown"));
                            resolve(sContent);
                        } catch (e) {
                            reject(new Error("Invalid completion response"));
                        }
                    } else if (oXhr.status === 401) {
                        // Token expired, clear cache and retry
                        TokenCache.clear();
                        reject(new Error("Token expired, please retry"));
                    } else {
                        Log.error("[AICoreClient] Completion failed: " + oXhr.status + " " + oXhr.responseText);
                        reject(new Error("Completion failed: " + oXhr.status));
                    }
                };
                
                oXhr.onerror = function () {
                    reject(new Error("Network error during completion"));
                };
                
                oXhr.send(JSON.stringify(oRequestBody));
            });
        },

        // ========================================================================
        // Trial Balance Specific Methods
        // ========================================================================

        /**
         * Generate variance commentary suggestion
         * @param {object} oVariance - Variance data
         * @returns {Promise<string>} Suggested commentary
         */
        suggestVarianceCommentary: function (oVariance) {
            var sPrompt = this._buildVariancePrompt(oVariance);
            
            return this.complete(sPrompt, {
                systemPrompt: "You are a financial analyst expert in trial balance analysis. " +
                             "Provide concise, professional variance commentary suitable for management reporting. " +
                             "Focus on identifying root causes and business impact.",
                maxTokens: 500,
                temperature: 0.3
            });
        },

        /**
         * Build variance analysis prompt
         * @private
         */
        _buildVariancePrompt: function (oVariance) {
            return "Analyze this variance and provide a brief commentary:\n\n" +
                   "Account: " + (oVariance.accountId || oVariance.account_id) + "\n" +
                   "Account Type: " + (oVariance.accountType || oVariance.account_type || "Unknown") + "\n" +
                   "Current Balance: $" + this._formatNumber(oVariance.currentBalance || oVariance.current_balance) + "\n" +
                   "Previous Balance: $" + this._formatNumber(oVariance.previousBalance || oVariance.previous_balance) + "\n" +
                   "Variance Amount: $" + this._formatNumber(oVariance.varianceAmount || oVariance.variance_amount) + "\n" +
                   "Variance %: " + this._formatPercent(oVariance.variancePercent || oVariance.variance_percent) + "\n" +
                   (oVariance.majorDriver ? "Suggested Driver: " + oVariance.majorDriver + "\n" : "") +
                   "\nProvide a 2-3 sentence commentary explaining this variance.";
        },

        /**
         * Identify major driver for variance
         * @param {object} oVariance - Variance data
         * @returns {Promise<object>} Driver analysis
         */
        identifyVarianceDriver: function (oVariance) {
            var sPrompt = "Analyze this variance and identify the most likely driver:\n\n" +
                         "Account: " + (oVariance.accountId || oVariance.account_id) + "\n" +
                         "Account Type: " + (oVariance.accountType || oVariance.account_type) + "\n" +
                         "Variance Amount: $" + this._formatNumber(oVariance.varianceAmount || oVariance.variance_amount) + "\n" +
                         "Variance %: " + this._formatPercent(oVariance.variancePercent || oVariance.variance_percent) + "\n\n" +
                         "Classify the driver as one of: VOLUME, PRICE, MIX, FX, ONE_TIME, TIMING, " +
                         "ACQUISITION_DISPOSAL, POLICY_CHANGE, OTHER\n\n" +
                         "Respond in JSON format: {\"driver\": \"CATEGORY\", \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}";
            
            return this.complete(sPrompt, {
                systemPrompt: "You are a financial analyst. Classify variance drivers accurately. " +
                             "Respond only with valid JSON.",
                maxTokens: 200,
                temperature: 0.2
            }).then(function (sResponse) {
                try {
                    // Extract JSON from response
                    var sJson = sResponse.match(/\{[\s\S]*\}/);
                    if (sJson) {
                        return JSON.parse(sJson[0]);
                    }
                    return { driver: "OTHER", confidence: 0.5, reasoning: sResponse };
                } catch (e) {
                    return { driver: "OTHER", confidence: 0.5, reasoning: sResponse };
                }
            });
        },

        /**
         * Analyze data quality issues
         * @param {array} aIssues - List of validation failures
         * @returns {Promise<string>} Analysis and recommendations
         */
        analyzeDataQualityIssues: function (aIssues) {
            var sIssueList = aIssues.map(function (issue, i) {
                return (i + 1) + ". " + issue.ruleId + ": " + issue.message;
            }).join("\n");
            
            var sPrompt = "Analyze these data quality issues from a trial balance validation:\n\n" +
                         sIssueList + "\n\n" +
                         "Provide:\n" +
                         "1. Root cause analysis\n" +
                         "2. Prioritized remediation steps\n" +
                         "3. Preventive measures";
            
            return this.complete(sPrompt, {
                systemPrompt: "You are a data quality expert specializing in financial data. " +
                             "Provide actionable recommendations.",
                maxTokens: 1000,
                temperature: 0.3
            });
        },

        /**
         * Generate trial balance summary
         * @param {object} oTrialBalance - Trial balance data
         * @returns {Promise<string>} Executive summary
         */
        generateTrialBalanceSummary: function (oTrialBalance) {
            var sPrompt = "Generate an executive summary for this trial balance:\n\n" +
                         "Company: " + (oTrialBalance.companyCode || "Unknown") + "\n" +
                         "Period: " + (oTrialBalance.period || "Unknown") + "\n" +
                         "Total Debits: $" + this._formatNumber(oTrialBalance.totalDebits) + "\n" +
                         "Total Credits: $" + this._formatNumber(oTrialBalance.totalCredits) + "\n" +
                         "Balance Difference: $" + this._formatNumber(oTrialBalance.balanceDifference) + "\n" +
                         "Is Balanced: " + (oTrialBalance.isBalanced ? "Yes" : "No") + "\n" +
                         "Account Count: " + (oTrialBalance.accountCount || 0) + "\n" +
                         "Material Variances: " + (oTrialBalance.materialVarianceCount || 0) + "\n" +
                         "Commentary Coverage: " + this._formatPercent(oTrialBalance.commentaryCoverage) + "\n\n" +
                         "Provide a 3-4 sentence executive summary highlighting key points and any concerns.";
            
            return this.complete(sPrompt, {
                systemPrompt: "You are a senior financial controller. Write clear, executive-level summaries.",
                maxTokens: 300,
                temperature: 0.4
            });
        },

        // ========================================================================
        // Utility Methods
        // ========================================================================

        /**
         * Format number for display
         * @private
         */
        _formatNumber: function (value) {
            if (value === null || value === undefined) return "0";
            return new Intl.NumberFormat('en-US', {
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }).format(value);
        },

        /**
         * Format percentage for display
         * @private
         */
        _formatPercent: function (value) {
            if (value === null || value === undefined) return "0%";
            return value.toFixed(1) + "%";
        },

        /**
         * Check if client is configured
         * @returns {boolean} Configuration status
         */
        isConfigured: function () {
            return !!(AICoreConfig.clientId && AICoreConfig.authUrl && AICoreConfig.baseUrl);
        },

        /**
         * Get current model name
         * @returns {string} Model identifier
         */
        getModelName: function () {
            return AICoreConfig.modelName;
        },

        /**
         * Clear token cache (for logout or refresh)
         */
        clearTokenCache: function () {
            TokenCache.clear();
        }

    });

    // Static method to get singleton instance
    var _instance = null;
    AICoreClient.getInstance = function (oConfig) {
        if (!_instance) {
            _instance = new AICoreClient(oConfig);
        }
        return _instance;
    };

    return AICoreClient;
});