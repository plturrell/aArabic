sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/format/DateFormat"
], function (Controller, MessageBox, MessageToast, DateFormat) {
    "use strict";

    return Controller.extend("hypershimmy.controller.Summary", {
        
        /**
         * Summary type descriptions
         * @private
         */
        _summaryTypeDescriptions: {
            brief: "Concise 1-2 paragraph overview (100-150 words). High-level summary of main points.",
            detailed: "Comprehensive 3-5 paragraph analysis (300-500 words). In-depth coverage of key topics.",
            executive: "Structured summary with Overview, Key Findings, and Recommendations (250-300 words). Ideal for decision-makers.",
            bullet_points: "5-8 key takeaways in bullet format with citations. Quick reference points.",
            comparative: "Compare and contrast analysis across sources (300-400 words). Highlights agreements and differences."
        },

        /**
         * Called when the controller is instantiated
         */
        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("summary").attachPatternMatched(this._onRouteMatched, this);
            
            // Initialize summary settings
            this._initializeSummarySettings();
            
            // Load saved settings
            this._loadSummarySettings();
        },

        /**
         * Initialize default summary settings
         * @private
         */
        _initializeSummarySettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Set defaults if not already set
            if (!oAppStateModel.getProperty("/summaryType")) {
                oAppStateModel.setProperty("/summaryType", "executive");
            }
            if (!oAppStateModel.getProperty("/summaryMaxLength")) {
                oAppStateModel.setProperty("/summaryMaxLength", 300);
            }
            if (!oAppStateModel.getProperty("/summaryTone")) {
                oAppStateModel.setProperty("/summaryTone", "professional");
            }
            if (oAppStateModel.getProperty("/summaryIncludeCitations") === undefined) {
                oAppStateModel.setProperty("/summaryIncludeCitations", true);
            }
            if (oAppStateModel.getProperty("/summaryIncludeKeyPoints") === undefined) {
                oAppStateModel.setProperty("/summaryIncludeKeyPoints", true);
            }
            if (oAppStateModel.getProperty("/summaryConfigExpanded") === undefined) {
                oAppStateModel.setProperty("/summaryConfigExpanded", true);
            }
            
            oAppStateModel.setProperty("/summaryFocusAreas", "");
            oAppStateModel.setProperty("/summaryGenerated", false);
            
            // Set initial description
            this._updateSummaryTypeDescription();
        },

        /**
         * Load summary settings from localStorage
         * @private
         */
        _loadSummarySettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            try {
                var sSettings = localStorage.getItem("hypershimmy.summarySettings");
                if (sSettings) {
                    var oSettings = JSON.parse(sSettings);
                    oAppStateModel.setProperty("/summaryType", oSettings.type || "executive");
                    oAppStateModel.setProperty("/summaryMaxLength", oSettings.maxLength || 300);
                    oAppStateModel.setProperty("/summaryTone", oSettings.tone || "professional");
                    oAppStateModel.setProperty("/summaryIncludeCitations", oSettings.includeCitations !== false);
                    oAppStateModel.setProperty("/summaryIncludeKeyPoints", oSettings.includeKeyPoints !== false);
                    
                    this._updateSummaryTypeDescription();
                }
            } catch (e) {
                console.error("Failed to load summary settings:", e);
            }
        },

        /**
         * Save summary settings to localStorage
         * @private
         */
        _saveSummarySettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            var oSettings = {
                type: oAppStateModel.getProperty("/summaryType") || "executive",
                maxLength: oAppStateModel.getProperty("/summaryMaxLength") || 300,
                tone: oAppStateModel.getProperty("/summaryTone") || "professional",
                includeCitations: oAppStateModel.getProperty("/summaryIncludeCitations") !== false,
                includeKeyPoints: oAppStateModel.getProperty("/summaryIncludeKeyPoints") !== false
            };
            
            try {
                localStorage.setItem("hypershimmy.summarySettings", JSON.stringify(oSettings));
            } catch (e) {
                console.error("Failed to save summary settings:", e);
            }
        },

        /**
         * Route matched handler
         * @param {sap.ui.base.Event} oEvent the route matched event
         * @private
         */
        _onRouteMatched: function (oEvent) {
            var sSourceId = oEvent.getParameter("arguments").sourceId;
            
            // Store current source ID
            this._currentSourceId = sSourceId;
            
            // Update app state
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            oAppStateModel.setProperty("/selectedSourceId", sSourceId);
            
            // Bind the view to the selected source
            var oView = this.getView();
            oView.bindElement({
                path: "/Sources('" + sSourceId + "')",
                parameters: {
                    $expand: "Summaries"
                }
            });
        },

        /**
         * Handler for summary type change
         * @param {sap.ui.base.Event} oEvent the change event
         */
        onSummaryTypeChange: function (oEvent) {
            this._updateSummaryTypeDescription();
            this._saveSummarySettings();
        },

        /**
         * Update summary type description
         * @private
         */
        _updateSummaryTypeDescription: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var sSummaryType = oAppStateModel.getProperty("/summaryType");
            var sDescription = this._summaryTypeDescriptions[sSummaryType] || "";
            
            oAppStateModel.setProperty("/summaryTypeDescription", sDescription);
        },

        /**
         * Handler for generate summary button
         */
        onGenerateSummary: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Get source IDs (for now, just use current source)
            var aSourceIds = [this._currentSourceId];
            
            // Get configuration
            var sSummaryType = oAppStateModel.getProperty("/summaryType");
            var iMaxLength = oAppStateModel.getProperty("/summaryMaxLength");
            var sTone = oAppStateModel.getProperty("/summaryTone");
            var bIncludeCitations = oAppStateModel.getProperty("/summaryIncludeCitations");
            var bIncludeKeyPoints = oAppStateModel.getProperty("/summaryIncludeKeyPoints");
            var sFocusAreas = oAppStateModel.getProperty("/summaryFocusAreas");
            
            // Parse focus areas
            var aFocusAreas = [];
            if (sFocusAreas && sFocusAreas.trim().length > 0) {
                aFocusAreas = sFocusAreas.split(",").map(function(s) {
                    return s.trim();
                }).filter(function(s) {
                    return s.length > 0;
                });
            }
            
            // Set busy state
            oAppStateModel.setProperty("/busy", true);
            oAppStateModel.setProperty("/summaryGenerated", false);
            oAppStateModel.setProperty("/summaryConfigExpanded", false);
            
            // Call OData Summary action
            this._callSummaryAction(
                aSourceIds,
                sSummaryType,
                iMaxLength,
                bIncludeCitations,
                bIncludeKeyPoints,
                sTone,
                aFocusAreas
            )
                .then(function(oResponse) {
                    // Process and display summary
                    this._displaySummary(oResponse);
                    
                    oAppStateModel.setProperty("/busy", false);
                    oAppStateModel.setProperty("/summaryGenerated", true);
                    
                    // Save settings
                    this._saveSummarySettings();
                    
                    MessageToast.show("Summary generated successfully");
                }.bind(this))
                .catch(function(oError) {
                    // Handle error
                    oAppStateModel.setProperty("/busy", false);
                    
                    var sErrorMessage = "Failed to generate summary. Please try again.";
                    if (oError.responseText) {
                        try {
                            var oErrorData = JSON.parse(oError.responseText);
                            if (oErrorData.error && oErrorData.error.message) {
                                sErrorMessage = oErrorData.error.message;
                            }
                        } catch (e) {
                            // Ignore JSON parse error
                        }
                    }
                    
                    MessageBox.error(sErrorMessage);
                }.bind(this));
        },

        /**
         * Call OData GenerateSummary action
         * @param {array} aSourceIds array of source IDs
         * @param {string} sSummaryType summary type
         * @param {number} iMaxLength max length in words
         * @param {boolean} bIncludeCitations include citations flag
         * @param {boolean} bIncludeKeyPoints include key points flag
         * @param {string} sTone tone of summary
         * @param {array} aFocusAreas focus areas
         * @returns {Promise} promise that resolves with summary response
         * @private
         */
        _callSummaryAction: function(
            aSourceIds,
            sSummaryType,
            iMaxLength,
            bIncludeCitations,
            bIncludeKeyPoints,
            sTone,
            aFocusAreas
        ) {
            return new Promise(function(resolve, reject) {
                // Prepare request payload
                var oPayload = {
                    SourceIds: aSourceIds,
                    SummaryType: sSummaryType,
                    MaxLength: iMaxLength,
                    IncludeCitations: bIncludeCitations,
                    IncludeKeyPoints: bIncludeKeyPoints,
                    Tone: sTone
                };
                
                // Add focus areas if provided
                if (aFocusAreas && aFocusAreas.length > 0) {
                    oPayload.FocusAreas = aFocusAreas;
                }
                
                // Call OData action
                jQuery.ajax({
                    url: "/odata/v4/research/GenerateSummary",
                    method: "POST",
                    contentType: "application/json",
                    data: JSON.stringify(oPayload),
                    success: function(oData) {
                        resolve(oData);
                    },
                    error: function(oError) {
                        reject(oError);
                    }
                });
            });
        },

        /**
         * Display summary in the UI
         * @param {object} oSummary the summary response
         * @private
         */
        _displaySummary: function (oSummary) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Store current summary
            oAppStateModel.setProperty("/currentSummary", oSummary);
            
            // Format summary text for display
            var sFormattedText = this._formatSummaryText(oSummary.SummaryText);
            oAppStateModel.setProperty("/formattedSummaryText", sFormattedText);
            
            // Set generated time
            var oDateFormat = DateFormat.getDateTimeInstance({
                pattern: "MMM dd, yyyy HH:mm:ss"
            });
            oAppStateModel.setProperty("/summaryGeneratedTime", oDateFormat.format(new Date()));
        },

        /**
         * Format summary text with proper HTML formatting
         * @param {string} sText the raw summary text
         * @returns {string} formatted HTML text
         * @private
         */
        _formatSummaryText: function (sText) {
            if (!sText) {
                return "";
            }
            
            // Escape HTML
            var sFormatted = sText
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;");
            
            // Convert markdown-style formatting
            sFormatted = sFormatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
            sFormatted = sFormatted.replace(/\*(.*?)\*/g, "<em>$1</em>");
            
            // Convert headings (markdown-style)
            sFormatted = sFormatted.replace(/^### (.*?)$/gm, "<h3>$1</h3>");
            sFormatted = sFormatted.replace(/^## (.*?)$/gm, "<h2>$1</h2>");
            sFormatted = sFormatted.replace(/^# (.*?)$/gm, "<h1>$1</h1>");
            
            // Convert bullet points
            sFormatted = sFormatted.replace(/^• (.*?)$/gm, "<li>$1</li>");
            sFormatted = sFormatted.replace(/^- (.*?)$/gm, "<li>$1</li>");
            
            // Wrap consecutive list items in ul tags
            sFormatted = sFormatted.replace(/(<li>.*?<\/li>\n?)+/g, function(match) {
                return "<ul>" + match + "</ul>";
            });
            
            // Convert line breaks (paragraphs)
            sFormatted = sFormatted.replace(/\n\n/g, "</p><p>");
            sFormatted = "<p>" + sFormatted + "</p>";
            
            // Convert single line breaks
            sFormatted = sFormatted.replace(/\n/g, "<br>");
            
            return sFormatted;
        },

        /**
         * Handler for export summary button
         */
        onExportSummary: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oSummary = oAppStateModel.getProperty("/currentSummary");
            
            if (!oSummary) {
                MessageToast.show("No summary to export");
                return;
            }
            
            // Create export data
            var sExportData = this._formatSummaryForExport(oSummary);
            
            // Create download link
            var oBlob = new Blob([sExportData], { type: "text/plain;charset=utf-8" });
            var sUrl = URL.createObjectURL(oBlob);
            var sFilename = "summary-" + oSummary.SummaryType + "-" + new Date().toISOString().split('T')[0] + ".txt";
            
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = sFilename;
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            URL.revokeObjectURL(sUrl);
            
            MessageToast.show("Summary exported successfully");
        },

        /**
         * Format summary for export
         * @param {object} oSummary the summary object
         * @returns {string} formatted export text
         * @private
         */
        _formatSummaryForExport: function (oSummary) {
            var aLines = [
                "HyperShimmy Research Summary",
                "=" .repeat(70),
                "",
                "Summary ID: " + oSummary.SummaryId,
                "Type: " + oSummary.SummaryType,
                "Word Count: " + oSummary.WordCount,
                "Confidence: " + (oSummary.Confidence * 100).toFixed(0) + "%",
                "Processing Time: " + oSummary.ProcessingTimeMs + "ms",
                "Generated: " + new Date().toLocaleString(),
                "",
                "=" .repeat(70),
                "",
                "SUMMARY",
                "-" .repeat(70),
                "",
                oSummary.SummaryText,
                ""
            ];
            
            // Add key points if available
            if (oSummary.KeyPoints && oSummary.KeyPoints.length > 0) {
                aLines.push("");
                aLines.push("KEY POINTS");
                aLines.push("-" .repeat(70));
                aLines.push("");
                
                oSummary.KeyPoints.forEach(function(oKeyPoint, idx) {
                    aLines.push((idx + 1) + ". " + oKeyPoint.Content);
                    aLines.push("   Category: " + oKeyPoint.Category + " | Importance: " + (oKeyPoint.Importance * 100).toFixed(0) + "%");
                    aLines.push("");
                });
            }
            
            // Add sources if available
            if (oSummary.SourceIds && oSummary.SourceIds.length > 0) {
                aLines.push("");
                aLines.push("SOURCES");
                aLines.push("-" .repeat(70));
                aLines.push("");
                oSummary.SourceIds.forEach(function(sSourceId, idx) {
                    aLines.push((idx + 1) + ". " + sSourceId);
                });
                aLines.push("");
            }
            
            aLines.push("");
            aLines.push("=" .repeat(70));
            aLines.push("End of summary export");
            
            return aLines.join("\n");
        },

        /**
         * Handler for copy summary button
         */
        onCopySummary: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oSummary = oAppStateModel.getProperty("/currentSummary");
            
            if (!oSummary || !oSummary.SummaryText) {
                MessageToast.show("No summary to copy");
                return;
            }
            
            // Copy summary text to clipboard
            this._copyToClipboard(oSummary.SummaryText);
        },

        /**
         * Copy text to clipboard
         * @param {string} sText the text to copy
         * @private
         */
        _copyToClipboard: function (sText) {
            // Use Clipboard API if available
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(sText)
                    .then(function () {
                        MessageToast.show("Summary copied to clipboard");
                    })
                    .catch(function (err) {
                        console.error("Failed to copy:", err);
                        MessageToast.show("Failed to copy summary");
                    });
            } else {
                // Fallback for older browsers
                var oTextArea = document.createElement("textarea");
                oTextArea.value = sText;
                oTextArea.style.position = "fixed";
                oTextArea.style.left = "-9999px";
                document.body.appendChild(oTextArea);
                oTextArea.select();
                
                try {
                    document.execCommand("copy");
                    MessageToast.show("Summary copied to clipboard");
                } catch (err) {
                    console.error("Failed to copy:", err);
                    MessageToast.show("Failed to copy summary");
                }
                
                document.body.removeChild(oTextArea);
            }
        },

        /**
         * Handler for source press in sources list
         * @param {sap.ui.base.Event} oEvent the press event
         */
        onSourcePress: function (oEvent) {
            var oItem = oEvent.getSource();
            var sSourceId = oItem.getBindingContext("appState").getObject();
            
            // Navigate to source detail
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("detail", {
                sourceId: sSourceId
            });
        },

        /**
         * Handler for navigation back button
         */
        onNavBack: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var sSourceId = oAppStateModel.getProperty("/selectedSourceId");
            
            // Navigate back to detail view
            oRouter.navTo("detail", {
                sourceId: sSourceId
            });
        }
    });
});
