sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/format/DateFormat",
    "sap/ui/model/json/JSONModel"
], function (Controller, MessageBox, MessageToast, DateFormat, JSONModel) {
    "use strict";

    return Controller.extend("hypershimmy.controller.Chat", {
        
        /**
         * Called when the controller is instantiated
         */
        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("chat").attachPatternMatched(this._onRouteMatched, this);
            
            // Initialize session ID
            this._sessionId = "session-" + Date.now();
            
            // Load chat settings from localStorage
            this._loadChatSettings();
            
            // Load persisted chat history
            this._loadChatHistory();
            
            // Initialize chat history rendering
            this._renderChatHistory();
        },

        /**
         * Load chat settings from localStorage
         * @private
         */
        _loadChatSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            try {
                var sSettings = localStorage.getItem("hypershimmy.chatSettings");
                if (sSettings) {
                    var oSettings = JSON.parse(sSettings);
                    oAppStateModel.setProperty("/chatMaxTokens", oSettings.maxTokens || 500);
                    oAppStateModel.setProperty("/chatTemperature", oSettings.temperature || 0.7);
                    oAppStateModel.setProperty("/chatIncludeSources", oSettings.includeSources !== false);
                }
            } catch (e) {
                console.error("Failed to load chat settings:", e);
            }
        },

        /**
         * Save chat settings to localStorage
         * @private
         */
        _saveChatSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            var oSettings = {
                maxTokens: oAppStateModel.getProperty("/chatMaxTokens") || 500,
                temperature: oAppStateModel.getProperty("/chatTemperature") || 0.7,
                includeSources: oAppStateModel.getProperty("/chatIncludeSources") !== false
            };
            
            try {
                localStorage.setItem("hypershimmy.chatSettings", JSON.stringify(oSettings));
            } catch (e) {
                console.error("Failed to save chat settings:", e);
            }
        },

        /**
         * Load chat history from localStorage
         * @private
         */
        _loadChatHistory: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            try {
                var sHistory = localStorage.getItem("hypershimmy.chatHistory." + this._sessionId);
                if (sHistory) {
                    var aChatHistory = JSON.parse(sHistory);
                    oAppStateModel.setProperty("/chatHistory", aChatHistory);
                }
            } catch (e) {
                console.error("Failed to load chat history:", e);
            }
        },

        /**
         * Save chat history to localStorage
         * @private
         */
        _saveChatHistory: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var aChatHistory = oAppStateModel.getProperty("/chatHistory") || [];
            
            try {
                localStorage.setItem(
                    "hypershimmy.chatHistory." + this._sessionId,
                    JSON.stringify(aChatHistory)
                );
            } catch (e) {
                console.error("Failed to save chat history:", e);
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
                    $expand: "ChatMessages"
                }
            });
            
            // Render chat history
            this._renderChatHistory();
        },

        /**
         * Render chat history from app state
         * @private
         */
        _renderChatHistory: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var aChatHistory = oAppStateModel.getProperty("/chatHistory") || [];
            var oContainer = this.byId("chatMessages");
            
            if (!oContainer) {
                return;
            }
            
            // Clear existing messages (except welcome message)
            oContainer.destroyItems();
            
            // Add welcome message container (shown when empty)
            var oWelcomeBox = new sap.m.VBox({
                visible: "{= ${appState>/chatHistory}.length === 0 }",
                alignItems: "Center",
                justifyContent: "Center",
                class: "sapUiLargeMarginTop",
                items: [
                    new sap.m.Title({
                        text: "{i18n>chatWelcomeTitle}",
                        class: "sapUiSmallMarginBottom"
                    }),
                    new sap.m.Text({
                        text: "{i18n>chatWelcomeText}",
                        class: "sapUiTinyMarginBottom"
                    })
                ]
            });
            oContainer.addItem(oWelcomeBox);
            
            // Render each message
            aChatHistory.forEach(function (oMessage) {
                var oMessageBox = this._createMessageBox(oMessage);
                oContainer.addItem(oMessageBox);
            }.bind(this));
            
            // Scroll to bottom
            this._scrollToBottom();
        },

        /**
         * Create a message box for display
         * @param {object} oMessage the message object
         * @returns {sap.m.VBox} the message box
         * @private
         */
        _createMessageBox: function (oMessage) {
            var sClass = oMessage.role === "user" ? "chatMessageUser" : "chatMessageAssistant";
            var sIcon = oMessage.role === "user" ? "sap-icon://person-placeholder" : "sap-icon://chatbot";
            
            var aItems = [
                new sap.m.HBox({
                    alignItems: "Center",
                    class: "sapUiTinyMarginBottom",
                    items: [
                        new sap.ui.core.Icon({
                            src: sIcon,
                            size: "1rem",
                            class: "sapUiTinyMarginEnd"
                        }),
                        new sap.m.Label({
                            text: oMessage.role === "user" ? "You" : "Assistant",
                            design: "Bold"
                        }),
                        new sap.m.Label({
                            text: " • " + this._formatTimestamp(oMessage.timestamp),
                            class: "sapUiTinyMarginBegin chatMessageTimestamp"
                        })
                    ]
                }),
                new sap.m.FormattedText({
                    htmlText: this._formatMessageContent(oMessage.content),
                    class: "chatMessageContent"
                })
            ];
            
            // Add metadata if available (for assistant messages)
            if (oMessage.role === "assistant" && oMessage.metadata) {
                aItems.push(this._createMetadataDisplay(oMessage.metadata));
            }
            
            // Add sources if available
            if (oMessage.sourceIds && oMessage.sourceIds.length > 0) {
                aItems.push(this._createSourcesDisplay(oMessage.sourceIds));
            }
            
            var oMessageBox = new sap.m.VBox({
                class: sClass,
                items: aItems
            });
            
            return oMessageBox;
        },

        /**
         * Format message content with proper line breaks and formatting
         * @param {string} sContent the raw content
         * @returns {string} formatted HTML content
         * @private
         */
        _formatMessageContent: function (sContent) {
            if (!sContent) {
                return "";
            }
            
            // Escape HTML
            var sEscaped = sContent
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;");
            
            // Convert markdown-style bold
            sEscaped = sEscaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
            
            // Convert line breaks
            sEscaped = sEscaped.replace(/\n/g, "<br>");
            
            return sEscaped;
        },

        /**
         * Create metadata display panel
         * @param {object} oMetadata the metadata object
         * @returns {sap.m.VBox} the metadata display
         * @private
         */
        _createMetadataDisplay: function (oMetadata) {
            var aItems = [];
            
            // Confidence indicator
            if (oMetadata.confidence !== undefined) {
                var sState = oMetadata.confidence > 0.7 ? "Success" : 
                            oMetadata.confidence > 0.5 ? "Warning" : "Error";
                aItems.push(new sap.m.ObjectStatus({
                    text: "Confidence: " + (oMetadata.confidence * 100).toFixed(0) + "%",
                    state: sState,
                    icon: "sap-icon://measurement-document"
                }));
            }
            
            // Query intent
            if (oMetadata.query_intent) {
                aItems.push(new sap.m.ObjectStatus({
                    text: "Intent: " + oMetadata.query_intent,
                    icon: "sap-icon://hello-world"
                }));
            }
            
            // Performance info
            if (oMetadata.total_time_ms) {
                aItems.push(new sap.m.ObjectStatus({
                    text: "Response time: " + oMetadata.total_time_ms + "ms",
                    icon: "sap-icon://performance"
                }));
            }
            
            if (aItems.length === 0) {
                return null;
            }
            
            return new sap.m.VBox({
                class: "sapUiTinyMarginTop chatMetadata",
                items: [
                    new sap.m.HBox({
                        wrap: "Wrap",
                        items: aItems.map(function(oItem) {
                            oItem.addStyleClass("sapUiTinyMarginEnd");
                            return oItem;
                        })
                    })
                ]
            });
        },

        /**
         * Create sources display panel
         * @param {array} aSources array of source IDs
         * @returns {sap.m.VBox} the sources display
         * @private
         */
        _createSourcesDisplay: function (aSources) {
            if (!aSources || aSources.length === 0) {
                return null;
            }
            
            var aSourceLinks = aSources.map(function(sSourceId) {
                return new sap.m.Link({
                    text: sSourceId,
                    press: function() {
                        MessageToast.show("Navigate to source: " + sSourceId);
                    }
                });
            });
            
            return new sap.m.VBox({
                class: "sapUiTinyMarginTop chatSources",
                items: [
                    new sap.m.Label({
                        text: "Sources:",
                        design: "Bold",
                        class: "sapUiTinyMarginBottom"
                    }),
                    new sap.m.HBox({
                        wrap: "Wrap",
                        items: aSourceLinks.map(function(oLink, idx) {
                            if (idx < aSourceLinks.length - 1) {
                                return new sap.m.HBox({
                                    items: [
                                        oLink,
                                        new sap.m.Text({ text: ", " })
                                    ]
                                });
                            }
                            return oLink;
                        })
                    })
                ]
            });
        },

        /**
         * Format timestamp for display
         * @param {number} timestamp the timestamp
         * @returns {string} formatted time
         * @private
         */
        _formatTimestamp: function (timestamp) {
            var oDateFormat = DateFormat.getTimeInstance({
                pattern: "HH:mm:ss"
            });
            return oDateFormat.format(new Date(timestamp));
        },

        /**
         * Scroll chat to bottom
         * @private
         */
        _scrollToBottom: function () {
            setTimeout(function () {
                var oScrollContainer = this.byId("chatMessagesContainer");
                if (oScrollContainer) {
                    var oScrollDelegate = oScrollContainer.getScrollDelegate();
                    if (oScrollDelegate) {
                        oScrollDelegate.scrollTo(0, 999999);
                    }
                }
            }.bind(this), 100);
        },

        /**
         * Handler for send message button
         */
        onSendMessage: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var sMessage = oAppStateModel.getProperty("/currentMessage");
            
            if (!sMessage || sMessage.trim().length === 0) {
                return;
            }
            
            // Add user message to chat history
            var aChatHistory = oAppStateModel.getProperty("/chatHistory") || [];
            var oUserMessage = {
                role: "user",
                content: sMessage.trim(),
                timestamp: Date.now()
            };
            aChatHistory.push(oUserMessage);
            
            // Clear input
            oAppStateModel.setProperty("/currentMessage", "");
            
            // Set busy state
            oAppStateModel.setProperty("/busy", true);
            
            // Update and save chat history
            oAppStateModel.setProperty("/chatHistory", aChatHistory);
            this._saveChatHistory();
            this._renderChatHistory();
            
            // Call OData Chat action
            this._callChatAction(sMessage.trim())
                .then(function(oResponse) {
                    // Add assistant response
                    var oAssistantMessage = {
                        role: "assistant",
                        content: oResponse.Content,
                        sourceIds: oResponse.SourceIds || [],
                        metadata: oResponse.Metadata ? JSON.parse(oResponse.Metadata) : null,
                        messageId: oResponse.MessageId,
                        timestamp: Date.now()
                    };
                    
                    aChatHistory.push(oAssistantMessage);
                    oAppStateModel.setProperty("/chatHistory", aChatHistory);
                    oAppStateModel.setProperty("/busy", false);
                    
                    this._saveChatHistory();
                    this._renderChatHistory();
                }.bind(this))
                .catch(function(oError) {
                    // Handle error
                    var sErrorMessage = "Sorry, I encountered an error processing your request.";
                    if (oError.responseText) {
                        try {
                            var oErrorData = JSON.parse(oError.responseText);
                            if (oErrorData.error && oErrorData.error.message) {
                                sErrorMessage += "\n\n" + oErrorData.error.message;
                            }
                        } catch (e) {
                            // Ignore JSON parse error
                        }
                    }
                    
                    aChatHistory.push({
                        role: "assistant",
                        content: sErrorMessage,
                        timestamp: Date.now(),
                        isError: true
                    });
                    
                    oAppStateModel.setProperty("/chatHistory", aChatHistory);
                    oAppStateModel.setProperty("/busy", false);
                    
                    this._renderChatHistory();
                    
                    MessageBox.error("Failed to get response from AI assistant. Please try again.");
                }.bind(this));
        },

        /**
         * Call OData Chat action
         * @param {string} sMessage the user message
         * @returns {Promise} promise that resolves with chat response
         * @private
         */
        _callChatAction: function(sMessage) {
            return new Promise(function(resolve, reject) {
                var oAppStateModel = this.getOwnerComponent().getModel("appState");
                var bIncludeSources = oAppStateModel.getProperty("/chatIncludeSources") !== false;
                
                // Prepare request payload
                var oPayload = {
                    SessionId: this._sessionId,
                    Message: sMessage,
                    IncludeSources: bIncludeSources,
                    MaxTokens: 500,
                    Temperature: 0.7
                };
                
                // Call OData action
                jQuery.ajax({
                    url: "/odata/v4/research/Chat",
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
            }.bind(this));
        },

        /**
         * Handler for clear chat button
         */
        onClearChat: function () {
            MessageBox.confirm(
                "Are you sure you want to clear the chat history?",
                {
                    title: "Clear Chat",
                    onClose: function (oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            var oAppStateModel = this.getOwnerComponent().getModel("appState");
                            oAppStateModel.setProperty("/chatHistory", []);
                            this._renderChatHistory();
                            MessageToast.show("Chat history cleared");
                        }
                    }.bind(this)
                }
            );
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
        },

        /**
         * Handler for export chat button
         */
        onExportChat: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var aChatHistory = oAppStateModel.getProperty("/chatHistory") || [];
            
            if (aChatHistory.length === 0) {
                MessageToast.show("No chat history to export");
                return;
            }
            
            // Create export data
            var sExportData = this._formatChatForExport(aChatHistory);
            
            // Create download link
            var oBlob = new Blob([sExportData], { type: "text/plain;charset=utf-8" });
            var sUrl = URL.createObjectURL(oBlob);
            var sFilename = "chat-export-" + new Date().toISOString().split('T')[0] + ".txt";
            
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = sFilename;
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            URL.revokeObjectURL(sUrl);
            
            MessageToast.show("Chat exported successfully");
        },

        /**
         * Format chat history for export
         * @param {array} aChatHistory the chat history
         * @returns {string} formatted export text
         * @private
         */
        _formatChatForExport: function (aChatHistory) {
            var aLines = [
                "HyperShimmy Chat Export",
                "Session: " + this._sessionId,
                "Exported: " + new Date().toLocaleString(),
                "=" .repeat(70),
                ""
            ];
            
            aChatHistory.forEach(function (oMessage, idx) {
                var sRole = oMessage.role === "user" ? "YOU" : "ASSISTANT";
                var sTimestamp = new Date(oMessage.timestamp).toLocaleTimeString();
                
                aLines.push("");
                aLines.push("[" + (idx + 1) + "] " + sRole + " (" + sTimestamp + ")");
                aLines.push("-".repeat(70));
                aLines.push(oMessage.content);
                
                // Add metadata for assistant messages
                if (oMessage.role === "assistant" && oMessage.metadata) {
                    aLines.push("");
                    aLines.push("Metadata:");
                    if (oMessage.metadata.confidence !== undefined) {
                        aLines.push("  - Confidence: " + (oMessage.metadata.confidence * 100).toFixed(0) + "%");
                    }
                    if (oMessage.metadata.query_intent) {
                        aLines.push("  - Intent: " + oMessage.metadata.query_intent);
                    }
                    if (oMessage.metadata.total_time_ms) {
                        aLines.push("  - Response time: " + oMessage.metadata.total_time_ms + "ms");
                    }
                }
                
                // Add sources
                if (oMessage.sourceIds && oMessage.sourceIds.length > 0) {
                    aLines.push("");
                    aLines.push("Sources: " + oMessage.sourceIds.join(", "));
                }
            });
            
            aLines.push("");
            aLines.push("=" .repeat(70));
            aLines.push("End of chat export");
            
            return aLines.join("\n");
        },

        /**
         * Handler for copy message content
         * @param {string} sContent the content to copy
         */
        onCopyMessage: function (sContent) {
            if (!sContent) {
                return;
            }
            
            // Use Clipboard API if available
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(sContent)
                    .then(function () {
                        MessageToast.show("Message copied to clipboard");
                    })
                    .catch(function (err) {
                        console.error("Failed to copy:", err);
                        MessageToast.show("Failed to copy message");
                    });
            } else {
                // Fallback for older browsers
                var oTextArea = document.createElement("textarea");
                oTextArea.value = sContent;
                oTextArea.style.position = "fixed";
                oTextArea.style.left = "-9999px";
                document.body.appendChild(oTextArea);
                oTextArea.select();
                
                try {
                    document.execCommand("copy");
                    MessageToast.show("Message copied to clipboard");
                } catch (err) {
                    console.error("Failed to copy:", err);
                    MessageToast.show("Failed to copy message");
                }
                
                document.body.removeChild(oTextArea);
            }
        },

        /**
         * Handler for regenerate last response
         */
        onRegenerateResponse: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var aChatHistory = oAppStateModel.getProperty("/chatHistory") || [];
            
            if (aChatHistory.length < 2) {
                MessageToast.show("No response to regenerate");
                return;
            }
            
            // Find the last user message
            var oLastUserMessage = null;
            for (var i = aChatHistory.length - 1; i >= 0; i--) {
                if (aChatHistory[i].role === "user") {
                    oLastUserMessage = aChatHistory[i];
                    break;
                }
            }
            
            if (!oLastUserMessage) {
                MessageToast.show("No user message found to regenerate from");
                return;
            }
            
            // Remove all messages after the last user message
            var iUserIndex = aChatHistory.indexOf(oLastUserMessage);
            aChatHistory = aChatHistory.slice(0, iUserIndex + 1);
            
            oAppStateModel.setProperty("/chatHistory", aChatHistory);
            oAppStateModel.setProperty("/busy", true);
            
            this._saveChatHistory();
            this._renderChatHistory();
            
            // Regenerate response
            this._callChatAction(oLastUserMessage.content)
                .then(function(oResponse) {
                    var oAssistantMessage = {
                        role: "assistant",
                        content: oResponse.Content,
                        sourceIds: oResponse.SourceIds || [],
                        metadata: oResponse.Metadata ? JSON.parse(oResponse.Metadata) : null,
                        messageId: oResponse.MessageId,
                        timestamp: Date.now()
                    };
                    
                    aChatHistory.push(oAssistantMessage);
                    oAppStateModel.setProperty("/chatHistory", aChatHistory);
                    oAppStateModel.setProperty("/busy", false);
                    
                    this._saveChatHistory();
                    this._renderChatHistory();
                    
                    MessageToast.show("Response regenerated");
                }.bind(this))
                .catch(function(oError) {
                    oAppStateModel.setProperty("/busy", false);
                    MessageBox.error("Failed to regenerate response. Please try again.");
                }.bind(this));
        },

        /**
         * Handler for chat settings button
         */
        onOpenSettings: function () {
            if (!this._oSettingsDialog) {
                this._oSettingsDialog = this._createSettingsDialog();
            }
            
            // Load current settings
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            this._oSettingsDialog.getModel("settings").setData({
                maxTokens: oAppStateModel.getProperty("/chatMaxTokens") || 500,
                temperature: oAppStateModel.getProperty("/chatTemperature") || 0.7,
                includeSources: oAppStateModel.getProperty("/chatIncludeSources") !== false
            });
            
            this._oSettingsDialog.open();
        },

        /**
         * Create settings dialog
         * @returns {sap.m.Dialog} the settings dialog
         * @private
         */
        _createSettingsDialog: function () {
            var oDialog = new sap.m.Dialog({
                title: "Chat Settings",
                contentWidth: "400px",
                content: [
                    new sap.m.VBox({
                        items: [
                            new sap.m.Label({
                                text: "Max Tokens:",
                                class: "sapUiTinyMarginTop"
                            }),
                            new sap.m.Slider({
                                min: 100,
                                max: 2000,
                                step: 100,
                                value: "{settings>/maxTokens}",
                                enableTickmarks: true,
                                width: "100%"
                            }),
                            new sap.m.Text({
                                text: "{settings>/maxTokens}",
                                class: "sapUiTinyMarginBottom"
                            }),
                            new sap.m.Label({
                                text: "Temperature:",
                                class: "sapUiSmallMarginTop"
                            }),
                            new sap.m.Slider({
                                min: 0,
                                max: 1,
                                step: 0.1,
                                value: "{settings>/temperature}",
                                enableTickmarks: true,
                                width: "100%"
                            }),
                            new sap.m.Text({
                                text: "{= ${settings>/temperature}.toFixed(1) }",
                                class: "sapUiTinyMarginBottom"
                            }),
                            new sap.m.CheckBox({
                                text: "Include source citations",
                                selected: "{settings>/includeSources}",
                                class: "sapUiSmallMarginTop"
                            })
                        ]
                    })
                ],
                beginButton: new sap.m.Button({
                    text: "Save",
                    type: "Emphasized",
                    press: function () {
                        this._onSaveSettings();
                        oDialog.close();
                    }.bind(this)
                }),
                endButton: new sap.m.Button({
                    text: "Cancel",
                    press: function () {
                        oDialog.close();
                    }
                })
            });
            
            oDialog.setModel(new JSONModel({}), "settings");
            
            return oDialog;
        },

        /**
         * Handler for save settings
         * @private
         */
        _onSaveSettings: function () {
            var oSettingsModel = this._oSettingsDialog.getModel("settings");
            var oSettings = oSettingsModel.getData();
            
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            oAppStateModel.setProperty("/chatMaxTokens", oSettings.maxTokens);
            oAppStateModel.setProperty("/chatTemperature", oSettings.temperature);
            oAppStateModel.setProperty("/chatIncludeSources", oSettings.includeSources);
            
            this._saveChatSettings();
            
            MessageToast.show("Settings saved");
        },

        /**
         * Handler for keyboard shortcuts in input
         * @param {sap.ui.base.Event} oEvent the keyboard event
         */
        onInputKeyPress: function (oEvent) {
            // Ctrl/Cmd + Enter to send message
            if ((oEvent.ctrlKey || oEvent.metaKey) && oEvent.keyCode === 13) {
                this.onSendMessage();
            }
        }
    });
});
