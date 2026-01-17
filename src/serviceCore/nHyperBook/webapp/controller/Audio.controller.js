sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/format/DateFormat"
], function (Controller, MessageBox, MessageToast, DateFormat) {
    "use strict";

    return Controller.extend("hypershimmy.controller.Audio", {
        
        /**
         * Called when the controller is instantiated
         */
        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("audio").attachPatternMatched(this._onRouteMatched, this);
            
            // Initialize audio settings
            this._initializeAudioSettings();
            
            // Load saved settings
            this._loadAudioSettings();
        },

        /**
         * Initialize default audio settings
         * @private
         */
        _initializeAudioSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Set defaults if not already set
            if (!oAppStateModel.getProperty("/audioVoice")) {
                oAppStateModel.setProperty("/audioVoice", "default");
            }
            if (!oAppStateModel.getProperty("/audioFormat")) {
                oAppStateModel.setProperty("/audioFormat", "mp3");
            }
            if (oAppStateModel.getProperty("/audioConfigExpanded") === undefined) {
                oAppStateModel.setProperty("/audioConfigExpanded", true);
            }
            
            oAppStateModel.setProperty("/audioText", "");
            oAppStateModel.setProperty("/audioGenerated", false);
            oAppStateModel.setProperty("/audioList", []);
            oAppStateModel.setProperty("/audioFileUrl", "");
        },

        /**
         * Load audio settings from localStorage
         * @private
         */
        _loadAudioSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            try {
                var sSettings = localStorage.getItem("hypershimmy.audioSettings");
                if (sSettings) {
                    var oSettings = JSON.parse(sSettings);
                    oAppStateModel.setProperty("/audioVoice", oSettings.voice || "default");
                    oAppStateModel.setProperty("/audioFormat", oSettings.format || "mp3");
                }
            } catch (e) {
                console.error("Failed to load audio settings:", e);
            }
        },

        /**
         * Save audio settings to localStorage
         * @private
         */
        _saveAudioSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            var oSettings = {
                voice: oAppStateModel.getProperty("/audioVoice") || "default",
                format: oAppStateModel.getProperty("/audioFormat") || "mp3"
            };
            
            try {
                localStorage.setItem("hypershimmy.audioSettings", JSON.stringify(oSettings));
            } catch (e) {
                console.error("Failed to save audio settings:", e);
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
                    $expand: "Audio"
                }
            });
            
            // Load audio list for this source
            this._loadAudioList();
        },

        /**
         * Load audio list from OData service
         * @private
         */
        _loadAudioList: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Call OData to get audio list
            jQuery.ajax({
                url: "/odata/v4/research/Audio?$filter=SourceId eq '" + this._currentSourceId + "'&$orderby=GeneratedAt desc",
                method: "GET",
                success: function (oData) {
                    var aAudioList = oData.value || [];
                    
                    // Format timestamps for display
                    var oDateFormat = DateFormat.getDateTimeInstance({
                        pattern: "MMM dd, yyyy HH:mm"
                    });
                    
                    aAudioList.forEach(function (oAudio) {
                        oAudio.generatedTimeFormatted = oDateFormat.format(
                            new Date(oAudio.GeneratedAt * 1000)
                        );
                    });
                    
                    oAppStateModel.setProperty("/audioList", aAudioList);
                }.bind(this),
                error: function (oError) {
                    console.error("Failed to load audio list:", oError);
                    oAppStateModel.setProperty("/audioList", []);
                }
            });
        },

        /**
         * Handler for generate audio button
         */
        onGenerateAudio: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Get configuration
            var sText = oAppStateModel.getProperty("/audioText");
            var sVoice = oAppStateModel.getProperty("/audioVoice");
            var sFormat = oAppStateModel.getProperty("/audioFormat");
            
            // Validate text
            if (!sText || sText.trim().length === 0) {
                MessageBox.error("Please enter text to convert to audio");
                return;
            }
            
            // Set busy state
            oAppStateModel.setProperty("/busy", true);
            oAppStateModel.setProperty("/audioGenerated", false);
            oAppStateModel.setProperty("/audioConfigExpanded", false);
            
            // Call OData GenerateAudio action
            this._callGenerateAudioAction(
                this._currentSourceId,
                sText,
                sVoice,
                sFormat
            )
                .then(function (oResponse) {
                    // Display audio
                    this._displayAudio(oResponse);
                    
                    oAppStateModel.setProperty("/busy", false);
                    oAppStateModel.setProperty("/audioGenerated", true);
                    
                    // Save settings
                    this._saveAudioSettings();
                    
                    // Reload audio list
                    this._loadAudioList();
                    
                    MessageToast.show("Audio generation initiated");
                }.bind(this))
                .catch(function (oError) {
                    // Handle error
                    oAppStateModel.setProperty("/busy", false);
                    
                    var sErrorMessage = "Failed to generate audio. Please try again.";
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
         * Call OData GenerateAudio action
         * @param {string} sSourceId source ID
         * @param {string} sText text to convert
         * @param {string} sVoice voice identifier
         * @param {string} sFormat audio format
         * @returns {Promise} promise that resolves with audio response
         * @private
         */
        _callGenerateAudioAction: function (sSourceId, sText, sVoice, sFormat) {
            return new Promise(function (resolve, reject) {
                // Prepare request payload
                var oPayload = {
                    SourceId: sSourceId,
                    Text: sText,
                    Voice: sVoice,
                    Format: sFormat
                };
                
                // Call OData action
                jQuery.ajax({
                    url: "/odata/v4/research/GenerateAudio",
                    method: "POST",
                    contentType: "application/json",
                    data: JSON.stringify(oPayload),
                    success: function (oData) {
                        resolve(oData);
                    },
                    error: function (oError) {
                        reject(oError);
                    }
                });
            });
        },

        /**
         * Display audio in the UI
         * @param {object} oAudioResponse the audio response
         * @private
         */
        _displayAudio: function (oAudioResponse) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Fetch full audio details from OData
            jQuery.ajax({
                url: "/odata/v4/research/Audio('" + oAudioResponse.AudioId + "')",
                method: "GET",
                success: function (oAudio) {
                    // Store current audio
                    oAppStateModel.setProperty("/currentAudio", oAudio);
                    
                    // Set audio file URL (for stub mode, this will be placeholder)
                    var sAudioUrl = "/audio/" + oAudio.AudioId + "." + oAudio.FilePath.split('.').pop();
                    oAppStateModel.setProperty("/audioFileUrl", sAudioUrl);
                    
                    // Set generated time
                    var oDateFormat = DateFormat.getDateTimeInstance({
                        pattern: "MMM dd, yyyy HH:mm:ss"
                    });
                    oAppStateModel.setProperty("/audioGeneratedTime", oDateFormat.format(new Date()));
                    
                    // Update audio player source
                    this._updateAudioPlayer();
                }.bind(this),
                error: function (oError) {
                    console.error("Failed to fetch audio details:", oError);
                    
                    // Use response data as fallback
                    var oAudio = {
                        AudioId: oAudioResponse.AudioId,
                        SourceId: this._currentSourceId,
                        Title: "Audio Overview",
                        FilePath: oAudioResponse.FilePath,
                        FileSize: 0,
                        DurationSeconds: 0.0,
                        SampleRate: 48000,
                        BitDepth: 24,
                        Channels: 2,
                        Provider: "audiolabshimmy",
                        Voice: oAppStateModel.getProperty("/audioVoice"),
                        GeneratedAt: Math.floor(Date.now() / 1000),
                        ProcessingTimeMs: null,
                        Status: oAudioResponse.Status,
                        Message: oAudioResponse.Message,
                        ErrorMessage: null
                    };
                    
                    oAppStateModel.setProperty("/currentAudio", oAudio);
                    oAppStateModel.setProperty("/audioFileUrl", "");
                }.bind(this)
            });
        },

        /**
         * Update audio player with new source
         * @private
         */
        _updateAudioPlayer: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var sAudioUrl = oAppStateModel.getProperty("/audioFileUrl");
            
            // Get audio player element
            var oAudioPlayer = document.getElementById("audioPlayer");
            if (oAudioPlayer) {
                oAudioPlayer.load();
                
                // Note: In stub mode, the audio file may not exist yet
                // This will be functional once AudioLabShimmy is integrated
            }
        },

        /**
         * Handler for audio selection in list
         * @param {sap.ui.base.Event} oEvent the selection change event
         */
        onAudioSelect: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oAudio = oItem.getBindingContext("appState").getObject();
            
            // Display selected audio
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            oAppStateModel.setProperty("/currentAudio", oAudio);
            oAppStateModel.setProperty("/audioGenerated", true);
            
            // Set audio file URL
            var sAudioUrl = "/audio/" + oAudio.AudioId + "." + oAudio.FilePath.split('.').pop();
            oAppStateModel.setProperty("/audioFileUrl", sAudioUrl);
            
            // Update audio player
            this._updateAudioPlayer();
            
            MessageToast.show("Audio loaded");
        },

        /**
         * Handler for play/pause button
         */
        onPlayPause: function () {
            var oAudioPlayer = document.getElementById("audioPlayer");
            if (oAudioPlayer) {
                if (oAudioPlayer.paused) {
                    oAudioPlayer.play()
                        .catch(function (error) {
                            console.error("Playback error:", error);
                            MessageToast.show("Audio file not available (waiting for AudioLabShimmy integration)");
                        });
                } else {
                    oAudioPlayer.pause();
                }
            }
        },

        /**
         * Handler for rewind button
         */
        onRewind: function () {
            var oAudioPlayer = document.getElementById("audioPlayer");
            if (oAudioPlayer) {
                oAudioPlayer.currentTime = Math.max(0, oAudioPlayer.currentTime - 10);
            }
        },

        /**
         * Handler for forward button
         */
        onForward: function () {
            var oAudioPlayer = document.getElementById("audioPlayer");
            if (oAudioPlayer) {
                oAudioPlayer.currentTime = Math.min(oAudioPlayer.duration, oAudioPlayer.currentTime + 10);
            }
        },

        /**
         * Handler for download audio button
         */
        onDownloadAudio: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oAudio = oAppStateModel.getProperty("/currentAudio");
            
            if (!oAudio) {
                MessageToast.show("No audio to download");
                return;
            }
            
            // In stub mode, show info message
            if (oAudio.Status === "pending_integration") {
                MessageBox.information(
                    "Audio download will be available once AudioLabShimmy integration is complete. " +
                    "The audio file is currently not generated.",
                    {
                        title: "Download Not Available"
                    }
                );
                return;
            }
            
            // Trigger download
            var sAudioUrl = "/audio/" + oAudio.AudioId + "." + oAudio.FilePath.split('.').pop();
            var sFilename = "audio-" + oAudio.AudioId + "." + oAudio.FilePath.split('.').pop();
            
            var oLink = document.createElement("a");
            oLink.href = sAudioUrl;
            oLink.download = sFilename;
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            
            MessageToast.show("Downloading audio file");
        },

        /**
         * Handler for delete audio button
         * @param {sap.ui.base.Event} oEvent the press event
         */
        onDeleteAudio: function (oEvent) {
            var oItem = oEvent.getSource().getParent();
            var oAudio = oItem.getBindingContext("appState").getObject();
            
            MessageBox.confirm(
                "Are you sure you want to delete this audio file?",
                {
                    title: "Confirm Deletion",
                    onClose: function (oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._deleteAudio(oAudio.AudioId);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * Delete audio via OData
         * @param {string} sAudioId the audio ID to delete
         * @private
         */
        _deleteAudio: function (sAudioId) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            jQuery.ajax({
                url: "/odata/v4/research/Audio('" + sAudioId + "')",
                method: "DELETE",
                success: function () {
                    MessageToast.show("Audio deleted successfully");
                    
                    // Reload audio list
                    this._loadAudioList();
                    
                    // Clear current audio if it was deleted
                    var oCurrentAudio = oAppStateModel.getProperty("/currentAudio");
                    if (oCurrentAudio && oCurrentAudio.AudioId === sAudioId) {
                        oAppStateModel.setProperty("/audioGenerated", false);
                        oAppStateModel.setProperty("/currentAudio", null);
                    }
                }.bind(this),
                error: function (oError) {
                    console.error("Failed to delete audio:", oError);
                    MessageBox.error("Failed to delete audio. Please try again.");
                }
            });
        },

        /**
         * Handler for refresh audio list button
         */
        onRefreshAudioList: function () {
            this._loadAudioList();
            MessageToast.show("Audio list refreshed");
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
