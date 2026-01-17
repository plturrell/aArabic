sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/format/DateFormat"
], function (Controller, MessageBox, MessageToast, DateFormat) {
    "use strict";

    return Controller.extend("hypershimmy.controller.Slides", {
        
        /**
         * Called when the controller is instantiated
         */
        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("slides").attachPatternMatched(this._onRouteMatched, this);
            
            // Initialize slides settings
            this._initializeSlidesSettings();
            
            // Load saved settings
            this._loadSlidesSettings();
        },

        /**
         * Initialize default slides settings
         * @private
         */
        _initializeSlidesSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Set defaults if not already set
            if (!oAppStateModel.getProperty("/presentationTheme")) {
                oAppStateModel.setProperty("/presentationTheme", "professional");
            }
            if (!oAppStateModel.getProperty("/presentationAudience")) {
                oAppStateModel.setProperty("/presentationAudience", "general");
            }
            if (!oAppStateModel.getProperty("/presentationDetail")) {
                oAppStateModel.setProperty("/presentationDetail", "medium");
            }
            if (!oAppStateModel.getProperty("/presentationNumSlides")) {
                oAppStateModel.setProperty("/presentationNumSlides", 7);
            }
            if (oAppStateModel.getProperty("/slidesConfigExpanded") === undefined) {
                oAppStateModel.setProperty("/slidesConfigExpanded", true);
            }
            
            oAppStateModel.setProperty("/presentationTitle", "");
            oAppStateModel.setProperty("/presentationGenerated", false);
            oAppStateModel.setProperty("/presentationList", []);
            oAppStateModel.setProperty("/currentSlides", []);
            oAppStateModel.setProperty("/currentSlideIndex", 0);
        },

        /**
         * Load slides settings from localStorage
         * @private
         */
        _loadSlidesSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            try {
                var sSettings = localStorage.getItem("hypershimmy.slidesSettings");
                if (sSettings) {
                    var oSettings = JSON.parse(sSettings);
                    oAppStateModel.setProperty("/presentationTheme", oSettings.theme || "professional");
                    oAppStateModel.setProperty("/presentationAudience", oSettings.audience || "general");
                    oAppStateModel.setProperty("/presentationDetail", oSettings.detail || "medium");
                    oAppStateModel.setProperty("/presentationNumSlides", oSettings.numSlides || 7);
                }
            } catch (e) {
                console.error("Failed to load slides settings:", e);
            }
        },

        /**
         * Save slides settings to localStorage
         * @private
         */
        _saveSlidesSettings: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            var oSettings = {
                theme: oAppStateModel.getProperty("/presentationTheme") || "professional",
                audience: oAppStateModel.getProperty("/presentationAudience") || "general",
                detail: oAppStateModel.getProperty("/presentationDetail") || "medium",
                numSlides: oAppStateModel.getProperty("/presentationNumSlides") || 7
            };
            
            try {
                localStorage.setItem("hypershimmy.slidesSettings", JSON.stringify(oSettings));
            } catch (e) {
                console.error("Failed to save slides settings:", e);
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
                    $expand: "Presentation"
                }
            });
            
            // Load presentation list for this source
            this._loadPresentationList();
        },

        /**
         * Load presentation list from OData service
         * @private
         */
        _loadPresentationList: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Call OData to get presentation list
            jQuery.ajax({
                url: "/odata/v4/research/Presentation?$filter=SourceId eq '" + this._currentSourceId + "'&$orderby=GeneratedAt desc",
                method: "GET",
                success: function (oData) {
                    var aPresentationList = oData.value || [];
                    
                    // Format timestamps for display
                    var oDateFormat = DateFormat.getDateTimeInstance({
                        pattern: "MMM dd, yyyy HH:mm"
                    });
                    
                    aPresentationList.forEach(function (oPresentation) {
                        oPresentation.generatedTimeFormatted = oDateFormat.format(
                            new Date(oPresentation.GeneratedAt * 1000)
                        );
                    });
                    
                    oAppStateModel.setProperty("/presentationList", aPresentationList);
                }.bind(this),
                error: function (oError) {
                    console.error("Failed to load presentation list:", oError);
                    oAppStateModel.setProperty("/presentationList", []);
                }
            });
        },

        /**
         * Handler for generate slides button
         */
        onGenerateSlides: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Get configuration
            var sTitle = oAppStateModel.getProperty("/presentationTitle");
            var sTheme = oAppStateModel.getProperty("/presentationTheme");
            var sAudience = oAppStateModel.getProperty("/presentationAudience");
            var sDetail = oAppStateModel.getProperty("/presentationDetail");
            var iNumSlides = oAppStateModel.getProperty("/presentationNumSlides");
            
            // Use default title if not provided
            if (!sTitle || sTitle.trim().length === 0) {
                sTitle = "Research Presentation";
            }
            
            // Set busy state
            oAppStateModel.setProperty("/busy", true);
            oAppStateModel.setProperty("/presentationGenerated", false);
            oAppStateModel.setProperty("/slidesConfigExpanded", false);
            
            // Call OData GenerateSlides action
            this._callGenerateSlidesAction(
                this._currentSourceId,
                sTitle,
                sTheme,
                sAudience,
                sDetail,
                iNumSlides
            )
                .then(function (oResponse) {
                    // Display presentation
                    this._displayPresentation(oResponse.PresentationId);
                    
                    oAppStateModel.setProperty("/busy", false);
                    oAppStateModel.setProperty("/presentationGenerated", true);
                    
                    // Save settings
                    this._saveSlidesSettings();
                    
                    // Reload presentation list
                    this._loadPresentationList();
                    
                    MessageToast.show("Presentation generated successfully with " + oResponse.NumSlides + " slides");
                }.bind(this))
                .catch(function (oError) {
                    // Handle error
                    oAppStateModel.setProperty("/busy", false);
                    
                    var sErrorMessage = "Failed to generate presentation. Please try again.";
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
         * Call OData GenerateSlides action
         * @param {string} sSourceId source ID
         * @param {string} sTitle presentation title
         * @param {string} sTheme theme identifier
         * @param {string} sAudience target audience
         * @param {string} sDetail detail level
         * @param {number} iNumSlides number of slides
         * @returns {Promise} promise that resolves with presentation response
         * @private
         */
        _callGenerateSlidesAction: function (sSourceId, sTitle, sTheme, sAudience, sDetail, iNumSlides) {
            return new Promise(function (resolve, reject) {
                // Prepare request payload
                var oPayload = {
                    SourceId: sSourceId,
                    Title: sTitle,
                    Theme: sTheme,
                    TargetAudience: sAudience,
                    DetailLevel: sDetail,
                    NumSlides: iNumSlides
                };
                
                // Call OData action
                jQuery.ajax({
                    url: "/odata/v4/research/GenerateSlides",
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
         * Display presentation in the UI
         * @param {string} sPresentationId the presentation ID
         * @private
         */
        _displayPresentation: function (sPresentationId) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            // Fetch full presentation details from OData
            jQuery.ajax({
                url: "/odata/v4/research/Presentation('" + sPresentationId + "')",
                method: "GET",
                success: function (oPresentation) {
                    // Store current presentation
                    oAppStateModel.setProperty("/currentPresentation", oPresentation);
                    
                    // Set generated time
                    var oDateFormat = DateFormat.getDateTimeInstance({
                        pattern: "MMM dd, yyyy HH:mm:ss"
                    });
                    oAppStateModel.setProperty("/presentationGeneratedTime", oDateFormat.format(new Date()));
                    
                    // Load slides for this presentation
                    this._loadSlides(sPresentationId);
                }.bind(this),
                error: function (oError) {
                    console.error("Failed to fetch presentation details:", oError);
                    MessageBox.error("Failed to load presentation details");
                }
            });
        },

        /**
         * Load slides for a presentation
         * @param {string} sPresentationId the presentation ID
         * @private
         */
        _loadSlides: function (sPresentationId) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            jQuery.ajax({
                url: "/odata/v4/research/Presentation('" + sPresentationId + "')/Slides?$orderby=SlideNumber asc",
                method: "GET",
                success: function (oData) {
                    var aSlides = oData.value || [];
                    
                    oAppStateModel.setProperty("/currentSlides", aSlides);
                    oAppStateModel.setProperty("/currentSlideIndex", 0);
                    
                    // Display first slide
                    if (aSlides.length > 0) {
                        this._displaySlide(0);
                    }
                }.bind(this),
                error: function (oError) {
                    console.error("Failed to load slides:", oError);
                    oAppStateModel.setProperty("/currentSlides", []);
                }
            });
        },

        /**
         * Display a specific slide
         * @param {number} iIndex slide index
         * @private
         */
        _displaySlide: function (iIndex) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var aSlides = oAppStateModel.getProperty("/currentSlides");
            
            if (iIndex >= 0 && iIndex < aSlides.length) {
                var oSlide = aSlides[iIndex];
                oAppStateModel.setProperty("/currentSlide", oSlide);
                oAppStateModel.setProperty("/currentSlideIndex", iIndex);
                
                // Convert content to HTML for display
                var sContentHtml = this._convertContentToHtml(oSlide.Content);
                oAppStateModel.setProperty("/currentSlideContentHtml", sContentHtml);
            }
        },

        /**
         * Convert slide content text to HTML
         * @param {string} sContent slide content
         * @returns {string} HTML formatted content
         * @private
         */
        _convertContentToHtml: function (sContent) {
            if (!sContent) {
                return "";
            }
            
            // Convert bullet points
            var sHtml = sContent.replace(/^• (.+)$/gm, "<li>$1</li>");
            
            // Wrap lists
            if (sHtml.includes("<li>")) {
                sHtml = "<ul>" + sHtml + "</ul>";
            }
            
            // Convert line breaks
            sHtml = sHtml.replace(/\n/g, "<br/>");
            
            return sHtml;
        },

        /**
         * Handler for presentation selection in list
         * @param {sap.ui.base.Event} oEvent the selection change event
         */
        onPresentationSelect: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oPresentation = oItem.getBindingContext("appState").getObject();
            
            // Display selected presentation
            this._displayPresentation(oPresentation.PresentationId);
            
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            oAppStateModel.setProperty("/presentationGenerated", true);
            
            MessageToast.show("Presentation loaded");
        },

        /**
         * Handler for slide selection in list
         * @param {sap.ui.base.Event} oEvent the selection change event
         */
        onSlideSelect: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oSlide = oItem.getBindingContext("appState").getObject();
            
            // Find index of selected slide
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var aSlides = oAppStateModel.getProperty("/currentSlides");
            var iIndex = aSlides.findIndex(function (s) {
                return s.SlideId === oSlide.SlideId;
            });
            
            if (iIndex >= 0) {
                this._displaySlide(iIndex);
            }
        },

        /**
         * Handler for previous slide button
         */
        onPreviousSlide: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var iCurrentIndex = oAppStateModel.getProperty("/currentSlideIndex");
            
            if (iCurrentIndex > 0) {
                this._displaySlide(iCurrentIndex - 1);
            }
        },

        /**
         * Handler for next slide button
         */
        onNextSlide: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var iCurrentIndex = oAppStateModel.getProperty("/currentSlideIndex");
            var aSlides = oAppStateModel.getProperty("/currentSlides");
            
            if (iCurrentIndex < aSlides.length - 1) {
                this._displaySlide(iCurrentIndex + 1);
            }
        },

        /**
         * Handler for open presentation button
         */
        onOpenPresentation: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oPresentation = oAppStateModel.getProperty("/currentPresentation");
            
            if (!oPresentation) {
                MessageToast.show("No presentation to open");
                return;
            }
            
            // Open presentation HTML file in new window
            var sUrl = "/" + oPresentation.FilePath;
            window.open(sUrl, "_blank");
            
            MessageToast.show("Opening presentation in new window");
        },

        /**
         * Handler for export standard button
         */
        onExportStandard: function () {
            this._exportPresentation(false);
        },

        /**
         * Handler for export with notes button
         */
        onExportWithNotes: function () {
            this._exportPresentation(true);
        },

        /**
         * Export presentation with options
         * @param {boolean} bIncludeNotes whether to include speaker notes
         * @private
         */
        _exportPresentation: function (bIncludeNotes) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oPresentation = oAppStateModel.getProperty("/currentPresentation");
            
            if (!oPresentation) {
                MessageToast.show("No presentation to export");
                return;
            }
            
            // Call ExportPresentation action
            var oPayload = {
                PresentationId: oPresentation.PresentationId,
                Format: "html",
                IncludeNotes: bIncludeNotes,
                Standalone: true,
                Compress: false
            };
            
            jQuery.ajax({
                url: "/odata/v4/research/ExportPresentation",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oPayload),
                success: function (oResponse) {
                    // Trigger download
                    var sExportUrl = "/" + oResponse.ExportPath;
                    var sFilename = oResponse.ExportPath.split('/').pop();
                    
                    var oLink = document.createElement("a");
                    oLink.href = sExportUrl;
                    oLink.download = sFilename;
                    document.body.appendChild(oLink);
                    oLink.click();
                    document.body.removeChild(oLink);
                    
                    MessageToast.show("Presentation exported (" + (oResponse.FileSize / 1024).toFixed(2) + " KB)");
                },
                error: function (oError) {
                    console.error("Failed to export presentation:", oError);
                    MessageBox.error("Failed to export presentation. Please try again.");
                }
            });
        },

        /**
         * Handler for download presentation button
         */
        onDownloadPresentation: function () {
            this.onExportStandard();
        },

        /**
         * Handler for share presentation button
         */
        onSharePresentation: function () {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            var oPresentation = oAppStateModel.getProperty("/currentPresentation");
            
            if (!oPresentation) {
                MessageToast.show("No presentation to share");
                return;
            }
            
            // Get presentation URL
            var sUrl = window.location.origin + "/" + oPresentation.FilePath;
            
            // Copy to clipboard
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(sUrl)
                    .then(function () {
                        MessageToast.show("Presentation URL copied to clipboard");
                    })
                    .catch(function () {
                        MessageToast.show("Failed to copy URL to clipboard");
                    });
            } else {
                // Fallback for older browsers
                MessageBox.information(
                    "Share this URL:\n\n" + sUrl,
                    {
                        title: "Share Presentation"
                    }
                );
            }
        },

        /**
         * Handler for delete presentation button
         * @param {sap.ui.base.Event} oEvent the press event
         */
        onDeletePresentation: function (oEvent) {
            var oItem = oEvent.getSource().getParent();
            var oPresentation = oItem.getBindingContext("appState").getObject();
            
            MessageBox.confirm(
                "Are you sure you want to delete this presentation? This will also delete all slides.",
                {
                    title: "Confirm Deletion",
                    onClose: function (oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._deletePresentation(oPresentation.PresentationId);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * Delete presentation via OData
         * @param {string} sPresentationId the presentation ID to delete
         * @private
         */
        _deletePresentation: function (sPresentationId) {
            var oAppStateModel = this.getOwnerComponent().getModel("appState");
            
            jQuery.ajax({
                url: "/odata/v4/research/Presentation('" + sPresentationId + "')",
                method: "DELETE",
                success: function () {
                    MessageToast.show("Presentation deleted successfully");
                    
                    // Reload presentation list
                    this._loadPresentationList();
                    
                    // Clear current presentation if it was deleted
                    var oCurrentPresentation = oAppStateModel.getProperty("/currentPresentation");
                    if (oCurrentPresentation && oCurrentPresentation.PresentationId === sPresentationId) {
                        oAppStateModel.setProperty("/presentationGenerated", false);
                        oAppStateModel.setProperty("/currentPresentation", null);
                        oAppStateModel.setProperty("/currentSlides", []);
                    }
                }.bind(this),
                error: function (oError) {
                    console.error("Failed to delete presentation:", oError);
                    MessageBox.error("Failed to delete presentation. Please try again.");
                }
            });
        },

        /**
         * Handler for refresh presentation list button
         */
        onRefreshPresentationList: function () {
            this._loadPresentationList();
            MessageToast.show("Presentation list refreshed");
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
