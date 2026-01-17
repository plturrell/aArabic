sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/Fragment",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/m/Dialog",
    "sap/m/Input",
    "sap/m/Select",
    "sap/m/Item",
    "sap/m/Button",
    "sap/m/VBox",
    "sap/m/Label"
], function (Controller, Fragment, MessageToast, MessageBox, Dialog, Input, Select, Item, Button, VBox, Label) {
    "use strict";

    return Controller.extend("hypershimmy.controller.Master", {
        
        /**
         * Called when the controller is instantiated
         */
        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("main").attachPatternMatched(this._onRouteMatched, this);
        },

        /**
         * Route matched handler
         * @param {sap.ui.base.Event} oEvent the route matched event
         * @private
         */
        _onRouteMatched: function (oEvent) {
            // Clear selection when navigating back to master
            var oList = this.byId("sourcesList");
            if (oList) {
                oList.removeSelections(true);
            }
        },

        /**
         * Handler for list item selection
         * @param {sap.ui.base.Event} oEvent the selection change event
         */
        onSelectionChange: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext();
            
            if (oContext) {
                var sSourceId = oContext.getProperty("Id");
                
                // Update app state
                var oAppStateModel = this.getOwnerComponent().getModel("appState");
                oAppStateModel.setProperty("/selectedSourceId", sSourceId);
                
                // Navigate to detail view with FlexibleColumnLayout
                var oRouter = this.getOwnerComponent().getRouter();
                oRouter.navTo("detail", {
                    sourceId: sSourceId
                });
            }
        },

        /**
         * Handler for add source button
         */
        onAddSource: function () {
            var that = this;
            
            // Create input fields
            var oUrlInput = new Input({
                placeholder: "Enter URL or file path",
                width: "100%"
            });
            
            var oTitleInput = new Input({
                placeholder: "Enter title (optional)",
                width: "100%"
            });
            
            var oTypeSelect = new Select({
                width: "100%",
                items: [
                    new Item({ key: "URL", text: "Web URL" }),
                    new Item({ key: "PDF", text: "PDF Document" }),
                    new Item({ key: "Text", text: "Text Document" })
                ]
            });
            
            // Create dialog
            var oDialog = new Dialog({
                title: "Add New Source",
                contentWidth: "500px",
                content: [
                    new VBox({
                        items: [
                            new Label({ text: "Source Type", required: true }),
                            oTypeSelect,
                            new Label({ text: "URL or Path", required: true, class: "sapUiTinyMarginTop" }),
                            oUrlInput,
                            new Label({ text: "Title", class: "sapUiTinyMarginTop" }),
                            oTitleInput
                        ]
                    })
                ],
                beginButton: new Button({
                    text: "Add",
                    type: "Emphasized",
                    press: function () {
                        var sUrl = oUrlInput.getValue();
                        var sTitle = oTitleInput.getValue();
                        var sType = oTypeSelect.getSelectedKey();
                        
                        if (!sUrl) {
                            MessageBox.error("Please enter a URL or path");
                            return;
                        }
                        
                        // In real implementation, this would create via OData
                        MessageToast.show("Source '" + (sTitle || sUrl) + "' added (mock)");
                        oDialog.close();
                        
                        // Refresh the list (mock - in real app would be automatic via OData binding)
                        that._addMockSource(sType, sUrl, sTitle);
                    }
                }),
                endButton: new Button({
                    text: "Cancel",
                    press: function () {
                        oDialog.close();
                    }
                }),
                afterClose: function () {
                    oDialog.destroy();
                }
            });
            
            oDialog.open();
        },

        /**
         * Add a mock source to the model (temporary for Day 5)
         * @param {string} sType the source type
         * @param {string} sUrl the URL
         * @param {string} sTitle the title
         * @private
         */
        _addMockSource: function (sType, sUrl, sTitle) {
            var oModel = this.getView().getModel();
            var aSources = oModel.getProperty("/Sources") || [];
            
            var sId = "source_" + Date.now();
            var oNewSource = {
                Id: sId,
                Title: sTitle || "New " + sType + " Source",
                SourceType: sType,
                Url: sUrl,
                Status: "Ready",
                Content: "This is sample content for the new source. In a real implementation, this would be fetched from the URL or file.",
                CreatedAt: new Date().toISOString(),
                UpdatedAt: new Date().toISOString()
            };
            
            aSources.push(oNewSource);
            oModel.setProperty("/Sources", aSources);
        },

        /**
         * Handler for upload file button (Day 17)
         */
        onUploadFile: function () {
            var oView = this.getView();
            
            // Load fragment if not already loaded
            if (!this._fileUploadDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "hypershimmy.view.fragments.FileUpload",
                    controller: this
                }).then(function (oDialog) {
                    this._fileUploadDialog = oDialog;
                    oView.addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._fileUploadDialog.open();
            }
        },

        /**
         * Handler for file selection change
         * @param {sap.ui.base.Event} oEvent the file change event
         */
        onFileChange: function (oEvent) {
            var oFileUploader = oEvent.getSource();
            var sFileName = oFileUploader.getValue();
            
            // Enable upload button if file is selected
            var oUploadButton = this.byId("uploadButton");
            if (oUploadButton) {
                oUploadButton.setEnabled(!!sFileName);
            }
            
            // Update status message
            var oMessageStrip = this.byId("uploadMessage");
            if (oMessageStrip && sFileName) {
                oMessageStrip.setText("File selected: " + sFileName);
                oMessageStrip.setType("Information");
                oMessageStrip.setVisible(true);
            }
        },

        /**
         * Handler for upload button press
         */
        onUploadPress: function () {
            var oFileUploader = this.byId("fileUploader");
            var oProgressIndicator = this.byId("uploadProgress");
            var oMessageStrip = this.byId("uploadMessage");
            var oUploadButton = this.byId("uploadButton");
            
            if (!oFileUploader.getValue()) {
                MessageBox.error("Please select a file to upload");
                return;
            }
            
            // Show progress indicator
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(0);
                oProgressIndicator.setDisplayValue("Uploading...");
                oProgressIndicator.setState("Information");
                oProgressIndicator.setVisible(true);
            }
            
            // Update message
            if (oMessageStrip) {
                oMessageStrip.setText("Uploading file...");
                oMessageStrip.setType("Information");
                oMessageStrip.setVisible(true);
            }
            
            // Disable upload button during upload
            if (oUploadButton) {
                oUploadButton.setEnabled(false);
            }
            
            // Start upload
            oFileUploader.upload();
            
            // Simulate progress (real progress would come from server)
            this._simulateUploadProgress();
        },

        /**
         * Simulate upload progress animation
         * @private
         */
        _simulateUploadProgress: function () {
            var oProgressIndicator = this.byId("uploadProgress");
            if (!oProgressIndicator) return;
            
            var iProgress = 0;
            var that = this;
            
            this._uploadProgressInterval = setInterval(function () {
                iProgress += 10;
                if (iProgress <= 90) {
                    oProgressIndicator.setPercentValue(iProgress);
                    oProgressIndicator.setDisplayValue("Uploading... " + iProgress + "%");
                }
            }, 200);
        },

        /**
         * Handler for upload complete
         * @param {sap.ui.base.Event} oEvent the upload complete event
         */
        onUploadComplete: function (oEvent) {
            var oFileUploader = oEvent.getSource();
            var oProgressIndicator = this.byId("uploadProgress");
            var oMessageStrip = this.byId("uploadMessage");
            var oUploadButton = this.byId("uploadButton");
            var oTitleInput = this.byId("fileTitleInput");
            
            // Clear progress interval
            if (this._uploadProgressInterval) {
                clearInterval(this._uploadProgressInterval);
                this._uploadProgressInterval = null;
            }
            
            // Get response
            var sResponse = oEvent.getParameter("response");
            var iStatus = oEvent.getParameter("status");
            
            try {
                // Parse JSON response
                var oResponse = JSON.parse(sResponse);
                
                if (oResponse.success) {
                    // Success
                    if (oProgressIndicator) {
                        oProgressIndicator.setPercentValue(100);
                        oProgressIndicator.setDisplayValue("Upload complete!");
                        oProgressIndicator.setState("Success");
                    }
                    
                    if (oMessageStrip) {
                        oMessageStrip.setText("File uploaded successfully: " + oResponse.filename);
                        oMessageStrip.setType("Success");
                    }
                    
                    // Add to sources list
                    var sTitle = oTitleInput ? oTitleInput.getValue() : "";
                    this._addUploadedSource(oResponse, sTitle);
                    
                    // Show success message
                    MessageToast.show("File uploaded: " + oResponse.filename);
                    
                    // Close dialog after 2 seconds
                    setTimeout(function () {
                        if (this._fileUploadDialog) {
                            this._fileUploadDialog.close();
                        }
                    }.bind(this), 2000);
                    
                } else {
                    // Error from server
                    if (oProgressIndicator) {
                        oProgressIndicator.setState("Error");
                        oProgressIndicator.setDisplayValue("Upload failed");
                    }
                    
                    if (oMessageStrip) {
                        oMessageStrip.setText("Upload failed: " + (oResponse.error || "Unknown error"));
                        oMessageStrip.setType("Error");
                    }
                    
                    MessageBox.error("Upload failed: " + (oResponse.error || "Unknown error"));
                    
                    // Re-enable upload button
                    if (oUploadButton) {
                        oUploadButton.setEnabled(true);
                    }
                }
                
            } catch (e) {
                // Parse error or network error
                if (oProgressIndicator) {
                    oProgressIndicator.setState("Error");
                    oProgressIndicator.setDisplayValue("Upload failed");
                }
                
                if (oMessageStrip) {
                    oMessageStrip.setText("Upload failed: " + (e.message || "Network error"));
                    oMessageStrip.setType("Error");
                }
                
                MessageBox.error("Upload failed: " + (e.message || "Network error"));
                
                // Re-enable upload button
                if (oUploadButton) {
                    oUploadButton.setEnabled(true);
                }
            }
        },

        /**
         * Add uploaded file as a source
         * @param {object} oUploadResponse the upload response data
         * @param {string} sTitle the user-provided title
         * @private
         */
        _addUploadedSource: function (oUploadResponse, sTitle) {
            var oModel = this.getView().getModel();
            var aSources = oModel.getProperty("/Sources") || [];
            
            // Determine source type from MIME type
            var sType = "Document";
            if (oUploadResponse.fileType.indexOf("pdf") >= 0) {
                sType = "PDF";
            } else if (oUploadResponse.fileType.indexOf("html") >= 0) {
                sType = "HTML";
            } else if (oUploadResponse.fileType.indexOf("text") >= 0) {
                sType = "Text";
            }
            
            var oNewSource = {
                Id: oUploadResponse.fileId,
                Title: sTitle || oUploadResponse.filename,
                SourceType: sType,
                Url: "uploads/" + oUploadResponse.fileId,
                Status: "Ready",
                Content: "Text extracted: " + oUploadResponse.textLength + " characters",
                Size: oUploadResponse.size,
                TextLength: oUploadResponse.textLength,
                CreatedAt: new Date().toISOString(),
                UpdatedAt: new Date().toISOString()
            };
            
            aSources.unshift(oNewSource); // Add to beginning of list
            oModel.setProperty("/Sources", aSources);
        },

        /**
         * Handler for cancel button
         */
        onCancelUpload: function () {
            if (this._fileUploadDialog) {
                this._fileUploadDialog.close();
            }
        },

        /**
         * Handler for dialog close
         */
        onFileUploadDialogClose: function () {
            // Reset dialog state
            var oFileUploader = this.byId("fileUploader");
            var oProgressIndicator = this.byId("uploadProgress");
            var oMessageStrip = this.byId("uploadMessage");
            var oUploadButton = this.byId("uploadButton");
            var oTitleInput = this.byId("fileTitleInput");
            
            if (oFileUploader) {
                oFileUploader.clear();
            }
            
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(0);
                oProgressIndicator.setDisplayValue("Ready to upload");
                oProgressIndicator.setState("None");
                oProgressIndicator.setVisible(false);
            }
            
            if (oMessageStrip) {
                oMessageStrip.setVisible(false);
            }
            
            if (oUploadButton) {
                oUploadButton.setEnabled(false);
            }
            
            if (oTitleInput) {
                oTitleInput.setValue("");
            }
            
            // Clear progress interval
            if (this._uploadProgressInterval) {
                clearInterval(this._uploadProgressInterval);
                this._uploadProgressInterval = null;
            }
        }
    });
});
