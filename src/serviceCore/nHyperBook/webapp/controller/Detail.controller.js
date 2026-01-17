sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/format/DateFormat",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/m/Dialog",
    "sap/m/TextArea",
    "sap/m/Button"
], function (Controller, DateFormat, MessageBox, MessageToast, Dialog, TextArea, Button) {
    "use strict";

    return Controller.extend("hypershimmy.controller.Detail", {
        
        formatter: {
            /**
             * Format date/time for display
             * @param {string} sDate the date string
             * @returns {string} formatted date/time
             */
            formatDateTime: function (sDate) {
                if (!sDate) return "";
                var oDateFormat = DateFormat.getDateTimeInstance({
                    pattern: "MMM d, yyyy 'at' HH:mm"
                });
                return oDateFormat.format(new Date(sDate));
            },

            /**
             * Get status state for ObjectStatus
             * @param {string} sStatus the status
             * @returns {string} the state
             */
            statusState: function (sStatus) {
                switch (sStatus) {
                    case "Ready":
                        return "Success";
                    case "Processing":
                        return "Warning";
                    case "Failed":
                        return "Error";
                    default:
                        return "None";
                }
            }
        },

        /**
         * Called when the controller is instantiated
         */
        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("detail").attachPatternMatched(this._onRouteMatched, this);
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
        },

        /**
         * Handler for navigation back button
         */
        onNavBack: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("main");
        },

        /**
         * Handler for open chat button
         */
        onOpenChat: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            var sSourceId = this._currentSourceId;
            
            if (sSourceId) {
                oRouter.navTo("chat", {
                    sourceId: sSourceId
                });
            }
        },

        /**
         * Handler for show actions button
         */
        onShowActions: function () {
            MessageToast.show("Actions menu - to be implemented");
        },

        /**
         * Handler for show full content
         */
        onShowFullContent: function () {
            var oView = this.getView();
            var oContext = oView.getBindingContext();
            
            if (!oContext) {
                return;
            }
            
            var sContent = oContext.getProperty("Content");
            var sTitle = oContext.getProperty("Title");
            
            // Create dialog with full content
            var oDialog = new Dialog({
                title: sTitle,
                contentWidth: "60%",
                contentHeight: "80%",
                resizable: true,
                draggable: true,
                content: [
                    new TextArea({
                        value: sContent,
                        width: "100%",
                        height: "100%",
                        editable: false
                    })
                ],
                beginButton: new Button({
                    text: "Close",
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
         * Handler for generate summary
         */
        onGenerateSummary: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            var sSourceId = this._currentSourceId;
            
            if (sSourceId) {
                oRouter.navTo("summary", {
                    sourceId: sSourceId
                });
            }
        },

        /**
         * Handler for generate mindmap
         */
        onGenerateMindmap: function () {
            MessageToast.show("Mindmap generation will be implemented in Week 8");
        },

        /**
         * Handler for generate audio
         */
        onGenerateAudio: function () {
            MessageToast.show("Audio generation will be implemented in Week 9");
        },

        /**
         * Handler for delete source
         */
        onDeleteSource: function () {
            var oView = this.getView();
            var oContext = oView.getBindingContext();
            
            if (!oContext) {
                return;
            }
            
            var sTitle = oContext.getProperty("Title");
            
            MessageBox.confirm(
                "Are you sure you want to delete the source '" + sTitle + "'?",
                {
                    title: "Delete Source",
                    onClose: function (oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            // In a real implementation, this would delete via OData
                            MessageToast.show("Source deleted (mock)");
                            
                            // Navigate back to master
                            var oRouter = this.getOwnerComponent().getRouter();
                            oRouter.navTo("main");
                        }
                    }.bind(this)
                }
            );
        }
    });
});
