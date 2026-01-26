sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast"
], function (Controller, History, JSONModel, MessageToast) {
    "use strict";

    return Controller.extend("galaxy.sim.controller.Lineage", {

        onInit: function () {
            // Initialize model with default values
            var oModel = new JSONModel({
                events: [],
                newEvent: {
                    eventType: "START",
                    jobName: "",
                    jobNamespace: "default",
                    runId: "",
                    inputs: [],
                    outputs: []
                },
                viewType: "Graph"
            });

            this.getView().setModel(oModel);
        },

        onNavBack: function () {
            var oHistory = History.getInstance();
            var sPreviousHash = oHistory.getPreviousHash();

            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                var oRouter = this.getOwnerComponent().getRouter();
                oRouter.navTo("home", {}, true);
            }
        },

        onBreadcrumbHome: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("home", {}, true);
        },

        onCanvasReady: function () {
            // Placeholder for lineage graph initialization
            // Will initialize D3.js or similar graph library here
            var oCanvas = document.getElementById("lineage-canvas");
            if (!oCanvas) {
                console.warn("Lineage canvas not found");
                return;
            }

            // Initialize graph rendering when library is available
            console.log("Lineage canvas ready for graph initialization");
        },

        onZoomIn: function () {
            MessageToast.show("Zoom In - Graph zoom will be implemented");
        },

        onZoomOut: function () {
            MessageToast.show("Zoom Out - Graph zoom will be implemented");
        },

        onZoomFit: function () {
            MessageToast.show("Fit to View - Graph fit will be implemented");
        },

        onViewChange: function (oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            var oModel = this.getView().getModel();
            oModel.setProperty("/viewType", sSelectedKey);

            if (sSelectedKey === "Graph") {
                MessageToast.show("Switched to Graph View");
            } else {
                MessageToast.show("Switched to Timeline View");
            }
        },

        onAddEvent: function () {
            var oModel = this.getView().getModel();
            var oNewEvent = oModel.getProperty("/newEvent");
            var aEvents = oModel.getProperty("/events");

            // Validate required fields
            if (!oNewEvent.jobName) {
                MessageToast.show("Job Name is required");
                return;
            }

            // Create event with timestamp and generated runId if not provided
            var oEvent = {
                eventTime: new Date().toISOString(),
                eventType: oNewEvent.eventType,
                job: {
                    namespace: oNewEvent.jobNamespace,
                    name: oNewEvent.jobName
                },
                run: {
                    runId: oNewEvent.runId || this._generateUUID()
                },
                inputs: oNewEvent.inputs || [],
                outputs: oNewEvent.outputs || [],
                producer: "galaxy-sim-lineage-ui"
            };

            // Add to events array
            aEvents.push(oEvent);
            oModel.setProperty("/events", aEvents);

            // Reset new event form
            oModel.setProperty("/newEvent", {
                eventType: "START",
                jobName: "",
                jobNamespace: "default",
                runId: "",
                inputs: [],
                outputs: []
            });

            MessageToast.show("Event added successfully");
        },

        onImportLineage: function () {
            MessageToast.show("Import Lineage - File picker will be implemented");
        },

        onExportJSON: function () {
            var oModel = this.getView().getModel();
            var aEvents = oModel.getProperty("/events");

            if (aEvents.length === 0) {
                MessageToast.show("No events to export");
                return;
            }

            var sJson = JSON.stringify(aEvents, null, 2);
            this._downloadFile(sJson, "lineage-events.json", "application/json");
            MessageToast.show("Events exported as JSON");
        },

        onExportOpenLineage: function () {
            var oModel = this.getView().getModel();
            var aEvents = oModel.getProperty("/events");

            if (aEvents.length === 0) {
                MessageToast.show("No events to export");
                return;
            }

            // Format as OpenLineage spec
            var aOpenLineageEvents = aEvents.map(function (oEvent) {
                return {
                    eventTime: oEvent.eventTime,
                    eventType: oEvent.eventType,
                    job: oEvent.job,
                    run: oEvent.run,
                    inputs: oEvent.inputs,
                    outputs: oEvent.outputs,
                    producer: oEvent.producer,
                    schemaURL: "https://openlineage.io/spec/1-0-5/OpenLineage.json#/definitions/RunEvent"
                };
            });

            var sJson = JSON.stringify(aOpenLineageEvents, null, 2);
            this._downloadFile(sJson, "openlineage-events.json", "application/json");
            MessageToast.show("Events exported in OpenLineage format");
        },

        onClearAll: function () {
            var oModel = this.getView().getModel();
            oModel.setProperty("/events", []);
            MessageToast.show("All events cleared");
        },

        _generateUUID: function () {
            return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
                var r = Math.random() * 16 | 0;
                var v = c === "x" ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        },

        _downloadFile: function (sContent, sFileName, sMimeType) {
            var oBlob = new Blob([sContent], { type: sMimeType });
            var sUrl = URL.createObjectURL(oBlob);
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = sFileName;
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            URL.revokeObjectURL(sUrl);
        }
    });
});

