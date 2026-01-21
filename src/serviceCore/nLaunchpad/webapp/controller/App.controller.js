sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (Controller, JSONModel, MessageToast, MessageBox) {
    "use strict";

    return Controller.extend("nLaunchpad.webapp.controller.App", {

        onInit: function () {
            // Launchpad state model
            var oLaunchpadModel = new JSONModel({
                busy: false,
                onlineCount: 0,
                totalCount: 8,
                searchQuery: ""
            });
            this.getView().setModel(oLaunchpadModel, "launchpad");

            // Apps status model
            var oAppsModel = new JSONModel({
                nCode: { online: true, port: 8087 },
                nAudioLab: { online: true, port: 8088 },
                nExtract: { online: true, port: 8089 },
                nLeanProof: { online: true, port: 8085 },
                nOpenaiServer: { online: true, port: 8081 },
                nHyperBook: { online: true, port: 8090 },
                nWorkflow: { online: true, port: 8082 },
                nWebServe: { online: true, port: 8080 }
            });
            this.getView().setModel(oAppsModel, "apps");

            // Check service status on init
            this._checkAllServicesStatus();
        },

        // --- Tile Press Handler ---
        onTilePress: function (oEvent) {
            var oTile = oEvent.getSource();
            var aCustomData = oTile.getCustomData();
            var sAppId = "", sUrl = "";

            aCustomData.forEach(function (oData) {
                if (oData.getKey() === "appId") sAppId = oData.getValue();
                if (oData.getKey() === "url") sUrl = oData.getValue();
            });

            var oAppsModel = this.getView().getModel("apps");
            var bOnline = oAppsModel.getProperty("/" + sAppId + "/online");

            if (bOnline) {
                MessageToast.show("Opening " + sAppId + "...");
                window.open(sUrl, "_blank");
            } else {
                MessageBox.warning("Service " + sAppId + " is currently offline. Please try again later.");
            }
        },

        // --- Search Functionality ---
        onSearchLiveChange: function (oEvent) {
            var sQuery = oEvent.getParameter("newValue").toLowerCase();
            this.getView().getModel("launchpad").setProperty("/searchQuery", sQuery);
            this._filterTiles(sQuery);
        },

        onSearch: function (oEvent) {
            var sQuery = oEvent.getParameter("query").toLowerCase();
            this._filterTiles(sQuery);
        },

        _filterTiles: function (sQuery) {
            var aPanels = this.byId("mainPage").getContent()[0].getContent();

            aPanels.forEach(function (oPanel) {
                if (oPanel.getMetadata().getName() === "sap.m.Panel") {
                    var oGridContainer = oPanel.getContent()[0];
                    if (oGridContainer && oGridContainer.getItems) {
                        var aItems = oGridContainer.getItems();
                        var bHasVisibleTile = false;

                        aItems.forEach(function (oTile) {
                            var sHeader = (oTile.getHeader() || "").toLowerCase();
                            var sSubheader = (oTile.getSubheader() || "").toLowerCase();
                            var bVisible = !sQuery ||
                                           sHeader.indexOf(sQuery) !== -1 ||
                                           sSubheader.indexOf(sQuery) !== -1;
                            oTile.setVisible(bVisible);
                            if (bVisible) bHasVisibleTile = true;
                        });

                        oPanel.setVisible(bHasVisibleTile || !sQuery);
                    }
                }
            });
        },

        // --- Service Status Check ---
        onRefreshStatus: function () {
            MessageToast.show("Refreshing service status...");
            this._checkAllServicesStatus();
        },

        _checkAllServicesStatus: function () {
            var that = this;
            var oLaunchpadModel = this.getView().getModel("launchpad");
            var oAppsModel = this.getView().getModel("apps");

            oLaunchpadModel.setProperty("/busy", true);

            var aApps = ["nCode", "nAudioLab", "nExtract", "nLeanProof", "nOpenaiServer", "nHyperBook", "nWorkflow", "nWebServe"];
            var iOnlineCount = 0;

            // For demo purposes, assume all services are online
            // In production, would check each service's health endpoint
            aApps.forEach(function (sAppId) {
                var iPort = oAppsModel.getProperty("/" + sAppId + "/port");
                that._checkServiceHealth(sAppId, iPort).then(function (bOnline) {
                    oAppsModel.setProperty("/" + sAppId + "/online", bOnline);
                    if (bOnline) iOnlineCount++;
                    oLaunchpadModel.setProperty("/onlineCount", iOnlineCount);
                });
            });

            setTimeout(function () {
                oLaunchpadModel.setProperty("/busy", false);
            }, 1000);
        },

        _checkServiceHealth: function (sAppId, iPort) {
            // In production, this would make actual health check requests
            // For now, assume all services are online (simulated)
            return new Promise(function (resolve) {
                setTimeout(function () {
                    // Simulate 80% online rate
                    resolve(Math.random() > 0.2);
                }, 100 + Math.random() * 200);
            });
        },

        // --- Settings ---
        onSettingsPress: function () {
            MessageToast.show("Settings dialog coming soon...");
        }
    });
});
