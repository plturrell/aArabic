sap.ui.define([
    "nAudioLab/webapp/controller/BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "nAudioLab/webapp/service/ServiceWrapper"
], function (BaseController, JSONModel, MessageToast, ServiceWrapper) {
    "use strict";

    return BaseController.extend("nAudioLab.webapp.controller.Detail", {
        onInit: function () {
            var oRouter = this.getRouter();
            oRouter.getRoute("detail").attachPatternMatched(this._onObjectMatched, this);
            this.setModel(new JSONModel({ 
                title: "Loading...", 
                isPlaying: false, 
                currentTime: 0,
                duration: 100,
                timeLabel: "00:00 / 03:45",
                busy: false
            }), "player");
        },

        _onObjectMatched: function (oEvent) {
            this.getOwnerComponent().getModel("appView").setProperty("/layout", "TwoColumnsMidExpanded");
            var sId = oEvent.getParameter("arguments").projectId;
            this._loadProject(sId);
        },

        _loadProject: function(sId) {
             var oModel = this.getModel("player");
             oModel.setProperty("/busy", true);
             
             ServiceWrapper.getProject(sId)
                .then(oProj => {
                    oModel.setProperty("/title", oProj.title);
                    // Reset player state
                    oModel.setProperty("/isPlaying", false);
                    oModel.setProperty("/currentTime", 0);
                    oModel.setProperty("/timeLabel", "00:00 / " + oProj.duration);
                    clearInterval(this._timer);
                })
                .finally(() => oModel.setProperty("/busy", false));
        },

        onPlayPause: function () {
            var oModel = this.getModel("player");
            var bPlaying = oModel.getProperty("/isPlaying");

            if (bPlaying) {
                clearInterval(this._timer);
                oModel.setProperty("/isPlaying", false);
                MessageToast.show("Paused");
            } else {
                oModel.setProperty("/isPlaying", true);
                MessageToast.show("Playing");
                this._timer = setInterval(() => {
                    var iCurrent = oModel.getProperty("/currentTime");
                    if (iCurrent >= 100) iCurrent = 0;
                    else iCurrent++;
                    
                    oModel.setProperty("/currentTime", iCurrent);
                }, 100);
            }
        },

        onFullScreen: function () {
             var oModel = this.getOwnerComponent().getModel("appView");
             var sLayout = oModel.getProperty("/layout");
             oModel.setProperty("/layout", sLayout === "MidColumnFullScreen" ? "TwoColumnsMidExpanded" : "MidColumnFullScreen");
        },

        onCloseDetail: function () {
            clearInterval(this._timer);
            this.getModel("player").setProperty("/isPlaying", false);
            this.getOwnerComponent().getModel("appView").setProperty("/layout", "OneColumn");
            this.getRouter().navTo("master");
        },
        
        onExit: function() {
            clearInterval(this._timer);
        }
    });
});
