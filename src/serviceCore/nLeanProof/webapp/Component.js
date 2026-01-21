sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "nLeanProof/webapp/model/models"
], function (UIComponent, Device, models) {
    "use strict";

    return UIComponent.extend("nLeanProof.webapp.Component", {
        metadata: {
            manifest: "json"
        },

        init: function () {
            // call the base component's init function
            UIComponent.prototype.init.apply(this, arguments);

            // Register custom Lean4 ACE editor mode
            this._registerLean4Mode();

            // enable routing
            this.getRouter().initialize();

            // set the device model
            this.setModel(models.createDeviceModel(), "device");
        },

        /**
         * Register the custom Lean4 syntax highlighting mode for ACE editor
         * This mode is used by the SAPUI5 CodeEditor control
         */
        _registerLean4Mode: function () {
            var sModulePath = sap.ui.require.toUrl("nLeanProof/webapp/lib/ace/mode-lean4.js");

            // Load the custom mode script
            var oScript = document.createElement("script");
            oScript.src = sModulePath;
            oScript.async = true;
            oScript.onload = function () {
                console.log("Lean4 ACE mode loaded successfully");
            };
            oScript.onerror = function () {
                console.warn("Failed to load Lean4 ACE mode");
            };
            document.head.appendChild(oScript);
        }
    });
});
